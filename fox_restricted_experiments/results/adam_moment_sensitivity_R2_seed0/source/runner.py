"""Train the paper's restricted answer model; save every finite-run outcome.

The three acquisition updates use the ordinary loss. Continuation is a
tractable polynomial schedule, not the existential SGD schedule in the proof.
Adam keeps its entire moment history and uses the full pair population.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

try:
    from .core import (Config, Model, LogAdam, acquisition_rates, gradient_step,
                       rates_at, native_gradients, boundary_log_odds, export_dataset)
except ImportError:
    from core import (Config, Model, LogAdam, acquisition_rates, gradient_step,
                      rates_at, native_gradients, boundary_log_odds, export_dataset)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def dump_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def fingerprint(params):
    digest = hashlib.sha256()
    for value in params:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate(c):
    if c.n < 2 or c.d < 1 or c.R < 2 or c.steps < 3:
        raise ValueError("Use n>=2, d>=1, integer R>=2, and steps>=3.")
    if any(type(getattr(c, k)) is not int for k in ("n", "d", "R", "steps", "pair_batch")):
        raise ValueError("Dimensions, lag, steps and pair batch must be integers.")
    if not 0 <= c.beta1 < 1 or not c.beta1**2 < c.beta2 < 1:
        raise ValueError("Adam requires 0<=beta1<1 and beta1**2<beta2<1.")
    if not 2/3 < c.power <= 1 or c.table_power <= 1:
        raise ValueError("Use scalar power in (2/3,1] and summable table power>1.")
    if not .5 < c.kcal < 1 or min(c.wo, c.wc) <= 0 or not math.isclose(c.kcal+c.wo+c.wc, 1):
        raise ValueError("Positive O/C weights and calibration>.5 must sum to one.")
    positive = ("sigma", "gamma", "c0", "q0", "u0", "h0", "am", "ag", "ax", "aw",
                "lr_sgd", "lr_adam", "offset", "table_lr", "eps_decay", "pair_batch")
    if any(not math.isfinite(getattr(c, k)) or getattr(c, k) <= 0 for k in positive):
        raise ValueError("Initialization scales and learning-rate settings must be positive and finite.")
    if c.gamma > 1/16 or c.batch_growth < 0:
        raise ValueError("Use acquisition gamma<=1/16 and nonnegative batch growth.")
    if c.ag**2 <= c.c0*c.am**2:
        raise ValueError("Binder multipliers must satisfy ag**2 > c0*am**2.")


def audit_native(model):
    """Compare independent native autodiff and analytical signed-log gradients."""
    actual, native_info = native_gradients(model)
    signed, info = model.gradients()
    errors = []
    for (sign, logabs), reference in zip(signed, actual):
        reconstructed = sign * logabs.exp()
        denominator = max(float(reference.abs().max()), 1e-300)
        errors.append(float((reconstructed-reference).abs().max())/denominator)
    native_logloss = float(native_info.get("logloss", native_info.get("log_loss")))
    return dict(relative_max_gradient_error_by_block=errors,
                absolute_log_loss_error=abs(float(info["logloss"])-native_logloss),
                passed=max(errors) < 2e-8 and abs(float(info["logloss"])-native_logloss) < 1e-10)


@torch.no_grad()
def snapshot(model, tags, step, S, initial_tables, examples, elapsed):
    q,p,u,v,x,w,m,g,h,rho,lrho = [float(v) for v in model.quantities()]
    gaps = model.gaps()[model.mask]
    delta = float(gaps.min())
    logloss = float(model.gradients()[1]["logloss"])
    _,_,_,so,sc = model.rows(gaps)
    norms = max(float(model.Q.norm(dim=1).max()), float(model.K.norm(dim=1).max()))
    movement = max(float((a-b).norm(dim=1).max()) for a,b in zip(model.params()[:2], initial_tables))
    return dict(**tags, step=step, S=S, examples=examples, elapsed_seconds=elapsed,
                log_loss=logloss, loss=math.exp(logloss), q=q,p=p,u=u,v=v,x=x,w=w,
                m=m,g=g,h=h,rho=rho,log_rho=lrho,m_rho=m*rho,
                delta_min=delta,delta_max=float(gaps.max()),row_norm_max=norms,
                table_movement=movement, content_horizon=1+m*delta/h,
                sg_balance=m*delta-(model.cfg.R+1)*h-math.log(m) if m>0 else None,
                train_overwrite_min_probability=float(torch.sigmoid(w*so).min()),
                train_recall_min_probability=float(torch.sigmoid(w*sc).min()),
                calibration_probability=float(torch.sigmoid(model.theta[-1])),
                certificate_valid=norms<=1 and delta>0 and w>=0 and m>=0,
                pair_count=model.cfg.n*(model.cfg.n-1), train_R=model.cfg.R)


@torch.no_grad()
def evaluate(model, tags, step, S, lags, prefixes):
    """Exhaustive key-pair witness curves, plus an arbitrary-prefix certificate.

    Probe: (a,-1)^P, (a,+1), (b,-1)^(r-1), ?a; complements have
    exactly the same correct-answer probability. This is not all streams.
    """
    _,_,_,_,_,w,m,_,h,rho,_ = model.quantities()
    gaps = model.gaps()[model.mask]
    valid = (float(model.Q.norm(dim=1).max())<=1 and float(model.K.norm(dim=1).max())<=1
             and float(gaps.min())>0 and float(w)>=0 and float(m)>=0)
    rows = []
    for lag in lags:
        logbound = (4*m*rho + torch.logaddexp(-m*gaps.min()+(lag-1)*h, -h)
                    -torch.log(-torch.expm1(-h)))
        bound = float(torch.sigmoid(-w*torch.tanh(logbound/2))) if valid else None
        for prefix in prefixes:
            logodds = boundary_log_odds(gaps,m,h,rho,lag,prefix)
            attention = torch.sigmoid(-logodds)
            probability = torch.sigmoid(-w*torch.tanh(logodds/2))
            rows.append(dict(**tags,step=step,S=S,lag=lag,prefix=prefix,
                             mean_probability=float(probability.mean()),min_probability=float(probability.min()),
                             mean_attention=float(attention.mean()),min_attention=float(attention.min()),
                             certified_probability_lower_bound=bound,
                             example_count=2*len(gaps),train_R=model.cfg.R))
    return rows


def run_suite(cfg, out_dir, seeds=(0,1,2), gate_modes=("learned","retrieval_frozen"),
              optimizers=("sgd","adam"), eval_lags=tuple(range(1,33)),
              prefixes=(0,16,64), log_every=1000, progress=True):
    """Return output path. Never replace an existing experiment.

    `steps` includes three acquisition updates. Scalar clock S starts after
    them. Sampling RNG and initial tensors are paired across gate variants.
    Both optimizers start from the Gaussian draw, not an Adam-trained fork.
    """
    _validate(cfg)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s<0 for s in seeds):
        raise ValueError("Provide distinct nonnegative integer seeds.")
    if not gate_modes or not set(gate_modes)<=set(("learned","retrieval_frozen","both_frozen")):
        raise ValueError("Unknown or empty gate modes.")
    if not optimizers or not set(optimizers)<=set(("sgd","adam")):
        raise ValueError("Choose sgd and/or adam.")
    if len(set(gate_modes))!=len(gate_modes) or len(set(optimizers))!=len(optimizers):
        raise ValueError("Duplicate branches are not allowed.")
    if not eval_lags or not prefixes or log_every<1 or any(type(x) is not int or x<1 for x in eval_lags) or any(type(x) is not int or x<0 for x in prefixes):
        raise ValueError("Use positive integer evaluation lags/log period and nonnegative prefixes.")
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {out}. Choose a new run tag.")
    out.mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir()
    source_dir = Path(__file__).resolve().parent
    source_bytes = {p.name:p.read_bytes() for p in source_dir.glob("*.py")}
    hashes = {name:hashlib.sha256(data).hexdigest() for name,data in source_bytes.items()}
    (out/"source").mkdir()
    for name,data in source_bytes.items():
        (out/"source"/name).write_bytes(data)
    dump_json(out/"config.json",dict(config=asdict(cfg),seeds=seeds,gate_modes=gate_modes,
              optimizers=optimizers,eval_lags=eval_lags,prefixes=prefixes,log_every=log_every,
              protocol="three ordinary-answer acquisition updates; practical polynomial continuation; no moment reset",
              numerical_backend="float64 parameters; exact signed-log analytic gradients and bias-corrected Adam buffers",
              sgd_sampling="full population during acquisition; growing iid ordered-pair batches thereafter",
              adam_sampling="full ordered-pair population at every update",
              epsilon="sigma**3 * exp(-eps_decay*(global_step-1)); outside sqrt(v)",
              source_sha256=hashes,torch_version=torch.__version__,numpy_version=np.__version__,
              python_version=platform.python_version(),cuda_available=torch.cuda.is_available(),
              theorem_dimension_at_failure_probability_005=math.ceil(144*math.log(16*cfg.n*(cfg.n-1)/.05)),
              limits=["Finite schedules do not implement existential SGD continuation envelopes or R>2 basin preparation.",
                      "Frozen answer theorem is R=2; larger R is an empirical extension.",
                      "No holdout key pairs: this tests length/lag distribution shift on the same vocabulary.",
                      "Finite-prefix probe minima are not minima over all possible streams."]))
    export_dataset(cfg, out/"dataset")
    histories,probabilities,runs,audits = [],[],[],[]
    first_hash_by_seed = {}
    for seed in seeds:
        for gate_mode in gate_modes:
            for optimizer in optimizers:
                c = replace(cfg,seed=seed,gate_mode=gate_mode)
                model = Model(c)
                tags = dict(seed=seed,gate_mode=gate_mode,optimizer=optimizer)
                label = f"seed{seed}_{gate_mode}_{optimizer}"
                initial_hash = fingerprint(model.params())
                first_hash_by_seed.setdefault(seed, initial_hash)
                if initial_hash != first_hash_by_seed[seed]:
                    raise AssertionError("Paired initial parameters differ across arms.")
                initial_tables = [p.clone() for p in model.params()[:2]]
                frozen_initial = model.theta.clone()
                acquisition = acquisition_rates(c,optimizer)
                adam = LogAdam(model.params(),c.beta1,c.beta2) if optimizer=="adam" else None
                rng = np.random.default_rng(seed+104729)
                start = time.monotonic()
                S, examples, completed = 0.,0,0
                status, reason, acquired = "finite_budget_complete", "", None
                native = audit_native(model)
                audits.append(dict(**tags,step=0,**native))
                if not native["passed"]:
                    raise AssertionError(f"Initial native gradient audit failed: {label}: {native}")
                def record():
                    row = snapshot(model,tags,completed,S,initial_tables,examples,time.monotonic()-start)
                    histories.append(row)
                    probabilities.extend(evaluate(model,tags,completed,S,eval_lags,prefixes))
                    return row
                record()
                for step in range(1,c.steps+1):
                    if time.monotonic()-start > c.max_seconds:
                        status,reason = "time_budget_stop", "per-branch wall-clock limit"
                        break
                    if step<=3:
                        base,rates,counts,batch = 0.,acquisition[step-1],None,c.n*(c.n-1)
                    else:
                        tail = step-4
                        base,rates = rates_at(c,optimizer,tail)
                        counts = None
                        batch = c.n*(c.n-1)
                        if optimizer=="sgd":
                            batch = math.ceil(c.pair_batch*(1+tail/c.offset)**c.batch_growth)
                            counts = torch.as_tensor(rng.multinomial(batch,np.full(c.n*(c.n-1),1/(c.n*(c.n-1)))),
                                                     device=c.device,dtype=torch.float64)
                    gradients,info = model.gradients(counts)
                    finite = all(bool(torch.isfinite(s).all()) and bool(torch.isfinite(l[s!=0]).all())
                                 for s,l in gradients)
                    if not finite or not math.isfinite(float(info["logloss"])):
                        status,reason = "numerical_stop","nonfinite signed-log loss or active gradient"
                        break
                    if float(info.get("table_log_span",0)) > 650:
                        status,reason = "numerical_stop","raw-table gradient aggregation dynamic range exceeded 650 log units"
                        break
                    previous = [p.clone() for p in model.params()]
                    if adam:
                        adam.step(model.params(),gradients,rates,3*math.log(c.sigma)-c.eps_decay*(step-1))
                    else:
                        gradient_step(model,gradients,rates)
                    if not all(bool(torch.isfinite(p).all()) for p in model.params()):
                        for p,old in zip(model.params(),previous):
                            p.copy_(old)
                        status,reason = "numerical_stop","nonfinite parameter update; parameters restored, optimizer state invalid"
                        break
                    frozen_indices = getattr(model,"frozen_scalar_indices",())
                    if frozen_indices and not torch.equal(model.theta[list(frozen_indices)],frozen_initial[list(frozen_indices)]):
                        raise AssertionError("A frozen coordinate moved.")
                    S += float(base)
                    examples += 4*batch+2*c.n
                    completed = step
                    if step==3:
                        acquired = bool((model.gaps()[model.mask]>0).all()) and float(model.theta[-1])>0
                        native = audit_native(model)
                        audits.append(dict(**tags,step=3,**native))
                        if not native["passed"]:
                            raise AssertionError(f"Acquired native gradient audit failed: {label}: {native}")
                        if not acquired:
                            status,reason = "acquisition_failed","three updates did not acquire all positive raw gaps and decoder orientation"
                    if step==3 or step%log_every==0 or step==c.steps or status!="finite_budget_complete":
                        row = record()
                        if progress:
                            print(f"{label}: {step}/{c.steps}, loss={row['loss']:.3g}, min_gap={row['delta_min']:.4g}, "
                                  f"horizon={row['content_horizon']:.2f}, S={S:.3g}",flush=True)
                    if status!="finite_budget_complete":
                        break
                if histories[-1]["step"] != completed or any(histories[-1][k]!=v for k,v in tags.items()):
                    record()
                run = dict(**tags,completed_steps=completed,requested_steps=c.steps,status=status,reason=reason,
                           initial_hash=initial_hash,final_hash=fingerprint(model.params()),
                           acquired_positive_gaps=acquired,train_R=c.R,S=S)
                runs.append(run)
                checkpoint = dict(config=asdict(c),model=[p.cpu() for p in model.params()],
                                  optimizer=None if adam is None else dict(t=adam.t,b1=adam.b1,b2=adam.b2,
                                       ms=[p.cpu() for p in adam.ms],ml=[p.cpu() for p in adam.ml],vl=[p.cpu() for p in adam.vl]),
                                  rng_state=rng.bit_generator.state,summary=run,
                                  optimizer_state_valid="restored" not in reason)
                torch.save(checkpoint,out/"checkpoints"/(label+".pt"))
                write_csv(out/"history.csv",histories)
                write_csv(out/"lag_probabilities.csv",probabilities)
                write_csv(out/"runs.csv",runs)
                dump_json(out/"native_gradient_audits.json",audits)
                if progress:
                    print(f"{label}: {status}"+(f" ({reason})" if reason else ""),flush=True)
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True)
    parser.add_argument("--n",type=int,default=8)
    parser.add_argument("--R",type=int,default=2)
    parser.add_argument("--d",type=int,default=2048)
    parser.add_argument("--steps",type=int,default=12000)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--max-lag",type=int,default=32)
    parser.add_argument("--log-every",type=int,default=1000)
    args = parser.parse_args()
    if args.device=="cpu":
        torch.set_num_threads(1)
    cfg = Config(n=args.n,R=args.R,d=args.d,steps=args.steps,device=args.device)
    path = run_suite(cfg,args.out,seeds=tuple(args.seeds),eval_lags=tuple(range(1,args.max_lag+1)),log_every=args.log_every)
    try:
        from .report import make_report
    except ImportError:
        from report import make_report
    make_report(path)


if __name__=="__main__":
    main()
