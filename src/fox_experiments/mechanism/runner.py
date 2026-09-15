"""Shared-acquisition controlled optimizer comparison."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
import torch
from .config import MechanismConfig
from .model import BindingFoX
from .tasks import templates, loss_function
from .diagnostics import snapshot, _numeric_diagnostics
from .io import _json, _csv, state_hash


def _parameter_groups(model, cfg):
    multipliers = {
        "q": cfg.alpha_m,
        "p": cfg.alpha_m,
        "u": cfg.alpha_g,
        "v": cfg.alpha_g,
        "z": cfg.alpha_g,
        "x": cfg.alpha_x,
        "w": cfg.alpha_w,
    }
    return [
        dict(
            params=[parameter],
            name=name,
            multiplier=multipliers.get(name, 1.0),
            lr=cfg.table_lr if name in ("Q", "K") else 0.001,
        )
        for name, parameter in model.named_parameters()
    ]


def run_mechanism(
    out_dir,
    profile="smoke",
    device="cpu",
    seeds=(0,),
    recall_lags=(2,),
    config_overrides=None,
    preset="practical",
):
    """Run six paired branches per seed/R; return artifact path strings.

    SGD here is full-batch gradient descent implemented by torch.optim.SGD.
    It intentionally omits the theorem's impractical stochastic batch schedule.
    All continuation optimizer moments reset at the common acquired checkpoint;
    gate reparameterization and optimizer choice are the paired interventions.
    """
    if profile not in ("smoke", "pilot"):
        raise ValueError("profile must be smoke or pilot")
    if preset not in ("practical", "appendix_s_native_probe"):
        raise ValueError("Unknown preset")
    cfg = MechanismConfig()
    if profile == "smoke":
        cfg = replace(
            cfg,
            n_keys=4,
            width=8,
            acquisition_steps=30,
            continuation_steps=20,
            log_every=10,
            radius_cap=64,
        )
    if preset == "appendix_s_native_probe":
        cfg = replace(cfg, eps0=1e-18, eps_decay=2.0, beta1=0.1, beta2=0.1)
    cfg = replace(cfg, **(config_overrides or {}))
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already contains a run; choose a new output directory."
        )
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config_record = dict(
        config=asdict(cfg),
        seeds=list(seeds),
        recall_lags=list(recall_lags),
        profile=profile,
        preset=preset,
        authoritative_source="learning_with_cot-3.pdf, Sections 2--6; Appendix S pp.124--125",
        protocol="Native finite transfer: common Adam acquisition, reset branch moments, matched functions; full-batch SGD/GD",
        schedule="eta=base*(1+j/tau)^(-.75); nu=table_lr*(1+j/tau)^(-2); Adam epsilon=eps0*exp(-eps_decay*j)",
        differences_from_appendix_S=[
            "n_keys/width and Gaussian scale downscaled",
            "300 practical Adam acquisition steps instead of three theorem-inspired updates",
            "reset moments at common fork (source Appendix S retains acquisition histories)",
            "native float64 rather than signed-log gradients/moments",
            "full pair exposure for all arms, not increasing stochastic SGD pair batches",
            "practical eps0=1e-8 and slower decay; native probe reuses 1e-18*exp(-2j) with j reset at fork",
            "R>2 changes the objective; no proved R-dependent cutoff is asserted",
        ],
        gate_rate_note="Same scalar multiplier alpha_g in both parameterizations; effective gate update speeds differ by design.",
        optional_standard_moments='config_overrides={"beta1": 0.9, "beta2": 0.99} runs a standard-moment sensitivity control.',
        torch_version=torch.__version__,
    )
    _json(out / "config.json", config_record)
    acquisition_rows, trace_rows, summary_rows, failure_rows = [], [], [], []
    for seed in seeds:
        scfg = replace(cfg, seed=seed)
        for recall_lag in recall_lags:
            model = BindingFoX(scfg, "factorized", device)
            data = templates(model, recall_lag)
            acquire = torch.optim.Adam(
                model.parameters(),
                lr=scfg.acquisition_lr,
                betas=(scfg.beta1, scfg.beta2),
                eps=scfg.eps0,
            )
            acquisition_failure = None
            for step in range(scfg.acquisition_steps):
                acquire.zero_grad(set_to_none=True)
                loss = loss_function(model, data)
                loss.backward()
                if not torch.isfinite(loss) or any(
                    not torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    acquisition_failure = "nonfinite_acquisition_loss_or_gradient"
                    break
                acquire.step()
            with torch.no_grad():
                gaps = model.gaps().clone()
                gaps.fill_diagonal_(torch.inf)
                pair_index = int(gaps.argmin())
                fixed_pair = (pair_index // scfg.n_keys, pair_index % scfg.n_keys)
            acquired = snapshot(model, data, scfg.acquisition_steps, 0.0, fixed_pair)
            acquired.update(
                seed=seed,
                recall_lag=recall_lag,
                parameter_hash=state_hash(model),
                qualified=acquired["delta_min"] > 0 and acquired["w"] > 0,
                acquisition_failure=acquisition_failure,
            )
            acquisition_rows.append(acquired)
            torch.save(
                dict(
                    model=model.state_dict(),
                    config=asdict(scfg),
                    optimizer=acquire.state_dict(),
                ),
                checkpoints / f"acquired_seed{seed}_R{recall_lag}.pt",
            )
            if acquisition_failure:
                failure_rows.append(
                    dict(seed=seed, recall_lag=recall_lag, reason=acquisition_failure)
                )
                continue
            for gate in ("factorized", "direct"):
                for arm in ("sgd", "adam_fixed", "adam_annealed"):
                    branch = model.converted(gate)
                    bdata = templates(branch, recall_lag)
                    groups = _parameter_groups(branch, scfg)
                    if arm == "sgd":
                        optimizer = torch.optim.SGD(
                            groups, momentum=0.0, weight_decay=0.0
                        )
                    else:
                        optimizer = torch.optim.Adam(
                            groups,
                            betas=(scfg.beta1, scfg.beta2),
                            eps=scfg.eps0,
                            weight_decay=0.0,
                            foreach=False,
                        )
                    initial_tables = (
                        branch.Q.detach().clone(),
                        branch.K.detach().clone(),
                    )
                    tag = dict(
                        seed=seed,
                        recall_lag=recall_lag,
                        gate=gate,
                        optimizer=arm,
                        objective_status=(
                            "original_R2"
                            if recall_lag == 2
                            else "exploratory_R_extension"
                        ),
                        unproved_candidate_boundary=recall_lag + 2,
                    )
                    first = snapshot(
                        branch, bdata, 0, 0.0, fixed_pair, initial_tables=initial_tables
                    )
                    first.update(tag)
                    first["candidate_balance_residual"] = (
                        (
                            first["m"] * first["delta_min"]
                            - (recall_lag + 1) * first["h"]
                            - math.log(first["m"])
                        )
                        if first["m"] > 0
                        else None
                    )
                    trace_rows.append(first)
                    cumulative_rate, stop_reason, completed = 0.0, None, 0
                    for j in range(scfg.continuation_steps):
                        base = scfg.sgd_lr if arm == "sgd" else scfg.adam_lr
                        eta = base * (1 + j / scfg.tau) ** -scfg.power
                        eps = (
                            scfg.eps0 * math.exp(-scfg.eps_decay * j)
                            if arm == "adam_annealed"
                            else scfg.eps0
                        )
                        if arm != "sgd" and eps == 0.0:
                            stop_reason = "scheduled_epsilon_underflow_to_zero"
                            break
                        for group in optimizer.param_groups:
                            group["lr"] = (
                                scfg.table_lr * (1 + j / scfg.tau) ** -scfg.table_power
                                if group["name"] in ("Q", "K")
                                else eta * group["multiplier"]
                            )
                            if arm != "sgd":
                                group["eps"] = eps
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(branch, bdata)
                        loss.backward()
                        if not torch.isfinite(loss) or any(
                            not torch.isfinite(p.grad).all()
                            for p in branch.parameters()
                        ):
                            stop_reason = "nonfinite_loss_or_gradient"
                            break
                        numeric = _numeric_diagnostics(branch, optimizer)
                        if numeric["gradient_zero_fraction"] == 1.0:
                            stop_reason = "all_native_gradients_zero"
                            break
                        if arm != "sgd" and numeric["gradient_square_underflow_count"]:
                            stop_reason = "native_gradient_square_underflow"
                            break
                        optimizer.step()
                        cumulative_rate += eta
                        completed = j + 1
                        if any(
                            not torch.isfinite(p).all() for p in branch.parameters()
                        ):
                            stop_reason = "nonfinite_parameter_after_update"
                            break
                        if (
                            completed % scfg.log_every == 0
                            or completed == scfg.continuation_steps
                        ):
                            row = snapshot(
                                branch,
                                bdata,
                                completed,
                                cumulative_rate,
                                fixed_pair,
                                optimizer,
                                initial_tables,
                            )
                            row.update(tag, base_lr=eta, epsilon=eps)
                            row["candidate_balance_residual"] = (
                                (
                                    row["m"] * row["delta_min"]
                                    - (recall_lag + 1) * row["h"]
                                    - math.log(row["m"])
                                )
                                if row["m"] > 0
                                else None
                            )
                            trace_rows.append(row)
                    final = snapshot(
                        branch,
                        bdata,
                        completed,
                        cumulative_rate,
                        fixed_pair,
                        optimizer,
                        initial_tables,
                    )
                    final.update(
                        tag,
                        stop_reason=stop_reason,
                        requested_steps=scfg.continuation_steps,
                        status=(
                            "numerical_stop"
                            if stop_reason
                            else "finite_budget_complete"
                        ),
                        acquisition_qualified=acquired["qualified"],
                    )
                    final["candidate_balance_residual"] = (
                        (
                            final["m"] * final["delta_min"]
                            - (recall_lag + 1) * final["h"]
                            - math.log(final["m"])
                        )
                        if final["m"] > 0
                        else None
                    )
                    summary_rows.append(final)
                    failure_rows.append(
                        dict(
                            **tag,
                            completed_steps=completed,
                            reason=stop_reason,
                            acquisition_qualified=acquired["qualified"],
                            interpretation="Finite budget ending never implies asymptotic convergence.",
                        )
                    )
                    torch.save(
                        dict(
                            model=branch.state_dict(),
                            optimizer=optimizer.state_dict(),
                            config=asdict(scfg),
                            gate=gate,
                            step=completed,
                            S=cumulative_rate,
                            fixed_pair=fixed_pair,
                            stop_reason=stop_reason,
                        ),
                        checkpoints / f"seed{seed}_R{recall_lag}_{gate}_{arm}.pt",
                    )
    _csv(out / "acquisition.csv", acquisition_rows)
    _csv(out / "traces.csv", trace_rows)
    _csv(out / "summary.csv", summary_rows)
    _json(out / "failures.json", failure_rows)
    return dict(
        config_json=str(out / "config.json"),
        acquisition_csv=str(out / "acquisition.csv"),
        traces_csv=str(out / "traces.csv"),
        summary_csv=str(out / "summary.csv"),
        failures_json=str(out / "failures.json"),
        checkpoint_dir=str(checkpoints),
    )
