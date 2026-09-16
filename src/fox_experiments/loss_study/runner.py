"""Paired objective experiments, with resumable updates and explicit diagnostics.

The answer-only and token-average objectives see exactly the same teacher-forced
sequence. Evaluation never feeds the answer token to the model. Checkpoints are
the source of truth; CSVs are rebuilt from them after an interruption or resume.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from fox_experiments.training.artifacts import (
    atomic_checkpoint,
    dump_json,
    exclusive_run_lock,
    state_hash,
    table,
)
from fox_experiments.training.optimizers import moment_diagnostics
from .config import branches
from .data import StudyData
from .model import build_model, model_diagnostics

POLICY = "paired-full-vocabulary-objective-study-v1"


def compute_losses(logits, batch, cfg):
    """Full-vocabulary CE; padding never contributes to either denominator.

    The weighted token mean is c*answer_only+(1-c)*other_tokens, where
    c=answer_weight*K/(N-K+answer_weight*K). Answer-only ignores answer_weight.
    """
    targets, mask = batch["targets"], batch["answer_mask"]
    if logits.shape[:2] != targets.shape or mask.shape != targets.shape:
        raise ValueError("Logits, targets and answer mask must share batch/time shape")
    valid = targets.ne(-100)
    answer = valid & mask.bool()
    other = valid & ~mask.bool()
    n, k = valid.sum(), answer.sum()
    if int(k) == 0:
        raise ValueError("Every training batch needs at least one valid answer")
    nll = F.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(targets)
    la = nll.masked_select(answer).sum() / k
    lo = nll.masked_select(other).sum() / other.sum().clamp_min(1)
    weighted_count = (n - k) + cfg.answer_weight * k
    coefficient = cfg.answer_weight * k / weighted_count
    return {
        "answer_only": la,
        "other_tokens": lo,
        "all_tokens": coefficient * la + (1 - coefficient) * lo,
        "unweighted_all_tokens": nll.masked_select(valid).mean(),
        "answer_fraction": k / n,
        "answer_coefficient": coefficient,
        "valid_count": n,
        "answer_count": k,
    }


def _category(name):
    if ".attn.gate." in name:
        return "first_gate" if "blocks.0." in name else "retrieval_gate"
    if any(part in name for part in ("output_head", "lm_head", "readout")):
        return "output"
    return "representation"


def make_optimizer(model, cfg, arm, lr):
    """Semantic parameter groups; native Adam uses coupled configured decay.

    paper_adamw is an explicitly bundled control: .9/.95 moments, epsilon1e-8,
    matrix-only .1 decoupled decay, unit multipliers and norm clipping at one.
    """
    grouped = {}
    multipliers = {
        "first_gate": cfg.first_gate_multiplier,
        "retrieval_gate": cfg.retrieval_gate_multiplier,
        "representation": cfg.representation_multiplier,
        "output": cfg.output_multiplier,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        category = _category(name)
        decay = (
            (0.1 if p.ndim >= 2 and "norm" not in name else 0.0)
            if arm == "paper_adamw"
            else cfg.weight_decay
        )
        grouped.setdefault((category, decay), []).append(p)
    groups = [
        {
            "params": params,
            "label": category + (f"_decay{decay}" if arm == "paper_adamw" else ""),
            "category": category,
            "base_multiplier": 1.0 if arm == "paper_adamw" else multipliers[category],
            "lr": lr * (1.0 if arm == "paper_adamw" else multipliers[category]),
            "weight_decay": decay,
        }
        for (category, decay), params in grouped.items()
    ]
    if arm == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0)
    if arm == "paper_adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if arm in ("adam_fixed", "adam_annealed"):
        return torch.optim.Adam(
            groups, lr=lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps0
        )
    raise ValueError(f"Unknown optimizer {arm}")


def schedule_optimizer(optimizer, cfg, arm, lr, step):
    """Set rates for zero-based update step and return base LR, epsilon.

    For the optional paper bundle, warmup occupies 10% of this finite run and
    is followed by cosine decay to zero. This is a control, not paper replication.
    Summable representations use TOTAL power representation_power, not its sum
    with the other groups' lr_power.
    """
    eps = 1e-8 if arm == "paper_adamw" else cfg.eps0
    if arm == "adam_annealed":
        eps *= math.exp(-cfg.epsilon_decay * step)
        if eps < np.finfo(np.float32).tiny:
            raise FloatingPointError("Annealed epsilon left normal fp32 range")
    factor = (1 + step / cfg.lr_offset) ** (-cfg.lr_power)
    if arm == "paper_adamw":
        warmup = max(1, int(math.ceil(cfg.steps * 0.1)))
        update = step + 1
        factor = (
            update / warmup
            if update <= warmup
            else 0.5
            * (1 + math.cos(math.pi * (update - warmup) / max(1, cfg.steps - warmup)))
        )
    for group in optimizer.param_groups:
        group_factor = factor
        if (
            arm != "paper_adamw"
            and group["category"] == "representation"
            and cfg.representation_schedule == "summable"
        ):
            group_factor = (1 + step / cfg.lr_offset) ** (-cfg.representation_power)
        group["lr"] = lr * group["base_multiplier"] * group_factor
        if arm in ("adam_fixed", "adam_annealed"):
            group["eps"] = eps
    return lr * factor, eps


def gradient_diagnostics(model, losses, objective):
    """Measure answer/non-answer components before the single actual backward.

    Norms are of data-loss gradients, before decay, clipping and Adam moments.
    Their weighted sum, not independently normalized Adam updates, is optimized.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    ga = torch.autograd.grad(
        losses["answer_only"], params, retain_graph=True, allow_unused=True
    )
    go = torch.autograd.grad(
        losses["other_tokens"], params, retain_graph=True, allow_unused=True
    )
    ca = 1.0 if objective == "answer_only" else float(losses["answer_coefficient"])
    co = 1 - ca
    groups = {
        category: []
        for category in (
            "all",
            "first_gate",
            "retrieval_gate",
            "output",
            "representation",
        )
    }
    for (name, _), a, o in zip(named, ga, go):
        groups["all"].append((a, o))
        groups[_category(name)].append((a, o))
    rows = []
    for category, pairs in groups.items():
        if not pairs:
            continue
        aa = sum(
            float(a.detach().double().square().sum()) for a, _ in pairs if a is not None
        )
        oo = sum(
            float(o.detach().double().square().sum()) for _, o in pairs if o is not None
        )
        dot = sum(
            float((a.detach().double() * o.detach().double()).sum())
            for a, o in pairs
            if a is not None and o is not None
        )
        an, on = math.sqrt(aa), math.sqrt(oo)
        rows.append(
            {
                "parameter_group": category,
                "answer_gradient_norm": an,
                "other_gradient_norm": on,
                "gradient_dot": dot,
                "gradient_cosine": dot / (an * on) if an * on else float("nan"),
                "answer_fraction": float(losses["answer_fraction"]),
                "answer_coefficient": ca,
                "other_coefficient": co,
                "weighted_answer_gradient_norm": ca * an,
                "weighted_other_gradient_norm": co * on,
                "combined_gradient_norm": math.sqrt(
                    max(0.0, ca * ca * aa + co * co * oo + 2 * ca * co * dot)
                ),
                "opposing_projection_ratio": (
                    -co * dot / (ca * aa) if ca * aa else float("nan")
                ),
            }
        )
    return rows


@torch.no_grad()
def evaluate(model, data, cfg, branch, step, split="short", device="cpu"):
    """Full-vocabulary next-token evaluation with answer excluded from inputs."""
    examples = data.evaluation_examples(split=split, seed=branch["seed"])
    rows, was_training = [], model.training
    model.eval()
    try:
        for start in range(0, len(examples), min(cfg.batch_size, 64)):
            chunk = examples[start : start + min(cfg.batch_size, 64)]
            lengths = [e.answer_index for e in chunk]
            x = torch.full(
                (len(chunk), max(lengths)), data.pad_id, dtype=torch.long, device=device
            )
            for i, ex in enumerate(chunk):
                x[i, : ex.answer_index] = torch.tensor(
                    ex.tokens[: ex.answer_index], device=device
                )
            logits = model(x, query_chunk=cfg.query_chunk)
            last = logits[
                torch.arange(len(chunk), device=device),
                torch.tensor(lengths, device=device) - 1,
            ].float()
            if not torch.isfinite(last).all():
                raise FloatingPointError("Nonfinite evaluation logits")
            target = torch.tensor(
                [e.tokens[e.answer_index] for e in chunk], device=device
            )
            logp = last.log_softmax(-1).gather(1, target[:, None]).squeeze(1)
            prediction = last.argmax(-1)
            for i, ex in enumerate(chunk):
                diagnostic = {}
                if ex.edit == "base" and ex.history_id == 0:
                    measured = model.query_diagnostics(
                        x[i : i + 1, : ex.answer_index],
                        [ex.target_position],
                        [ex.conflict_position],
                    )
                    diagnostic = {
                        "query_" + name: float(value.float().mean())
                        for name, value in measured.items()
                    }
                rows.append(
                    {
                        **branch,
                        "step": step,
                        "split": split,
                        "lag": ex.lag,
                        "actual_lag": ex.actual_lag,
                        "prefix": ex.prefix,
                        "actual_prefix": ex.metadata.get("actual_prefix", ex.prefix),
                        "token_lag_query": ex.answer_index - 1 - ex.target_position,
                        "stale_count": sum(
                            1
                            for key, _ in ex.metadata.get("records", [])[
                                : ex.metadata.get("target_record", 0)
                            ]
                            if key == ex.metadata.get("query_key")
                        ),
                        "example_sha256": hashlib.sha256(
                            np.asarray(ex.tokens, dtype="<i8").tobytes()
                        ).hexdigest(),
                        "history_id": ex.history_id,
                        "edit": ex.edit,
                        "template": ex.template,
                        "answer_index": ex.answer_index,
                        "target_position": ex.target_position,
                        "conflict_position": ex.conflict_position,
                        "correct": int(prediction[i] == target[i]),
                        "probability": float(logp[i].exp()),
                        "nll": float(-logp[i]),
                        "prediction_id": int(prediction[i]),
                        "answer_id": int(target[i]),
                        **diagnostic,
                    }
                )
    finally:
        model.train(was_training)
    return rows


def summarize_evaluations(rows, cfg):
    """Joint-edit short qualification and a radius on the TESTED grid only."""
    import pandas as pd

    if not rows:
        return []
    frame = pd.DataFrame(rows)
    result = []
    for (branch, step), raw in frame.groupby(["branch", "step"], sort=False):
        metadata = raw.iloc[0]
        joint = raw.groupby(
            ["split", "lag", "prefix", "history_id"], as_index=False
        ).agg(correct=("correct", "min"), probability=("probability", "min"))
        joint["confident"] = (joint.probability >= cfg.probability_threshold).astype(
            float
        )
        panels = joint.groupby(["split", "lag", "prefix"], as_index=False).agg(
            accuracy=("correct", "mean"),
            confidence=("confident", "mean"),
            histories=("history_id", "nunique"),
        )
        short = panels[panels.split == "short"]
        short_prefixes = sorted(
            {0, cfg.train_max_prefix}
            | {p for p in cfg.eval_prefixes if p <= cfg.train_max_prefix}
        )
        short_expected = {(l, p) for l in cfg.short_lags for p in short_prefixes}
        short_present = {
            (int(r.lag), int(r.prefix))
            for r in short.itertuples()
            if r.histories == cfg.eval_histories
        }
        complete = short_expected == short_present
        worst = float(short.accuracy.min()) if len(short) else float("nan")
        qualified = complete and worst >= cfg.qualification_threshold
        long = panels[panels.split == "long"]
        expected = {(l, p) for l in cfg.eval_lags for p in cfg.eval_prefixes}
        present = {
            (int(r.lag), int(r.prefix))
            for r in long.itertuples()
            if r.histories == cfg.eval_histories
        }
        long_complete = expected == present
        radius, next_failed, censored = float("nan"), float("nan"), False
        status = "short_unqualified" if complete else "short_incomplete"
        if qualified and not long_complete:
            status = "long_incomplete"
        elif qualified:
            radius = 0
            for lag in cfg.eval_lags:
                if (
                    float(long[long.lag == lag].accuracy.min())
                    < cfg.qualification_threshold
                ):
                    next_failed = lag
                    break
                radius = lag
            censored = radius == max(cfg.eval_lags)
            status = (
                "right_censored_at_grid_ceiling"
                if censored
                else "next_tested_lag_failed"
            )
        base_short = raw[(raw.split == "short") & (raw.edit == "base")]
        result.append(
            {
                "branch": branch,
                **{
                    k: metadata[k]
                    for k in ("seed", "gate_mode", "optimizer", "objective")
                },
                "step": int(step),
                "short_complete": bool(complete),
                "short_qualified": bool(qualified),
                "short_joint_accuracy_worst": worst,
                "short_joint_accuracy_mean": (
                    float(short.accuracy.mean()) if len(short) else float("nan")
                ),
                "short_base_accuracy": float(base_short.correct.mean()),
                "short_nll": float(raw[raw.split == "short"].nll.mean()),
                "short_confident_joint_fraction_worst": (
                    float(short.confidence.min()) if len(short) else float("nan")
                ),
                "long_complete": bool(long_complete),
                "tested_grid_radius": radius,
                "next_failed_tested_lag": next_failed,
                "grid_ceiling": max(cfg.eval_lags),
                "right_censored": bool(censored),
                "radius_status": status,
                "radius_direction": "higher is better; tested lag grid, all declared prefixes and edits",
                "qualification_threshold": cfg.qualification_threshold,
                "probability_threshold": cfg.probability_threshold,
                "software_only": cfg.profile == "smoke",
            }
        )
    return result


def _json_identity(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _source_fingerprints():
    package = Path(__file__).resolve().parents[1]
    files = sorted(
        set((package / "loss_study").glob("*.py"))
        | set((package / "models").glob("*.py"))
        | set((package / "data").glob("*.py"))
    )
    files.extend(
        [package / "training" / "optimizers.py", package / "training" / "artifacts.py"]
    )
    return {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def _nongate_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if ".attn.gate." not in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _save(path, model, optimizer, state):
    atomic_checkpoint(
        path,
        {
            **state,
            "model": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng": _rng_state(),
        },
    )


def _train_branch(cfg, data, branch, directory, device, signature, initial, resume):
    """Checkpoint at declared intervals; safe replay requires identical spec/runtime."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "last.pt"
    model = build_model(cfg, data, branch["gate_mode"], branch["seed"]).to(device)
    lr = cfg.branch_learning_rates.get(
        branch["branch"], cfg.learning_rates[branch["optimizer"]]
    )
    optimizer = make_optimizer(model, cfg, branch["optimizer"], lr)
    row_branch = {
        **branch,
        **initial,
        "data_sha256": data.fingerprint()["data_sha256"],
        "run_signature": signature,
    }
    state = {
        "signature": signature,
        "branch": branch,
        "initial": initial,
        "step": 0,
        "training": [],
        "gradients": [],
        "evaluation": [],
    }
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["signature"] != signature or saved["branch"] != branch:
            raise ValueError(
                "Checkpoint specification changed; use a fresh output directory"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer_state"])
        _restore_rng(saved["rng"])
        state = {k: saved[k] for k in state}
    else:
        torch.manual_seed(branch["seed"])
        np.random.seed(branch["seed"])
        random.seed(branch["seed"])
        # Persist a resumable initialization before any evaluation can fail.
        _save(path, model, optimizer, state)
    checkpoint_step = state["step"]
    expected_evaluation = (
        checkpoint_step == 0
        or checkpoint_step % cfg.eval_every == 0
        or checkpoint_step == cfg.steps
    )
    evaluated_splits = {
        row["split"] for row in state["evaluation"] if row["step"] == checkpoint_step
    }
    if expected_evaluation and evaluated_splits != {"short", "long"}:
        pending = []
        for split in ("short", "long"):
            pending.extend(
                evaluate(model, data, cfg, row_branch, checkpoint_step, split, device)
            )
        state["evaluation"] = [
            row for row in state["evaluation"] if row["step"] != checkpoint_step
        ] + pending
        _save(path, model, optimizer, state)
    for step in range(state["step"], cfg.steps):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = data.batch(step, branch["seed"], device)
        lr_now, eps_now = schedule_optimizer(
            optimizer, cfg, branch["optimizer"], lr, step
        )
        logits = model(batch["input_ids"], query_chunk=cfg.query_chunk)
        losses = compute_losses(logits, batch, cfg)
        loss = losses[branch["objective"]]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at update {step + 1}")
        if step == 0 or (step + 1) % cfg.diagnostic_every == 0 or step + 1 == cfg.steps:
            state["gradients"].extend(
                {**row_branch, "step": step + 1, **row}
                for row in gradient_diagnostics(model, losses, branch["objective"])
            )
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable
        ):
            raise FloatingPointError(f"Nonfinite gradient at update {step + 1}")
        clip = 1.0 if branch["optimizer"] == "paper_adamw" else cfg.clip_grad
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable, clip if clip is not None else float("inf")
        )
        before = [p.detach().clone() for p in trainable]
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in trainable):
            raise FloatingPointError(f"Nonfinite parameter at update {step + 1}")
        relative = max(
            float((p.detach() - old).norm() / old.norm().clamp_min(1e-8))
            for p, old in zip(trainable, before)
        )
        gate_relative = [
            float((p.detach() - old).norm() / old.norm().clamp_min(1e-8))
            for (name, p), old in zip(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad], before
            )
            if ".attn.gate." in name
        ]
        token_hash = hashlib.sha256(
            batch["input_ids"].detach().cpu().numpy().tobytes()
            + batch["targets"].detach().cpu().numpy().tobytes()
        ).hexdigest()
        row = {
            **row_branch,
            "step": step + 1,
            "loss": float(loss.detach()),
            **{k: float(v.detach()) for k, v in losses.items()},
            "all_tokens_answer_coefficient": float(
                losses["answer_coefficient"].detach()
            ),
            "answer_coefficient": (
                1.0
                if branch["objective"] == "answer_only"
                else float(losses["answer_coefficient"].detach())
            ),
            "base_learning_rate": lr,
            "scheduled_base_learning_rate": lr_now,
            "epsilon": eps_now,
            "unclipped_gradient_norm": float(grad_norm),
            "clip_grad": clip,
            "weight_decay": (
                cfg.weight_decay if branch["optimizer"] != "paper_adamw" else 0.1
            ),
            "max_relative_parameter_step": relative,
            "max_relative_gate_step": max(gate_relative, default=0.0),
            "gate_step_over_half": max(gate_relative, default=0.0) > 0.5,
            "batch_sha256": token_hash,
            "update_seconds": time.perf_counter() - start,
        }
        if step == 0 or (step + 1) % cfg.diagnostic_every == 0 or step + 1 == cfg.steps:
            row.update(moment_diagnostics(optimizer))
            row.update(model_diagnostics(model, batch))
        state["training"].append(row)
        state["step"] = step + 1
        needs_eval = (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps
        # Save before evaluation too: a failed long evaluation does not lose updates.
        if (step + 1) % cfg.checkpoint_every == 0 or needs_eval:
            _save(path, model, optimizer, state)
        if needs_eval:
            pending = []
            for split in ("short", "long"):
                pending.extend(
                    evaluate(model, data, cfg, row_branch, step + 1, split, device)
                )
            state["evaluation"].extend(pending)
            _save(path, model, optimizer, state)
    # Resume an evaluation failure after the last update without repeating updates.
    if not any(
        row["step"] == cfg.steps and row["split"] == "long"
        for row in state["evaluation"]
    ):
        pending = []
        for split in ("short", "long"):
            pending.extend(
                evaluate(model, data, cfg, row_branch, cfg.steps, split, device)
            )
        state["evaluation"].extend(pending)
        _save(path, model, optimizer, state)
    return state


def _write_artifacts(out, cfg, branch_status):
    training, gradients, evaluation = [], [], []
    for item in branch_status:
        path = out / item["branch"] / "last.pt"
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            training.extend(saved["training"])
            gradients.extend(saved["gradients"])
            evaluation.extend(saved["evaluation"])
            item["completed_steps"] = saved["step"]
    summaries = summarize_evaluations(evaluation, cfg)
    summary_keys = {row["branch"] for row in summaries}
    for item in branch_status:
        if item["branch"] not in summary_keys:
            summaries.append(
                {
                    **item,
                    "step": item.get("completed_steps", 0),
                    "short_qualified": False,
                    "tested_grid_radius": float("nan"),
                    "radius_status": "no_evaluation_available",
                    "software_only": cfg.profile == "smoke",
                }
            )
    lookup = {r["branch"]: r for r in branch_status}
    for row in summaries:
        row["branch_status"] = lookup[row["branch"]]["status"]
        row["failure"] = lookup[row["branch"]].get("failure", "")
    for name, rows, empty in (
        ("training", training, {"branch": "", "step": ""}),
        ("gradients", gradients, {"branch": "", "step": ""}),
        ("eval", evaluation, {"branch": "", "step": ""}),
        ("summary", summaries, {"branch": "", "step": ""}),
        ("branches", branch_status, {"branch": "", "status": ""}),
    ):
        table(out / f"{name}.csv", rows if rows else [empty])


def run_loss_study(
    config, out_dir, device="cpu", data_dir=None, resume=False, corpus=None
):
    """Run every configured paired branch; failures stay visible and resumable."""
    cfg, out = config, Path(out_dir)
    device = torch.device(device)
    data = StudyData(cfg, data_dir=data_dir, corpus=corpus)
    spec = _json_identity(
        {
            "policy": POLICY,
            "config": cfg.to_dict(),
            "sources": _source_fingerprints(),
            "data": data.fingerprint(),
            "device_type": device.type,
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
            },
            "initialization": "same non-gates and matched constant-gate function per seed",
            "evaluation": "full-vocabulary argmax, all applicable causal edits; finite tested lag grid",
            "software_only": cfg.profile == "smoke",
        }
    )
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    statuses, initial_cache = [], {}
    with exclusive_run_lock(out):
        spec_path = out / "run_specification.json"
        if spec_path.exists():
            if not resume:
                raise FileExistsError(
                    "Output already contains a run; use resume=True or a fresh directory"
                )
            if json.loads(spec_path.read_text()) != spec:
                raise ValueError(
                    "Run specification changed (config/source/runtime/data/device); use a fresh directory"
                )
        elif any(p.name != ".run.lock" for p in out.iterdir()):
            raise FileExistsError(
                "Output directory is not empty and has no matching run specification"
            )
        else:
            dump_json(spec_path, spec)
            dump_json(out / "config.json", cfg.to_dict())
            dump_json(out / "data_description.json", data.describe())
        for branch in branches(cfg):
            status = {
                **branch,
                "status": "running",
                "completed_steps": 0,
                "failure": "",
            }
            statuses.append(status)
            try:
                initial_model = build_model(
                    cfg, data, branch["gate_mode"], branch["seed"]
                )
                initial = {
                    "initial_state_hash": state_hash(initial_model),
                    "shared_non_gate_initial_hash": _nongate_hash(initial_model),
                }
                cache_key = (branch["seed"], branch["gate_mode"])
                x = data.batch(0, branch["seed"], "cpu")["input_ids"][:2]
                with torch.no_grad():
                    logits = initial_model(x, query_chunk=cfg.query_chunk).float()
                if cache_key in initial_cache:
                    torch.testing.assert_close(
                        logits, initial_cache[cache_key][1], atol=0, rtol=0
                    )
                other_mode = (
                    "direct_constant"
                    if branch["gate_mode"] == "factorized_constant"
                    else "factorized_constant"
                )
                if (
                    branch["gate_mode"] != "original_data"
                    and (branch["seed"], other_mode) in initial_cache
                ):
                    other = initial_cache[(branch["seed"], other_mode)]
                    if other[0] != initial["shared_non_gate_initial_hash"]:
                        raise ValueError("Paired initial non-gate parameters differ")
                    torch.testing.assert_close(logits, other[1], atol=1e-6, rtol=1e-6)
                initial_cache[cache_key] = (
                    initial["shared_non_gate_initial_hash"],
                    logits,
                )
                status.update(initial)
                del initial_model
                saved = _train_branch(
                    cfg,
                    data,
                    branch,
                    out / branch["branch"],
                    device,
                    signature,
                    initial,
                    resume,
                )
                status.update(status="complete", completed_steps=saved["step"])
            except KeyboardInterrupt:
                status.update(status="interrupted", failure="KeyboardInterrupt")
                _write_artifacts(out, cfg, statuses)
                dump_json(
                    out / "status.json",
                    {
                        "status": "interrupted",
                        "branches": statuses,
                        "software_only": cfg.profile == "smoke",
                    },
                )
                raise
            except Exception as error:
                status.update(
                    status="failed", failure=f"{type(error).__name__}: {error}"
                )
                failure_path = out / branch["branch"] / "failure.txt"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(traceback.format_exc())
                status["traceback_file"] = str(failure_path.resolve())
            _write_artifacts(out, cfg, statuses)
            dump_json(
                out / "status.json",
                {
                    "status": "running",
                    "branches": statuses,
                    "software_only": cfg.profile == "smoke",
                },
            )
        failures = [r for r in statuses if r["status"] != "complete"]
        dump_json(
            out / "status.json",
            {
                "status": "failed" if failures else "complete",
                "branches": statuses,
                "failures": failures,
                "software_only": cfg.profile == "smoke",
                "interpretation": (
                    "Software smoke only"
                    if cfg.profile == "smoke"
                    else "Finite-grid empirical study; not an asymptotic or infinite-generalization proof"
                ),
            },
        )
    return {
        "out_dir": str(out.resolve()),
        "status": "failed" if failures else "complete",
        "failures": failures,
        "summary": str((out / "summary.csv").resolve()),
        "eval": str((out / "eval.csv").resolve()),
    }
