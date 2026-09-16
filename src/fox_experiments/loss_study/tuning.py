"""Optional equal-budget learning-rate search using short validation only."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from .config import branches
from .data import StudyData
from .model import build_model
from .runner import compute_losses, make_optimizer, schedule_optimizer


@torch.no_grad()
def short_validation(model, data, config, seed, device):
    """Different namespace from confirmation/test; never evaluates long lags."""
    examples = data.evaluation_examples(split="tune", seed=seed)
    rows = []
    model.eval()
    for start in range(0, len(examples), config.batch_size):
        subset = examples[start : start + config.batch_size]
        batch = data.collate(subset, device=device)
        logits = model(batch["input_ids"], query_chunk=config.query_chunk).float()
        chosen = logits[batch["answer_mask"]]
        targets = batch["targets"][batch["answer_mask"]]
        probabilities = chosen.softmax(-1).gather(1, targets[:, None]).squeeze(1)
        correct = chosen.argmax(-1).eq(targets)
        for example, ok, prob in zip(subset, correct.tolist(), probabilities.tolist()):
            rows.append(
                dict(
                    history_id=example.history_id,
                    lag=example.lag,
                    prefix=example.prefix,
                    correct=ok,
                    probability=prob,
                )
            )
    frame = pd.DataFrame(rows)
    joint = frame.groupby(["lag", "prefix", "history_id"]).correct.all()
    return dict(
        short_joint_accuracy=float(joint.mean()),
        short_worst_panel_accuracy=float(joint.groupby(level=[0, 1]).mean().min()),
        short_mean_probability=float(frame.probability.mean()),
    )


def tune_learning_rates(
    config,
    out_dir,
    device="cpu",
    data_dir=None,
    corpus=None,
    steps=250,
    multipliers=(1 / 3, 1.0, 3.0),
):
    """Return a NEW config with per-branch rates, plus a reviewable trial ledger.

    Tuning is optional. It changes rates independently across objectives, so the
    tuned study is distinct from the core same-rate loss intervention. Every
    branch/trial gets the same number of training updates and held-out short
    cases. Selection never sees long-context evaluation.
    """
    if steps < 1 or not multipliers or any(x <= 0 for x in multipliers):
        raise ValueError("Need positive tuning steps and candidate multipliers")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh tuning directory")
    out.mkdir(parents=True, exist_ok=True)
    base = replace(config, branch_learning_rates={})
    data = StudyData(base, data_dir=data_dir, corpus=corpus)
    package = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*.py"))
    }
    specification = dict(
        config=base.to_dict(),
        steps=steps,
        multipliers=list(multipliers),
        data=data.fingerprint(),
        sources=sources,
        torch=str(torch.__version__),
        device=str(device),
        selection="short worst-panel joint accuracy, joint accuracy, probability",
    )
    (out / "tuning_specification.json").write_text(json.dumps(specification, indent=2))
    trials, selected = [], {}
    for spec in branches(base):
        eligible = []
        for multiplier in multipliers:
            lr = base.learning_rates[spec["optimizer"]] * multiplier
            row = dict(**spec, learning_rate=lr, steps=steps, status="failed")
            try:
                model = build_model(base, data, spec["gate_mode"], spec["seed"]).to(
                    device
                )
                optimizer = make_optimizer(model, base, spec["optimizer"], lr)
                for step in range(steps):
                    model.train()
                    batch = data.batch(step, spec["seed"], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    schedule_optimizer(optimizer, base, spec["optimizer"], lr, step)
                    losses = compute_losses(
                        model(batch["input_ids"], query_chunk=base.query_chunk),
                        batch,
                        base,
                    )
                    loss = losses[spec["objective"]]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite tuning loss")
                    loss.backward()
                    clip = 1.0 if spec["optimizer"] == "paper_adamw" else base.clip_grad
                    norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip if clip is not None else float("inf")
                    )
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite tuning gradient")
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in model.parameters()):
                        raise FloatingPointError("Nonfinite tuning parameters")
                row.update(short_validation(model, data, base, spec["seed"], device))
                row["status"] = "ok"
                eligible.append(row)
            except (FloatingPointError, RuntimeError) as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            trials.append(row)
            pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
        if not eligible:
            raise RuntimeError(
                f"All LR candidates failed for {spec['branch']}; inspect trials.csv"
            )
        best = max(
            eligible,
            key=lambda r: (
                r["short_worst_panel_accuracy"],
                r["short_joint_accuracy"],
                r["short_mean_probability"],
            ),
        )
        selected[spec["branch"]] = best["learning_rate"]
        print(
            f"Tuned {spec['branch']}: lr={best['learning_rate']:.4g}, "
            f"short worst-panel={best['short_worst_panel_accuracy']:.3f}",
            flush=True,
        )
    result = replace(base, branch_learning_rates=selected)
    (out / "selected_config.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result
