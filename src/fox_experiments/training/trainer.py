"""The small-scale training loop: one deterministic, resumable optimizer segment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from fox_experiments.data import LatestWriteData
from fox_experiments.models import FoXLM
from .artifacts import atomic_checkpoint, dump_json, finite_json_row, table
from .config import ExperimentConfig
from .optimizers import make_optimizer, moment_diagnostics, set_optimizer_schedule


def train_segment(
    model: FoXLM,
    data: LatestWriteData,
    config: ExperimentConfig,
    arm: str,
    learning_rate: float,
    steps: int,
    seed: int,
    out_dir: str | Path,
    task_probability: float | None = None,
    checkpoint_steps: Sequence[int] = (),
) -> tuple[list[dict[str, Any]], str]:
    """Train a segment, retaining moments and replaying its data when resumed.

    Each call starts with a fresh optimizer unless ``out_dir/resume.pt`` exists.
    Segment specifications are immutable: changing settings requires a new output
    directory. Data batches depend only on seed, update, and microbatch indices.
    Model checkpoints are saved at explicit evaluation steps as well as the most
    recent save interval. An interrupted process can lose at most one interval.

    Checkpoints are trusted local artifacts produced by this repository. Do not
    load arbitrary third-party pickle checkpoints through this resume path.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    checkpoint_steps = tuple(sorted(set(checkpoint_steps)))
    if any(step < 1 or step > steps for step in checkpoint_steps):
        raise ValueError("checkpoint_steps must be in [1, steps]")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model, config, arm, learning_rate)
    device = next(model.parameters()).device
    specification = {
        "arm": arm,
        "lr": learning_rate,
        "steps": steps,
        "seed": seed,
        "task_probability": task_probability,
        "checkpoint_steps": checkpoint_steps,
        "config": config.to_dict(),
        "parameter_shapes": {
            name: list(parameter.shape) for name, parameter in model.named_parameters()
        },
    }
    specification_text = json.dumps(specification, sort_keys=True)
    signature = hashlib.sha256(specification_text.encode()).hexdigest()
    config_path = out / "config.json"
    if config_path.exists():
        previous = json.dumps(json.loads(config_path.read_text()), sort_keys=True)
        if previous != specification_text:
            raise ValueError("Segment config mismatch; use a new run directory")
    else:
        dump_json(config_path, specification)

    history: list[dict[str, Any]] = []
    start, cumulative_lr, elapsed_offset = 0, 0.0, 0.0
    resume_path = out / "resume.pt"
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume config mismatch; use a new run directory")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        cumulative_lr = saved["S"]
        history = saved["history"]
        elapsed_offset = saved["seconds"]
        for checkpoint_step in checkpoint_steps:
            path = out / f"checkpoint_{checkpoint_step:06d}.pt"
            if checkpoint_step <= start and not path.exists():
                raise RuntimeError(
                    f"Evaluation checkpoint is missing from resumed segment: {path}"
                )

    started = time.perf_counter()
    status, reason = "complete", None
    for step in range(start, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        try:
            current_lr, epsilon = set_optimizer_schedule(
                optimizer, config, arm, learning_rate, step, steps
            )
        except FloatingPointError as error:
            status, reason = "numerical_stop", str(error)
            break

        loss_total = 0.0
        for microbatch in range(config.accumulation):
            inputs, targets, weights = data.batch(
                config, step, microbatch, seed, task_probability=task_probability
            )
            inputs = inputs.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            loss = model.loss(inputs, targets, loss_mask=weights, logit_chunk=64)
            loss = loss / config.accumulation
            if not torch.isfinite(loss):
                status, reason = "numerical_failure", "nonfinite loss"
                break
            loss.backward()
            loss_total += float(loss.detach())
        if status != "complete":
            break

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            status, reason = "numerical_failure", "nonfinite gradient"
            break
        gradient_norm = math.sqrt(
            sum(float(gradient.detach().square().sum()) for gradient in gradients)
        )
        if arm == "paper_adamw":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cumulative_lr += current_lr
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            status, reason = "numerical_failure", "nonfinite parameter"
            break

        completed = step + 1
        save_update = (
            completed % config.save_every == 0
            or completed == steps
            or completed in checkpoint_steps
        )
        diagnostics = (
            moment_diagnostics(optimizer)
            if step < 2 or completed % config.save_every == 0 or completed == steps
            else {}
        )
        record = finite_json_row(
            {
                "step": completed,
                "loss": loss_total,
                "S": cumulative_lr,
                "lr": current_lr,
                "epsilon": epsilon,
                "gradient_norm": gradient_norm,
                "seconds": elapsed_offset + time.perf_counter() - started,
                **diagnostics,
            }
        )
        history.append(record)
        if save_update:
            saved = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": completed,
                "S": cumulative_lr,
                "history": history,
                "signature": signature,
                "seconds": record["seconds"],
            }
            # Write the evaluation snapshot first. A crash can then only replay an
            # already saved update, never leave resume ahead of a required snapshot.
            if completed in checkpoint_steps:
                atomic_checkpoint(
                    out / f"checkpoint_{completed:06d}.pt",
                    {key: saved[key] for key in ("model", "step", "S")},
                )
            atomic_checkpoint(resume_path, saved)
            table(out / "training.csv", history)

    dump_json(
        out / "status.json",
        {
            "status": status,
            "reason": reason,
            "completed_steps": len(history),
            "requested_steps": steps,
            "software_only": config.profile == "smoke",
        },
    )
    return history, status
