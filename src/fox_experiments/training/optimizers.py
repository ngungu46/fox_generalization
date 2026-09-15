"""Optimizer interventions and diagnostics, separate from the training loop."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ExperimentConfig


def parameter_groups(
    model: nn.Module,
    config: ExperimentConfig,
    learning_rate: float,
    practical: bool = False,
) -> list[dict[str, Any]]:
    """Assign gate/readout multipliers, or the paper-style AdamW decay groups.

    The practical control decays matrix weights but excludes norms and biases.
    Theory-inspired arms apply no weight decay; first-layer forgetting, later
    forgetting, and the output head receive explicit learning-rate multipliers.
    """
    if practical:
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if parameter.ndim >= 2 and "norm" not in name:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        return [
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": 0.1,
                "label": "weights",
            },
            {
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "label": "norms_and_gates",
            },
        ]

    groups: dict[tuple[str, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        is_gate = ".attn.gate." in name
        first_layer = any(prefix in name for prefix in ("blocks.0.", "layers.0."))
        if is_gate and first_layer:
            label, multiplier = "first_gate", config.first_gate_multiplier
        elif is_gate:
            label, multiplier = "retrieval_gate", config.retrieval_gate_multiplier
        elif any(part in name for part in ("output_head", "lm_head", "readout")):
            label, multiplier = "output", config.output_multiplier
        else:
            label, multiplier = "representation", 1.0
        groups.setdefault((label, multiplier), []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": learning_rate * multiplier,
            "base_multiplier": multiplier,
            "label": label,
        }
        for (label, multiplier), parameters in groups.items()
    ]


def make_optimizer(
    model: nn.Module,
    config: ExperimentConfig,
    arm: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Build one arm. Every trial and continuation starts fresh optimizer state."""
    groups = parameter_groups(
        model, config, learning_rate, practical=arm == "paper_adamw"
    )
    if arm == "sgd":
        return torch.optim.SGD(groups, lr=learning_rate, momentum=0, weight_decay=0)
    if arm == "paper_adamw":
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    if arm in ("adam_fixed", "adam_annealed"):
        return torch.optim.Adam(
            groups,
            lr=learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps0,
            weight_decay=0,
        )
    raise ValueError(f"Unknown optimizer arm: {arm}")


def set_optimizer_schedule(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    arm: str,
    learning_rate: float,
    step: int,
    total_steps: int,
) -> tuple[float, float]:
    """Update schedules and return base LR and epsilon for the current update.

    Annealed epsilon is native fp32 arithmetic. A very small epsilon cannot
    reproduce the source draft's signed-log asymptotics, so that range is stopped.
    """
    if arm == "paper_adamw":
        warmup = max(1, min(20, total_steps // 10))
        multiplier = min(1.0, (step + 1) / warmup)
        multiplier *= 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * multiplier
    else:
        multiplier = (1 + step / config.lr_offset) ** (-config.lr_power)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * multiplier * group["base_multiplier"]

    epsilon = config.eps0
    if arm == "adam_annealed":
        epsilon *= math.exp(-config.epsilon_decay * step)
        if epsilon < np.finfo(np.float32).tiny:
            raise FloatingPointError(
                "Scheduled epsilon is below normal fp32 range. "
                "Use the controlled float64 mechanism experiment to extend the study."
            )
    if arm.startswith("adam"):
        for group in optimizer.param_groups:
            group["eps"] = epsilon
    return learning_rate * multiplier, epsilon


def moment_diagnostics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Measure how often epsilon dominates Adam's bias-corrected denominator."""
    rows: dict[str, float] = {}
    for group in optimizer.param_groups:
        ratios, zero_fractions = [], []
        epsilon = group.get("eps", 0)
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "exp_avg_sq" not in state:
                continue
            step = float(state["step"])
            root_second_moment = (
                state["exp_avg_sq"] / (1 - group["betas"][1] ** step)
            ).sqrt()
            zero_fractions.append(float((root_second_moment == 0).float().mean()))
            if epsilon > 0:
                ratios.append((root_second_moment / epsilon).detach().flatten())
        if ratios:
            all_ratios = torch.cat(ratios)
            label = group["label"]
            rows[label + "_sqrt_v_over_eps_median"] = float(all_ratios.median())
            rows[label + "_epsilon_dominated_fraction"] = float(
                (all_ratios <= 1).float().mean()
            )
            rows[label + "_zero_moment_fraction"] = float(np.mean(zero_fractions))
    return rows
