"""Scalar, optimizer and finite-range diagnostics."""

from __future__ import annotations

import math
import numpy as np
import torch
from .tasks import loss_function
from .bounds import sufficient_error_bound, contiguous_radius, witness_error


@torch.no_grad()
def _numeric_diagnostics(model, optimizer):
    """Native arithmetic instrumentation; there is no hidden gradient rescaling."""
    output = {}
    gradient_count = zero_count = square_underflow = 0
    scalar_signs = {}
    for group in optimizer.param_groups:
        name, p = group["name"], group["params"][0]
        if p.grad is None:
            continue
        gradient = p.grad.detach()
        gradient_count += gradient.numel()
        zero_count += int((gradient == 0).sum())
        square_underflow += int(((gradient != 0) & (gradient.square() == 0)).sum())
        if gradient.numel() == 1:
            scalar_signs["grad_" + name] = float(gradient)
        state = optimizer.state.get(p, {})
        if "exp_avg_sq" in state:
            step = int(state["step"])
            rms = (state["exp_avg_sq"] / (1 - group["betas"][1] ** step)).sqrt()
            eps = group["eps"]
            ratio = rms / eps if eps else torch.where(rms > 0, torch.inf, torch.nan)
            output[name + "_rms_over_eps_min"] = float(ratio.min())
            output[name + "_eps_dominated_fraction"] = float(
                (rms <= eps).double().mean()
            )
    output.update(scalar_signs)
    output.update(
        gradient_zero_fraction=zero_count / max(1, gradient_count),
        gradient_square_underflow_count=square_underflow,
    )
    return output


@torch.no_grad()
def snapshot(
    model, data, step, cumulative_rate, fixed_pair, optimizer=None, initial_tables=None
):
    m, g, h, rho = (float(x) for x in model.quantities())
    mask = ~torch.eye(model.config.n_keys, dtype=torch.bool, device=model.Q.device)
    gaps = model.gaps()[mask]
    delta = float(gaps.min())
    cfg = model.config
    errors = sufficient_error_bound(model, np.arange(1, cfg.radius_cap + 1))
    radius, censored = contiguous_radius(errors, cfg.error_threshold)
    result = dict(
        step=step,
        S=cumulative_rate,
        loss=float(loss_function(model, data)),
        m=m,
        g=g,
        h=h,
        rho=rho,
        m_rho=m * rho,
        delta_min=delta,
        w=float(model.w),
        m_over_h=m / h,
        min_margin_over_h=m * delta / h,
        positive_gap_fraction=float((gaps > 0).double().mean()),
        q_norm_max=float(model.Q.norm(dim=1).max()),
        k_norm_max=float(model.K.norm(dim=1).max()),
        sufficient_radius=radius,
        radius_cap=cfg.radius_cap,
        radius_right_censored=censored,
        error_threshold=cfg.error_threshold,
        bound_eligible=bool(np.isfinite(errors).all()),
        fixed_pair_a=fixed_pair[0],
        fixed_pair_b=fixed_pair[1],
        fixed_lag4_witness_error=witness_error(model, 4, 0, fixed_pair),
        m_over_S2=m / cumulative_rate**2 if cumulative_rate else None,
        g_over_S2=g / cumulative_rate**2 if cumulative_rate else None,
        h_over_S=h / cumulative_rate if cumulative_rate else None,
        balance_residual=m * delta - 3 * h - math.log(m) if m > 0 else None,
    )
    for name, item in zip(("overwrite", "recall"), data):
        result[name + "_min_probability"] = float(model(*item).sigmoid().min())
    result["calibration_probability"] = float(model.w.sigmoid())
    recall_lag = data[1][0].shape[1]
    for lag in sorted(
        set(
            (
                1,
                2,
                3,
                4,
                5,
                8,
                16,
                32,
                64,
                recall_lag - 1,
                recall_lag,
                recall_lag + 1,
                recall_lag + 2,
                recall_lag + 3,
            )
        )
    ):
        result[f"bound_error_r{lag}"] = float(sufficient_error_bound(model, [lag])[0])
        for older in (0, 8, 100000, math.inf):
            label = "inf" if math.isinf(older) else str(older)
            result[f"witness_error_r{lag}_N{label}"] = witness_error(model, lag, older)
    if initial_tables is not None:
        result["Q_max_row_movement"] = float(
            (model.Q - initial_tables[0]).norm(dim=1).max()
        )
        result["K_max_row_movement"] = float(
            (model.K - initial_tables[1]).norm(dim=1).max()
        )
    if optimizer is not None:
        result.update(_numeric_diagnostics(model, optimizer))
    return result
