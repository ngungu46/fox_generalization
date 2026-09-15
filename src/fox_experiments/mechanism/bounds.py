"""Analytic diagnostics valid only for the controlled binding model."""

from __future__ import annotations

import math
import numpy as np
import torch


@torch.no_grad()
def sufficient_error_bound(model, lags):
    """Answer-error upper bound for ALL finite older prefixes, with actual norms.

    A <= exp(4*m*rho*RQ*RK) [exp(-m*delta+(r-1)*h)+exp(-h)]/(1-exp(-h)).
    A is competitor/target attention odds, so error <= sigmoid(w*tanh(log(A)/2)).
    The result is a sufficient mathematical bound for this architecture only.
    """
    m, _, h, rho = (float(x) for x in model.quantities())
    mask = ~torch.eye(model.config.n_keys, dtype=torch.bool, device=model.Q.device)
    delta = float(model.gaps()[mask].min())
    w = float(model.w)
    if min(m, h, delta, w) <= 0 or not all(
        math.isfinite(x) for x in (m, h, delta, w, rho)
    ):
        return np.full(np.asarray(lags).shape, np.nan)
    rq = float(model.Q.norm(dim=1).max())
    rk = float(model.K.norm(dim=1).max())
    log_a = (
        4 * m * rho * rq * rk
        + np.logaddexp(-m * delta + (np.asarray(lags) - 1) * h, -h)
        - math.log(-math.expm1(-h))
    )
    return np.exp(-np.logaddexp(0.0, -w * np.tanh(log_a / 2)))


def contiguous_radius(errors, threshold):
    """No interpolation and no skipping failed lags; return radius and cap censoring."""
    values = np.asarray(errors)
    passed = np.isfinite(values) & (values <= threshold)
    failures = np.flatnonzero(~passed)
    radius = int(failures[0]) if len(failures) else len(values)
    return radius, bool(len(values) > 0 and radius == len(values))


def _log_geometric(h, count):
    if count == 0:
        return -math.inf
    return (0.0 if math.isinf(count) else math.log(-math.expm1(-count * h))) - math.log(
        -math.expm1(-h)
    )


@torch.no_grad()
def witness_error(model, lag, older=0, pair=None):
    """Exact error for a^- repeated N, a^+, b^- repeated r-1, query a.

    If pair is None maximize over all ordered distinct pairs. Infinite N uses
    the exact geometric limit, which is also the supremum over finite N in
    this specific all-wrong-competitor witness family.
    """
    m, _, h, rho = (float(x) for x in model.quantities())
    gaps = model.gaps()
    if pair is None:
        mask = ~torch.eye(model.config.n_keys, dtype=torch.bool, device=model.Q.device)
        deltas = gaps[mask].cpu().numpy()
    else:
        deltas = np.asarray([float(gaps[pair[0], pair[1]])])
    log_a = np.full(len(deltas), -h + _log_geometric(h, older))
    if lag >= 2:
        log_a = np.logaddexp(log_a, -m * (1 - rho) * deltas + h)
    if lag >= 3:
        log_a = np.logaddexp(
            log_a, -m * deltas + (lag - 1) * h + _log_geometric(h, lag - 2)
        )
    errors = np.exp(-np.logaddexp(0.0, -float(model.w) * np.tanh(log_a / 2)))
    return float(errors.max())
