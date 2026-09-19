"""Uncapped, arbitrary-context diagnostics for the paper's answer error.

E_t(r) is the supremum of 1-P(correct) over *all finite streams* whose latest
queried record has lag at most r.  An infinite stale-prefix witness gives a
lower bound on this supremum (by taking limits of finite witnesses).  The
manuscript's attention-odds transfer inequality gives an upper bound whenever
its row-norm, positive-gap, and sign hypotheses hold. Neither bound is called
an exact computation of E_t(r).

No sequence of length r is materialized. Errors are evaluated as
sigmoid(-correct_logit), and log errors as logsigmoid(-correct_logit), never
by subtracting a saturated probability from one.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

try:
    from .core import _geom_log
except ImportError:
    from core import _geom_log


@dataclass
class ErrorSnapshot:
    gaps: torch.Tensor
    m: torch.Tensor
    h: torch.Tensor
    rho: torch.Tensor
    w: torch.Tensor
    delta_min: float
    max_row_norm: float
    bound_valid: bool
    bound_invalid_reason: str


@torch.no_grad()
def snapshot(model):
    """Copy compact quantities once, avoiding repeated device synchronizations."""
    quantities = torch.stack(list(model.quantities()[5:10])).detach().cpu()
    w, m, _, h, rho = quantities.unbind()
    gaps = model.gaps()[model.mask].detach().cpu()
    row_norm = float(torch.maximum(model.Q.norm(dim=1).max(), model.K.norm(dim=1).max()))
    delta = float(gaps.min())
    reasons = []
    if not bool(torch.isfinite(quantities).all()) or not bool(torch.isfinite(gaps).all()) or not math.isfinite(row_norm):
        reasons.append("nonfinite_model")
    if row_norm > 1:
        reasons.append("row_norm_exceeds_one")
    if delta <= 0:
        reasons.append("nonpositive_matching_gap")
    if float(m) < 0:
        reasons.append("negative_matching_scale")
    if float(w) < 0:
        reasons.append("negative_decoder")
    if float(h) <= 0:
        reasons.append("nonpositive_forgetting")
    if not 0 <= float(rho) <= 1:
        reasons.append("invalid_binding_probability")
    return ErrorSnapshot(gaps, m, h, rho, w, delta, row_norm, not reasons, ";".join(reasons))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _error_from_log_odds(log_odds, w):
    # z=(1-A)/(1+A)=-tanh(log(A)/2), so error=sigmoid(w*tanh(log(A)/2)).
    return F.logsigmoid(w * torch.tanh(log_odds / 2))


def _float_integer(value):
    # Torch first interprets Python ints as int64. Convert explicitly so lags
    # beyond 2**63 still work without changing the recorded Python integer.
    try:
        return float(value)
    except OverflowError:
        return math.inf


def _witness_log_odds(state, lag):
    result = (-state.h + _geom_log(math.inf, state.h)).expand_as(state.gaps)
    if lag >= 2:
        result = torch.logaddexp(result, -state.m * (1 - state.rho) * state.gaps + state.h)
    if lag >= 3:
        result = torch.logaddexp(result, -state.m * state.gaps + _float_integer(lag - 1) * state.h
                                + _geom_log(_float_integer(lag - 2), state.h))
    return result


def _log_uniform_odds(state, lag):
    log_tail = _geom_log(math.inf, state.h)
    competing = -state.m * state.delta_min + _float_integer(lag - 1) * state.h
    return 4 * state.m * state.rho + torch.logaddexp(competing, -state.h) + log_tail


@torch.no_grad()
def error_bounds(state, lag):
    """Lower witness and conditional upper bound on E_t(lag).

    The witness is (a,-1)^k,(a,+1),(b,-1)^(lag-1), maximized over all
    a!=b, as k tends to infinity. For w>=0 its error increases with lag:
    extending the suffix adds positive non-target attention odds while leaving
    the target-relative old tail unchanged. Thus it is a valid lower bound for
    the nested lag<=r sets. Positive matching gaps are needed for the stated
    uniform upper certificate, not for this witness construction.
    """
    _positive_integer(lag, "lag")
    odds = _witness_log_odds(state, lag)
    log_lower = float(_error_from_log_odds(odds, state.w).max())
    log_upper = float(_error_from_log_odds(_log_uniform_odds(state, lag), state.w)) if state.bound_valid else None
    return {
        "lag": lag,
        "lower_error": math.exp(log_lower),
        "upper_error": math.exp(log_upper) if log_upper is not None else None,
        "log_lower_error": log_lower,
        "log_upper_error": log_upper,
        "bound_valid": state.bound_valid,
        "bound_invalid_reason": state.bound_invalid_reason,
        "lower_bound_method": "infinite_stale_prefix_max_all_pairs",
        "upper_bound_method": "uniform_arbitrary_stream_transfer" if state.bound_valid else "",
    }


def scalar_snapshot(delta_min, m, h, rho, w, row_norm_max=1.):
    """Reconstruct the diagnostics state from a scalar checkpoint history.

    For m,w>=0, each wrong-key term decreases with the raw gap, so the
    minimum-gap pair attains the all-pairs maximum witness error. Other signs
    are rejected because delta_min alone no longer identifies that maximum.
    """
    if m < 0 or w < 0:
        raise ValueError("scalar_snapshot requires m,w>=0; use the full snapshot for other signs")
    values = (delta_min, m, h, rho, w, row_norm_max)
    reasons = []
    if not all(math.isfinite(value) for value in values):
        reasons.append("nonfinite_model")
    if delta_min <= 0:
        reasons.append("nonpositive_matching_gap")
    if row_norm_max > 1:
        reasons.append("row_norm_exceeds_one")
    if h <= 0:
        reasons.append("nonpositive_forgetting")
    if not 0 <= rho <= 1:
        reasons.append("invalid_binding_probability")
    tensors = torch.tensor([m, h, rho, w], dtype=torch.float64)
    return ErrorSnapshot(torch.tensor([delta_min], dtype=torch.float64), *tensors.unbind(),
                         delta_min, row_norm_max, not reasons, ";".join(reasons))


def scalar_error_bounds(delta_min, m, h, rho, w, lag, row_norm_max=1.):
    """Compute E_t(lag) bounds directly from saved scalar diagnostics."""
    return error_bounds(scalar_snapshot(delta_min, m, h, rho, w, row_norm_max), lag)


def certified_radius(state, error_target):
    """Largest integer radius certified by the analytic upper bound.

    Zero means the bound cannot certify even lag one; None means its
    hypotheses fail. This is *not* the exact maximal generalizing radius.
    The analytic inverse is checked against adjacent integers to make the
    floor decision robust near a floating-point boundary.
    """
    if not math.isfinite(error_target) or not 0 < error_target < .5:
        raise ValueError("error_target must lie strictly between zero and 0.5")
    result = {"error_target": error_target, "certified_radius": None,
              "bound_valid": state.bound_valid, "bound_invalid_reason": state.bound_invalid_reason,
              "radius_definition": "max_radius_certified_by_uniform_upper_bound"}
    if not state.bound_valid:
        return result
    w, m, h, rho = (float(value) for value in (state.w, state.m, state.h, state.rho))
    required_margin = math.log1p(-error_target) - math.log(error_target)
    if w <= required_margin:
        result["certified_radius"] = 0
        return result
    required_z = required_margin / w
    allowed_log_odds = math.log1p(-required_z) - math.log1p(required_z)
    log_denominator = math.log(-math.expm1(-h))
    remaining = allowed_log_odds - 4 * m * rho + log_denominator
    if remaining <= -h:
        result["certified_radius"] = 0
        return result
    # log(exp(remaining)-exp(-h)); expm1 is accurate near equality.
    log_recent_allowance = remaining + math.log(-math.expm1(-h - remaining))
    radius_real = 1 + (m * state.delta_min + log_recent_allowance) / h
    if not math.isfinite(radius_real):
        result["bound_valid"] = False
        result["bound_invalid_reason"] = "radius_inverse_exceeds_float64_range"
        return result
    radius = max(0, math.floor(radius_real))
    target_log = math.log(error_target)
    def passes(r):
        return float(_error_from_log_odds(_log_uniform_odds(state, r), state.w)) <= target_log
    # In ordinary ranges these loops execute zero or one times. Above 2**53,
    # adjacent integers cannot be distinguished in float64; keep a conservative
    # representable radius rather than spending time walking the plateau.
    if radius > 2 ** 53:
        radius = max(0, math.floor(math.nextafter(radius_real, -math.inf)))
        result["radius_rounding"] = "conservative_float64_spacing"
    else:
        while radius > 0 and not passes(radius):
            radius -= 1
        while radius < 2 ** 53 and passes(radius + 1):
            radius += 1
        result["radius_rounding"] = "adjacent_integer_verified"
    result["certified_radius"] = radius
    return result


def _moving_lag(state, theta, gap, learned):
    if not math.isfinite(gap) or gap <= 0 or float(state.h) <= 0 or not math.isfinite(float(state.m / state.h)):
        return 1
    # Learned reference: floor(theta*m*delta/h-1). Frozen reference:
    # floor(1+theta*m*delta/h). Clamping makes early checkpoints evaluable;
    # a flat/clamped radius is not evidence for a diverging-radius theorem.
    return max(1, math.floor(theta * float(state.m / state.h) * gap + (-1 if learned else 1)))


@torch.no_grad()
def evaluate_asymptotics(model, tags, step, S, reference_gap=None,
                         fixed_lags=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                         theta_values=(.25, .5, .75), clock_coefficients=(.001, .003, .01),
                         error_targets=(.1, .01, .001)):
    """Return flat error-trajectory rows and certified-radius rows.

    ``adaptive_theta`` uses the current minimum gap. ``reference_theta``
    uses a gap recorded just after acquisition; its retention is logged, never
    assumed. Neither is an exogenous schedule. ``clock`` is predeclared as
    floor(1+c*S**a), with a=1 learned / a=2 frozen. The learning-rate clock
    S is fixed by the training schedule, and c is never fitted to test error.
    No evaluation lag cap is applied to any schedule.
    """
    return evaluate_snapshot(snapshot(model), model.cfg.R, model.cfg.gate_mode, tags, step, S,
                             reference_gap, fixed_lags, theta_values, clock_coefficients, error_targets)


@torch.no_grad()
def evaluate_snapshot(state, train_R, gate_mode, tags, step, S, reference_gap=None,
                      fixed_lags=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                      theta_values=(.25, .5, .75), clock_coefficients=(.001, .003, .01),
                      error_targets=(.1, .01, .001)):
    """The same evaluator for current models and saved scalar histories.

    Scalar histories must include their actual minimum matching gap, scale,
    gate, binding probability, decoder, and maximum raw row norm. No state is
    interpolated or extrapolated between saved checkpoints.
    """
    _positive_integer(train_R, "train_R")
    if gate_mode not in ("learned", "retrieval_frozen", "both_frozen"):
        raise ValueError("Unknown gate mode")
    if not math.isfinite(S) or S < 0:
        raise ValueError("S must be nonnegative and finite")
    for theta in theta_values:
        if not math.isfinite(theta) or not 0 < theta < 1:
            raise ValueError("theta_values must be in (0,1)")
    for coefficient in clock_coefficients:
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError("clock_coefficients must be positive and finite")
    learned = gate_mode == "learned"
    power = 1 if learned else 2
    common = {**tags, "step": step, "S": S, "delta_min": state.delta_min,
              "max_row_norm": state.max_row_norm, "m": float(state.m), "h": float(state.h),
              "w": float(state.w), "m_rho": float(state.m * state.rho),
              "reference_gap": reference_gap,
              "reference_gap_retained": state.delta_min >= reference_gap if reference_gap is not None else None}
    probes = []
    lags = set(fixed_lags) | {train_R + offset for offset in range(4)}
    for lag in sorted(lags):
        _positive_integer(lag, "fixed_lag")
        probes.append(("fixed", lag, None, None, None))
    for theta in theta_values:
        probes.append(("adaptive_theta", _moving_lag(state, theta, state.delta_min, learned), theta, None, None))
        if reference_gap is not None and math.isfinite(reference_gap) and reference_gap > 0:
            probes.append(("reference_theta", _moving_lag(state, theta, reference_gap, learned), theta, None, None))
    for coefficient in clock_coefficients:
        lag = max(1, math.floor(1 + coefficient * S ** power))
        probes.append(("clock", lag, None, power, coefficient))
    cache = {}
    error_rows = []
    for probe, lag, theta, clock_power, coefficient in probes:
        if lag not in cache:
            cache[lag] = error_bounds(state, lag)
        error_rows.append({**common, "probe": probe, "theta": theta, "clock_power": clock_power,
                           "coefficient": coefficient, **cache[lag]})
    radius_rows = [{**common, **certified_radius(state, target)} for target in error_targets]
    return error_rows, radius_rows
