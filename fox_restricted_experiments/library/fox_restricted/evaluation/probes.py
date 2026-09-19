"""Notebook probes: three fixed lags and a fixed multiple of m_t / h_t."""
import math

from ..legacy.asymptotics import snapshot, evaluate_snapshot, error_bounds


def convergence_probes(model, tags, step, S, reference_gap, theta):
    state = snapshot(model)
    rows, radii = evaluate_snapshot(
        state, model.cfg.R, model.cfg.gate_mode, tags, step, S,
        reference_gap=reference_gap,
        fixed_lags=tuple(model.cfg.R + i for i in (1, 2, 3)),
        theta_values=(), clock_coefficients=())
    # The coefficient is fixed at the acquisition checkpoint. It is never
    # adjusted using later test errors. A missing positive acquired gap means
    # this particular theorem-motivated moving probe is unavailable.
    coefficient = theta * reference_gap if reference_gap is not None and reference_gap > 0 else None
    lag = None
    if coefficient is not None and math.isfinite(float(state.m / state.h)):
        lag = max(1, math.floor(coefficient * float(state.m / state.h)))
    common = dict(tags, step=step, S=S, probe="theory_ratio", theta=theta,
                  coefficient=coefficient, reference_gap=reference_gap,
                  reference_gap_retained=(state.delta_min >= reference_gap) if reference_gap is not None else None,
                  m=float(state.m), h=float(state.h), delta_min=state.delta_min,
                  w=float(state.w), m_rho=float(state.m * state.rho),
                  max_row_norm=state.max_row_norm,
                  lag_definition="max(1, floor(theta * acquired_delta_min * m_t / h_t))")
    if lag is not None:
        rows.append(dict(common, **error_bounds(state, lag)))
    else:
        rows.append(dict(common, lag=None, lower_error=None, upper_error=None,
                         log_lower_error=None, log_upper_error=None,
                         bound_valid=False, bound_invalid_reason="positive_acquired_gap_unavailable"))
    return rows, radii
