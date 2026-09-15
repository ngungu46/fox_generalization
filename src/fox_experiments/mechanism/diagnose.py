"""Explain saved mechanism radii without changing parameters or rerunning training.

The uniform certificate and the explicit witness family answer different questions.
A failed sufficient bound alone is inconclusive; a failing explicit stream is a
counterexample at this checkpoint. Neither establishes an optimizer's asymptotic
limit. This module reads CSV/JSON only and never loads pickled checkpoints.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _error(log_odds, w):
    return np.exp(-np.logaddexp(0.0, -w * np.tanh(log_odds / 2)))


def _radius(errors, threshold):
    failed = np.flatnonzero(~np.isfinite(errors) | (errors > threshold))
    return int(failed[0]) if len(failed) else len(errors)


def _witness_errors(m, delta, h, rho, w, cap, *, older_infinite):
    """Exact worst ordered-pair witness, including the first binding transition.

    Stream: optional infinitely long limit of (a,-1), then (a,+1), then
    r-1 copies of (b,-1). The minimum raw gap is worst when m,w>0.
    The infinite-prefix value is the supremum of this finite-prefix family.
    """
    lags = np.arange(1, cap + 1)
    log_denom = math.log(-math.expm1(-h))
    log_a = np.full(cap, -h - log_denom if older_infinite else -np.inf)
    if cap >= 2:
        log_a[1:] = np.logaddexp(log_a[1:], -m * (1-rho) * delta + h)
    if cap >= 3:
        counts = lags[2:] - 2
        log_geom = np.log(-np.expm1(-counts * h)) - log_denom
        log_a[2:] = np.logaddexp(
            log_a[2:], -m * delta + (lags[2:]-1) * h + log_geom
        )
    return _error(log_a, w)


def diagnose_mechanism(out_dir) -> pd.DataFrame:
    """Return one readable diagnosis per saved branch; do not write any files.

    Reads ``summary.csv`` and optional ``config.json``/``traces.csv``. The decoder
    floor is sigmoid(-abs(w)), a necessary bound for binary values in [-1,1].
    A positive w, matching margin and forgetting slope are required by the
    certificate used in this experiment. Missing/nonfinite inputs are reported
    as unavailable, never converted into an apparent measured zero radius.
    """
    out = Path(out_dir)
    summary = pd.read_csv(out / "summary.csv")
    config = {}
    if (out / "config.json").exists():
        raw = json.loads((out / "config.json").read_text())
        config = raw.get("config", raw)
    traces = (pd.read_csv(out / "traces.csv")
              if (out / "traces.csv").exists() else pd.DataFrame())
    results = []
    for row in summary.to_dict("records"):
        result = {key: row.get(key) for key in ("seed", "recall_lag", "gate", "optimizer")}
        values = {key: _number(row.get(key)) for key in ("m", "delta_min", "h", "rho", "w")}
        m, delta, h, rho, w = [values[k] for k in ("m", "delta_min", "h", "rho", "w")]
        threshold = _number(row.get("error_threshold", config.get("error_threshold", .01)))
        cap_value = _number(row.get("radius_cap", config.get("radius_cap", 256)))
        cap_valid = math.isfinite(cap_value) and 1 <= cap_value <= 1_000_000 and cap_value.is_integer()
        threshold_valid = math.isfinite(threshold) and 0 < threshold < .5
        finite = all(math.isfinite(value) for value in values.values())
        state_valid = finite and 0 <= rho <= 1 and cap_valid and threshold_valid
        positive_decoder = math.isfinite(w) and w > 0
        matching_eligible = finite and m > 0 and delta > 0 and h > 0
        eligible = state_valid and positive_decoder and matching_eligible
        floor = float(np.exp(-np.logaddexp(0.0, abs(w)))) if math.isfinite(w) else math.nan
        required = math.log((1-threshold)/threshold) if threshold_valid else math.nan
        result.update(values)
        result.update(
            error_threshold=threshold,
            decoder_error_floor=floor,
            best_possible_answer_probability=1-floor,
            decoder_w_required=required,
            positive_decoder=positive_decoder,
            matching_margin_eligible=matching_eligible,
            diagnostic_inputs_valid=state_valid,
            decoder_can_reach_threshold=bool(eligible and floor <= threshold),
            saved_sufficient_radius=_number(row.get("sufficient_radius")),
            saved_bound_error_lag1=_number(row.get("bound_error_r1")),
            overwrite_min_probability=_number(row.get("overwrite_min_probability")),
            recall_min_probability=_number(row.get("recall_min_probability")),
            acquisition_qualified=row.get("acquisition_qualified"),
            stop_reason=row.get("stop_reason"),
            witness_no_prefix_radius=math.nan,
            witness_infinite_prefix_radius=math.nan,
            witness_infinite_prefix_lag1_error=math.nan,
            witness_radius_cap=cap_value,
            witness_no_prefix_right_censored=False,
            witness_infinite_prefix_right_censored=False,
            initial_w=math.nan,
            initial_decoder_error_floor=math.nan,
        )
        if not traces.empty:
            selected = traces
            for key in ("seed", "recall_lag", "gate", "optimizer"):
                if key in selected and key in row:
                    selected = selected[selected[key] == row[key]]
            if len(selected) and "w" in selected:
                selected = selected.sort_values("step") if "step" in selected else selected
                initial_w = _number(selected.iloc[0]["w"])
                result["initial_w"] = initial_w
                if math.isfinite(initial_w):
                    result["initial_decoder_error_floor"] = float(np.exp(-np.logaddexp(0, abs(initial_w))))
        if eligible:
            cap = int(cap_value)
            for infinite, label in ((False, "no_prefix"), (True, "infinite_prefix")):
                errors = _witness_errors(m, delta, h, rho, w, cap, older_infinite=infinite)
                radius = _radius(errors, threshold)
                result[f"witness_{label}_radius"] = radius
                result[f"witness_{label}_right_censored"] = radius == cap
                if infinite:
                    result["witness_infinite_prefix_lag1_error"] = float(errors[0])
        if not state_valid:
            diagnosis = "invalid_or_missing_diagnostic_inputs"
        elif not positive_decoder:
            diagnosis = "nonpositive_decoder_certificate_ineligible"
        elif not matching_eligible:
            diagnosis = "matching_or_forgetting_certificate_ineligible"
        elif floor > threshold:
            diagnosis = "decoder_confidence_below_threshold"
        elif result["witness_infinite_prefix_lag1_error"] > threshold:
            diagnosis = "stale_prefix_fails_at_lag1"
        elif result["saved_sufficient_radius"] == 0:
            diagnosis = "sufficient_bound_not_passed_no_failure_proved"
        elif result["saved_sufficient_radius"] > 0:
            diagnosis = "positive_sufficient_radius"
        else:
            diagnosis = "saved_certificate_unavailable"
        result["diagnosis"] = diagnosis
        result["interpretation"] = (
            "Radius 0 means no positive lag certified at the configured error threshold; "
            "it does not mean zero accuracy. Explicit witness radii concern this witness "
            "family only. A finite checkpoint does not determine an asymptotic boundary."
        )
        results.append(result)
    return pd.DataFrame(results)
