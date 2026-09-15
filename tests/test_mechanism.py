"""Independent numerical checks of the controlled mechanism and its certificate."""

import itertools
import json
import math
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from fox_experiments.mechanism import (
    BindingFoX,
    MechanismConfig,
    contiguous_radius,
    loss_function,
    run_mechanism,
    sufficient_error_bound,
    templates,
    witness_error,
)


def matched_model():
    model = BindingFoX(MechanismConfig(n_keys=2, width=2))
    with torch.no_grad():
        model.Q.copy_(torch.eye(2, dtype=torch.float64) * 1.7)
        model.K.copy_(model.Q)
        model.q.fill_(2.0)
        model.p.fill_(2.0)
        model.w.fill_(2.0)
        model.x.fill_(0.3)
    return model


def test_conversion_preserves_actual_binding_gate_and_full_function():
    model = matched_model()
    other = model.converted("direct")
    for a, b in zip(model.quantities(), other.quantities()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for recall_lag in (2, 4, 8):
        for data in templates(model, recall_lag):
            torch.testing.assert_close(model(*data), other(*data), rtol=0, atol=0)
    assert set(dict(other.named_parameters())) == {"Q", "K", "q", "p", "z", "x", "w"}


def test_original_R2_odds_match_forward_answer_loss():
    model = matched_model()
    m, _, h, rho = model.quantities()
    delta = model.gaps()[0, 1]
    odds_o = torch.exp(delta * m * rho) * (
        torch.exp(-2 * h) + torch.exp(-3 * h)
    ) + torch.exp(-delta * m * (1 - 2 * rho) - h)
    odds_c = torch.exp(-delta * m * (1 - rho) + h)
    expected = 0.6 * torch.nn.functional.softplus(-model.w)
    for odds in (odds_o, odds_c):
        expected = expected + 0.2 * torch.nn.functional.softplus(
            -model.w * (1 - odds) / (1 + odds)
        )
    torch.testing.assert_close(loss_function(model, templates(model)), expected)


def test_witness_formula_includes_first_binder_transition_and_old_tail():
    model = matched_model()
    for lag in (1, 2, 4, 8):
        for older in (0, 1, 7, 1000):
            keys = torch.tensor([[0] * (older + 1) + [1] * (lag - 1)])
            values = -torch.ones_like(keys, dtype=torch.float64)
            values[0, older] = 1.0
            explicit = float(
                torch.sigmoid(-model(keys, values, torch.tensor([0]))).detach()
            )
            assert abs(explicit - witness_error(model, lag, older, (0, 1))) < 2e-13
        assert (
            witness_error(model, lag, 1000, (0, 1))
            <= witness_error(model, lag, math.inf, (0, 1)) + 1e-14
        )


def test_norm_corrected_bound_covers_exhaustive_arbitrary_prefixes():
    # Norms deliberately exceed one, and prefixes include arbitrary keys/values.
    model = matched_model()
    for length in range(1, 6):
        rows, values, lags, labels = [], [], [], []
        for records in itertools.product(range(4), repeat=length):
            keys = [record // 2 for record in records]
            if 0 not in keys:
                continue
            vals = [float(2 * (record % 2) - 1) for record in records]
            target = max(i for i, key in enumerate(keys) if key == 0)
            rows.append(keys)
            values.append(vals)
            lags.append(length - target)
            labels.append(vals[target])
        logits = model(
            torch.tensor(rows),
            torch.tensor(values, dtype=torch.float64),
            torch.zeros(len(rows), dtype=torch.long),
        )
        actual = torch.sigmoid(-torch.tensor(labels) * logits).detach().numpy()
        upper = sufficient_error_bound(model, np.asarray(lags))
        assert np.all(actual <= upper + 1e-12)


def test_radius_is_contiguous_and_right_censored():
    assert contiguous_radius([0.001, 0.1, 0.001], 0.01) == (1, False)
    assert contiguous_radius([0.001, 0.001], 0.01) == (2, True)
    assert contiguous_radius([float("nan"), 0.001], 0.01) == (0, False)


def test_tiny_runner_saves_six_function_matched_arms(tmp_path):
    output = run_mechanism(
        tmp_path,
        config_overrides=dict(
            acquisition_steps=2,
            continuation_steps=2,
            log_every=1,
            radius_cap=8,
            eps_decay=1e6,
        ),
    )
    import csv

    with open(output["summary_csv"]) as handle:
        summaries = list(csv.DictReader(handle))
    with open(output["traces_csv"]) as handle:
        first_rows = [r for r in csv.DictReader(handle) if r["step"] == "0"]
    assert len(summaries) == 6
    assert len(first_rows) == 6
    for name in ("m", "g", "rho", "delta_min", "w", "loss"):
        assert max(float(r[name]) for r in first_rows) == min(
            float(r[name]) for r in first_rows
        )
    with open(output["failures_json"]) as handle:
        failures = json.load(handle)
    assert len(failures) == 6
    assert (
        sum(row["reason"] == "scheduled_epsilon_underflow_to_zero" for row in failures)
        == 2
    )
