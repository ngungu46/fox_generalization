"""Read-only diagnostics distinguish genuine causes of a zero certificate."""

import json
import math

import pandas as pd
import pytest

from fox_experiments.mechanism.diagnose import diagnose_mechanism


def saved_run(tmp_path, **overrides):
    row = dict(seed=0, recall_lag=2, gate="factorized", optimizer="sgd",
               m=20., delta_min=1., h=3., rho=1e-10, w=8.,
               sufficient_radius=5, error_threshold=.01, radius_cap=16)
    row.update(overrides)
    pd.DataFrame([row]).to_csv(tmp_path / "summary.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps({"config": {"error_threshold": .01}}))
    pd.DataFrame([{**row, "step": 0, "w": 3.},
                  {**row, "step": 1000}]).to_csv(tmp_path / "traces.csv", index=False)
    return row


def test_decoder_floor_blocks_even_a_perfect_no_prefix_witness(tmp_path):
    saved_run(tmp_path, w=4.1355, sufficient_radius=0)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = diagnose_mechanism(tmp_path).iloc[0]
    assert result.diagnosis == "decoder_confidence_below_threshold"
    assert result.decoder_error_floor == pytest.approx(1/(1+math.exp(4.1355)))
    assert result.decoder_w_required == pytest.approx(math.log(99))
    assert result.witness_no_prefix_radius == 0
    assert result.initial_w == 3
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_stale_prefix_can_fail_when_decoder_and_no_prefix_retrieval_pass(tmp_path):
    saved_run(tmp_path, w=6.020834, h=1.9774, sufficient_radius=0)
    result = diagnose_mechanism(tmp_path).iloc[0]
    assert result.diagnosis == "stale_prefix_fails_at_lag1"
    assert result.decoder_error_floor < .01
    assert result.witness_no_prefix_radius > 0
    assert result.witness_infinite_prefix_radius == 0
    assert result.witness_infinite_prefix_lag1_error == pytest.approx(.012693, abs=1e-6)


def test_a_failed_sufficient_bound_does_not_invent_a_witness_failure(tmp_path):
    saved_run(tmp_path, sufficient_radius=0)
    result = diagnose_mechanism(tmp_path).iloc[0]
    assert result.diagnosis == "sufficient_bound_not_passed_no_failure_proved"
    assert result.witness_no_prefix_radius > 0
    assert result.witness_infinite_prefix_radius > 0
    assert result.witness_infinite_prefix_lag1_error < .01


def test_negative_or_nonfinite_decoder_is_not_a_measured_zero_radius(tmp_path):
    saved_run(tmp_path, w=-8., sufficient_radius=0)
    result = diagnose_mechanism(tmp_path).iloc[0]
    assert result.decoder_error_floor == pytest.approx(1/(1+math.exp(8)))
    assert not result.positive_decoder
    assert result.diagnosis == "nonpositive_decoder_certificate_ineligible"
    assert math.isnan(result.witness_no_prefix_radius)
    saved_run(tmp_path, w=float("nan"), sufficient_radius=0)
    result = diagnose_mechanism(tmp_path).iloc[0]
    assert result.diagnosis == "invalid_or_missing_diagnostic_inputs"
    assert math.isnan(result.decoder_error_floor)
    assert math.isnan(result.witness_no_prefix_radius)
