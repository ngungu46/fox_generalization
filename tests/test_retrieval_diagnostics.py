"""A zero conjunction or unqualified run must not become a claimed boundary."""

import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fox_experiments.evaluation.aggregation import (
    aggregate_histories,
    finite_grid_ranges,
)
from fox_experiments.evaluation.diagnostics import diagnose_predictions, diagnose_run
from fox_experiments.models import FoXLM, ModelConfig
from fox_experiments.training import ExperimentConfig, run_experiment


def constant_guess_rows():
    rows = []
    for lag in (32, 64):
        for variant in ("base", "latest_flip", "stale_flip", "irrelevant_flip"):
            rows.append(
                {
                    "branch": "b",
                    "gate": "direct_constant",
                    "optimizer": "sgd",
                    "seed_id": 0,
                    "checkpoint_step": 400,
                    "S": 1.0,
                    "short_qualified": False,
                    "lag": lag,
                    "prefix_copies": 0,
                    "kind": "conflict",
                    "history_id": "0",
                    "variant": variant,
                    "prediction": 0,
                    "answer": int(variant == "latest_flip"),
                    "correct": int(variant != "latest_flip"),
                    "probability": 0.1,
                    "nll": np.log(10),
                }
            )
    return pd.DataFrame(rows)


def test_constant_guess_has_nonzero_answers_but_zero_all_edit_success():
    row = diagnose_predictions(constant_guess_rows()).iloc[0]
    assert row.base_accuracy == 1
    assert row.individual_answer_accuracy == 0.75
    assert row.all_edit_history_accuracy == 0
    assert row.latest_edit_prediction_change_rate == 0
    assert row.distinct_prediction_tokens == 1
    assert "constant_prediction" in row.diagnosis


def test_zero_range_distinguishes_unqualified_failed_and_invalid_data():
    _, panels = aggregate_histories(constant_guess_rows())
    row = finite_grid_ranges(panels).iloc[0]
    assert row.largest_passing_grid_lag == 0
    assert row.first_failed_tested_lag == 32
    assert row.interpretation == "short_rule_not_acquired"
    panels.short_qualified = True
    assert (
        finite_grid_ranges(panels).iloc[0].interpretation == "first_tested_lag_failed"
    )
    panels.loc[0, "history_accuracy"] = np.nan
    row = finite_grid_ranges(panels).iloc[0]
    assert row.largest_passing_grid_lag == 0
    assert row.interpretation == "invalid_measurement"


def test_diagnosis_uses_results_only_and_preserves_raw_bytes(tmp_path):
    raw = constant_guess_rows()
    path = tmp_path / "retrieval_raw.csv"
    raw.to_csv(path, index=False)
    original = path.read_bytes()
    pd.DataFrame([{"qualified": False}]).to_csv(
        tmp_path / "acquisition.csv", index=False
    )
    result = diagnose_run(tmp_path)
    assert result["constant_final_branches"] == 1
    assert result["unqualified_final_branches"] == 1
    assert path.read_bytes() == original
    assert (tmp_path / "diagnostics/diagnosis.md").is_file()


def test_acquisition_guard_skips_all_lr_trials_and_long_tests(tmp_path):
    from fox_experiments.data import Corpus
    from types import SimpleNamespace

    config = replace(
        ExperimentConfig.for_profile("smoke"),
        gate_modes=("direct_constant",),
        optimizer_arms=("sgd",),
        include_paper_control=False,
    )
    model = FoXLM(ModelConfig(d_model=16, n_heads=2, n_layers=2))
    array = np.zeros((4, 128), dtype=np.uint16)
    data = SimpleNamespace(
        answers=[0, 1], corpus=Corpus(array, array, array), protocol_version="test"
    )
    source_result = (model, {"status": "complete", "qualified": False})
    with (
        patch(
            "fox_experiments.training.experiment._acquire_source",
            return_value=source_result,
        ),
        patch(
            "fox_experiments.training.experiment._select_learning_rate",
            side_effect=AssertionError("must skip trials"),
        ),
    ):
        result = run_experiment(data, config, tmp_path / "run", allow_unqualified=False)
    assert result["status"] == "acquisition_failed"
    assert not (tmp_path / "run/retrieval_raw.csv").exists()
    failures = json.loads((tmp_path / "run/failures.json").read_text())
    assert failures[0]["status"] == "short_acquisition_failed"
    specification = json.loads((tmp_path / "run/run_specification.json").read_text())
    assert specification["acquisition_policy"] == {"allow_unqualified": False}
    with pytest.raises(ValueError, match="acquisition policy"):
        run_experiment(
            data, config, tmp_path / "run", resume=True, allow_unqualified=True
        )
