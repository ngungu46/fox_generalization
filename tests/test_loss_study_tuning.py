"""Short-only selection must not consume the long generalization test."""

from dataclasses import replace
import json

import pandas as pd
import pytest
import torch

from fox_experiments.loss_study.config import LossStudyConfig, branches, intervention
from fox_experiments.loss_study.data import StudyData
from fox_experiments.loss_study.tuning import tune_learning_rates


def test_tuning_uses_only_short_validation_and_retains_both_objectives(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    cfg = replace(
        LossStudyConfig.for_profile("smoke"),
        gate_modes=("factorized_constant",),
        optimizers=("adam_fixed",),
    )
    original = StudyData.evaluation_examples
    seen = []

    def checked(self, split="short", seed=0):
        assert (
            split == "tune"
        ), "Learning-rate selection accessed confirmation or long test"
        seen.append(split)
        return original(self, split=split, seed=seed)

    monkeypatch.setattr(StudyData, "evaluation_examples", checked)
    result = tune_learning_rates(
        cfg, tmp_path / "tuning", steps=1, multipliers=(0.5, 1)
    )
    assert len(seen) == 4
    assert set(result.branch_learning_rates) == {r["branch"] for r in branches(cfg)}
    trials = pd.read_csv(tmp_path / "tuning/trials.csv")
    assert len(trials) == 4
    assert trials.status.eq("ok").all()
    assert set(trials.objective) == {"answer_only", "all_tokens"}
    assert not cfg.branch_learning_rates  # No mutation of core matched-rate config.
    saved = json.loads((tmp_path / "tuning/selected_config.json").read_text())
    assert saved["branch_learning_rates"] == result.branch_learning_rates
    with pytest.raises(FileExistsError):
        tune_learning_rates(cfg, tmp_path / "tuning", steps=1)


def test_interventions_preserve_literal_template_envelope():
    cfg = intervention(LossStudyConfig.for_profile("colab"), "theory_template_support")
    assert cfg.train_max_lag == 2 and cfg.train_max_prefix == 3
    assert cfg.short_lags == (1, 2)
    with pytest.raises(ValueError, match="Literal theory"):
        replace(cfg, train_max_lag=4)
    with pytest.raises(ValueError, match="parser"):
        replace(cfg, data_mode="text_background")
