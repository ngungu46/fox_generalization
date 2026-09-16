"""Never turn censored, unqualified or unpaired outcomes into numerical effects."""

import json
import pandas as pd

from fox_experiments.loss_study.comparison import compare_conditions
from fox_experiments.loss_study.config import LossStudyConfig


def test_condition_comparison_keeps_missing_and_censored_effects(tmp_path):
    cfg = LossStudyConfig.for_profile("smoke").to_dict()
    rows = [
        dict(
            seed=0,
            gate_mode="factorized_constant",
            optimizer="sgd",
            objective="answer_only",
            step=2,
            short_qualified=True,
            short_joint_accuracy_worst=1.0,
            tested_grid_radius=2,
            right_censored=False,
            branch_status="complete",
        )
    ]
    runs = {}
    for label, qualified, censored in (
        ("baseline", True, False),
        ("unqualified", False, False),
        ("ceiling", True, True),
        ("different_code", True, False),
    ):
        out = tmp_path / label
        out.mkdir()
        (out / "config.json").write_text(json.dumps(cfg))
        spec = {
            "sources": {
                "runner.py": "changed" if label == "different_code" else "same"
            },
            "data": {"protocol": "same"},
        }
        (out / "run_specification.json").write_text(json.dumps(spec))
        frame = pd.DataFrame(rows)
        frame["short_qualified"] = qualified
        frame["right_censored"] = censored
        frame.to_csv(out / "summary.csv", index=False)
        runs[label] = out
    result = compare_conditions(runs).set_index("condition")
    assert result.loc["baseline", "radius_effect"] == 0
    assert result.loc["unqualified", "status"] == "short_unqualified"
    assert result.loc["ceiling", "status"] == "censored_compare_accuracy_instead"
    assert (
        result.loc["different_code", "status"]
        == "changed_or_unverified_source_provenance"
    )
    assert (
        result.loc[["unqualified", "ceiling", "different_code"], "radius_effect"]
        .isna()
        .all()
    )
