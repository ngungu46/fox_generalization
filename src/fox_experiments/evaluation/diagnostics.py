"""Explain zero retrieval scores from saved predictions, without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregation import HISTORY_KEYS, aggregate_histories, finite_grid_ranges


def diagnose_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    """Separate guessing, edit sensitivity, and the conjunction of all edits.

    A constant answer can occasionally be correct but cannot pass both a base
    history and its latest-value edit. These diagnostics retain that distinction.
    They do not replace the primary all-edit score or change its threshold.
    """
    histories, panels = aggregate_histories(raw)
    ranges = finite_grid_ranges(panels)
    rows = []
    identity = [key for key in HISTORY_KEYS if key not in ("short_qualified", "S")]
    for (branch, step), frame in raw.groupby(["branch", "checkpoint_step"]):
        history = histories[
            (histories.branch == branch) & (histories.checkpoint_step == step)
        ]
        measured = ranges[(ranges.branch == branch) & (ranges.step == step)]
        by_variant = frame.groupby("variant").correct.mean()
        predictions = frame.pivot(
            index=identity, columns="variant", values="prediction"
        )
        changed = np.nan
        if {"base", "latest_flip"} <= set(predictions.columns):
            paired = predictions[["base", "latest_flip"]].dropna()
            if len(paired):
                changed = float((paired.base != paired.latest_flip).mean())
        count = frame.prediction.value_counts()
        qualified = bool(frame.short_qualified.astype(str).str.lower().eq("true").all())
        zero_all_edits = bool((history.all_edits_correct == 0).all())
        radius = (
            int(measured.largest_passing_grid_lag.iloc[0]) if len(measured) else None
        )
        reasons = []
        if not qualified:
            reasons.append("short_rule_not_acquired")
        if len(count) == 1:
            reasons.append("constant_prediction_on_entire_test_panel")
        if changed == 0:
            reasons.append("prediction_does_not_respond_to_latest_value_edits")
        if zero_all_edits and frame.correct.mean() > 0:
            reasons.append("some_answers_correct_but_no_history_passes_all_edits")
        if radius == 0:
            reasons.append("no_initial_tested_lag_passes_all_prefix_panels")
        rows.append(
            {
                "branch": branch,
                "checkpoint_step": step,
                "gate": frame.gate.iloc[0],
                "optimizer": frame.optimizer.iloc[0],
                "short_qualified": qualified,
                "n_predictions": len(frame),
                "n_base_histories": len(history),
                "individual_answer_accuracy": float(frame.correct.mean()),
                "base_accuracy": by_variant.get("base", np.nan),
                "latest_flip_accuracy": by_variant.get("latest_flip", np.nan),
                "all_edit_history_accuracy": float(history.all_edits_correct.mean()),
                "latest_edit_prediction_change_rate": changed,
                "distinct_prediction_tokens": len(count),
                "most_common_prediction_token": int(count.index[0]),
                "most_common_prediction_fraction": float(count.iloc[0] / len(frame)),
                "mean_answer_probability": float(frame.probability.mean()),
                "mean_answer_nll": float(frame.nll.mean()),
                "largest_passing_tested_lag": radius,
                "diagnosis": (
                    "; ".join(reasons) if reasons else "inspect_lag_and_prefix_panels"
                ),
            }
        )
    return pd.DataFrame(rows)


def diagnose_run(out_dir: str | Path, destination: str | Path | None = None) -> dict:
    """Write a diagnostic table/report beside a run or in a separate directory.

    Existing raw predictions, summaries, weights and scientific scores are never
    changed. A results-only ZIP is sufficient; checkpoints are not required.
    """
    source = Path(out_dir)
    output = Path(destination) if destination is not None else source / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    lines = ["# Retrieval diagnosis", "", "Source: `" + str(source.resolve()) + "`", ""]
    acquisition_path = source / "acquisition.csv"
    if acquisition_path.exists():
        acquisition = pd.read_csv(acquisition_path)
        if "qualified" in acquisition:
            passed = acquisition.qualified.astype(str).str.lower().eq("true")
            lines.append(
                f"Short acquisition: **{int(passed.sum())}/{len(passed)} sources qualified**."
            )
            if not passed.all():
                lines.append(
                    "Unqualified sources do not establish an optimizer generalization boundary."
                )
            lines.append("")
    raw_path = source / "retrieval_raw.csv"
    result = {"source": str(source), "report": str(output / "diagnosis.md")}
    if raw_path.exists() and raw_path.stat().st_size:
        raw = pd.read_csv(raw_path)
        diagnostics = diagnose_predictions(raw)
        csv_path = output / "retrieval_diagnosis.csv"
        diagnostics.to_csv(csv_path, index=False)
        last = diagnostics[
            diagnostics.checkpoint_step
            == diagnostics.groupby("branch").checkpoint_step.transform("max")
        ]
        lines += [
            "## Final checkpoint in each branch",
            "",
            "| Branch | Base accuracy | All edits | Changed after latest edit | Most common token ID | Fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in last.itertuples():
            lines.append(
                f"| {row.branch} | {row.base_accuracy:.3f} | {row.all_edit_history_accuracy:.3f} | "
                f"{row.latest_edit_prediction_change_rate:.3f} | {row.most_common_prediction_token} | "
                f"{row.most_common_prediction_fraction:.3f} |"
            )
        lines += [
            "",
            "A zero all-edit score can coexist with nonzero individual accuracy. "
            "If the prediction never changes after the latest value changes, it cannot correctly answer both versions.",
            "",
            "A tested range of zero means the first tested lag failed the conjunction over prefix panels. "
            "It is not an estimate of a boundary at lag zero; smaller untested distances may pass.",
        ]
        result["csv"] = str(csv_path)
        result["constant_final_branches"] = int(
            (last.distinct_prediction_tokens == 1).sum()
        )
        result["unqualified_final_branches"] = int((~last.short_qualified).sum())
    else:
        lines += [
            "No saved retrieval predictions yet. Inspect acquisition before launching continuation."
        ]
        result["status"] = "no_retrieval_predictions"
    lm_path = source / "lm_raw.csv"
    if lm_path.exists():
        lm = pd.read_csv(lm_path)
        lines += [
            "",
            "## Language-model loss",
            "",
            f"Saved token/context NLL spans {lm.nll.min():.6g}–{lm.nll.max():.6g} nats.",
            "Language-model NLL, full-vocabulary retrieval accuracy, and a controlled model's "
            "99% sufficient radius are different quantities. A flat context curve does not mean zero loss.",
        ]
    (output / "diagnosis.md").write_text("\n".join(lines) + "\n")
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
