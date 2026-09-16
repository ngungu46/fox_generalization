"""Compare named saved interventions without silently replacing missing runs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_conditions(runs, baseline="baseline", destination=None):
    """Return seed-matched final outcomes and every changed config field.

    This is a descriptive table. One-at-a-time changes do not prove necessity;
    censored radii and unqualified branches never receive a numerical radius
    effect. Run identifiers/config differences remain visible to the reader.
    """
    if baseline not in runs:
        raise ValueError("The baseline must be present in runs")
    keys = ["seed", "gate_mode", "optimizer", "objective"]
    loaded = {}
    for name, directory in runs.items():
        path = Path(directory)
        cfg = json.loads((path / "config.json").read_text())
        frame = pd.read_csv(path / "summary.csv")
        if not set(keys + ["step"]).issubset(frame):
            raise ValueError(f"{name} lacks a complete branch summary schema")
        final = frame.sort_values("step").groupby(keys, as_index=False).tail(1)
        spec_file = path / "run_specification.json"
        spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
        loaded[name] = (cfg, final, spec)
    base_cfg, base, base_spec = loaded[baseline]
    outputs = []
    for name, (cfg, frame, spec) in loaded.items():
        changed = {
            key: {"baseline": base_cfg.get(key), "condition": cfg.get(key)}
            for key in sorted(set(base_cfg) | set(cfg))
            if base_cfg.get(key) != cfg.get(key)
        }
        paired = base.merge(
            frame,
            on=keys,
            how="outer",
            suffixes=("_baseline", "_condition"),
            indicator=True,
            validate="one_to_one",
        )
        for _, row in paired.iterrows():
            result = {key: row[key] for key in keys}
            result.update(
                condition=name,
                baseline=baseline,
                changed_fields=json.dumps(changed, sort_keys=True),
                status="missing_pair",
                radius_effect=np.nan,
                short_accuracy_effect=np.nan,
                baseline_path=str(Path(runs[baseline]).resolve()),
                condition_path=str(Path(runs[name]).resolve()),
            )
            if row["_merge"] == "both":
                for label in ("baseline", "condition"):
                    for metric in (
                        "step",
                        "short_qualified",
                        "short_joint_accuracy_worst",
                        "tested_grid_radius",
                        "right_censored",
                        "branch_status",
                    ):
                        result[f"{label}_{metric}"] = row.get(
                            f"{metric}_{label}", np.nan
                        )
                result["short_accuracy_effect"] = row.get(
                    "short_joint_accuracy_worst_condition", np.nan
                ) - row.get("short_joint_accuracy_worst_baseline", np.nan)
                complete = all(
                    row.get(f"branch_status_{label}") == "complete"
                    for label in ("baseline", "condition")
                )
                fit = all(
                    row.get(f"short_qualified_{label}") == True
                    for label in ("baseline", "condition")
                )
                censored = any(
                    row.get(f"right_censored_{label}") == True
                    for label in ("baseline", "condition")
                )
                grid_match = all(
                    cfg.get(k) == base_cfg.get(k)
                    for k in (
                        "eval_lags",
                        "eval_prefixes",
                        "eval_histories",
                        "qualification_threshold",
                    )
                )
                same_sources = bool(base_spec.get("sources")) and spec.get(
                    "sources"
                ) == base_spec.get("sources")
                same_protocol = bool(
                    base_spec.get("data", {}).get("protocol")
                ) and spec.get("data", {}).get("protocol") == base_spec.get(
                    "data", {}
                ).get(
                    "protocol"
                )
                result["same_sources"] = same_sources
                result["same_generator_protocol"] = same_protocol
                if not same_sources or not same_protocol:
                    result["status"] = "changed_or_unverified_source_provenance"
                elif not complete:
                    result["status"] = "incomplete_branch"
                elif (
                    cfg["steps"] != base_cfg["steps"]
                    or row.step_baseline != row.step_condition
                ):
                    result["status"] = "different_training_budget"
                elif not grid_match:
                    result["status"] = "different_evaluation_panel"
                elif not fit:
                    result["status"] = "short_unqualified"
                elif censored:
                    result["status"] = "censored_compare_accuracy_instead"
                else:
                    result["status"] = "paired_finite_grid"
                    result["radius_effect"] = (
                        row.tested_grid_radius_condition
                        - row.tested_grid_radius_baseline
                    )
            outputs.append(result)
    result = pd.DataFrame(outputs)
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result
