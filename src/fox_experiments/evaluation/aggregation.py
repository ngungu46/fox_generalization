"""History-level accuracy, finite-grid ranges, and paired optimizer comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_KEYS = [
    "branch",
    "gate",
    "optimizer",
    "seed_id",
    "checkpoint_step",
    "S",
    "short_qualified",
    "lag",
    "prefix_copies",
    "kind",
    "history_id",
]
PANEL_KEYS = HISTORY_KEYS[:-1]


def aggregate_histories(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all edit variants from a base history together in the success event."""
    histories = (
        raw.groupby(HISTORY_KEYS, dropna=False)
        .agg(
            all_edits_correct=("correct", "min"),
            mean_nll=("nll", "mean"),
            worst_probability=("probability", "min"),
        )
        .reset_index()
    )
    panels = (
        histories.groupby(PANEL_KEYS, dropna=False)
        .agg(
            history_accuracy=("all_edits_correct", "mean"),
            n_histories=("all_edits_correct", "size"),
            mean_nll=("mean_nll", "mean"),
        )
        .reset_index()
    )
    return histories, panels


def finite_grid_ranges(panels: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Stop at the first failing tested lag across all tested stale-prefix sizes.

    Passing the largest tested lag is right-censored, not an infinite-context
    finding. Untested smaller/intervening lags are not claimed to pass.
    """
    rows = []
    conflict = panels[panels.kind == "conflict"]
    for (branch, step), frame in conflict.groupby(["branch", "checkpoint_step"]):
        grid = sorted(frame.lag.unique())
        radius = 0
        first_failed_lag = None
        invalid_measurement = False
        for lag in grid:
            accuracy = frame[frame.lag == lag].history_accuracy.to_numpy()
            if not np.isfinite(accuracy).all() or accuracy.min() < threshold:
                first_failed_lag = lag
                invalid_measurement = not np.isfinite(accuracy).all()
                break
            radius = lag
        qualified = bool(frame.short_qualified.astype(str).str.lower().eq("true").all())
        if invalid_measurement:
            interpretation = "invalid_measurement"
        elif not qualified:
            interpretation = "short_rule_not_acquired"
        elif radius == 0:
            interpretation = "first_tested_lag_failed"
        elif radius == max(grid):
            interpretation = "test_ceiling_reached"
        else:
            interpretation = "finite_tested_range"
        rows.append(
            {
                "branch": branch,
                "step": step,
                "largest_passing_grid_lag": radius,
                "first_tested_lag": min(grid),
                "last_tested_lag": max(grid),
                "right_censored": radius == max(grid),
                "short_qualified": qualified,
                "first_failed_tested_lag": first_failed_lag,
                "interpretation": interpretation,
                "threshold": threshold,
                "note": "Only tested grid points; intervening and smaller untested lags are not certified",
            }
        )
    return pd.DataFrame(rows)


def paired_optimizer_contrasts(
    histories: pd.DataFrame,
    bootstrap_samples: int = 500,
    seed: int = 17,
) -> pd.DataFrame:
    """Exploratory paired differences; resample seeds and histories, never edits.

    One training seed cannot estimate training-seed variability. These pointwise
    bootstrap intervals neither correct for testing many lags nor certify a range.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for gate in histories.gate.unique():
        frame = histories[(histories.gate == gate) & (histories.kind == "conflict")]
        frame = frame[frame.checkpoint_step == frame.checkpoint_step.max()]
        index = ["seed_id", "history_id", "lag", "prefix_copies"]
        paired = frame.pivot(
            index=index, columns="optimizer", values="all_edits_correct"
        ).reset_index()
        for adam in ("adam_fixed", "adam_annealed"):
            if adam not in paired or "sgd" not in paired:
                continue
            for (lag, prefix), group in paired.groupby(["lag", "prefix_copies"]):
                group = group.dropna(subset=[adam, "sgd"])
                if group.empty:
                    continue
                seeds = group.seed_id.unique()
                bootstrap_differences = []
                for _ in range(bootstrap_samples):
                    values = []
                    for sampled_seed in rng.choice(seeds, len(seeds), replace=True):
                        seed_rows = group[group.seed_id == sampled_seed]
                        differences = (seed_rows[adam] - seed_rows.sgd).to_numpy()
                        values.extend(
                            rng.choice(differences, len(differences), replace=True)
                        )
                    bootstrap_differences.append(np.mean(values))
                rows.append(
                    {
                        "gate": gate,
                        "adam": adam,
                        "lag": lag,
                        "prefix_copies": prefix,
                        "difference": (group[adam] - group.sgd).mean(),
                        "ci_low": np.quantile(bootstrap_differences, 0.025),
                        "ci_high": np.quantile(bootstrap_differences, 0.975),
                        "seeds": len(seeds),
                        "note": "Exploratory pointwise paired bootstrap; one seed cannot estimate training-seed variability",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "gate",
            "adam",
            "lag",
            "prefix_copies",
            "difference",
            "ci_low",
            "ci_high",
            "seeds",
            "note",
        ],
    )
