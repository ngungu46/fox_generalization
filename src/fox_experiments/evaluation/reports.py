"""Build reusable finite-grid tables and figures from saved experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .aggregation import (
    aggregate_histories,
    finite_grid_ranges,
    paired_optimizer_contrasts,
)
from .plotting import plot_language_modeling, plot_retrieval_panels
from .diagnostics import diagnose_run


def summarize_results(
    out_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Recreate analysis without loading or retraining a model.

    Reports are exploratory. Read short_qualified, failures.json, and the training
    budget before interpreting a passing finite grid as evidence about the theory.
    A finite empirical sweep cannot establish infinite-distance generalization.
    """
    output = Path(out_dir)
    required = [output / "retrieval_raw.csv", output / "lm_raw.csv"]
    missing = [
        path.name for path in required if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"No completed checkpoint evaluation for {missing}. "
            "Run or resume training first; inspect failures.json and stage status.json files."
        )
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    raw = pd.read_csv(required[0])
    language_modeling = pd.read_csv(required[1])
    histories, panels = aggregate_histories(raw)
    threshold = 0.9
    if (output / "config.json").exists():
        threshold = json.loads((output / "config.json").read_text()).get(
            "short_threshold", 0.9
        )
    ranges = finite_grid_ranges(panels, threshold=threshold)
    contrasts = paired_optimizer_contrasts(histories)
    histories.to_csv(output / "history_level.csv", index=False)
    panels.to_csv(output / "retrieval_panels.csv", index=False)
    ranges.to_csv(output / "finite_grid_ranges.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    software_only = bool(raw.software_only.eq(True).all())
    paths = plot_retrieval_panels(panels, figures, software_only)
    paths.append(plot_language_modeling(language_modeling, figures, software_only))
    diagnose_run(output)
    return panels, ranges, contrasts, paths
