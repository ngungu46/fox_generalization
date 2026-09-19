"""Public one-experiment entry point; the notebook never contains training code."""
import csv
from pathlib import Path

from ..study import Run
from ..legacy.a100_runner import run_large_suite, _atomic_csv
from ..legacy.asymptotics import scalar_snapshot, error_bounds


def _add_ratio_probes(path, theta):
    """Evaluate the fixed m/h schedule from actual saved scalar checkpoints."""
    import math
    with (path / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    with (path / "asymptotic_errors.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["probe"] != "theory_ratio"]
    for row in history:
        reference = row.get("acquired_reference_gap")
        if not reference or float(reference) <= 0:
            rows.append(dict(seed=row["seed"], gate_mode=row["gate_mode"], optimizer=row["optimizer"],
                             step=row["step"], S=row["S"], probe="theory_ratio", theta=theta,
                             coefficient=None, reference_gap=reference, lag=None,
                             lower_error=None, upper_error=None, log_lower_error=None, log_upper_error=None,
                             bound_valid=False, bound_invalid_reason="positive_acquired_gap_unavailable"))
            continue
        state = scalar_snapshot(float(row["delta_min"]), float(row["m"]), float(row["h"]),
                                float(row["rho"]), float(row["w"]), float(row["row_norm_max"]))
        coefficient = theta * float(reference)
        lag = max(1, math.floor(coefficient * float(state.m / state.h)))
        rows.append(dict(seed=row["seed"], gate_mode=row["gate_mode"], optimizer=row["optimizer"],
                         step=row["step"], S=row["S"], probe="theory_ratio", theta=theta,
                         coefficient=coefficient, reference_gap=float(reference),
                         reference_gap_retained=state.delta_min >= float(reference),
                         m=float(state.m), h=float(state.h), delta_min=state.delta_min,
                         lag_definition="max(1, floor(theta * acquired_delta_min * m_t / h_t))",
                         **error_bounds(state, lag)))
    _atomic_csv(path / "asymptotic_errors.csv", rows)


def train(experiment, output_root, *, resume=True, progress=True, max_updates=None):
    """Train one named arm over its seeds, or resume its exact saved state.

    Pair data uses the unchanged, audited finite-population runner. Random
    data uses actual mixed-key/mixed-value stream derivatives in both the
    three acquisition updates and continuation; it has no hidden pair-data
    warm start. Both implementations retain every Adam buffer on resume.
    """
    path = Path(output_root).expanduser().resolve() / experiment.name
    if experiment.dataset == "random":
        from .random_training import run_random_experiment
        run_random_experiment(experiment, path, resume=resume, progress=progress,
                              max_updates=max_updates)
    else:
        from ..legacy.a100_runner import _atomic_json
        probe_manifest = path / "notebook_probes.json"
        expected = {"theta": experiment.theta,
                    "definition": "max(1, floor(theta * acquired_delta_min * m_t / h_t))"}
        if probe_manifest.exists():
            import json
            if json.loads(probe_manifest.read_text()) != expected:
                raise ValueError("Moving-probe definition changed; use a new output folder")
        try:
            run_large_suite(
                experiment.config, path, seeds=experiment.seeds,
                gate_modes=(experiment.gate_mode,), optimizers=(experiment.optimizer,),
                eval_lags=tuple(experiment.config.R + i for i in (1, 2, 3)), prefixes=(0, 64),
                log_every=experiment.log_every, checkpoint_every=experiment.checkpoint_every,
                check_every=experiment.check_every, resume=resume, progress=progress,
                fixed_lags=tuple(experiment.config.R + i for i in (1, 2, 3)),
                theta_values=(experiment.theta,), clock_coefficients=(.01,), max_updates=max_updates)
        finally:
            if (path / "history.csv").exists():
                _add_ratio_probes(path, experiment.theta)
                _atomic_json(probe_manifest, expected)
    return Run(path, experiment.name)


def summarize(runs):
    """Read actual completion/failure statuses; a stopped arm is never omitted."""
    records = []
    for run in runs:
        with (Path(run.path) / "runs.csv").open() as handle:
            records.extend(dict(experiment=run.name, **row) for row in csv.DictReader(handle))
    return records
