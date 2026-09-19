"""Compute E_t(r) envelopes from existing scalar histories without retraining."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "library"))

from fox_restricted.legacy import asymptotics
from fox_restricted.legacy.asymptotics import scalar_snapshot, evaluate_snapshot
from fox_restricted.legacy.runner import write_csv, dump_json


def reanalyze(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty analysis output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    R = int(config["config"]["R"])
    config["analysis_only"] = True
    config["analysis_source"] = str(source.resolve())
    config["analysis_history_sha256"] = hashlib.sha256((source / "history.csv").read_bytes()).hexdigest()
    config["analysis_method"] = "Uniform upper and infinite-prefix witness lower from recorded scalars; no retraining"
    config["analysis_code_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(asymptotics.__file__))
    }
    dump_json(output / "config.json", config)
    for name in ("history.csv", "runs.csv", "lag_probabilities.csv"):
        if (source / name).exists():
            (output / name).write_bytes((source / name).read_bytes())
    with (source / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    acquired_gaps = {
        (row["seed"], row["gate_mode"], row["optimizer"]): float(row["delta_min"])
        for row in history if int(row["step"]) == 3
    }
    errors, radii = [], []
    for row in history:
        key = tuple(row[name] for name in ("seed", "gate_mode", "optimizer"))
        tags = dict(seed=int(row["seed"]), gate_mode=row["gate_mode"], optimizer=row["optimizer"])
        state = scalar_snapshot(*(float(row[name]) for name in ("delta_min", "m", "h", "rho", "w")),
                                row_norm_max=float(row["row_norm_max"]))
        current_errors, current_radii = evaluate_snapshot(
            state, R, row["gate_mode"], tags, int(row["step"]), float(row["S"]),
            reference_gap=acquired_gaps.get(key) if int(row["step"]) >= 3 else None,
            fixed_lags=(1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512),
            clock_coefficients=(.001, .003, .01, .03, .1),
        )
        errors.extend(current_errors)
        radii.extend(current_radii)
    write_csv(output / "asymptotic_errors.csv", errors)
    write_csv(output / "certified_radii.csv", radii)
    return str(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(reanalyze(args.source, args.output))
