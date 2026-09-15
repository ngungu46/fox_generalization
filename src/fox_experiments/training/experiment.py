"""Coordinate acquisition, matched optimizer branches, and held-out evaluation.

This module owns experiment order and artifacts. Model equations live in models/,
data generation in data/, update rules in optimizers.py, and metrics in evaluation/.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from fox_experiments.data import LatestWriteData
from fox_experiments.models import FoXLM, ModelConfig
from fox_experiments.evaluation.metrics import evaluate_branch, lm_evaluate, short_score
from .artifacts import (
    atomic_checkpoint,
    dataset_identity,
    dump_json,
    exclusive_run_lock,
    finite_json_row,
    read_table,
    state_hash,
    table,
)
from .config import ExperimentConfig, branch_specs, budget, source_gate_for
from .trainer import train_segment


def _implementation_identity() -> dict[str, str]:
    """Record source contents so cached results never silently mix implementations."""
    package = Path(__file__).resolve().parents[1]
    result = {}
    for folder in ("models", "data", "training", "evaluation"):
        for path in sorted((package / folder).rglob("*.py")):
            result[str(path.relative_to(package))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def _initialize_run(
    out: Path,
    data: LatestWriteData,
    config: ExperimentConfig,
    device: str,
    resume: bool,
) -> None:
    """Require deliberate reuse and validate the config, data, and implementation."""
    populated = any(path.name != ".run.lock" for path in out.iterdir())
    if populated and not resume:
        raise FileExistsError(
            f"Run directory already contains artifacts: {out}. "
            "Choose a new run name, or pass resume=True to continue this exact run."
        )
    signature = {
        "config": config.to_dict(),
        "dataset": dataset_identity(data),
        "implementation": _implementation_identity(),
    }
    signature_path = out / "run_specification.json"
    if populated:
        if not signature_path.exists():
            raise ValueError(
                "Existing directory has no run specification; choose a new output directory"
            )
        previous = json.loads(signature_path.read_text())
        if json.dumps(previous, sort_keys=True) != json.dumps(
            signature, sort_keys=True
        ):
            raise ValueError(
                "Resume configuration, dataset, or implementation differs from the saved run. "
                "Choose a new run name to keep results separate."
            )
    else:
        dump_json(signature_path, signature)
        dump_json(out / "config.json", config.to_dict())
        dump_json(out / "budget.json", budget(config))
        dump_json(
            out / "environment.json",
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
        )


def _acquire_source(
    data: LatestWriteData,
    config: ExperimentConfig,
    out: Path,
    device: str,
    seed: int,
    source_gate: str,
) -> tuple[FoXLM, dict[str, Any]]:
    """Acquire one shared short-task function before changing parameterization."""
    model = FoXLM(
        ModelConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            gate_mode=source_gate,
            g0=config.g0,
            other_g0=config.other_g0,
            seed=seed,
        )
    ).to(device)
    directory = out / f"seed{seed}" / f"acquire_{source_gate}"
    _, status = train_segment(
        model,
        data,
        config,
        "paper_adamw",
        config.acquisition_lr,
        config.pretrain_steps,
        seed,
        directory / "natural_lm",
        task_probability=0.0,
    )
    if status == "complete":
        _, status = train_segment(
            model,
            data,
            config,
            "paper_adamw",
            config.acquisition_lr,
            config.acquisition_steps,
            seed + 10000,
            directory / "mixed_task",
        )
    metrics_path = directory / "acquisition_metrics.json"
    cached = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    if status == "complete" and cached and cached["status"] == "complete":
        row = cached
    else:
        metrics = (
            short_score(model, data, config, "confirm", seed)
            if status == "complete"
            else {}
        )
        row = {
            "seed": seed,
            "source_gate": source_gate,
            "status": status,
            "qualified": status == "complete"
            and metrics["worst_accuracy"] >= config.short_threshold,
            **{key: value for key, value in metrics.items() if key != "panels"},
        }
        dump_json(metrics_path, row)
    if status == "complete":
        model_path = directory / "acquired_model.pt"
        if not model_path.exists():
            atomic_checkpoint(model_path, model.state_dict())
        if not row["qualified"]:
            print(
                f"Acquisition {seed}/{source_gate}: short rule not acquired; branches remain diagnostic.",
                flush=True,
            )
    return model, row


def _select_learning_rate(
    base: FoXLM,
    data: LatestWriteData,
    config: ExperimentConfig,
    branch: Path,
    arm: str,
    seed: int,
) -> tuple[int, float] | None:
    """Tune only on the tune split, then discard trial weights and moments."""
    rates = config.sgd_lrs if arm == "sgd" else config.adam_lrs
    candidates = []
    for index, learning_rate in enumerate(rates):
        directory = branch / f"trial{index}"
        metrics_path = directory / "metrics.json"
        cached = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        if cached and cached["status"] == "complete":
            row = cached
        else:
            model = copy.deepcopy(base)
            _, status = train_segment(
                model,
                data,
                config,
                arm,
                learning_rate,
                config.trial_steps,
                seed + 20000,
                directory,
            )
            score = (
                short_score(model, data, config, "tune", seed)
                if status == "complete"
                else None
            )
            row = {
                "branch": branch.name,
                "trial": index,
                "lr": learning_rate,
                "status": status,
                **(
                    {key: value for key, value in score.items() if key != "panels"}
                    if score
                    else {}
                ),
            }
            dump_json(metrics_path, row)
            del model
        if row["status"] == "complete":
            candidates.append(
                (row["worst_accuracy"], -row["mean_nll"], index, learning_rate)
            )
    if not candidates:
        return None
    winner = max(candidates)
    selection = {
        "trial": winner[2],
        "lr": winner[3],
        "criterion": "worst tune accuracy, then mean tune NLL",
    }
    if not (branch / "selected_lr.json").exists():
        dump_json(branch / "selected_lr.json", selection)
    return winner[2], winner[3]


def _evaluate_checkpoint(
    model: FoXLM,
    data: LatestWriteData,
    config: ExperimentConfig,
    branch: Path,
    device: str,
    seed: int,
    gate: str,
    arm: str,
    step: int,
    initial_hash: str,
    learning_rate: float,
    lr_index: int,
) -> None:
    """Cache a complete evaluation before marking it available for aggregation."""
    directory = branch / "evaluation" / f"step{step:06d}"
    if (directory / "complete.json").exists():
        return
    saved = torch.load(
        branch / "continuation" / f"checkpoint_{step:06d}.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(saved["model"])
    short = short_score(model, data, config, "confirm", seed)
    probes = evaluate_branch(model, data, config, seed)
    language_modeling = lm_evaluate(model, data, config, "test", 800000 + seed * 100)
    metadata = {
        "branch": branch.name,
        "gate": gate,
        "optimizer": arm,
        "seed_id": seed,
        "checkpoint_step": step,
        "S": saved["S"],
        "short_qualified": short["worst_accuracy"] >= config.short_threshold,
        "software_only": config.profile == "smoke",
    }
    probes = [finite_json_row({**row, **metadata}) for row in probes]
    language_modeling = [{**row, **metadata} for row in language_modeling]
    rates = config.sgd_lrs if arm == "sgd" else config.adam_lrs
    summary = {
        **metadata,
        "selected_lr": learning_rate,
        "lr_grid_endpoint": lr_index in (0, len(rates) - 1),
        "initial_state_hash": initial_hash,
        "source_gate": source_gate_for(gate),
        "confirm_worst_accuracy": short["worst_accuracy"],
        "confirm_nll": short["mean_nll"],
        "test_base_accuracy": float(
            np.mean([row["correct"] for row in probes if row["variant"] == "base"])
        ),
    }
    table(directory / "retrieval_raw.csv", probes)
    table(directory / "lm_raw.csv", language_modeling)
    dump_json(directory / "summary.json", summary)
    dump_json(directory / "complete.json", {"status": "complete", "step": step})
    print(
        branch.name,
        "step",
        step,
        "short min",
        round(short["worst_accuracy"], 3),
        "test mean",
        round(summary["test_base_accuracy"], 3),
        flush=True,
    )


def _aggregate_cached_results(out: Path) -> None:
    """Rebuild tables from all completed shards, preserving earlier resumed results."""
    acquisition = [
        json.loads(path.read_text())
        for path in sorted(out.glob("seed*/acquire_*/acquisition_metrics.json"))
    ]
    trials = [
        json.loads(path.read_text())
        for path in sorted(out.glob("seed*_*/trial*/metrics.json"))
    ]
    summaries, probes, language_modeling = [], [], []
    for marker in sorted(out.glob("seed*_*/evaluation/step*/complete.json")):
        directory = marker.parent
        summaries.append(json.loads((directory / "summary.json").read_text()))
        probes.extend(read_table(directory / "retrieval_raw.csv"))
        language_modeling.extend(read_table(directory / "lm_raw.csv"))
    table(out / "acquisition.csv", acquisition)
    table(out / "lr_trials.csv", trials)
    # Do not replace already present research outputs with empty partial tables.
    if summaries:
        table(out / "summary.csv", summaries)
        table(out / "retrieval_raw.csv", probes)
        table(out / "lm_raw.csv", language_modeling)


def _result_paths(out: Path) -> dict[str, str]:
    return {
        "output_dir": str(out),
        "summary": str(out / "summary.csv"),
        "retrieval": str(out / "retrieval_raw.csv"),
        "lm": str(out / "lm_raw.csv"),
        "failures": str(out / "failures.json"),
    }


def run_experiment(
    data: LatestWriteData,
    config: ExperimentConfig,
    out_dir: str | Path,
    device: str = "cpu",
    *,
    resume: bool = False,
) -> dict[str, str]:
    """Run the full small-scale study with explicit, checked resume semantics.

    Acquisition is shared by paired factorized/direct constant-gate branches.
    Learning rates are chosen on tune data. Confirm data qualifies the short rule;
    the frozen lag/prefix test grid is never used for learning-rate selection.
    Unqualified branches remain visible as diagnostics. Read failures.json before
    interpreting outputs. Set resume=True only to continue this exact run.
    """
    out = Path(out_dir)
    with exclusive_run_lock(out):
        _initialize_run(out, data, config, device, resume)
        status_path = out / "run_status.json"
        if (
            status_path.exists()
            and json.loads(status_path.read_text())["status"] == "complete"
        ):
            _aggregate_cached_results(out)
            return _result_paths(out)
        dump_json(
            status_path,
            {"status": "running", "software_only": config.profile == "smoke"},
        )
        failures = []
        for seed in config.seeds:
            sources = {}
            source_gates = {source_gate_for(gate) for gate, _ in branch_specs(config)}
            for source_gate in sorted(source_gates):
                source, row = _acquire_source(
                    data, config, out, device, seed, source_gate
                )
                _aggregate_cached_results(out)
                if row["status"] != "complete":
                    failures.append(row)
                    continue
                sources[source_gate] = source
            for gate, arm in branch_specs(config):
                source_gate = source_gate_for(gate)
                if source_gate not in sources:
                    continue
                base = sources[source_gate].converted(gate)
                branch = out / f"seed{seed}_{gate}_{arm}"
                branch.mkdir(exist_ok=True)
                check_example = data.example(
                    "tune", 1234, config.train_lags[0], config.train_length
                )
                check = torch.tensor([check_example["tokens"]], device=device)
                # The intervention must preserve the starting function before any
                # optimizer update. Raw parameter hashes need not match across gates.
                with torch.no_grad():
                    torch.testing.assert_close(
                        base(check), sources[source_gate](check), rtol=2e-5, atol=2e-5
                    )
                selection = _select_learning_rate(base, data, config, branch, arm, seed)
                _aggregate_cached_results(out)
                if selection is None:
                    failures.append(
                        {"branch": branch.name, "status": "all_lr_trials_failed"}
                    )
                    continue
                lr_index, learning_rate = selection
                model = copy.deepcopy(base)
                initial_hash = state_hash(model)
                checkpoint_steps = sorted(
                    {max(1, config.continuation_steps // 2), config.continuation_steps}
                )
                _, status = train_segment(
                    model,
                    data,
                    config,
                    arm,
                    learning_rate,
                    config.continuation_steps,
                    seed + 30000,
                    branch / "continuation",
                    checkpoint_steps=checkpoint_steps,
                )
                if status != "complete":
                    failures.append({"branch": branch.name, "status": status})
                    continue
                for step in checkpoint_steps:
                    _evaluate_checkpoint(
                        model,
                        data,
                        config,
                        branch,
                        device,
                        seed,
                        gate,
                        arm,
                        step,
                        initial_hash,
                        learning_rate,
                        lr_index,
                    )
                    _aggregate_cached_results(out)
                del model, base
            del sources
        dump_json(out / "failures.json", failures)
        dump_json(
            status_path,
            {
                "status": "complete" if not failures else "complete_with_failures",
                "failed_branches_or_sources": len(failures),
                "software_only": config.profile == "smoke",
            },
        )
        return _result_paths(out)
