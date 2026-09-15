"""Use the same finite lag/prefix probes for full and downscaled checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import task_config
from .trainer import load_model


def evaluate(config, checkpoint, output, split="test", *, corpus=None, allow_cpu=False):
    from fox_experiments.data import LatestWriteData
    from fox_experiments.evaluation.metrics import (
        short_score,
        evaluate_branch,
        lm_evaluate,
    )
    from .data import NativeCorpus
    import tiktoken

    if split not in ("tune", "confirm", "test"):
        raise ValueError("Use tune, confirm, or test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Full evaluation requires CUDA; CPU is reserved for explicit software tests"
        )
    model, saved = load_model(checkpoint, device)
    # Preserve the predeclared task/evaluation grid stored with the checkpoint.
    # Only local storage destinations may change when moving checkpoints.
    if saved["specification"]["config"]["task"] != config.task:
        raise ValueError(
            "Evaluation task/grid changed from training config; create an explicitly documented follow-up study"
        )
    trained = saved["specification"]["config"]
    if trained["model"] != config.model or trained["precision"] != config.precision:
        raise ValueError(
            "Evaluation model/precision config must match the checkpoint's training specification"
        )
    data = LatestWriteData(
        (
            corpus
            if corpus is not None
            else NativeCorpus(config.data_root, config.strict_data)
        ),
        tiktoken.get_encoding("gpt2"),
    )
    task = task_config(config)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    spec = saved["specification"]
    seed = spec["config"]["seed"]
    metadata = {
        "branch": f"seed{seed}_{spec['stage']}_{spec['gate']}_{spec['arm']}",
        "gate": spec["gate"],
        "optimizer": spec["arm"],
        "stage": spec["stage"],
        "seed_id": spec["config"]["seed"],
        "checkpoint_step": saved["step"],
        "S": saved["S"],
        "software_only": allow_cpu,
        "short_threshold": task.short_threshold,
        "checkpoint": str(Path(checkpoint).resolve()),
    }
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config.precision == "bf16"
        else nullcontext()
    )
    with amp, torch.no_grad():
        short = short_score(
            model, data, task, "confirm" if split == "test" else split, seed
        )
        metadata["short_qualified"] = short["worst_accuracy"] >= task.short_threshold
        summary = {
            **metadata,
            "confirm_worst_accuracy": short["worst_accuracy"],
            "confirm_nll": short["mean_nll"],
            "selected_lr": spec["learning_rate"],
            "selection_split": split,
            "lr_grid_endpoint": False,
        }
        if split == "test":
            probes = evaluate_branch(model, data, task, seed)
            natural = lm_evaluate(model, data, task, "test", 800000 + seed * 100)
            probe_frame = pd.DataFrame([{**row, **metadata} for row in probes])
            probe_frame.to_csv(directory / "retrieval_raw.csv", index=False)
            pd.DataFrame([{**row, **metadata} for row in natural]).to_csv(
                directory / "lm_raw.csv", index=False
            )
            summary["test_base_accuracy"] = float(
                np.mean([row["correct"] for row in probes if row["variant"] == "base"])
            )
        else:
            pd.DataFrame(short["panels"]).to_csv(
                directory / "short_panels.csv", index=False
            )
    pd.DataFrame([summary]).to_csv(directory / "summary.csv", index=False)
    (directory / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, default=str), flush=True)
    return summary


def collect_reports(run_root):
    """Combine completed test evaluations, keeping checkpoint and seed IDs."""
    root = Path(run_root)
    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    collected = {}
    for name in ("summary.csv", "retrieval_raw.csv", "lm_raw.csv"):
        frames = []
        for path in sorted(root.glob("**/evaluation/*/" + name)):
            frames.append(pd.read_csv(path))
        if frames:
            result = pd.concat(frames, ignore_index=True)
            if name == "summary.csv" and "selection_split" in result:
                result = result[result.selection_split == "test"]
            if "stage" in result:
                source = result[result.stage == "source"]
                if len(source):
                    source.to_csv(destination / ("source_" + name), index=False)
                result = result[result.stage == "continuation"]
            if not len(result):
                continue
            if name == "summary.csv":
                thresholds = result.short_threshold.unique()
                if len(thresholds) != 1:
                    raise ValueError(
                        "Keep runs with different qualification thresholds in separate analysis roots"
                    )
                (destination / "config.json").write_text(
                    json.dumps({"short_threshold": float(thresholds[0])})
                )
            result.to_csv(destination / name, index=False)
            collected[name] = str(destination / name)
    if not collected:
        return {"status": "no_evaluations", "output": str(destination)}
    reports = []
    if "retrieval_raw.csv" in collected and "lm_raw.csv" in collected:
        from fox_experiments.evaluation import summarize_results

        _, _, _, paths = summarize_results(destination)
        reports = [str(path) for path in paths]
    return {
        "status": "collected",
        "output": str(destination),
        "files": collected,
        "plots": reports,
    }
