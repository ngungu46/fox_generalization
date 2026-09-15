"""Saved-result length comparisons and plots; no training is performed here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PaperBaselineConfig


def paired_paper_comparison(raw):
    """Pair final losses against original FoX + paper AdamW on identical targets.

    ``nll_difference = branch - paper`` is better when negative.
    ``extra_context_gain_difference = branch_gain - paper_gain`` is better when
    positive. Missing controls or target pairs produce a status and missing
    differences, never zero-filled evidence of equivalence.
    """
    columns = [
        "arm",
        "optimizer",
        "gate_mode",
        "checkpoint_step",
        "context_length",
        "train_length",
        "is_extrapolation",
        "status",
        "paired_documents",
        "branch_nll",
        "paper_nll",
        "nll_difference",
        "branch_context_gain",
        "paper_context_gain",
        "extra_context_gain_difference",
    ]
    final = raw[(raw.metric == "same_target_suffix") & (raw.checkpoint_step > 0)]
    reference = final[
        (final.optimizer == "paper_adamw") & (final.gate_mode == "original_data")
    ]
    keys = [
        "checkpoint_step",
        "sample",
        "doc_id",
        "window_offset",
        "context_length",
        "train_length",
        "target_start_offset",
        "target_count",
        "target_token_sha256",
    ]
    if "seed_id" in raw.columns:
        keys.append("seed_id")
    rows = []
    for (arm, step, context), frame in final.groupby(
        ["arm", "checkpoint_step", "context_length"]
    ):
        baseline = reference[
            (reference.checkpoint_step == step) & (reference.context_length == context)
        ]
        row = {
            "arm": arm,
            "optimizer": frame.optimizer.iloc[0],
            "gate_mode": frame.gate_mode.iloc[0],
            "checkpoint_step": step,
            "context_length": context,
            "train_length": frame.train_length.iloc[0],
            "is_extrapolation": bool(context > frame.train_length.iloc[0]),
            "branch_nll": frame.nll.mean(),
            "branch_context_gain": frame.nll_gain_vs_train_context.mean(),
            "status": "missing_paper_baseline",
            "paired_documents": 0,
            "paper_nll": np.nan,
            "nll_difference": np.nan,
            "paper_context_gain": np.nan,
            "extra_context_gain_difference": np.nan,
        }
        if not baseline.empty:
            paired = frame.merge(
                baseline[keys + ["nll", "nll_gain_vs_train_context"]],
                on=keys,
                how="left",
                suffixes=("", "_paper"),
                validate="one_to_one",
                indicator=True,
            )
            complete = paired._merge.eq("both")
            row["paired_documents"] = int(paired.loc[complete, "doc_id"].nunique())
            row["status"] = "missing_pairs"
            if complete.all() and len(frame) == len(baseline):
                row.update(
                    status="ok",
                    paper_nll=paired.nll_paper.mean(),
                    nll_difference=(paired.nll - paired.nll_paper).mean(),
                    paper_context_gain=paired.nll_gain_vs_train_context_paper.mean(),
                    extra_context_gain_difference=(
                        paired.nll_gain_vs_train_context
                        - paired.nll_gain_vs_train_context_paper
                    ).mean(),
                )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def summarize_paper_comparison(out_dir):
    """Regenerate pure-LM plots; retrieval qualification never gates these metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(out_dir)
    config = PaperBaselineConfig.from_json(output / "config.json")
    raw = pd.read_csv(output / "lm_raw.csv")
    # Archived original-only outputs remain analyzable without relabeling models.
    if "gate_mode" not in raw:
        raw["gate_mode"] = "original_data"
    if "optimizer" not in raw:
        raw["optimizer"] = raw.arm.where(raw.checkpoint_step > 0, "initial")
    if "stage" not in raw:
        raw["stage"] = np.where(raw.checkpoint_step > 0, "final", "initial")
    suffix = raw[raw.metric == "same_target_suffix"]
    means = suffix.groupby(
        ["arm", "optimizer", "gate_mode", "stage", "checkpoint_step", "context_length"],
        as_index=False,
    ).agg(
        mean_nll=("nll", "mean"),
        mean_nll_gain=("nll_gain_vs_train_context", "mean"),
        documents=("sample", "nunique"),
    )
    means.to_csv(output / "context_summary.csv", index=False)
    comparisons = paired_paper_comparison(raw)
    comparisons.to_csv(output / "comparison_vs_paper.csv", index=False)

    def pretty_label(gate, optimizer):
        gate_label = {
            "original_data": "FoX",
            "factorized_constant": "Factorized",
            "direct_constant": "Direct",
        }[gate]
        optimizer_label = {
            "initial": "initial",
            "paper_adamw": "paper AdamW",
            "adam_fixed": "Adam fixed",
            "adam_annealed": "Adam annealed",
            "sgd": "SGD",
        }[optimizer]
        return f"{gate_label}: {optimizer_label}"

    def render(gate_mode):
        plot_raw = raw[raw.gate_mode == gate_mode]
        if gate_mode != "original_data":
            paper_reference = raw[
                (raw.gate_mode == "original_data") & (raw.optimizer == "paper_adamw")
            ]
            plot_raw = pd.concat([plot_raw, paper_reference], ignore_index=True)
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for arm, frame in plot_raw.groupby("arm"):
            style = "--" if frame.checkpoint_step.eq(0).all() else "-"
            curve_label = pretty_label(frame.gate_mode.iloc[0], frame.optimizer.iloc[0])
            per_position = frame[frame.metric == "per_position"].copy()
            per_position["position_bin"] = ((per_position.position - 1) // 32) * 32 + 1
            positional = per_position.groupby("position_bin").agg(
                position=("position", "mean"), nll=("nll", "mean")
            )
            axes[0].plot(positional.position, positional.nll, style, label=curve_label)
            contexts = means[means.arm == arm].sort_values("context_length")
            axes[1].plot(
                contexts.context_length,
                contexts.mean_nll,
                style + "o",
                label=curve_label,
            )
            axes[2].plot(
                contexts.context_length,
                contexts.mean_nll_gain,
                style + "o",
                label=curve_label,
            )
        for axis in axes:
            axis.axvline(
                config.train_length,
                color="black",
                linestyle="--",
                alpha=0.55,
                label="Training context",
            )
            axis.grid(alpha=0.2)
        axes[0].set(
            title="Next-token loss by position",
            xlabel="Target position (32-token bin centers)",
            ylabel="NLL (nats)",
            xlim=(1, max((*config.eval_contexts, config.train_length))),
        )
        axes[1].set(
            title="Identical target suffix",
            xlabel="Reset context window (tokens)",
            ylabel="NLL (nats)",
        )
        axes[2].set(
            title="Utility of additional context",
            xlabel="Reset context window (tokens)",
            ylabel="NLL gain vs training context",
        )
        axes[2].axhline(0, color="black", linewidth=0.7)
        axes[1].set_xscale("log", base=2)
        axes[2].set_xscale("log", base=2)
        label = (
            "Software validation only"
            if config.profile == "smoke"
            else "Finite FoX model/optimizer comparison; rates untuned"
        )
        figure.suptitle(f"{gate_mode} — {label}")
        handles, labels = axes[1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        figure.tight_layout(rect=(0, 0.14, 1, 0.94))
        path = output / (
            "paper_context_comparison.png"
            if len(config.model_variants) == 1
            else f"paper_context_{gate_mode}.png"
        )
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return str(path)

    paths = [render(gate) for gate in config.model_variants]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    available = comparisons[comparisons.status == "ok"]
    for arm, frame in available.groupby("arm"):
        if (
            frame.optimizer.iloc[0] == "paper_adamw"
            and frame.gate_mode.iloc[0] == "original_data"
        ):
            continue
        frame = frame.sort_values("context_length")
        curve_label = pretty_label(frame.gate_mode.iloc[0], frame.optimizer.iloc[0])
        axes[0].plot(
            frame.context_length, frame.nll_difference, "o-", label=curve_label
        )
        axes[1].plot(
            frame.context_length,
            frame.extra_context_gain_difference,
            "o-",
            label=curve_label,
        )
    for axis in axes:
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvline(config.train_length, color="black", linestyle="--", alpha=0.5)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Reset context window (tokens)")
        axis.grid(alpha=0.2)
        if available.empty:
            axis.text(
                0.5,
                0.5,
                "Paper baseline or matched targets unavailable",
                transform=axis.transAxes,
                ha="center",
            )
    axes[0].set(
        title="NLL difference: branch − paper",
        ylabel="NLL difference (negative is better)",
    )
    axes[1].set(
        title="Additional-context utility vs paper",
        ylabel="Gain difference (positive is better)",
    )
    figure.suptitle(
        "Software validation only"
        if config.profile == "smoke"
        else "Paired differences vs original FoX + paper AdamW"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.21, 1, 0.94))
    difference_path = output / "comparison_vs_paper.png"
    figure.savefig(difference_path, dpi=150)
    plt.close(figure)
    paths.append(str(difference_path))
    return {
        "context_summary": str(output / "context_summary.csv"),
        "figures": paths,
        "comparison_vs_paper": str(output / "comparison_vs_paper.csv"),
        "summary": str(output / "summary.csv"),
    }
