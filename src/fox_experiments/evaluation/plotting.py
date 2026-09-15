"""Standalone PNG figures for natural-text probes and controlled mechanisms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


def plot_retrieval_panels(
    panels: pd.DataFrame,
    output: Path,
    software_only: bool,
) -> list[Path]:
    """Show each stale-prefix size separately so it cannot hide target-distance failures."""
    paths = []
    for gate in panels.gate.unique():
        frame = panels[(panels.gate == gate) & (panels.kind == "conflict")]
        frame = frame[frame.checkpoint_step == frame.checkpoint_step.max()]
        prefixes = sorted(frame.prefix_copies.unique())
        figure, axes = plt.subplots(
            1, len(prefixes), figsize=(4 * len(prefixes), 3.3), squeeze=False
        )
        for axis, prefix in zip(axes[0], prefixes):
            for optimizer, group in frame[frame.prefix_copies == prefix].groupby(
                "optimizer"
            ):
                accuracy = group.groupby("lag").history_accuracy.mean()
                axis.plot(accuracy.index, accuracy.values, "o-", label=optimizer)
            axis.set_xscale("log", base=2)
            axis.set_ylim(-0.03, 1.03)
            axis.set_title(f"{prefix} extra stale writes")
            axis.set_xlabel("Target lag (tokens)")
            axis.set_ylabel("All-edit history accuracy")
            axis.grid(alpha=0.2)
        axes[0, -1].legend(fontsize=7)
        label = (
            "software validation only" if software_only else "finite empirical probe"
        )
        figure.suptitle(f"{gate} — {label}")
        figure.tight_layout()
        path = output / f"{gate}_lag_prefix.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path)
    return paths


def plot_language_modeling(
    language_modeling: pd.DataFrame,
    output: Path,
    software_only: bool,
) -> Path:
    """Compare per-position NLL and the utility of more context for one fixed target."""
    latest = language_modeling[
        language_modeling.checkpoint_step == language_modeling.checkpoint_step.max()
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for branch, frame in latest.groupby("branch"):
        positions = frame[frame.metric == "per_position"].copy()
        positions["bin"] = (positions.position // 32) * 32
        mean_loss = positions.groupby("bin").nll.mean()
        axes[0].plot(mean_loss.index, mean_loss.values, label=branch, alpha=0.8)
        contexts = (
            frame[frame.metric == "same_target_context"]
            .groupby("context_length")
            .nll.mean()
        )
        axes[1].plot(contexts.index, contexts.values, "o-", alpha=0.8)
    axes[0].set(
        title="Per-position next-token loss", xlabel="Position", ylabel="NLL (nats)"
    )
    axes[1].set(
        title="Same target, more context", xlabel="Context tokens", ylabel="NLL (nats)"
    )
    axes[1].set_xscale("log", base=2)
    for axis in axes:
        axis.ticklabel_format(axis="y", useOffset=False, style="plain")
    figure.legend(loc="lower center", bbox_to_anchor=(0.5, 0.01), fontsize=7, ncol=2)
    figure.subplots_adjust(bottom=0.36, top=0.82)
    if software_only:
        figure.suptitle("Software validation only — untrained tiny models")
    path = output / "language_modeling.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_mechanism(out_dir: str | Path) -> list[Path]:
    """Plot measured trajectories; a zero sufficient radius is not zero accuracy."""
    output = Path(out_dir)
    traces = pd.read_csv(output / "traces.csv")
    paths = []
    for recall_lag, frame in traces.groupby("recall_lag"):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        for (gate, optimizer, seed), group in frame.groupby(
            ["gate", "optimizer", "seed"]
        ):
            style = "-" if gate == "factorized" else "--"
            label = f"{optimizer}, {gate}, seed {seed}"
            axes[0].plot(group.step, group.min_margin_over_h, style, label=label)
            axes[1].plot(group.step, group.sufficient_radius, style)
            error_column = f"witness_error_r{int(recall_lag) + 2}_Ninf"
            axes[2].semilogy(group.step, group[error_column].clip(lower=1e-16), style)
        axes[0].set(
            title="Matching / forgetting",
            xlabel="Continuation updates",
            ylabel="m × minimum raw gap / h",
        )
        axes[1].set(
            title="Sufficient 99% radius",
            xlabel="Continuation updates",
            ylabel="Contiguous certified record lag",
        )
        axes[2].set(
            title="Lag R+2, old-prefix limit",
            xlabel="Continuation updates",
            ylabel="Answer error",
        )
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.title.set_fontsize(11)
        axes[1].set_ylim(0, max(1, float(frame.sufficient_radius.max()) + 1))
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        figure.suptitle(
            f"Controlled mechanism, training R={int(recall_lag)} — finite trajectories"
        )
        figure.legend(loc="lower center", bbox_to_anchor=(0.5, 0), fontsize=7, ncol=3)
        figure.subplots_adjust(bottom=0.29, top=0.83, wspace=0.65)
        path = output / f"mechanism_R{int(recall_lag)}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
