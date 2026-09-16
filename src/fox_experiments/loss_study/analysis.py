"""Readable, saved-artifact-only plots for the paired loss study."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame.dropna(subset=["branch"]) if "branch" in frame else frame


def _label(gate, optimizer, seed=None):
    gate = {
        "factorized_constant": "factorized",
        "direct_constant": "direct",
        "original_data": "input gate",
    }.get(gate, gate)
    optimizer = {
        "adam_fixed": "Adam fixed ε",
        "adam_annealed": "Adam annealed ε",
        "sgd": "SGD",
        "paper_adamw": "AdamW bundle",
    }.get(optimizer, optimizer)
    return f"{gate} / {optimizer}" + (f" / seed {seed}" if seed is not None else "")


def summarize_loss_study(out_dir):
    """Plot all branches, including short-unqualified branches and failures.

    No zero is substituted for an ineligible radius. Confidence is not conflated
    with argmax accuracy, and passing the last lag is explicitly right-censored.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    status = json.loads((out / "status.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    raw, summary, grad = (
        _read(out / name) for name in ("eval.csv", "summary.csv", "gradients.csv")
    )
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    figures = []
    objectives = cfg["objectives"]
    colors = {
        "sgd": "#3b6fb6",
        "adam_fixed": "#d77922",
        "adam_annealed": "#25896e",
        "paper_adamw": "#9063ad",
    }
    styles = {"factorized_constant": "-", "direct_constant": "--", "original_data": ":"}
    software = (
        " — SOFTWARE SMOKE, no scientific inference"
        if cfg["profile"] == "smoke"
        else ""
    )

    def save(fig, name):
        path = directory / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(str(path.resolve()))

    if len(raw) and "split" in raw:
        final = raw[raw.step == raw.groupby("branch").step.transform("max")]
        long = final[final.split == "long"]
        if len(long):
            joint = long.groupby(
                [
                    "branch",
                    "objective",
                    "gate_mode",
                    "optimizer",
                    "seed",
                    "lag",
                    "prefix",
                    "history_id",
                ],
                as_index=False,
            ).correct.min()
            panel = joint.groupby(
                [
                    "branch",
                    "objective",
                    "gate_mode",
                    "optimizer",
                    "seed",
                    "lag",
                    "prefix",
                ],
                as_index=False,
            ).correct.mean()
            worst = panel.groupby(
                ["branch", "objective", "gate_mode", "optimizer", "seed", "lag"],
                as_index=False,
            ).correct.min()
            fig, axes = plt.subplots(
                1,
                len(objectives),
                figsize=(8 * len(objectives), 5.5),
                squeeze=False,
                sharey=True,
            )
            for ax, objective in zip(axes[0], objectives):
                for (_, gate, optimizer, seed), rows in worst[
                    worst.objective == objective
                ].groupby(["branch", "gate_mode", "optimizer", "seed"]):
                    ax.plot(
                        rows.lag,
                        rows.correct,
                        marker="o",
                        color=colors[optimizer],
                        linestyle=styles[gate],
                        label=_label(
                            gate, optimizer, seed if len(cfg["seeds"]) > 1 else None
                        ),
                    )
                ax.axhline(
                    cfg["qualification_threshold"],
                    color="gray",
                    linestyle=":",
                    label="passing threshold",
                )
                ax.axvline(
                    cfg["train_max_lag"], color="black", linewidth=1, linestyle=":"
                )
                ax.set(
                    xlabel="Target-inclusive lag (records; tested points)",
                    ylabel="Joint-edit accuracy, worst prefix",
                    ylim=(-0.03, 1.03),
                    title=objective.replace("_", " "),
                )
                ax.set_xscale("log", base=2)
                ax.set_xticks(cfg["eval_lags"], cfg["eval_lags"])
                ax.legend(
                    fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2
                )
            fig.suptitle(
                "Final retrieval — unqualified branches remain visible" + software
            )
            fig.tight_layout()
            save(fig, "accuracy_by_lag.png")

            # Keep the two size variables visible instead of only taking their
            # worst case. These are empirical panels, not tail certificates.
            branch_groups = list(panel.groupby("branch", sort=True))
            ncols = 3
            nrows = (len(branch_groups) + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False
            )
            for ax, (branch, rows) in zip(axes.flat, branch_groups):
                grid = rows.pivot(index="prefix", columns="lag", values="correct")
                grid = grid.reindex(
                    index=cfg["eval_prefixes"], columns=cfg["eval_lags"]
                )
                heatmap = ax.imshow(
                    grid.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis"
                )
                first = rows.iloc[0]
                ax.set_title(
                    first.objective
                    + "\n"
                    + _label(first.gate_mode, first.optimizer, first.seed),
                    fontsize=9,
                )
                ax.set_xticks(range(len(grid.columns)), grid.columns)
                ax.set_yticks(range(len(grid.index)), grid.index)
                ax.set_xlabel("Target-inclusive record lag")
                ax.set_ylabel("Older prefix records")
                for y in range(len(grid.index)):
                    for x in range(len(grid.columns)):
                        value = grid.iloc[y, x]
                        if np.isfinite(value):
                            ax.text(
                                x,
                                y,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="black" if value > 0.65 else "white",
                            )
            for ax in list(axes.flat)[len(branch_groups) :]:
                ax.axis("off")
            fig.suptitle(
                "Joint-edit accuracy: lag × older prefix; last available checkpoints"
                + software
            )
            fig.tight_layout(rect=(0, 0, 0.92, 0.95))
            color_axis = fig.add_axes([0.94, 0.15, 0.012, 0.65])
            fig.colorbar(heatmap, cax=color_axis, label="Joint-edit accuracy")
            save(fig, "lag_prefix_accuracy.png")

    if len(summary) and "short_joint_accuracy_worst" in summary:
        fig, axes = plt.subplots(
            2, len(objectives), figsize=(8 * len(objectives), 9), squeeze=False
        )
        for col, objective in enumerate(objectives):
            for (_, gate, optimizer, seed), rows in summary[
                summary.objective == objective
            ].groupby(["branch", "gate_mode", "optimizer", "seed"]):
                rows = rows.sort_values("step")
                style = dict(
                    color=colors[optimizer],
                    linestyle=styles[gate],
                    label=_label(
                        gate, optimizer, seed if len(cfg["seeds"]) > 1 else None
                    ),
                )
                axes[0, col].plot(
                    rows.step, rows.short_joint_accuracy_worst, marker=".", **style
                )
                axes[1, col].plot(
                    rows.step, rows.tested_grid_radius, marker="o", **style
                )
                at_ceiling = rows[rows.right_censored == True]  # noqa: E712
                axes[1, col].scatter(
                    at_ceiling.step,
                    at_ceiling.tested_grid_radius,
                    marker="^",
                    s=65,
                    color=colors[optimizer],
                )
            axes[0, col].axhline(
                cfg["qualification_threshold"], color="gray", linestyle=":"
            )
            axes[0, col].set(
                title=objective.replace("_", " "),
                ylim=(-0.03, 1.03),
                ylabel="Short joint-edit accuracy (worst panel)",
                xlabel="Optimizer updates",
            )
            axes[1, col].set(
                ylabel="Radius on tested lag grid",
                xlabel="Optimizer updates",
                ylim=(-0.5, max(cfg["eval_lags"]) * 1.1),
            )
            axes[0, col].set_xlim(0, cfg["steps"])
            axes[1, col].set_xlim(0, cfg["steps"])
            axes[1, col].text(
                0.02,
                0.98,
                "Missing radius = short-unqualified/incomplete\n▲ = ≥ tested ceiling; gaps between lags untested",
                transform=axes[1, col].transAxes,
                va="top",
                fontsize=8,
            )
            axes[1, col].legend(
                fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2
            )
        fig.suptitle("Acquisition and finite-grid extrapolation" + software)
        fig.tight_layout()
        save(fig, "short_fit_and_radius.png")

    if len(grad) and "parameter_group" in grad:
        gate_grad = grad[grad.parameter_group == "first_gate"]
        fig, axes = plt.subplots(
            2, len(objectives), figsize=(8 * len(objectives), 9), squeeze=False
        )
        for col, objective in enumerate(objectives):
            for (_, gate, optimizer, seed), rows in gate_grad[
                gate_grad.objective == objective
            ].groupby(["branch", "gate_mode", "optimizer", "seed"]):
                style = dict(
                    color=colors[optimizer],
                    linestyle=styles[gate],
                    label=_label(
                        gate, optimizer, seed if len(cfg["seeds"]) > 1 else None
                    ),
                )
                rows = rows.sort_values("step")
                axes[0, col].plot(rows.step, rows.gradient_cosine, marker=".", **style)
                # Ratio exposes actual weighting, not just raw directional cosine.
                denominator = rows.weighted_answer_gradient_norm.replace(0, np.nan)
                ratio = rows.weighted_other_gradient_norm / denominator
                axes[1, col].plot(rows.step, ratio, marker=".", **style)
            axes[0, col].axhline(0, color="gray", linewidth=1)
            axes[0, col].set(
                title=objective.replace("_", " "),
                ylabel="First gate: cosine(∇answer, ∇other)",
                ylim=(-1.05, 1.05),
                xlabel="Update (before step)",
            )
            axes[1, col].set(
                ylabel="Weighted other / answer gradient norm",
                xlabel="Update (before step)",
            )
            axes[1, col].set_yscale("symlog", linthresh=0.1)
            axes[1, col].legend(
                fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2
            )
        fig.suptitle(
            "Objective gradient conflict and dilution — first forgetting gate"
            + software
        )
        fig.tight_layout()
        save(fig, "gradient_conflict.png")
    result = {
        "out_dir": str(out.resolve()),
        "status": status["status"],
        "failures": status.get(
            "failures",
            [r for r in status.get("branches", []) if r["status"] not in ("complete",)],
        ),
        "summary": str((out / "summary.csv").resolve()),
        "eval": str((out / "eval.csv").resolve()),
        "training": str((out / "training.csv").resolve()),
        "gradients": str((out / "gradients.csv").resolve()),
        "branches": str((out / "branches.csv").resolve()),
        "figures": figures,
        "software_only": cfg["profile"] == "smoke",
    }
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
