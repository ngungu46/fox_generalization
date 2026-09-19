"""Plot measured moving-radius error envelopes for the two local Adam pilots."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "library"))
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plot(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1] / "results"
    choices = (("pilot_R2_seed0", "Adam β=(0.9, 0.999)", "#596c80"),
               ("adam_moment_sensitivity_R2_seed0", "Adam β=(0.1, 0.1)", "#b84e16"))
    out = root / "adam_moment_comparison"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), sharex=True)
    final = []
    for column, gate in enumerate(("learned", "retrieval_frozen")):
        for name, label, color in choices:
            with (root / name / "asymptotic_reanalysis" / "asymptotic_errors.csv").open() as handle:
                selected = [r for r in csv.DictReader(handle) if r["gate_mode"] == gate
                            and r["optimizer"] == "adam" and r["probe"] == "adaptive_theta"
                            and float(r["theta"]) == .75]
            selected.sort(key=lambda r: int(r["step"]))
            valid = [r for r in selected if r["upper_error"]]
            xs = [int(r["step"]) / 1000 for r in valid]
            lower = [float(r["log_lower_error"]) / math.log(10) for r in valid]
            upper = [float(r["log_upper_error"]) / math.log(10) for r in valid]
            axes[0, column].plot(xs, upper, color=color, label=label, lw=2.2)
            axes[0, column].plot(xs, lower, color=color, ls=":", lw=1)
            axes[0, column].fill_between(xs, lower, upper, color=color, alpha=.18)
            axes[1, column].step([int(r["step"]) / 1000 for r in selected],
                                 [int(r["lag"]) for r in selected], where="post", color=color, lw=2.2)
            final.append(dict(pilot=name, gate_mode=gate, **{
                k: selected[-1][k] for k in ("step", "S", "lag", "lower_error", "upper_error")
            }))
        axes[0, column].set_title("Learned retrieval gate" if gate == "learned" else "Frozen retrieval gate, h₀=1")
        axes[0, column].axhline(-2, ls="--", lw=.8, color=".55", label="Error 0.01")
        axes[0, column].legend(frameon=False, fontsize=9, loc="lower left")
        axes[0, column].set_ylabel("log₁₀ Eₜ(rₜ): lower / upper bounds")
        axes[1, column].set_ylabel("Actual moving lag rₜ")
        axes[1, column].set_xlabel("Training updates (thousands)")
        for ax in axes[:, column]:
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(alpha=.2)
    fig.suptitle("Adam at growing lags: finite pilot sensitivity to moment settings", fontsize=15)
    fig.text(.5, .018, "N=8, d=2,048, R=2, seed 0. Moving lag uses θ=0.75; solid = upper bound, dotted = lower.\n"
             "Matched Gaussian initialization; changing moments affects both acquisition and continuation. These are CPU pilots, not A100 runs.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"moving_error_comparison.{extension}", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    (out / "final_comparison.json").write_text(json.dumps(final, indent=2) + "\n")
    return str(out / "moving_error_comparison.png")


if __name__ == "__main__":
    print(make_plot())
