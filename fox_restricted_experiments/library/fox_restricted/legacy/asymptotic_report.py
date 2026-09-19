"""Plot certified brackets for worst-case answer error, using stored log errors.

The unknown supremum E_t(r) is bracketed by an analytic witness lower bound
and a uniform upper certificate. Neither endpoint is an empirical estimate of
that supremum. Invalid certificates are displayed with the trivial upper 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


ARMS = (("learned", "sgd"), ("learned", "adam"),
        ("retrieval_frozen", "sgd"), ("retrieval_frozen", "adam"))
GATES = {"learned": "Learned retrieval", "retrieval_frozen": "Frozen retrieval"}
PALETTE = tuple(plt.get_cmap("tab10").colors)


def _float(value, default=math.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _read(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _valid(row):
    return str(row.get("bound_valid", "")).lower() in {"true", "1", "1.0", "yes"}


def _fixed(row):
    return str(row.get("probe", "")).lower().startswith("fixed")


def _probe(row):
    name = str(row.get("probe", "unknown"))
    if _fixed(row):
        return f"fixed r={_float(row.get('lag')):g}"
    parameter = row.get("theta")
    if parameter not in (None, ""):
        return f"{name}, θ={_float(parameter):g}"
    parameter = row.get("coefficient")
    if parameter not in (None, ""):
        return f"{name}, c={_float(parameter):g}"
    return name


def _curves(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("gate_mode")), str(row.get("optimizer")),
                 str(row.get("seed")), _probe(row))].append(row)
    return {key: sorted(values, key=lambda row: _float(row.get("step"), -1))
            for key, values in grouped.items()}


def _log_error(row, side):
    if side == "upper" and not _valid(row):
        return 0.0
    value = _float(row.get(f"log_{side}_error"))
    if math.isnan(value):
        raw = _float(row.get(f"{side}_error"))
        if raw > 0:
            value = math.log(raw)
    return value / math.log(10)


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _axes_style(ax):
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def _clock_axis(ax, xkey="S"):
    threshold = 1 if xkey == "S" else 10
    largest = ax.dataLim.xmax
    if math.isfinite(largest) and largest > 2 * threshold:
        ax.set_xscale("symlog", linthresh=threshold)
    ax.set_xlim(left=0)


def _error_plot(rows, fixed, xkey, path):
    rows = [row for row in rows if _fixed(row) == fixed]
    grouped = _curves(rows)
    probes = sorted({_probe(row) for row in rows})
    colors = {probe: PALETTE[i % len(PALETTE)] for i, probe in enumerate(probes)}
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.5), squeeze=False)
    for ax, arm in zip(axes.ravel(), ARMS):
        seen = set()
        has_invalid = False
        for (gate, opt, seed, probe), values in grouped.items():
            if (gate, opt) != arm:
                continue
            points = [(row, _float(row.get(xkey)), _log_error(row, "lower"), _log_error(row, "upper"))
                      for row in values]
            points = [p for p in points if math.isfinite(p[1])]
            if not points:
                continue
            x = np.array([p[1] for p in points])
            lower = np.array([p[2] for p in points])
            upper = np.array([p[3] for p in points])
            color = colors[probe]
            ax.plot(x, upper, color=color, lw=1.5, alpha=.78,
                    label=probe if probe not in seen else "_nolegend_")
            ax.plot(x, lower, color=color, lw=1, ls=":", alpha=.75)
            ax.fill_between(x, lower, upper, color=color, alpha=.055,
                            where=np.isfinite(lower) & np.isfinite(upper))
            invalid = [p[1] for p in points if not _valid(p[0])]
            if invalid:
                ax.scatter(invalid, np.zeros(len(invalid)), marker="x", color=color, s=16, zorder=5)
                has_invalid = True
            seen.add(probe)
        if not seen:
            ax.text(.5, .5, "No saved bracket measurements", transform=ax.transAxes, ha="center", color=".4")
        else:
            handles, labels = ax.get_legend_handles_labels()
            handles += [Line2D([], [], color=".3", lw=1.6), Line2D([], [], color=".3", ls=":")]
            labels += ["Upper endpoint", "Witness lower endpoint"]
            if has_invalid:
                handles.append(Line2D([], [], color=".3", marker="x", ls=""))
                labels.append("Uncertified: trivial upper = 1")
            ax.legend(handles, labels, fontsize=7, ncol=2, frameon=False, loc="best")
        _clock_axis(ax, xkey)
        ax.set(title=f"{GATES[arm[0]]} · {arm[1].upper()}",
               xlabel="Continuation learning-rate sum S" if xkey == "S" else "Optimizer update t",
               ylabel="log₁₀(error bound)")
        ax.set_ylim(top=.15)
        ax.axhline(-2, color=".6", ls="--", lw=.6)
        _axes_style(ax)
    subject = "Fixed radii: Eₜ(r)" if fixed else "Moving radii: Eₜ(rₜ)"
    fig.suptitle(subject + " bracketed by a witness and a uniform certificate", fontsize=14)
    fig.text(.5, .016, "Each seed is a separate envelope, not a seed confidence interval. "
             "E is unknown inside the bracket; log errors are used directly without subtracting a probability from 1.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .95))
    _save(fig, path)


def _radius_plot(errors, radii, path):
    fig, axes = plt.subplots(4, 2, figsize=(13.4, 13.5), squeeze=False)
    moving = [row for row in errors if not _fixed(row)]
    probes = sorted({_probe(row) for row in moving})
    colors = {probe: PALETTE[i % len(PALETTE)] for i, probe in enumerate(probes)}
    for row_index, arm in enumerate(ARMS):
        ax = axes[row_index, 0]
        seen = set()
        for (gate, opt, seed, probe), values in _curves(moving).items():
            if (gate, opt) != arm:
                continue
            points = [(_float(row.get("S")), _float(row.get("lag"))) for row in values]
            points = [p for p in points if all(math.isfinite(v) for v in p)]
            if points:
                ax.plot(*zip(*points), color=colors[probe], alpha=.7, lw=1.3,
                        label=probe if probe not in seen else "_nolegend_")
                seen.add(probe)
        ax.set(title=f"{GATES[arm[0]]} · {arm[1].upper()} — actual probe radius",
               xlabel="Continuation sum S", ylabel="Actual integer radius rₜ")
        if seen:
            ax.legend(fontsize=7, ncol=2, frameon=False)
        else:
            ax.text(.5, .5, "No moving-radius observations", transform=ax.transAxes, ha="center", color=".4")
        ax = axes[row_index, 1]
        threshold_groups = defaultdict(list)
        for row in radii:
            if (row.get("gate_mode"), row.get("optimizer")) == arm:
                threshold_groups[(str(row.get("seed")), str(row.get("error_target", row.get("threshold"))))].append(row)
        seen = set()
        has_uncertified = False
        for (seed, threshold), values in sorted(threshold_groups.items()):
            values.sort(key=lambda row: _float(row.get("step"), -1))
            xs = [_float(row.get("S")) for row in values]
            ys = [_float(row.get("certified_radius"), 0) for row in values]
            threshold_index = sorted({key[1] for key in threshold_groups}).index(threshold)
            color = PALETTE[threshold_index % len(PALETTE)]
            ax.plot(xs, ys, color=color, alpha=.75, lw=1.4,
                    label=f"E ≤ {_float(threshold):g}" if threshold not in seen else "_nolegend_")
            invalid = [(x, y) for x, y, row in zip(xs, ys, values) if not _valid(row)]
            if invalid:
                ax.scatter(*zip(*invalid), color=color, marker="x", s=15)
                has_uncertified = True
            seen.add(threshold)
        ax.set(title="Certified radius from uniform bound; no evaluation-lag cap",
               xlabel="Continuation sum S", ylabel="Largest certified integer radius")
        if seen:
            if has_uncertified:
                ax.plot([], [], color=".3", marker="x", ls="", label="Unavailable certificate")
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(.5, .5, "No saved certified radii", transform=ax.transAxes, ha="center", color=".4")
        for ax in axes[row_index]:
            _clock_axis(ax)
            largest = ax.dataLim.ymax
            if math.isfinite(largest) and largest > 16:
                ax.set_yscale("symlog", linthresh=4)
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(0, max(1.1, 1.08 * largest) if math.isfinite(largest) else 1.1)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            _axes_style(ax)
    fig.suptitle("Measured radius trajectories and certified generalization range", fontsize=14)
    fig.text(.5, .007, "Clock-based radii are chosen probes; adaptive radii use measured gaps. "
             "Observed growth over a finite window does not establish rₜ → ∞. Radius 0 certifies no positive radius.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .975))
    _save(fig, path)


def _fmt(value):
    value = _float(value)
    if value == -math.inf:
        return "−∞"
    return f"{value:.5g}" if math.isfinite(value) else "unavailable"


def _optimizer_plot(history, config, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    grouped = defaultdict(list)
    for row in history:
        grouped[(row.get("gate_mode"), row.get("optimizer"), str(row.get("seed")))].append(row)
    columns = (("q", "am", 1.0), ("u", "ag", 1.5), ("w", "aw", 0.1))
    for gate_index, gate in enumerate(("learned", "retrieval_frozen")):
        for column_index, (parameter, multiplier_name, default) in enumerate(columns):
            ax = axes[gate_index, column_index]
            direction_ax = ax.twinx()
            seen = set()
            direction_seen = False
            nominal = _float(config.get(multiplier_name), default)
            for (row_gate, optimizer, seed), values in sorted(grouped.items()):
                if row_gate != gate:
                    continue
                points, directions = [], []
                for row in sorted(values, key=lambda item: _float(item.get("step"), -1)):
                    S = _float(row.get("S"))
                    if not math.isfinite(S) or S <= 0:
                        continue
                    ratio = _float(row.get(parameter + "_over_S"))
                    if not math.isfinite(ratio):
                        ratio = _float(row.get(parameter)) / S
                    if math.isfinite(ratio):
                        points.append((S, ratio))
                    direction = _float(row.get("adam_direction_" + parameter))
                    if optimizer == "adam" and math.isfinite(direction):
                        directions.append((S, -direction))
                if points:
                    ax.plot(*zip(*points), color={"sgd": "#2166ac", "adam": "#d95f02"}.get(optimizer, ".3"),
                            lw=1.5, alpha=.75,
                            label=f"{optimizer.upper()} {parameter}/S" if optimizer not in seen else "_nolegend_")
                    seen.add(optimizer)
                if directions:
                    direction_ax.plot(*zip(*directions), color="#762a83", ls="--", lw=1.2, alpha=.6,
                                      label="Adam normalized growth direction" if not direction_seen else "_nolegend_")
                    direction_seen = True
            ax.axhline(nominal, color=".3", ls=":", lw=1,
                       label=f"Adam nominal {multiplier_name} = {nominal:g}")
            direction_ax.axhline(1, color="#762a83", ls=":", lw=.6, alpha=.4)
            ax.set(title=f"{GATES[gate]} · {parameter}", xlabel="Continuation learning-rate sum S",
                   ylabel=f"{parameter}/S")
            direction_ax.set_ylabel("−m̂ / (√v̂ + ε)", color="#762a83")
            direction_ax.tick_params(axis="y", colors="#762a83", labelsize=8)
            direction_ax.spines["top"].set_visible(False)
            direction_low, direction_high = direction_ax.dataLim.ymin, direction_ax.dataLim.ymax
            direction_ax.set_ylim(min(0, 1.05 * direction_low) if math.isfinite(direction_low) else 0,
                                  max(1.1, 1.05 * direction_high) if math.isfinite(direction_high) else 1.1)
            direction_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            if not direction_seen:
                direction_ax.set_ylim(0, 1.1)
                ax.text(.02, .03, "Adam directions unavailable in these logs", transform=ax.transAxes,
                        fontsize=7, color=".4")
            if not seen:
                ax.text(.5, .5, "No saved positive-S ratio observations", transform=ax.transAxes,
                        ha="center", fontsize=8, color=".4")
            _clock_axis(ax)
            _axes_style(ax)
            handles, labels = ax.get_legend_handles_labels()
            more_handles, more_labels = direction_ax.get_legend_handles_labels()
            ax.legend(handles + more_handles, labels + more_labels, fontsize=7, frameon=False)
    fig.suptitle("Optimizer tracking: cumulative parameter ratios and current Adam direction", fontsize=14)
    fig.text(.5, .012, "Left axes: parameter/S, including the acquisition offset. Right axes: signed normalized growth direction. "
             "Nominal references are conditional Adam asymptotic values, not finite-run guarantees.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .95))
    _save(fig, path)


def _report(errors, radii, runs):
    invalid_count = sum(not _valid(row) for row in errors)
    lines = ["# Worst-case generalization error: finite-run bracket report", "",
             "Define E_t(r) as the supremum of 1 − P_t(correct answer) over every finite record stream "
             "whose latest assignment to the queried key is at lag at most r. This E is a model error, "
             "not Adam's denominator epsilon.", "",
             "The unknown E_t(r) is bounded below by an infinite-prefix witness supremum and above by "
             "a conditional uniform certificate over arbitrary finite streams. The infinite-prefix witness "
             "is a limit of finite streams and therefore supplies a valid lower bound on the supremum. "
             "Finite-prefix sample minima and means are reported separately by the ordinary report.", "",
             "Plots use the saved natural logarithms divided by log(10), so tiny errors remain visible "
             "without evaluating 1 − a probability rounded to one. Solid and dotted curves are the upper "
             "and lower endpoints for each seed. Their shaded interval brackets the true supremum; "
             "it is not a confidence interval or a numerical estimate of E.", "",
             f"Saved error rows: {len(errors)}. Rows without a valid uniform certificate: {invalid_count}. "
             "Every invalid upper certificate is shown explicitly as the trivial upper bound E ≤ 1 "
             "with a cross marker; no uncertified upper curve is silently omitted.", "",
             "## What the trajectories test", "",
             "- Fixed-radius probes examine error reduction at an unchanged lag threshold.",
             "- Clock probes choose a radius proportional to S for learned retrieval and to S² for frozen "
             "retrieval (with their saved coefficients and integer rounding). These prescribed clocks do "
             "not assert that each optimizer supports that growth rate. In particular, frozen-SGD clock "
             "probes are stress tests; its ordinary-answer theorem does not give a sharp time law.",
             "- Adaptive probes use the measured content gap and forgetting scale. Their actual integer "
             "radii are plotted. A decreasing upper error at a radius that stays bounded does not establish "
             "generalization along a radius tending to infinity.",
             "- Certified radii invert the uniform bound at each requested error threshold and are not "
             "clipped to the largest finite evaluation lag. Radius zero means no positive radius is certified.",
             "- R = 2 is the proved frozen-gate setting. The theoretical asymptotic conclusions require "
             "their sufficient initialization, acquisition, continuation, and basin hypotheses. These "
             "practical polynomial schedules are finite experiments, not literal implementations of the "
             "existential conservative SGD schedule. Larger R is an extension experiment.",
             "- The frozen intervention holds retrieval h fixed and continues training the binding gate. "
             "Uniform frozen-gate success needs h₀ > log(2) and the certificate's other conditions.", "",
             "`asymptotic_optimizer_diagnostics.png` compares q/S, u/S and w/S with their configured "
             "nominal Adam multipliers, and separately shows the signed normalized Adam growth directions. "
             "Ratios include finite acquisition offsets. Missing moment directions are labeled unavailable. "
             "Changing a beta preset changes both acquisition and continuation dynamics, not only a later memory window.", "",
             "## Latest saved brackets", "",
             "Values are log10(error). A more negative upper endpoint is stronger evidence of a small "
             "worst-case error at that saved radius; finite trajectories do not establish a limit.", "",
             "| Seed | Gate | Optimizer | Probe | Step | S | Radius | log10 lower | log10 upper | Valid uniform bound |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (gate, opt, seed, probe), values in sorted(_curves(errors).items()):
        row = values[-1]
        lines.append("| " + " | ".join([seed, gate, opt, probe, _fmt(row.get("step")), _fmt(row.get("S")),
                                        _fmt(row.get("lag")), _fmt(_log_error(row, "lower")),
                                        _fmt(_log_error(row, "upper")), str(_valid(row))]) + " |")
    if not errors:
        lines.append("| — | — | — | No saved data | — | — | — | unavailable | unavailable | False |")
    lines.extend(["", "## Run status", "", "| Seed | Gate | Optimizer | Completed steps | Status | Reason |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for row in runs:
        lines.append("| " + " | ".join(str(row.get(key, "unavailable")).replace("|", "/").replace("\n", " ")
                                             for key in ("seed", "gate_mode", "optimizer", "completed_steps", "status", "reason")) + " |")
    if not runs:
        lines.append("| — | — | — | — | No saved status | — |")
    lines.extend(["", "Each branch is plotted through its own last saved checkpoint. Stopped runs can have "
                  "different budgets. Inspect `runs.csv`, `history.csv`, `asymptotic_errors.csv`, "
                  "`certified_radii.csv`, and the checkpoint metadata before comparing arms.", ""])
    return "\n".join(lines)


def make_asymptotic_report(out_dir):
    """Create standalone PNG/PDF figures and a Markdown interpretation guide."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = _read(out_dir / "asymptotic_errors.csv")
    radii = _read(out_dir / "certified_radii.csv")
    runs = _read(out_dir / "runs.csv")
    history = _read(out_dir / "history.csv")
    config_path = out_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = config_data.get("config", config_data)
    paths = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        for fixed, label in ((True, "fixed"), (False, "moving")):
            for xkey, xname in (("step", "step"), ("S", "S")):
                name = f"asymptotic_{label}_error_by_{xname}"
                path = out_dir / (name + ".png")
                _error_plot(errors, fixed, xkey, path)
                paths[name] = str(path)
                paths[name + "_pdf"] = str(path.with_suffix(".pdf"))
        path = out_dir / "asymptotic_radius_trajectories.png"
        _radius_plot(errors, radii, path)
        paths["radius_trajectories"] = str(path)
        paths["radius_trajectories_pdf"] = str(path.with_suffix(".pdf"))
        path = out_dir / "asymptotic_optimizer_diagnostics.png"
        _optimizer_plot(history, config, path)
        paths["optimizer_diagnostics"] = str(path)
        paths["optimizer_diagnostics_pdf"] = str(path.with_suffix(".pdf"))
    path = out_dir / "asymptotic_report.md"
    path.write_text(_report(errors, radii, runs), encoding="utf-8")
    paths["report"] = str(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(make_asymptotic_report(args.out_dir), indent=2))
