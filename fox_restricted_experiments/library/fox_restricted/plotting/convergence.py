"""Readable, one-experiment figures for fixed and growing recall radii.

The saved measurements bracket the *unknown* worst-case error E_t(r). A
stale-prefix witness gives a lower bound; the uniform certificate gives an
upper bound. The filled area is therefore a mathematical bracket, not a
confidence interval. Each random seed is drawn separately.

These functions only read saved CSV files. They do not retrain the model or
construct an arbitrarily long sequence when plotting a large radius.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


_GATE_NAMES = {
    "learned": "Learned forgetting",
    "retrieval_frozen": "ALiBi (frozen forgetting)",
}
_X_LABELS = {"step": "Optimizer update t", "S": "Continuation learning-rate sum S"}


def _directory(run_or_path) -> Path:
    """Accept either the public RunResult object or an ordinary folder path."""
    return Path(getattr(run_or_path, "path", run_or_path)).expanduser()


def _read_errors(directory: Path) -> list[dict]:
    path = directory / "asymptotic_errors.csv"
    if not path.exists():
        raise FileNotFoundError(f"No error measurements at {path}. Train the run first.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value, default=math.nan) -> float:
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


def _certificate_valid(row: dict) -> bool:
    return str(row.get("bound_valid", "")).lower() in {"true", "1", "1.0", "yes"}


def _log_error(row: dict, side: str) -> float:
    # An invalid certificate does not establish zero error. Show its always
    # valid replacement, E <= 1, and visibly mark the missing certificate.
    if side == "upper" and not _certificate_valid(row):
        return 0.0
    value = _number(row.get(f"log_{side}_error"))
    if not math.isnan(value):
        return value
    probability = _number(row.get(f"{side}_error"))
    if 0 < probability <= 1:
        return math.log(probability)
    return -math.inf if probability == 0 else math.nan


def _arm(row: dict) -> tuple[str, str, str]:
    return (str(row.get("gate_mode", "unknown")),
            str(row.get("optimizer", "unknown")), str(row.get("seed", "0")))


def _arm_label(arm: tuple[str, str, str]) -> str:
    gate, optimizer, seed = arm
    return f"{_GATE_NAMES.get(gate, gate)} / {optimizer.upper()} / seed {seed}"


def _experiment_label(rows: list[dict]) -> str:
    families = {(_arm(row)[0], _arm(row)[1]) for row in rows}
    seeds = {_arm(row)[2] for row in rows}
    if len(families) != 1:
        return "All saved arms and seeds"
    gate, optimizer = next(iter(families))
    seed_label = f"seed {next(iter(seeds))}" if len(seeds) == 1 else f"{len(seeds)} seeds"
    return f"{_GATE_NAMES.get(gate, gate)} / {optimizer.upper()} / {seed_label}"


def _config_lag(directory: Path) -> int:
    path = directory / "config.json"
    if not path.exists():
        raise ValueError("Pass lags explicitly when config.json is unavailable.")
    config = json.loads(path.read_text(encoding="utf-8"))
    config = config.get("config", config)
    if "R" not in config:
        raise ValueError("config.json has no training lag R; pass lags explicitly.")
    return int(config["R"])


def _ordered_groups(rows: list[dict], fixed: bool) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        key = (*_arm(row), int(row["lag"])) if fixed else _arm(row)
        groups[key].append(row)
    return {key: sorted(values, key=lambda row: _number(row.get("step")))
            for key, values in sorted(groups.items())}


def _style_axes(axes, x: str) -> None:
    for axis in axes:
        axis.set_xlabel(_X_LABELS[x])
        axis.set_xlim(left=0)
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set(ylabel="Worst-case error Eₜ(r)", ylim=(-.02, 1.02))
    axes[0].axhline(0, color=".55", linewidth=.6)
    axes[0].axhline(1, color=".55", linewidth=.6)
    axes[1].set(ylabel="log₁₀(error bound)")
    axes[1].set_ylim(top=.05)


def _draw_bracket(axes, values, x, color, label, probability_axis=None):
    usable = [row for row in values if math.isfinite(_number(row.get(x)))]
    times = np.array([_number(row[x]) for row in usable])
    log_lower = np.array([_log_error(row, "lower") for row in usable])
    log_upper = np.array([_log_error(row, "upper") for row in usable])
    lower, upper = np.exp(log_lower), np.exp(log_upper)
    for axis, lo, hi in ((axes[0], lower, upper),
                         (axes[1], log_lower / math.log(10), log_upper / math.log(10))):
        axis.plot(times, hi, color=color, linewidth=1.6, alpha=.85, label=label)
        axis.plot(times, lo, color=color, linewidth=1.1, linestyle=":", alpha=.85)
        axis.fill_between(times, lo, hi, color=color, alpha=.10,
                          where=np.isfinite(lo) & np.isfinite(hi))
    invalid = [time for time, row in zip(times, usable) if not _certificate_valid(row)]
    if invalid:
        axes[0].scatter(invalid, np.ones(len(invalid)), color=color, marker="x", s=17, zorder=5)
        axes[1].scatter(invalid, np.zeros(len(invalid)), color=color, marker="x", s=17, zorder=5)
    if probability_axis is not None:
        # P_correct,worst = 1-E. expm1 avoids cancellation when E is near 1.
        correct_lower, correct_upper = -np.expm1(log_upper), -np.expm1(log_lower)
        probability_axis.plot(times, correct_lower, color=color, linewidth=1.6, alpha=.85)
        probability_axis.plot(times, correct_upper, color=color, linewidth=1.1,
                              linestyle=":", alpha=.85)
        probability_axis.fill_between(times, correct_lower, correct_upper, color=color, alpha=.1)
    return bool(invalid)


def _finish(fig, axes, title, invalid, directory, stem, save):
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", fontsize=8, frameon=False)
    endpoints = [Line2D([], [], color=".25", linewidth=1.6),
                 Line2D([], [], color=".25", linestyle=":")]
    names = ["Uniform upper error bound", "Witness lower error bound"]
    if invalid:
        endpoints.append(Line2D([], [], color=".25", marker="x", linestyle=""))
        names.append("Uncertified: trivial upper = 1")
    if handles:
        fig.legend(endpoints, names, loc="lower center", ncol=len(names),
                   bbox_to_anchor=(.5, .04), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.text(.5, .018, "Shading brackets the unknown worst-case error; it is not a confidence interval. "
             "Each seed is drawn separately.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .11, 1, .93))
    if save:
        target = directory / "figures" / f"{stem}.png" if save is True else Path(save)
        if not target.suffix:
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    return fig


def plot_fixed_lags(run_or_path, lags=None, *, x="step", probability=False, save=False):
    """Plot E_t(R+1), E_t(R+2), E_t(R+3) for a saved experiment.

    ``lags`` overrides the three default radii. ``x='S'`` uses accumulated
    continuation learning rate. Set ``probability=True`` to add the equivalent
    worst-case correct-probability bracket, 1-E, as a third panel. The first two
    panels always show error in [0,1] and log10(error), respectively.

    Return an open matplotlib Figure for notebook display. ``save=True`` also
    writes PNG and PDF under the run's ``figures`` folder; a path can be supplied
    to choose another output location.
    """
    if x not in _X_LABELS:
        raise ValueError("x must be 'step' or 'S'.")
    directory = _directory(run_or_path)
    if lags is None:
        R = _config_lag(directory)
        lags = (R + 1, R + 2, R + 3)
    requested = tuple(dict.fromkeys(int(lag) for lag in lags))
    if not requested or any(lag < 1 for lag in requested):
        raise ValueError("lags must contain one or more positive integers.")
    rows = [row for row in _read_errors(directory)
            if str(row.get("probe", "")).startswith("fixed")
            and _number(row.get("lag")) in requested]
    present = {int(row["lag"]) for row in rows}
    if missing := set(requested) - present:
        raise ValueError(f"No saved fixed-lag measurements for {sorted(missing)} in {directory}.")
    groups = _ordered_groups(rows, fixed=True)
    family_count = len({_arm(row)[:2] for row in rows})
    panels = 3 if probability else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.5), squeeze=False)
    axes = axes[0]
    palette = plt.get_cmap("tab10")
    colors = {lag: palette(i % 10) for i, lag in enumerate(requested)}
    invalid = False
    seen_labels = set()
    for key, values in groups.items():
        lag = key[-1]
        label = f"r = {lag}" if family_count == 1 else f"r = {lag}; {_arm_label(key[:3])}"
        curve_label = "_nolegend_" if label in seen_labels else label
        seen_labels.add(label)
        invalid |= _draw_bracket(axes, values, x, colors[lag], curve_label,
                                 probability_axis=axes[2] if probability else None)
    _style_axes(axes, x)
    if probability:
        axes[2].set(ylabel="Worst-case correct probability 1 − Eₜ(r)", ylim=(-.02, 1.02),
                    title="Probability of the correct answer")
    return _finish(fig, axes, f"Fixed-lag generalization · {_experiment_label(rows)}", invalid,
                   directory, "fixed_lag_convergence", save)


def plot_moving_lag(run_or_path, theta=.5, *, x="step", save=False):
    """Plot E_t(r_t) and r_t=max(1,floor(theta*delta_ref*m_t/h_t)).

    This reads the ``theory_ratio`` probe. Its coefficient uses the gap fixed
    after acquisition, rather than silently changing the coefficient at each
    evaluation. Increasing r_t and decreasing error must be assessed together.
    ``theta`` must be between zero and one, and must have been evaluated during
    training. No radius is clipped to a finite evaluation-sequence length.
    A recorded probe without a positive acquired gap is annotated as
    unavailable, so one unsuccessful acquisition does not interrupt a notebook.

    Return an open matplotlib Figure. See ``plot_fixed_lags`` for x/save options.
    """
    if x not in _X_LABELS:
        raise ValueError("x must be 'step' or 'S'.")
    if not 0 < theta < 1:
        raise ValueError("theta must be strictly between 0 and 1.")
    directory = _directory(run_or_path)
    requested_rows = [row for row in _read_errors(directory)
                      if row.get("probe") == "theory_ratio"
                      and math.isclose(_number(row.get("theta")), theta,
                                       rel_tol=1e-10, abs_tol=1e-12)]
    if not requested_rows:
        raise ValueError(f"No theory_ratio probe with theta={theta:g} in {directory}.")
    rows = [row for row in requested_rows if math.isfinite(_number(row.get("lag")))
            and _number(row.get("lag")) >= 1]
    requested_arms = {_arm(row) for row in requested_rows}
    unavailable_arms = requested_arms - {_arm(row) for row in rows}
    groups = _ordered_groups(rows, fixed=False)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), squeeze=False)
    axes = axes[0]
    invalid = False
    family_count = len({_arm(row)[:2] for row in requested_rows})
    for index, (arm, values) in enumerate(groups.items()):
        color = plt.get_cmap("tab10")(index % 10)
        label = f"seed {arm[2]}" if family_count == 1 else _arm_label(arm)
        invalid |= _draw_bracket(axes, values, x, color, label)
        times = [_number(row.get(x)) for row in values]
        radii = [_number(row.get("lag")) for row in values]
        axes[2].plot(times, radii, color=color, linewidth=1.6, alpha=.85, label=label)
    _style_axes(axes, x)
    axes[2].set(ylabel="Actual integer radius rₜ", ylim=(0, None),
                title="rₜ = max(1, ⌊θ δref mₜ/hₜ⌋)")
    if rows and max(_number(row.get("lag"), 1) for row in rows) > 100:
        axes[2].set_yscale("symlog", linthresh=10)
    if not rows:
        message = ("Positive acquired gap unavailable\n"
                   f"{len(unavailable_arms)} of {len(requested_arms)} seed runs\n"
                   "No moving radius or error evaluated")
        for axis in axes:
            axis.text(.5, .5, message, ha="center", va="center",
                      transform=axis.transAxes, fontsize=10, color=".35")
    elif unavailable_arms:
        message = (f"{len(unavailable_arms)} of {len(requested_arms)} seed runs unavailable:\n"
                   "positive acquired gap unavailable")
        axes[2].text(.03, .97, message, ha="left", va="top", transform=axes[2].transAxes,
                     fontsize=8, color=".35", bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    axes[0].set_ylabel("Worst-case error Eₜ(rₜ)")
    return _finish(fig, axes, f"Growing-lag generalization · {_experiment_label(requested_rows)} · θ = {theta:g}", invalid,
                   directory, f"moving_lag_convergence_theta_{theta:g}", save)
