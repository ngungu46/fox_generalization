"""Plot the measured restricted-model experiment without importing the trainer.

Only numpy and matplotlib are required.  CSV files remain the source of truth;
figures never extrapolate an optimizer trajectory beyond its saved checkpoints.
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
import numpy as np


COLORS = {"sgd": "#2166ac", "adam": "#d95f02"}
GATE_NAMES = {
    "learned": "Learned retrieval gate",
    "retrieval_frozen": "Frozen retrieval gate (ALiBi)",
    "both_frozen": "Both gates frozen (extra control)",
}
KEY_FIELDS = ("seed", "gate_mode", "optimizer")


def _number(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path):
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row):
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def _group(rows):
    result = defaultdict(list)
    for row in rows:
        result[_key(row)].append(row)
    return result


def _latest(rows):
    """Select the last evaluated checkpoint separately for every branch."""
    result = []
    for branch in _group(rows).values():
        step = max((_number(row.get("step"), -1) for row in branch), default=-1)
        result.extend(row for row in branch if _number(row.get("step"), -1) == step)
    return result


def _mean(values):
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _minimum(values):
    values = [v for v in values if math.isfinite(v)]
    return min(values) if values else float("nan")


def _per_seed_lag(rows, metric="min_probability"):
    # Each CSV row already aggregates ordered key pairs for one prefix. Taking
    # a second minimum therefore gives a minimum over the finite test panel.
    groups = defaultdict(list)
    for row in rows:
        groups[(_key(row), _number(row.get("lag")))].append(_number(row.get(metric)))
    result = defaultdict(dict)
    for (branch, lag), values in groups.items():
        if math.isfinite(lag):
            result[branch][lag] = _minimum(values)
    return result


def _per_seed_mean_lag(rows):
    groups = defaultdict(list)
    for row in rows:
        value = _number(row.get("mean_probability"))
        weight = _number(row.get("example_count"), 1)
        lag = _number(row.get("lag"))
        if math.isfinite(value) and math.isfinite(weight) and weight > 0 and math.isfinite(lag):
            groups[(_key(row), lag)].append((value, weight))
    result = defaultdict(dict)
    for (key, lag), values in groups.items():
        result[key][lag] = sum(v * w for v, w in values) / sum(w for _, w in values)
    return result


def _format(value, digits=4):
    value = _number(value)
    return f"{value:.{digits}g}" if math.isfinite(value) else "unavailable"


def _save(fig, path, pdf=False):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _empty(ax, message="No saved observations"):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, color="0.4")


def _finish_axes(axes):
    for ax in np.asarray(axes).ravel():
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=8, frameon=False,
                      ncol=2 if len(unique) > 6 else 1)


def _lag_plot(final_rows, gates, R, path):
    fig, axes = plt.subplots(1, len(gates), figsize=(6.3 * len(gates), 4.6), squeeze=False)
    series = _per_seed_lag(final_rows)
    mean_series = _per_seed_mean_lag(final_rows)
    for ax, gate in zip(axes.ravel(), gates):
        has_data = False
        for opt, color in COLORS.items():
            branches = [vals for key, vals in series.items() if key[1:] == (gate, opt)]
            if not branches:
                continue
            has_data = True
            for values in branches:
                xs = sorted(values)
                ax.plot(xs, [values[x] for x in xs], color=color, alpha=.18, lw=.9)
            xs = sorted({lag for values in branches for lag in values})
            # Unequal finite run budgets remain explicit in report.md. This
            # envelope is a seed range, not a confidence interval.
            batches = [[v[x] for v in branches if x in v and math.isfinite(v[x])] for x in xs]
            means = [_mean(batch) for batch in batches]
            lower = [_minimum(batch) for batch in batches]
            upper = [max(batch) if batch else float("nan") for batch in batches]
            ax.fill_between(xs, lower, upper, color=color, alpha=.12)
            ax.plot(xs, means, color=color, lw=2.2, label=f"{opt.upper()}: seed mean of minima")
            mean_branches = [vals for key, vals in mean_series.items() if key[1:] == (gate, opt)]
            if mean_branches:
                ax.plot(xs, [_mean([vals[x] for vals in mean_branches if x in vals]) for x in xs],
                        color=color, ls="--", lw=1.3, label=f"{opt.upper()}: mean probability")
        for threshold in (.5, .9):
            ax.axhline(threshold, color=".55", linestyle=":", lw=.8)
            ax.text(.985, threshold + .01, f"{threshold:g}", transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=8, color=".4")
        if R is not None:
            ax.axvline(R, color=".3", linestyle="--", lw=1, label=f"Training lag R = {R}")
            if gate == "learned" and R >= 2:
                ax.axvline(R + 2, color="#6a51a3", linestyle=":", lw=1.4,
                           label=("SGD horizon 4 (asymptotic theorem)" if R == 2
                                  else "SGD horizon R + 2 (conditional asymptotic)"))
        if not has_data:
            _empty(ax)
        ax.set(title=GATE_NAMES.get(gate, gate), xlabel="Target lag r (records)",
               ylabel="Correct-answer probability", ylim=(-.025, 1.045))
    fig.suptitle("Generalization at the latest saved checkpoint of each run", fontsize=14)
    fig.text(.5, .015, "Solid: seed mean of finite-panel minima. Dashed: mean over tested examples, then seeds. "
             "Band: range of seed minima. Finite runs do not establish asymptotic limits.", ha="center", fontsize=8)
    _finish_axes(axes)
    fig.tight_layout(rect=(0, .06, 1, .94))
    _save(fig, path, pdf=True)


def _training_plot(rows, history, gates, R, path):
    fig, axes = plt.subplots(len(gates), 2, figsize=(12.7, 3.65 * len(gates)), squeeze=False)
    history_lookup = {(_key(row), _number(row.get("step"))): row for row in history}
    chosen_lags = [R, R + 2, R + 3, 2 * R + 4] if R is not None else []
    if not chosen_lags:
        chosen_lags = sorted({_number(row.get("lag")) for row in rows if math.isfinite(_number(row.get("lag")))})[:4]
    chosen_lags = list(dict.fromkeys(chosen_lags))
    lag_colors = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    aggregated = defaultdict(list)
    for row in rows:
        aggregated[(_key(row), _number(row.get("step")), _number(row.get("lag")))].append(row)
    curves = defaultdict(list)
    for (branch, step, lag), batch in aggregated.items():
        source = history_lookup.get((branch, step), {})
        S = _number(batch[0].get("S"))
        if not math.isfinite(S):
            S = _number(source.get("S"))
        value = _minimum([_number(row.get("min_probability")) for row in batch])
        if math.isfinite(S) and math.isfinite(value):
            curves[(branch, lag)].append((S, step, value))
    for row_index, gate in enumerate(gates):
        for col_index, opt in enumerate(COLORS):
            ax = axes[row_index, col_index]
            has_data = False
            seed_styles = {}
            for lag, color in zip(chosen_lags, lag_colors):
                branches = [(key, values) for (key, r), values in curves.items() if key[1:] == (gate, opt) and r == lag]
                for branch_index, (key, values) in enumerate(sorted(branches)):
                    values = sorted(values, key=lambda point: point[1])
                    linestyle = ("-", "--", "-.", ":")[branch_index % 4]
                    seed_styles[key[0]] = linestyle
                    ax.plot([p[0] for p in values], [p[2] for p in values], color=color,
                            lw=1.5, alpha=.8, ls=linestyle,
                            label=f"r = {lag:g}" if branch_index == 0 else "_nolegend_")
                    has_data = True
            for seed, linestyle in sorted(seed_styles.items()):
                ax.plot([], [], color=".4", ls=linestyle, lw=1.2, label=f"seed {seed}")
            ax.axhline(.5, color=".55", ls=":", lw=.8)
            ax.axhline(.9, color=".55", ls=":", lw=.8)
            ax.set_xscale("symlog", linthresh=1)
            ax.set(title=f"{GATE_NAMES.get(gate, gate)} · {opt.upper()}",
                   xlabel="Continuation learning-rate sum S (own run schedule)",
                   ylabel="Minimum correct-answer probability", ylim=(-.025, 1.045))
            if not has_data:
                _empty(ax, "No saved probabilities at the selected lags")
    fig.suptitle("Probability during training at selected test lags", fontsize=14)
    fig.text(.5, .012, "Each seed keeps its own learning-rate sum; curves are not aligned or averaged across different S values.",
             ha="center", fontsize=9)
    _finish_axes(axes)
    fig.tight_layout(rect=(0, .045, 1, .95))
    _save(fig, path)


def _diagnostic_plot(history, final_rows, gates, R, path):
    fig, axes = plt.subplots(len(gates), 4, figsize=(19, 3.9 * len(gates)), squeeze=False)
    specifications = [
        ("log_loss", "Log ordinary answer loss", "ln(loss)"),
        ("content_horizon", "Content horizon", "1 + m δ_min / h"),
        ("m_rho", "Binding-contamination scale", "m ρ"),
    ]
    for gate_index, gate in enumerate(gates):
        for metric_index, (metric, title, ylabel) in enumerate(specifications):
            ax = axes[gate_index, metric_index]
            has_data = False
            for key, branch in sorted(_group(history).items()):
                if key[1] != gate:
                    continue
                points = []
                for row in sorted(branch, key=lambda row: _number(row.get("step"), -1)):
                    x, y = _number(row.get("S")), _number(row.get(metric))
                    if metric == "log_loss" and not math.isfinite(y):
                        loss = _number(row.get("loss"))
                        y = math.log(loss) if loss > 0 else float("nan")
                    if math.isfinite(x) and math.isfinite(y):
                        points.append((x, y))
                if points:
                    ax.plot(*zip(*points), color=COLORS.get(key[2], ".3"), lw=1.5,
                            alpha=.8, label=f"{key[2].upper()}, seed {key[0]}")
                    has_data = True
            ax.set_xscale("symlog", linthresh=1)
            if metric == "m_rho":
                ax.set_yscale("symlog", linthresh=1e-7)
            if metric == "content_horizon" and R is not None:
                ax.axhline(R, color=".5", ls="--", lw=.8, label="Training lag R")
                if gate == "learned" and R >= 2:
                    ax.axhline(R + 2, color="#6a51a3", ls=":", lw=1,
                               label=("SGD asymptote 4 (R = 2)" if R == 2
                                      else "SGD R + 2 (conditional, R > 2)"))
            ax.set(title=f"{GATE_NAMES.get(gate, gate)}\n{title}", xlabel="Continuation sum S", ylabel=ylabel)
            if not has_data:
                _empty(ax)
        ax = axes[gate_index, 3]
        for metric, label, linestyle in [("min_probability", "answer", "-"), ("min_attention", "attention", "--")]:
            series = _per_seed_lag(final_rows, metric)
            for opt, color in COLORS.items():
                branches = [vals for key, vals in series.items() if key[1:] == (gate, opt)]
                xs = sorted({lag for values in branches for lag in values})
                if xs:
                    ax.plot(xs, [_mean([v[x] for v in branches if x in v]) for x in xs],
                            color=color, ls=linestyle, lw=1.8, label=f"{opt.upper()} {label}")
        ax.axhline(.5, color=".55", ls=":", lw=.8)
        ax.set(title="Latest checkpoint: attention vs answer", xlabel="Target lag r (records)",
               ylabel="Seed mean of finite-panel minima", ylim=(-.025, 1.045))
        if not final_rows:
            _empty(ax)
    fig.suptitle("Mechanism diagnostics from saved checkpoints", fontsize=15)
    fig.text(.5, .012, "A large content horizon is a diagnostic, not a probability certificate; "
             "answer confidence also depends on binding, old prefixes, and output scale.", ha="center", fontsize=9)
    _finish_axes(axes)
    fig.tight_layout(rect=(0, .045, 1, .95))
    _save(fig, path)


def _markdown(config_data, history, lag_rows, runs, R):
    config = config_data.get("config", config_data)
    final_rows = _latest(lag_rows)
    final_history = {_key(row): row for row in _latest(history)}
    settings = []
    for name in ("n", "d", "R", "steps"):
        if name in config:
            settings.append(f"{name}={config[name]}")
    if "R" not in config and R is not None:
        settings.append(f"R={R}")
    lines = ["# Restricted-model empirical report", "",
             "These results describe the saved finite training runs. They can test consistency with the theory; "
             "they do not prove the asymptotic optimizer claims.", ""]
    if settings:
        lines.extend(["Settings: " + ", ".join(settings) + ". See `config.json` for initialization, optimizer, and evaluation settings.", ""])
    prefixes = sorted({_number(row.get("prefix")) for row in lag_rows if math.isfinite(_number(row.get("prefix")))})
    lags = sorted({_number(row.get("lag")) for row in lag_rows if math.isfinite(_number(row.get("lag")))})
    lines.extend([
        "- Checkpoint step 0 is the paired Gaussian initialization; step 3 ends optimizer-specific acquisition. "
        "Continuation uses each optimizer's own learning-rate sum S.",
        "- The frozen-retrieval comparison freezes only the retrieval forgetting parameter h. "
        "The binding gate remains trainable. Any `both_frozen` arm is a separate extra control.",
        "- The SGD continuation schedule is a practical experiment schedule, not the literal conservative theorem schedule.",
        "- Frozen-gate arbitrary-prefix success requires the theorem's conditions, including h₀ > log(2). "
        "Positive matching gaps, suitable row bounds, controlled binding contamination, and positive output scale must also hold.",
        "- Learned-gate SGD's asymptotic horizon is 4 in the main R = 2 theorem. "
        "The R + 2 reference for R > 2 is conditional on the generalized theorem's hypotheses. "
        "At the boundary, output probability must be checked separately from target attention.",
        "- The fully proved frozen-retrieval answer result uses R=2. Frozen R>2 and learned-Adam R>2 "
        "runs are empirical extension tests here; the notebook also omits the additional R>2 SGD basin preparation.",
        "- Test prefix lengths: " + (", ".join(f"{p:g}" for p in prefixes) or "none saved") + ". "
        "Test lags: " + (", ".join(f"{r:g}" for r in lags) or "none saved") + ".", "",
        "## Run completion and latest observations", "",
        "Plots use the latest saved evaluation separately for each branch. A stopped run may therefore have an earlier "
        "checkpoint than a completed run. Seed envelopes are observed ranges, not confidence intervals.", "",
        "| Seed | Gate | Optimizer | Completed / requested | Status | Evaluation step | Reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    final_steps = {_key(row): _format(row.get("step"), 8) for row in final_rows}
    branch_runs = list(runs)
    known = {_key(row) for row in branch_runs}
    for key, row in final_history.items():
        if key not in known:
            branch_runs.append(dict(row, completed_steps=row.get("step", ""), requested_steps="unavailable"))
    for row in sorted(branch_runs, key=_key):
        key = _key(row)
        reason = str(row.get("reason", "")).replace("|", "/").replace("\n", " ") or "—"
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {row.get('completed_steps', 'unavailable')} / "
                     f"{row.get('requested_steps', 'unavailable')} | {row.get('status', 'unavailable')} | "
                     f"{final_steps.get(key, 'none')} | {reason} |")
    if not branch_runs:
        lines.append("| — | — | — | — | No saved runs | — | — |")
    lines.extend(["", "## Generalization at selected lags", "",
                  "For each seed, the observed minimum is over all tested ordered key pairs and prefix lengths. "
                  "The table gives the mean and worst of those per-seed minima. It does not cover untested prefixes. "
                  "An analytic certificate, when present, is a separate conditional arbitrary-prefix lower bound; "
                  "blank or unavailable certificates are not empirical failures.", "",
                  "| Gate | Optimizer | Lag | Seeds | Mean probability | Mean of observed minima | Worst observed minimum | Analytic lower bound (worst seed) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    selected = set([R, R + 2, R + 3, 2 * R + 4]) if R is not None else set(lags[:4])
    series = _per_seed_lag(final_rows)
    mean_series = _per_seed_mean_lag(final_rows)
    certificates = _per_seed_lag(final_rows, "certified_probability_lower_bound")
    groups = defaultdict(list)
    for key, values in series.items():
        for lag, value in values.items():
            if lag in selected:
                groups[(key[1], key[2], lag)].append((key, value))
    for (gate, opt, lag), observations in sorted(groups.items()):
        values = [value for _, value in observations]
        bounds = [certificates.get(key, {}).get(lag, float("nan")) for key, _ in observations]
        mean_probability = _mean([mean_series.get(key, {}).get(lag, float("nan")) for key, _ in observations])
        # Never label a partial collection as a certificate across every seed.
        bound = min(bounds) if bounds and all(math.isfinite(v) for v in bounds) else float("nan")
        lines.append(f"| {gate} | {opt} | {lag:g} | {len(values)} | {_format(mean_probability)} | {_format(_mean(values))} | "
                     f"{_format(_minimum(values))} | {_format(bound)} |")
    if not groups:
        lines.append("| — | — | — | 0 | unavailable | unavailable | unavailable | unavailable |")
    lines.extend(["", "## Final mechanism diagnostics", "",
                  "| Seed | Gate | Optimizer | Step | ln(loss) | δ_min | Max row norm | h | mρ | Content horizon | SGD balance |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for key, row in sorted(final_history.items()):
        metrics = [row.get(name) for name in ("step", "log_loss", "delta_min", "row_norm_max", "h", "m_rho", "content_horizon", "sg_balance")]
        lines.append("| " + " | ".join([*key, *[_format(v) for v in metrics]]) + " |")
    if not final_history:
        lines.append("| — | — | — | — | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
    lines.extend(["", "The content horizon is 1 + m δ_min / h; the SGD balance is m δ_min − (R + 1)h − log(m). "
                  "These are mechanism diagnostics, not substitutes for the actual answer probabilities. "
                  "`theory_diagnostics.png` displays target attention separately from correct-answer probability, "
                  "since the two can behave differently.", "",
                  "## Saved figures", "",
                  "- `generalization_by_lag.png` and `.pdf`: probability versus lag at each branch's latest evaluation.",
                  "- `probability_by_training.png`: selected-lag probabilities along each saved training trajectory.",
                  "- `theory_diagnostics.png`: log loss, content horizon, binding contamination, and attention versus answer probability.",
                  "", "The CSV files contain all measurements and stopped-run statuses. Unobserved checkpoints are never filled in.", ""])
    return "\n".join(lines)


def make_report(out_dir):
    """Read an experiment directory and return the generated artifact paths.

    Empty or stopped experiments still produce explicit, readable placeholder
    figures and a status report. Nothing is imputed as a successful outcome.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = _read_csv(out_dir / "history.csv")
    lag_rows = _read_csv(out_dir / "lag_probabilities.csv")
    runs = _read_csv(out_dir / "runs.csv")
    config_path = out_dir / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = config_data.get("config", config_data)
    R = _number(config.get("R", config.get("train_R")))
    if not math.isfinite(R):
        R = next((_number(row.get("train_R")) for row in history + runs if math.isfinite(_number(row.get("train_R")))), float("nan"))
    R = int(R) if math.isfinite(R) else None
    observed_gates = {str(row.get("gate_mode")) for row in history + lag_rows + runs if row.get("gate_mode")}
    observed_gates.update(config_data.get("gate_modes", []))
    gates = [gate for gate in GATE_NAMES if gate in observed_gates]
    gates += sorted(observed_gates - set(gates))
    gates = gates or ["learned", "retrieval_frozen"]
    paths = {
        "generalization_by_lag": str(out_dir / "generalization_by_lag.png"),
        "generalization_by_lag_pdf": str(out_dir / "generalization_by_lag.pdf"),
        "probability_by_training": str(out_dir / "probability_by_training.png"),
        "theory_diagnostics": str(out_dir / "theory_diagnostics.png"),
        "report": str(out_dir / "report.md"),
    }
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11}):
        _lag_plot(_latest(lag_rows), gates, R, Path(paths["generalization_by_lag"]))
        _training_plot(lag_rows, history, gates, R, Path(paths["probability_by_training"]))
        _diagnostic_plot(history, _latest(lag_rows), gates, R, Path(paths["theory_diagnostics"]))
    Path(paths["report"]).write_text(_markdown(config_data, history, lag_rows, runs, R), encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="Directory containing experiment CSV files")
    args = parser.parse_args()
    for name, path in make_report(args.out_dir).items():
        print(f"{name}: {path}")
