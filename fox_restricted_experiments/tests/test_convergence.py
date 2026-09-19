"""Plot semantics: correct probes, stable log errors, and invalid certificates."""
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from fox_restricted.plotting.convergence import plot_fixed_lags, plot_moving_lag


def save_rows(directory: Path, rows):
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"config": {"R": 2}}))
    with (directory / "asymptotic_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def row(lag=3, step=0, probe="fixed", theta="", **updates):
    value = dict(seed=0, gate_mode="learned", optimizer="sgd", step=step, S=step / 10,
                 probe=probe, theta=theta, lag=lag, lower_error=.01, upper_error=.02,
                 log_lower_error=math.log(.01), log_upper_error=math.log(.02), bound_valid=True)
    value.update(updates)
    return value


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_default_lags_are_R_plus_one_to_three(tmp_path):
    save_rows(tmp_path, [row(lag=lag) for lag in (1, 2, 3, 4, 5, 6)])
    fig = plot_fixed_lags(SimpleNamespace(path=tmp_path))
    assert len(fig.axes) == 2
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert labels == ["r = 3", "r = 4", "r = 5"]
    assert fig.axes[0].get_ylim() == (-.02, 1.02)


def test_underflow_keeps_saved_log_error_and_probability_orientation(tmp_path):
    save_rows(tmp_path, [row(log_lower_error=-10000, log_upper_error=-9000,
                            lower_error=0, upper_error=0)])
    fig = plot_fixed_lags(tmp_path, lags=(3,), probability=True)
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [-9000 / math.log(10)])
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(), [-10000 / math.log(10)])
    np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), [1])
    save_rows(tmp_path, [row(log_lower_error=math.log(.2), log_upper_error=math.log(.8))])
    fig = plot_fixed_lags(tmp_path, lags=(3,), probability=True)
    np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(), [.2])
    np.testing.assert_allclose(fig.axes[2].lines[1].get_ydata(), [.8])


def test_invalid_certificate_is_trivial_one_and_marked(tmp_path):
    save_rows(tmp_path, [row(bound_valid=False, upper_error="", log_upper_error="")])
    fig = plot_fixed_lags(tmp_path, lags=(3,))
    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(), [1])
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), [0])
    assert "Uncertified: trivial upper = 1" in [text.get_text() for text in fig.legends[0].texts]


def test_moving_plot_selects_requested_probe_and_preserves_radius(tmp_path):
    rows = [row(lag="", step=0, probe="theory_ratio", theta=.5, optimizer="adam",
                bound_valid=False, lower_error="", upper_error="",
                log_lower_error="", log_upper_error=""),
            row(lag=1, step=3, probe="theory_ratio", theta=.5, optimizer="adam"),
            row(lag=10**12, step=1000, probe="theory_ratio", theta=.5, optimizer="adam"),
            row(lag=7, step=1000, probe="theory_ratio", theta=.75, optimizer="adam"),
            row(lag=9, step=1000, probe="adaptive_theta", theta=.5, optimizer="adam")]
    save_rows(tmp_path, rows)
    fig = plot_moving_lag(tmp_path, theta=.5, x="S", save=True)
    np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), [1, 10**12])
    np.testing.assert_allclose(fig.axes[2].lines[0].get_xdata(), [.3, 100])
    assert (tmp_path / "figures" / "moving_lag_convergence_theta_0.5.png").exists()
    assert (tmp_path / "figures" / "moving_lag_convergence_theta_0.5.pdf").exists()


def test_missing_probe_or_lag_fails_instead_of_silently_plotting_other_curves(tmp_path):
    save_rows(tmp_path, [row(lag=3, probe="adaptive_theta", theta=.5)])
    with pytest.raises(ValueError, match="No theory_ratio"):
        plot_moving_lag(tmp_path)
    with pytest.raises(ValueError, match="No saved fixed-lag"):
        plot_fixed_lags(tmp_path)
    with pytest.raises(ValueError, match="strictly between"):
        plot_moving_lag(tmp_path, theta=1)


def test_recorded_unavailable_radius_returns_annotated_figure_without_fake_curve(tmp_path):
    save_rows(tmp_path, [row(lag="", probe="theory_ratio", theta=.5, optimizer="adam",
                            bound_valid=False, lower_error="", upper_error="",
                            log_lower_error="", log_upper_error="")])
    fig = plot_moving_lag(tmp_path)
    assert len(fig.axes) == 3
    assert not fig.axes[1].lines
    assert not fig.axes[2].lines
    for axis in fig.axes:
        assert "Positive acquired gap unavailable" in axis.texts[0].get_text()
        assert "1 of 1 seed runs" in axis.texts[0].get_text()


def test_partly_unavailable_seeds_are_counted_and_not_filled(tmp_path):
    save_rows(tmp_path, [row(lag="", probe="theory_ratio", theta=.5, seed=0, optimizer="adam"),
                        row(lag=4, probe="theory_ratio", theta=.5, seed=1, optimizer="adam")])
    fig = plot_moving_lag(tmp_path)
    assert len(fig.axes[2].lines) == 1
    np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), [4])
    assert "1 of 2 seed runs unavailable" in fig.axes[2].texts[0].get_text()
