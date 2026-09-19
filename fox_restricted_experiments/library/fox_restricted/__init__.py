"""Restricted FoX generalization experiments.

Use the public experiment API for training and plotting. The scientific kernel
remains isolated in ``legacy`` so existing checkpoint hashes remain valid.
"""
from .legacy.core import Config
from .data.random_streams import RandomStreamConfig
from .study import Experiment, Run, a100_config, make_experiments
from .training.experiment import train, summarize
from .plotting.convergence import plot_fixed_lags, plot_moving_lag
from .runtime import check_runtime, prepare_output, benchmark
from .training.performance import PerformanceConfig
from .profiling import profile_experiment, format_profile

__version__ = "0.3.0"
__all__ = [
    "Config", "RandomStreamConfig", "Experiment", "Run", "a100_config",
    "make_experiments", "train", "summarize", "plot_fixed_lags", "plot_moving_lag",
    "check_runtime", "prepare_output", "benchmark",
    "PerformanceConfig", "profile_experiment", "format_profile",
]
