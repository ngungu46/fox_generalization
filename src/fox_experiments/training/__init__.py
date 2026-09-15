"""Public small-scale training API used by notebooks and command-line entry points."""

from .config import ExperimentConfig, budget
from .experiment import run_experiment
from .optimizers import make_optimizer, parameter_groups
from .trainer import train_segment

__all__ = [
    "ExperimentConfig",
    "budget",
    "run_experiment",
    "make_optimizer",
    "parameter_groups",
    "train_segment",
]
