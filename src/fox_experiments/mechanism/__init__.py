"""Controlled theory diagnostics, separate from ordinary text-model evaluation."""

from .config import MechanismConfig
from .model import BindingFoX
from .tasks import templates, loss_function
from .bounds import sufficient_error_bound, contiguous_radius, witness_error
from .runner import run_mechanism

__all__ = [
    "MechanismConfig",
    "BindingFoX",
    "templates",
    "loss_function",
    "sufficient_error_bound",
    "contiguous_radius",
    "witness_error",
    "run_mechanism",
]
