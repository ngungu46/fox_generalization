"""Small paired answer-only / full-stream objective experiment."""

from .config import LossStudyConfig, branches, intervention
from .runner import run_loss_study
from .analysis import summarize_loss_study

__all__ = [
    "LossStudyConfig",
    "branches",
    "intervention",
    "run_loss_study",
    "summarize_loss_study",
]
