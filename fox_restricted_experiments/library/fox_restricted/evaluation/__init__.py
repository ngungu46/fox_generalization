"""Worst-case answer-error bounds, certified radii, and diagnostic reports."""
from ..legacy.asymptotics import (
    ErrorSnapshot, certified_radius, error_bounds, evaluate_asymptotics,
    evaluate_snapshot, scalar_error_bounds, scalar_snapshot, snapshot,
)


def make_asymptotic_report(*args, **kwargs):
    """Load the historical file-based reporter only when requested."""
    from ..legacy.asymptotic_report import make_asymptotic_report as build
    return build(*args, **kwargs)


def make_report(*args, **kwargs):
    """Keep historical reporting from changing the notebook plotting backend."""
    from ..legacy.report import make_report as build
    return build(*args, **kwargs)

__all__ = ["ErrorSnapshot", "certified_radius", "error_bounds", "evaluate_asymptotics",
           "evaluate_snapshot", "scalar_error_bounds", "scalar_snapshot", "snapshot",
           "make_asymptotic_report", "make_report"]
