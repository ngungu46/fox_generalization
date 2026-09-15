"""Original FoX versus paired gate/optimizer interventions, from initialization."""

from .config import PaperBaselineConfig, branch_specs
from .runner import run_paper_comparison, summarize_paper_comparison

__all__ = [
    "PaperBaselineConfig",
    "branch_specs",
    "run_paper_comparison",
    "summarize_paper_comparison",
]
