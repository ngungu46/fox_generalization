"""Default-FoX paper recipe versus alternative optimizers, from initialization."""

from .config import PaperBaselineConfig
from .runner import run_paper_comparison, summarize_paper_comparison

__all__ = ["PaperBaselineConfig", "run_paper_comparison", "summarize_paper_comparison"]
