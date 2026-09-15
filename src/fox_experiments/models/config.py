"""Configuration for the shared Colab and cluster language model."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

GATE_MODES = (
    "factorized_constant",
    "direct_constant",
    "original_data",
    "factorized_data",
)


@dataclass
class ModelConfig:
    """Architecture and gate settings, independent of optimizer settings.

    ``attention_backend='sdpa'`` is the portable PyTorch implementation.
    ``'upstream'`` selects the supplied paper's fused GPU kernel. The latter
    is imported lazily, so the pilot does not need Triton installed.
    """

    vocab_size: int = 50257
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    gate_mode: str = "factorized_constant"
    g0: float = 0.1
    other_g0: float = 0.1
    seed: int = 0
    ff_hidden: Optional[int] = None
    dropout: float = 0.0
    norm_eps: float = 1e-5
    gate_weight_std: float = 0.02
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}")
        if self.d_model < 1 or self.n_heads < 1 or self.d_model % self.n_heads:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if (
            self.n_layers < 1
            or self.vocab_size < 2
            or self.g0 <= 0
            or self.other_g0 <= 0
        ):
            raise ValueError(
                "n_layers >= 1, vocab_size >= 2, and both initial decay rates > 0 are required"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.attention_backend not in ("sdpa", "upstream"):
            raise ValueError("attention_backend must be 'sdpa' or 'upstream'")
        if self.attention_backend == "upstream" and self.dropout != 0:
            raise ValueError("The upstream attention kernel requires dropout=0")
