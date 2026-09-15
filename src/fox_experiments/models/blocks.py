"""Pre-normalized residual blocks, RMS normalization, and SwiGLU."""

from __future__ import annotations
import math
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .attention import FoXAttention
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        width = cfg.ff_hidden or 64 * math.ceil((8 * cfg.d_model / 3) / 64)
        self.w1 = nn.Linear(cfg.d_model, width, bias=False)
        self.w3 = nn.Linear(cfg.d_model, width, bias=False)
        self.w2 = nn.Linear(width, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w3(x)) * self.w1(x))


class FoXBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = FoXAttention(cfg, index)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, query_chunk: Optional[int] = None) -> Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), query_chunk=query_chunk))
        return x + self.dropout(self.ff(self.ff_norm(x)))
