"""Exact causal forgetting attention with optional query chunking.

For a query at i and a key at j <= i, the logit is
q_i @ k_j / sqrt(head_width) - sum(decay[j + 1:i + 1]).
The SDPA implementation is a transparent reference for Colab; query chunking
reduces temporary attention-map memory while retaining every past key.
"""

from __future__ import annotations
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .config import ModelConfig
from .gates import ForgetGate


def forgetting_cost(
    decay: Tensor, query_start: int = 0, query_end: Optional[int] = None
) -> Tensor:
    """Prefix difference [B,H,Q,T]; diagonal is 0 and past costs are positive.

    Entries for future keys are negative and must receive a causal mask before
    softmax. Keeping this primitive unmasked makes its indexing easy to test.
    """
    prefix = decay.float().cumsum(-1)
    end = decay.shape[-1] if query_end is None else query_end
    return prefix[..., query_start:end, None] - prefix[..., None, :]


class FoXAttention(nn.Module):
    """Multi-head attention with learned cumulative log-retention bias."""

    def __init__(self, cfg: ModelConfig, block_index: int):
        super().__init__()
        self.n_heads, self.d_head = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        self.attention_backend = cfg.attention_backend
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mode = cfg.gate_mode if block_index == 0 else "original_data"
        initial_decay = cfg.g0 if block_index == 0 else cfg.other_g0
        self.gate = ForgetGate(
            cfg.d_model,
            cfg.n_heads,
            mode,
            initial_decay,
            cfg.seed + 100003 + block_index,
            cfg.gate_weight_std,
        )

    def projected(self, x: Tensor):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(self, x: Tensor, query_chunk: Optional[int] = None) -> Tensor:
        q, k, v = self.projected(x)
        length = x.shape[1]
        if length < 1:
            raise ValueError("Sequences must contain at least one token")
        chunk = length if query_chunk is None else query_chunk
        if chunk < 1:
            raise ValueError("query_chunk must be positive")
        decay = self.gate(x)
        if self.attention_backend == "upstream":
            # Fused kernel uses log retention; our gate returns positive decay.
            # It handles the complete causal sequence without a dense mask, so
            # query_chunk is only relevant to the SDPA reference path below.
            from fox_experiments.full_training.attention import (
                upstream_forgetting_attention,
            )

            attended = upstream_forgetting_attention(q, k, v, log_fgate=-decay)
            result = attended.transpose(1, 2).contiguous().view_as(x)
            return self.out(result)

        prefix = decay.float().cumsum(-1)
        key_positions = torch.arange(length, device=x.device)
        outputs = []
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            bias = prefix[..., None, :] - prefix[..., start:end, None]
            query_positions = torch.arange(start, end, device=x.device)
            bias = bias.masked_fill(
                key_positions[None, :] > query_positions[:, None], float("-inf")
            )
            # FP32 bias is accepted with lower precision QKV by SDPA; the
            # prefix subtraction is always done in FP32 before the call.
            attended = F.scaled_dot_product_attention(
                q[..., start:end, :],
                k,
                v,
                attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            outputs.append(attended)
        result = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view_as(x)
        return self.out(result)
