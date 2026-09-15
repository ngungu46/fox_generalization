"""Exact token-level cross entropy without a full saved vocabulary tensor."""

from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def per_token_nll(
    hidden: Tensor,
    targets: Tensor,
    output_head: nn.Module,
    chunk_size: int = 128,
    checkpoint_logits: bool = True,
) -> Tensor:
    """NLL with shape targets.shape; -100 labels contribute zero.

    Output vocabulary logits are computed in token chunks. During training,
    checkpointing recomputes each chunk on backward so saved softmax activations
    do not scale as batch * context * vocabulary. This is exact cross entropy.
    """
    if hidden.shape[:-1] != targets.shape:
        raise ValueError(
            "hidden/targets shapes must match except the final hidden dimension"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    flat_h, flat_y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)

    def calculate(h, y):
        return F.cross_entropy(
            output_head(h).float(), y, reduction="none", ignore_index=-100
        )

    pieces = []
    for start in range(0, flat_y.numel(), chunk_size):
        h, y = flat_h[start : start + chunk_size], flat_y[start : start + chunk_size]
        if checkpoint_logits and torch.is_grad_enabled():
            pieces.append(checkpoint(calculate, h, y, use_reentrant=False))
        else:
            pieces.append(calculate(h, y))
    if not pieces:
        return hidden.sum(-1) * 0
    return torch.cat(pieces).reshape_as(targets)
