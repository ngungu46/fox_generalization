"""Downscaled architecture port of upstream's default FoX (LLaMA).

The audited upstream files are ``configs/model/forgetting_transformer.py`` and
``src/forgetting_transformer/model/forgetting_transformer/{configuration,
modeling}_forgetting_transformer.py`` in the official repository:
https://github.com/zhixuan-lin/forgetting-transformer

This factory reproduces their default architecture choices and initialization
distributions at the requested smaller dimensions. It does not reproduce the
upstream seeded parameter values, complete training pipeline, or all numerical
kernels. The shared local model combines Q/K/V projections, uses separate
SwiGLU input projections, computes forget-gate projections in FP32, and has no
KV cache. SDPA, normalization, SwiGLU, and cross entropy are portable PyTorch
implementations. The optional ``upstream`` attention backend uses the existing
fused-kernel adapter; it does not make the other components bitwise identical.

Upstream parameterizes retention as ``logsigmoid(W_retention x + b_retention)``.
The local model uses positive decay ``softplus(W_decay x + b_decay)``, then
negates that decay in the attention bias. The two are function-equivalent
under ``(W_decay, b_decay) = -(W_retention, b_retention)``. Zero biases and
zero-mean Gaussian weights therefore give the same initialization distribution.

No FoX-Pro additions, RoPE, QK normalization, token shifts, output gates,
output normalization, local windows, or pruning are enabled here.
"""

from __future__ import annotations

import math

import torch

from fox_experiments.models import FoXLM, ModelConfig


def build_paper_fox(
    d_model: int = 128,
    n_layers: int = 2,
    n_heads: int = 2,
    vocab_size: int = 50257,
    seed: int = 0,
    attention_backend: str = "sdpa",
    gradient_checkpointing: bool = False,
) -> FoXLM:
    """Build an ordinary data-dependent FoX baseline at a chosen small size.

    Defaults match the upstream FoX (LLaMA) architectural configuration: affine
    pre-RMSNorm with epsilon 1e-6; SwiGLU hidden ratio 4, with intermediate size
    rounded up to a multiple of 256; untied embeddings/output; bias-free QKV,
    attention output, and MLP projections; and trainable data-dependent forget
    gates in every block. Every gate bias is exactly zero, giving decay log(2)
    on zero input, while its weights have standard deviation 0.02.

    All projection and embedding weights have the upstream N(0, 0.02^2)
    initialization distribution, including residual output projections. The
    factory deliberately leaves ``FoXLM`` and its existing callers unchanged.
    ``model.config`` contains the selected architecture and runtime settings;
    recreating this initialization requires this factory, not bare ``FoXLM``.
    Loading a saved state dict works with either once shapes match.
    """
    # Follow upstream's integer truncation before rounding, rather than
    # approximating the advertised hidden_ratio with a different FF multiple.
    intermediate = int(d_model * 4 * 2 / 3)
    intermediate = 256 * ((intermediate + 255) // 256)
    config = ModelConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=vocab_size,
        gate_mode="original_data",
        g0=math.log(2.0),
        other_g0=math.log(2.0),
        seed=seed,
        ff_hidden=intermediate,
        norm_eps=1e-6,
        gate_weight_std=0.02,
        dropout=0.0,
        attention_backend=attention_backend,
        gradient_checkpointing=gradient_checkpointing,
    )
    model = FoXLM(config)
    with torch.no_grad():
        for block in model.blocks:
            # Avoid any inverse-softplus roundoff in the upstream zero bias.
            block.attn.gate.b.zero_()
            # Shared FoXLM initializes these two projections with an additional
            # 1/sqrt(2L). Undo that experimental choice only in this baseline.
            block.attn.out.weight.mul_(math.sqrt(2 * n_layers))
            block.ff.w2.weight.mul_(math.sqrt(2 * n_layers))
    return model


__all__ = ["build_paper_fox"]
