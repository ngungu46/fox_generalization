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

import copy
from dataclasses import replace
import math

import torch

from fox_experiments.models import FoXLM, ForgetGate, ModelConfig, inverse_softplus


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


def build_comparison_fox(
    variant: str,
    shared_paper_model: FoXLM,
    first_g0: float = 2.944102615,
    other_g0: float = 0.1,
) -> FoXLM:
    """Create one gate variant while preserving a shared paper-FoX backbone.

    ``original_data`` returns an independent, unchanged copy of the source.
    ``factorized_constant`` replaces the first block's gate by
    ``softplus(u_h * v_h)``, initialized with equal positive factors, and
    ``direct_constant`` uses a function-matched ``softplus(b_h)`` gate. Later
    blocks in both constant arms keep the source's data-dependent gate matrices
    but initialize their biases to ``inverse_softplus(other_g0)``. Thus
    ``other_g0`` is their decay at zero input, not at every input.

    Every non-gate parameter is copied bitwise; the input source is untouched.
    The original-data and constant variants intentionally have different
    initial functions. Only the two constant arms are function-matched. Their
    first-gate parameter counts also differ from the original data-dependent
    gate. Positive balanced factors reflect one prescribed initialization
    condition; they do not establish all assumptions of the theory.

    New first-gate parameters use FP32 on the source device, preserving exact
    pairing under the shared model's FP32 gate arithmetic. All other tensors
    retain the source's device and dtype. Model configuration records the new
    gate mode and initial decay settings, and can be used to construct the
    correct parameter shapes before loading this variant's state dict.
    """
    variants = ("original_data", "factorized_constant", "direct_constant")
    if variant not in variants:
        raise ValueError(f"variant must be one of {variants}")
    if not isinstance(shared_paper_model, FoXLM):
        raise TypeError("shared_paper_model must be a FoXLM")
    if shared_paper_model.config.gate_mode != "original_data" or any(
        block.attn.gate.mode != "original_data" for block in shared_paper_model.blocks
    ):
        raise ValueError("The shared paper model must use original_data in every block")
    if variant == "original_data":
        return copy.deepcopy(shared_paper_model)
    if not math.isfinite(first_g0) or first_g0 <= math.log(2.0):
        raise ValueError("Constant comparison arms require finite first_g0 > log(2)")
    if not math.isfinite(other_g0) or other_g0 <= 0:
        raise ValueError("other_g0 must be finite and positive")

    model = copy.deepcopy(shared_paper_model)
    model.config = replace(
        model.config, gate_mode=variant, g0=float(first_g0), other_g0=float(other_g0)
    )
    source_gate = shared_paper_model.blocks[0].attn.gate
    gate = ForgetGate(
        model.config.d_model,
        model.config.n_heads,
        "factorized_constant",
        first_g0,
        model.config.seed + 100003,
        model.config.gate_weight_std,
    ).to(device=source_gate.b.device, dtype=torch.float32)
    if variant == "direct_constant":
        direct_gate = ForgetGate(
            model.config.d_model,
            model.config.n_heads,
            "direct_constant",
            first_g0,
            model.config.seed + 100003,
            model.config.gate_weight_std,
        ).to(device=source_gate.b.device, dtype=torch.float32)
        with torch.no_grad():
            # Match the actual FP32 product, including sqrt roundoff, so the
            # two parameterizations have identical logits at the intervention.
            direct_gate.b.copy_(gate.u * gate.v)
        gate = direct_gate
    gate.train(source_gate.training)
    model.blocks[0].attn.gate = gate
    with torch.no_grad():
        for block in model.blocks[1:]:
            block.attn.gate.b.fill_(inverse_softplus(other_g0))
    return model


__all__ = ["build_paper_fox", "build_comparison_fox"]
