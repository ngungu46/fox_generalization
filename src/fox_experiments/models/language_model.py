"""Small causal FoX language model shared by training and evaluation.

There are no positional embeddings, QK normalization, record boundaries, key
identity indicators, or retrieval labels in the model. Probe positions are
used only by the explicitly separate diagnostics method.
"""

from __future__ import annotations
import copy
import math
from dataclasses import replace
from typing import Optional
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from .blocks import FoXBlock, RMSNorm
from .config import GATE_MODES, ModelConfig
from .gates import ForgetGate
from .losses import per_token_nll


class FoXLM(nn.Module):
    """Untied GPT-2-vocabulary LM with a controlled first-layer gate.

    ``converted`` changes the first gate parameterization at a shared
    checkpoint while preserving its function. State-dict names are identical
    for the portable and upstream attention backends.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = copy.deepcopy(config)
        # All nn.Linear/Embedding creation and initialization consume exactly
        # the same RNG stream for every gate arm. Gates use private RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.blocks = nn.ModuleList(
                [FoXBlock(config, i) for i in range(config.n_layers)]
            )
            self.norm = RMSNorm(config.d_model, config.norm_eps)
            self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            for block in self.blocks:
                nn.init.normal_(
                    block.attn.out.weight, std=0.02 / math.sqrt(2 * config.n_layers)
                )
                nn.init.normal_(
                    block.ff.w2.weight, std=0.02 / math.sqrt(2 * config.n_layers)
                )

    def encode(self, tokens: Tensor, query_chunk: Optional[int] = None) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, nonempty time]")
        x = self.embedding(tokens)
        for block in self.blocks:
            if (
                self.config.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                x = checkpoint(block, x, query_chunk=query_chunk, use_reentrant=False)
            else:
                x = block(x, query_chunk=query_chunk)
        return self.norm(x)

    def logits(self, hidden: Tensor) -> Tensor:
        return self.output_head(hidden)

    def forward(self, tokens: Tensor, query_chunk: Optional[int] = None) -> Tensor:
        return self.logits(self.encode(tokens, query_chunk=query_chunk))

    def loss(
        self,
        tokens: Tensor,
        targets: Tensor,
        loss_mask: Optional[Tensor] = None,
        query_chunk: Optional[int] = None,
        logit_chunk: int = 128,
    ) -> Tensor:
        hidden = self.encode(tokens, query_chunk=query_chunk)
        nll = per_token_nll(hidden, targets, self.output_head, chunk_size=logit_chunk)
        weights = (targets != -100).to(nll.dtype)
        if loss_mask is not None:
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must have the same shape as targets")
            if torch.any(loss_mask < 0):
                raise ValueError("loss_mask weights must be nonnegative")
            weights = weights * loss_mask.to(nll.dtype)
        return (nll * weights).sum() / weights.sum().clamp_min(1)

    def gate_parameters(self, layer: Optional[int] = None):
        """All forgetting-gate parameters, or one block's gate parameters."""
        blocks = self.blocks if layer is None else [self.blocks[layer]]
        return [
            parameter for block in blocks for parameter in block.attn.gate.parameters()
        ]

    def nongate_parameters(self):
        ids = {id(parameter) for parameter in self.gate_parameters()}
        return [
            parameter for parameter in self.parameters() if id(parameter) not in ids
        ]

    def converted(self, gate_mode: str) -> "FoXLM":
        """Copy a checkpoint and change only its first gate, preserving function.

        Constant<->data conversions are intentionally rejected: they change the
        hypothesis class and are not the paired intervention in this experiment.
        Direct->factorized uses u=v=sqrt(raw) for each positive raw bias.
        Nonpositive biases use u=1, v=raw, an asymmetric sensitivity arm outside
        the theorem's prescribed symmetric positive initial region. Balancing a
        checkpoint alone does not establish the theorem's remaining conditions.
        """
        old = self.blocks[0].attn.gate
        constant = {"direct_constant", "factorized_constant"}
        data = {"original_data", "factorized_data"}
        if gate_mode not in GATE_MODES:
            raise ValueError(f"Unknown gate mode {gate_mode}")
        if not ({old.mode, gate_mode} <= constant or {old.mode, gate_mode} <= data):
            raise ValueError(
                "Conversions must remain within constant or data gate families"
            )
        result = copy.deepcopy(self)
        result.config = replace(self.config, gate_mode=gate_mode)
        if old.mode == gate_mode:
            return result
        reference = next(old.parameters())
        new = ForgetGate(
            self.config.d_model,
            self.config.n_heads,
            gate_mode,
            self.config.g0,
            self.config.seed + 100003,
            self.config.gate_weight_std,
        )
        new = new.to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            if old.mode == "direct_constant":
                positive = old.b > 0
                balanced = old.b.clamp_min(0).sqrt()
                new.u.copy_(torch.where(positive, balanced, torch.ones_like(old.b)))
                new.v.copy_(torch.where(positive, balanced, old.b))
            elif old.mode == "factorized_constant":
                new.b.copy_(old.u * old.v)
            elif old.mode == "original_data":
                new.u.fill_(1)
                new.weight.copy_(old.weight)
                new.v.copy_(old.b)
            else:
                new.weight.copy_(old.weight * old.u[:, None])
                new.b.copy_(old.v * old.u)
        new.train(old.training)
        result.blocks[0].attn.gate = new
        return result

    @torch.no_grad()
    def last_query_diagnostics(
        self,
        tokens: Tensor,
        target_positions: Tensor,
        conflict_positions: Optional[Tensor] = None,
        layer: int = 0,
        query_chunk: Optional[int] = 128,
    ):
        """Inspect final-query attention at one block; labels never enter forward.

        Metrics have shape [batch, heads]. The content margin is a QK-only score
        difference, and integrated_target_forgetting_cost is the positive decay
        sum between the supplied target token and final query. Missing conflicts
        use index -1 and return NaN. Best-other includes every other visible token
        (including the query itself), without using record or key metadata.
        """
        if not 0 <= layer < len(self.blocks):
            raise ValueError("layer outside model")
        was_training = self.training
        try:
            self.eval()
            x = self.embedding(tokens)
            for block in self.blocks[:layer]:
                x = block(x, query_chunk=query_chunk)
            block = self.blocks[layer]
            normalized = block.attn_norm(x)
            q, k, _ = block.attn.projected(normalized)
            decay = block.attn.gate(normalized)
            with torch.autocast(device_type=x.device.type, enabled=False):
                content = (
                    q[:, :, -1:, :].float() @ k.float().transpose(-1, -2)
                ).squeeze(-2) / math.sqrt(block.attn.d_head)
                prefix = decay.cumsum(-1)
                cost = prefix[..., -1:] - prefix
                attention = torch.softmax(content - cost, dim=-1)
            batch, heads, length = content.shape
            positions = torch.as_tensor(
                target_positions, device=tokens.device, dtype=torch.long
            ).reshape(-1)
            if positions.numel() == 1:
                positions = positions.expand(batch)
            if positions.numel() != batch or torch.any(
                (positions < 0) | (positions >= length)
            ):
                raise ValueError(
                    "target_positions must provide a valid index for each sequence"
                )
            index = positions[:, None, None].expand(-1, heads, 1)
            target_content = content.gather(-1, index).squeeze(-1)
            other = content.clone().scatter(-1, index, float("-inf"))
            metrics = {
                "target_attention": attention.gather(-1, index).squeeze(-1),
                "integrated_target_forgetting_cost": cost.gather(-1, index).squeeze(-1),
                "target_content_score": target_content,
                "content_margin_vs_best_other": target_content - other.max(-1).values,
                "mean_decay": decay.mean(-1),
                "query_decay": decay[..., -1],
                "lag": (length - 1 - positions)[:, None].expand(-1, heads),
            }
            if conflict_positions is None:
                conflicts = torch.full(
                    (batch,), -1, device=tokens.device, dtype=torch.long
                )
            else:
                conflicts = torch.as_tensor(
                    conflict_positions, device=tokens.device, dtype=torch.long
                ).reshape(-1)
                if conflicts.numel() == 1:
                    conflicts = conflicts.expand(batch)
                if conflicts.numel() != batch or torch.any(
                    (conflicts < -1) | (conflicts >= length)
                ):
                    raise ValueError(
                        "conflict_positions must be -1 or a valid index per sequence"
                    )
            conflict_index = conflicts.clamp_min(0)[:, None, None].expand(-1, heads, 1)
            conflict_content = content.gather(-1, conflict_index).squeeze(-1)
            nan = torch.full_like(target_content, float("nan"))
            metrics["content_margin_vs_conflict"] = torch.where(
                conflicts[:, None] >= 0, target_content - conflict_content, nan
            )
            metrics["conflict_attention"] = torch.where(
                conflicts[:, None] >= 0,
                attention.gather(-1, conflict_index).squeeze(-1),
                nan,
            )
            metrics["target_minus_conflict_logit"] = torch.where(
                conflicts[:, None] >= 0,
                (content - cost).gather(-1, index).squeeze(-1)
                - (content - cost).gather(-1, conflict_index).squeeze(-1),
                nan,
            )
            return metrics
        finally:
            self.train(was_training)
