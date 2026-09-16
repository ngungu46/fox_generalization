"""Tiny FoX models with explicit token-format routing for the objective study.

``record`` routing supplies a parser: the binding block sees nearby KEY token
types, and later blocks retrieve completed VALUE token types. It never receives
the queried key's correct match, answer labels, target positions, or loss mask.
``causal`` routing removes that parser and uses ordinary full causal attention.
The masks are the same for the answer-only and all-token training objectives.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fox_experiments.models import FoXAttention, FoXLM, ForgetGate, ModelConfig


def _membership(tokens: Tensor, ids: tuple[int, ...]) -> Tensor:
    identifiers = torch.as_tensor(ids, device=tokens.device, dtype=tokens.dtype)
    return (tokens[..., None] == identifiers).any(-1)


def format_attention_mask(
    tokens: Tensor,
    layer: int,
    token_format: dict,
    routing: str,
    query_start: int = 0,
    query_end: int | None = None,
) -> Tensor:
    """Boolean [B,Q,T] visibility determined solely by input token types.

    In block zero, ``local3`` means backward distance at most THREE, so a
    VALUE row can inspect KEY tokens at offsets one and three. Other rows see
    all non-padding tokens at distances 0..3. Later blocks see all VALUE tokens
    at or before the query. Empty rows fall back to their own current token.
    """
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError("tokens must be a nonempty [batch, time] tensor")
    if routing not in ("record", "causal"):
        raise ValueError("routing must be record or causal")
    end = tokens.shape[1] if query_end is None else query_end
    if not 0 <= query_start < end <= tokens.shape[1]:
        raise ValueError("Invalid query slice")
    keys = torch.arange(tokens.shape[1], device=tokens.device)
    queries = torch.arange(query_start, end, device=tokens.device)
    distance = queries[:, None] - keys[None, :]
    causal = (distance >= 0)[None, :, :]
    # In causal text mode PAD and BOS may both be GPT-2 EOT. Right-padding
    # tokens are future keys for every valid prediction, so ordinary causal
    # masking preserves BOS/internal EOT without leaking padding into outputs.
    allowed = causal.expand(tokens.shape[0], -1, -1)
    if routing == "record":
        allowed = allowed & tokens.ne(token_format["pad_id"])[:, None, :]
        if layer == 0:
            allowed = allowed & (distance <= 3)[None, :, :]
            query_is_value = _membership(
                tokens[:, query_start:end], token_format["value_ids"]
            )
            key_candidates = _membership(tokens, token_format["key_ids"])
            allowed = allowed & (
                ~query_is_value[..., None] | key_candidates[:, None, :]
            )
        else:
            allowed = (
                allowed & _membership(tokens, token_format["value_ids"])[:, None, :]
            )
    diagonal = keys[None, :] == queries[:, None]
    return allowed | (~allowed.any(-1, keepdim=True) & diagonal[None, :, :])


class StudyAttention(FoXAttention):
    """Shared FoX projections with supplied format routing and named ablations."""

    def __init__(self, original: FoXAttention, index: int, study_config, token_format):
        # Reuse already initialized parameters; changing routing consumes no RNG
        # and does not alter the non-gate initialization of paired model arms.
        nn.Module.__init__(self)
        self.n_heads, self.d_head = original.n_heads, original.d_head
        self.dropout = original.dropout
        self.attention_backend = "sdpa"
        self.qkv, self.out, self.gate = original.qkv, original.out, original.gate
        self.layer_index = index
        self.routing = study_config.routing
        self.token_format = copy.deepcopy(token_format)
        self.qk_normalization = study_config.qk_norm
        self.output_gate = None
        if study_config.output_gate:
            # The zero logits initially multiply each head's output by .5.
            # This is an explicit architecture ablation, not function matching.
            with torch.random.fork_rng(devices=[]):
                self.output_gate = nn.Linear(
                    self.qkv.in_features, self.n_heads, bias=True
                )
            nn.init.zeros_(self.output_gate.weight)
            nn.init.zeros_(self.output_gate.bias)

    def projected(self, x):
        q, k, v = super().projected(x)
        if self.qk_normalization:
            # Keep the ordinary 1/sqrt(d_head) attention scaling; there is no
            # learned compensating gain in this bounded-content ablation.
            q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
            k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)
        return q, k, v

    def forward(self, x, tokens, query_chunk=None):
        q, k, v = self.projected(x)
        length = x.shape[1]
        chunk = length if query_chunk is None else query_chunk
        if chunk < 1:
            raise ValueError("query_chunk must be positive")
        prefix = self.gate(x).float().cumsum(-1)
        output_scale = None
        if self.output_gate is not None:
            output_scale = self.output_gate(x).float().sigmoid().transpose(1, 2)
        pieces = []
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            allowed = format_attention_mask(
                tokens, self.layer_index, self.token_format, self.routing, start, end
            )
            bias = prefix[..., None, :] - prefix[..., start:end, None]
            bias = bias.masked_fill(~allowed[:, None], float("-inf"))
            attended = F.scaled_dot_product_attention(
                q[..., start:end, :],
                k,
                v,
                attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
            if output_scale is not None:
                attended = attended * output_scale[..., start:end, None].to(
                    attended.dtype
                )
            pieces.append(attended)
        output = torch.cat(pieces, dim=2).transpose(1, 2).contiguous().view_as(x)
        return self.out(output)


class LossStudyFoXLM(FoXLM):
    """FoXLM whose forward signature contains only input token IDs.

    ``study_config`` and ``token_format`` are plain serializable metadata. No
    dataset, tokenizer, gold-position array, or objective weights are retained.
    """

    def __init__(self, model_config, study_config, token_format):
        super().__init__(model_config)
        self.study_config = asdict(study_config)
        self.token_format = copy.deepcopy(token_format)
        for index, block in enumerate(self.blocks):
            original = block.attn
            if index > 0 and study_config.retrieval_gate == "constant":
                original.gate = ForgetGate(
                    model_config.d_model,
                    model_config.n_heads,
                    "direct_constant",
                    study_config.retrieval_g0,
                    model_config.seed + 100003 + index,
                )
            block.attn = StudyAttention(original, index, study_config, token_format)

    def format_attention_mask(self, tokens, layer=0, query_start=0, query_end=None):
        return format_attention_mask(
            tokens,
            layer,
            self.token_format,
            self.study_config["routing"],
            query_start,
            query_end,
        )

    @staticmethod
    def _run_block(block, x, tokens, query_chunk):
        x = x + block.dropout(block.attn(block.attn_norm(x), tokens, query_chunk))
        return x + block.dropout(block.ff(block.ff_norm(x)))

    def encode(self, tokens, query_chunk=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must have shape [batch, nonempty time]")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = self._run_block(block, x, tokens, query_chunk)
        return self.norm(x)

    @torch.no_grad()
    def query_diagnostics(
        self,
        tokens,
        target_positions,
        conflict_positions=None,
        layer=None,
        query_chunk=128,
    ):
        """Attention at the last non-padding input token, after token-only routing.

        Target/conflict indices select entries in an already computed attention
        distribution. They never enter the model or its attention mask.
        """
        layer = len(self.blocks) - 1 if layer is None else layer
        if not 0 <= layer < len(self.blocks):
            raise ValueError("layer outside model")
        was_training = self.training
        try:
            self.eval()
            x = self.embedding(tokens)
            for block in self.blocks[:layer]:
                x = self._run_block(block, x, tokens, query_chunk)
            attn = self.blocks[layer].attn
            x = self.blocks[layer].attn_norm(x)
            q, k, _ = attn.projected(x)
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            query = (
                positions[None]
                .expand_as(tokens)
                .masked_fill(tokens.eq(self.token_format["pad_id"]), -1)
                .max(-1)
                .values
            )
            if torch.any(query < 0):
                raise ValueError("Diagnostics need at least one non-padding token")
            q_index = query[:, None, None, None].expand(
                -1, attn.n_heads, 1, attn.d_head
            )
            content = (
                q.gather(2, q_index).float() @ k.float().transpose(-1, -2)
            ).squeeze(-2) / math.sqrt(attn.d_head)
            decay = attn.gate(x)
            prefix = decay.cumsum(-1)
            cost = (
                prefix.gather(-1, query[:, None, None].expand(-1, attn.n_heads, 1))
                - prefix
            )
            allowed = (
                self.format_attention_mask(tokens, layer)
                .gather(1, query[:, None, None].expand(-1, 1, tokens.shape[1]))
                .squeeze(1)
            )
            logit = (content - cost).masked_fill(~allowed[:, None], float("-inf"))
            attention = logit.softmax(-1)

            def indices(value, missing=False):
                result = torch.as_tensor(
                    value, device=tokens.device, dtype=torch.long
                ).reshape(-1)
                if result.numel() == 1:
                    result = result.expand(tokens.shape[0])
                low = -1 if missing else 0
                if result.numel() != tokens.shape[0] or torch.any(
                    (result < low) | (result >= tokens.shape[1])
                ):
                    raise ValueError("Invalid diagnostic positions")
                return result

            target = indices(target_positions)
            conflict = indices(
                -1 if conflict_positions is None else conflict_positions, missing=True
            )
            target_index = target[:, None, None].expand(-1, attn.n_heads, 1)
            conflict_index = conflict.clamp_min(0)[:, None, None].expand_as(
                target_index
            )
            target_content = content.gather(-1, target_index).squeeze(-1)
            other = content.masked_fill(~allowed[:, None], float("-inf")).scatter(
                -1, target_index, float("-inf")
            )
            return {
                "target_attention": attention.gather(-1, target_index).squeeze(-1),
                "target_visible": allowed.gather(-1, target[:, None]).expand(
                    -1, attn.n_heads
                ),
                "integrated_target_forgetting_cost": cost.gather(
                    -1, target_index
                ).squeeze(-1),
                "content_margin_vs_best_other": target_content - other.max(-1).values,
                "content_margin_vs_conflict": torch.where(
                    conflict[:, None] >= 0,
                    target_content - content.gather(-1, conflict_index).squeeze(-1),
                    torch.full_like(target_content, float("nan")),
                ),
                "mean_decay": decay.mean(-1),
                "lag": (query - target)[:, None].expand(-1, attn.n_heads),
            }
        finally:
            self.train(was_training)

    def last_query_diagnostics(
        self,
        tokens,
        target_positions,
        conflict_positions=None,
        layer=0,
        query_chunk=128,
    ):
        return self.query_diagnostics(
            tokens, target_positions, conflict_positions, layer, query_chunk
        )


def build_model(config, data, gate_mode, seed):
    """Build paired models; direct binding gate copies the factorized product."""
    if gate_mode not in ("factorized_constant", "direct_constant", "original_data"):
        raise ValueError("Unsupported first-gate mode")
    token_format = {
        "key_ids": tuple(int(value) for value in data.key_ids),
        "value_ids": tuple(int(value) for value in data.value_ids),
        "query_id": None if data.query_id is None else int(data.query_id),
        "bos_id": int(data.bos_id),
        "pad_id": int(data.pad_id),
    }
    if not token_format["key_ids"] or not token_format["value_ids"]:
        raise ValueError("Nonempty KEY and VALUE token sets are required")
    if set(token_format["key_ids"]) & set(token_format["value_ids"]):
        raise ValueError("KEY and VALUE token sets must be disjoint")
    initial_mode = (
        "original_data" if gate_mode == "original_data" else "factorized_constant"
    )
    model_config = ModelConfig(
        vocab_size=data.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        gate_mode=initial_mode,
        g0=config.first_g0,
        other_g0=config.retrieval_g0,
        seed=seed,
    )
    model = LossStudyFoXLM(model_config, config, token_format)
    if gate_mode == "direct_constant":
        model = model.converted("direct_constant")
    return model


@torch.no_grad()
def model_diagnostics(model, batch=None):
    """Scalar gate/factor and content-scale measurements for a saved checkpoint.

    If a batch is supplied, gate statistics use its input tokens; otherwise data
    gates are measured at zero input. Changes across saved rows show drift.
    These quantities are measurements, not a theoretical retrieval certificate.
    """
    if isinstance(batch, Tensor):
        tokens = batch
    elif isinstance(batch, Mapping):
        tokens = batch.get("input_ids")
    else:
        tokens = getattr(batch, "input_ids", None)
    device = next(model.parameters()).device
    if tokens is not None:
        tokens = tokens.to(device)
    x = model.embedding(tokens) if tokens is not None else None
    result = {}
    for index, block in enumerate(model.blocks):
        attn = block.attn
        inputs = (
            block.attn_norm(x)
            if x is not None
            else torch.zeros(1, 1, model.config.d_model, device=device)
        )
        values = attn.gate(inputs)
        valid = tokens.ne(model.token_format["pad_id"]) if tokens is not None else None
        active = (
            values.masked_select(valid[:, None].expand_as(values))
            if valid is not None
            else values.flatten()
        )
        prefix = f"layer{index}_"
        result[prefix + "gate_mean"] = float(active.mean())
        result[prefix + "gate_min"] = float(active.min())
        result[prefix + "gate_max"] = float(active.max())
        q_weight, k_weight, _ = attn.qkv.weight.chunk(3, dim=0)
        result[prefix + "q_weight_rms"] = float(q_weight.square().mean().sqrt())
        result[prefix + "k_weight_rms"] = float(k_weight.square().mean().sqrt())
        if index == 0 and attn.gate.mode == "factorized_constant":
            result["binding_u_mean"] = float(attn.gate.u.mean())
            result["binding_v_mean"] = float(attn.gate.v.mean())
            result["binding_factor_balance_max"] = float(
                (attn.gate.u - attn.gate.v).abs().max()
            )
        if x is not None:
            q, k, _ = attn.projected(inputs)
            result[prefix + "q_norm_mean"] = float(q.float().norm(dim=-1).mean())
            result[prefix + "k_norm_mean"] = float(k.float().norm(dim=-1).mean())
            if attn.output_gate is not None:
                result[prefix + "output_gate_mean"] = float(
                    attn.output_gate(inputs).sigmoid().mean()
                )
            x = model._run_block(block, x, tokens, model.study_config["query_chunk"])
    result["first_gate_mean"] = result["layer0_gate_mean"]
    result["retrieval_gate_mean"] = result[f"layer{len(model.blocks) - 1}_gate_mean"]
    return result


__all__ = [
    "LossStudyFoXLM",
    "build_model",
    "format_attention_mask",
    "model_diagnostics",
]
