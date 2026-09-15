"""Held-out latest-write and natural-text measurements.

Target lag and the amount of stale prefix are independent test axes. Every edit
panel holds the background and geometry fixed. Query chunking changes evaluation
memory, not the attended context: every query still sees its full causal past.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from fox_experiments.data import LatestWriteData
from fox_experiments.models import FoXLM

if TYPE_CHECKING:
    from fox_experiments.training.config import ExperimentConfig


@torch.no_grad()
def score_examples(
    model: FoXLM,
    examples: list[dict[str, Any]],
    config: ExperimentConfig,
    diagnostics: bool = False,
) -> list[dict[str, Any]]:
    """Score all-vocabulary answers, with optional attention diagnostics."""
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(examples), 2):
        batch = examples[start : start + 2]
        inputs = torch.tensor([example["tokens"] for example in batch], device=device)
        hidden = model.encode(inputs, query_chunk=config.query_chunk)
        logits = model.logits(hidden[:, -1])
        log_probability = logits.log_softmax(-1)
        predictions = logits.argmax(-1)
        for index, example in enumerate(batch):
            answer = example["answer"]
            rows.append(
                {
                    "variant": example.get("variant", "base"),
                    "correct": int(predictions[index] == answer),
                    "prediction": int(predictions[index]),
                    "answer": answer,
                    "probability": float(log_probability[index, answer].exp()),
                    "nll": float(-log_probability[index, answer]),
                    "lag": example["lag"],
                    "prefix_copies": example["prefix_copies"],
                    "length": len(example["tokens"]),
                    "seed": example["seed"],
                    "doc_id": example["doc_id"],
                    "offset": example["offset"],
                    "kind": example["kind"],
                    "style": example["style"],
                }
            )
    if diagnostics and examples:
        _add_attention_diagnostics(model, examples[0], config, rows[0])
    return rows


def _add_attention_diagnostics(
    model: FoXLM,
    example: dict[str, Any],
    config: ExperimentConfig,
    row: dict[str, Any],
) -> None:
    """Inspect candidate binding and retrieval blocks on the base example."""
    device = next(model.parameters()).device
    inputs = torch.tensor([example["tokens"]], device=device)
    target_position = torch.tensor([example["target_position"]], device=device)
    for layer in (0, config.n_layers - 1):
        diagnostics = model.last_query_diagnostics(
            inputs,
            target_position,
            conflict_positions=torch.tensor(
                [example["irrelevant_position"]], device=device
            ),
            layer=layer,
            query_chunk=config.query_chunk,
        )
        for name, value in diagnostics.items():
            if torch.is_tensor(value) and value.numel():
                row[f"layer{layer}_{name}_mean"] = float(value.float().mean())
        if example["stale_positions"]:
            stale_diagnostics = model.last_query_diagnostics(
                inputs,
                target_position,
                conflict_positions=torch.tensor(
                    [example["stale_positions"][-1]], device=device
                ),
                layer=layer,
                query_chunk=config.query_chunk,
            )
            row[f"layer{layer}_content_margin_vs_stale_same_key"] = float(
                stale_diagnostics["content_margin_vs_conflict"].float().mean()
            )


def short_score(
    model: FoXLM,
    data: LatestWriteData,
    config: ExperimentConfig,
    split: str,
    seed: int,
) -> dict[str, Any]:
    """Measure the worst edit/lag accuracy for LR tuning or short-task confirmation."""
    panels = []
    for lag in config.train_lags:
        for history in range(config.validation_histories):
            example = data.example(
                split, 500000 + seed * 1000 + history, int(lag), config.train_length
            )
            panels.extend(score_examples(model, data.edits(example), config))
    variants = ("base", "latest_flip", "stale_flip", "irrelevant_flip")
    worst_accuracy = min(
        np.mean(
            [
                row["correct"]
                for row in panels
                if row["lag"] == lag and row["variant"] == variant
            ]
        )
        for lag in config.train_lags
        for variant in variants
    )
    return {
        "worst_accuracy": float(worst_accuracy),
        "mean_nll": float(np.mean([row["nll"] for row in panels])),
        "mean_accuracy": float(np.mean([row["correct"] for row in panels])),
        "panels": panels,
    }


@torch.no_grad()
def lm_evaluate(
    model: FoXLM,
    data: LatestWriteData,
    config: ExperimentConfig,
    split: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute per-position loss and same-target utility from additional context."""
    model.eval()
    device = next(model.parameters()).device
    rows = []
    length = config.lm_eval_length
    for sample in range(config.lm_eval_rows):
        sequence, document, offset = data.corpus.sample(
            split, length + 1, np.random.default_rng(seed + sample)
        )
        inputs = torch.tensor([sequence[:-1]], device=device)
        targets = torch.tensor([sequence[1:]], device=device)
        hidden = model.encode(inputs, query_chunk=config.query_chunk)
        losses = []
        for start in range(0, length, 64):
            logits = model.logits(hidden[:, start : start + 64])
            token_losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets[:, start : start + 64].reshape(-1),
                reduction="none",
            )
            losses.extend(token_losses.cpu().tolist())
        for position, nll in enumerate(losses, start=1):
            rows.append(
                {
                    "metric": "per_position",
                    "sample": sample,
                    "doc_id": document,
                    "offset": offset,
                    "position": position,
                    "context_length": position,
                    "nll": nll,
                }
            )
        # Every shorter suffix predicts exactly the same last target token.
        for context in sorted({32, 64, config.train_length, length}):
            if context > length:
                continue
            suffix_hidden = model.encode(
                inputs[:, -context:], query_chunk=config.query_chunk
            )
            logits = model.logits(suffix_hidden[:, -1])
            rows.append(
                {
                    "metric": "same_target_context",
                    "sample": sample,
                    "doc_id": document,
                    "offset": offset,
                    "position": length,
                    "context_length": context,
                    "nll": float(F.cross_entropy(logits, targets[:, -1])),
                }
            )
    return rows


def evaluate_branch(
    model: FoXLM,
    data: LatestWriteData,
    config: ExperimentConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate the frozen, predeclared lag × stale-prefix grid on the test split."""
    rows = []
    base_length = max(config.train_length, max(config.test_lags) + 64)
    for lag in config.test_lags:
        for prefix_copies in config.prefix_copies:
            kinds = ("conflict", "singleton") if prefix_copies == 0 else ("conflict",)
            for kind in kinds:
                for history in range(config.test_histories):
                    example = data.example(
                        "test",
                        900000 + seed * 1000 + history,
                        int(lag),
                        base_length,
                        int(prefix_copies),
                        kind=kind,
                    )
                    panel = score_examples(
                        model, data.edits(example), config, diagnostics=history == 0
                    )
                    for row in panel:
                        row["history_id"] = f"{seed}_{history}"
                        row["base_length"] = base_length
                    rows.extend(panel)
    return rows
