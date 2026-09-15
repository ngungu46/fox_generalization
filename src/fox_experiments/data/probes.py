"""Latest-write retrieval tasks mixed into ordinary token prediction.

All records and queries use plain text and the GPT-2 vocabulary. The model
receives only token IDs. Target/conflict positions below are evaluation
metadata, never extra model inputs.
"""

from __future__ import annotations

import copy
from typing import Protocol

import numpy as np
import torch


class ProbeBatchConfig(Protocol):
    """Training settings used by the batch builder, without a trainer import."""

    mixture_probability: float
    batch_size: int
    train_lags: tuple[int, ...]
    train_length: int
    answer_weight: float


class LatestWriteData:
    """Interleave natural text with plain-text updates at an exact token lag.

    An example asks for the latest value assigned to a named key. Old writes
    precede the latest write; an unrelated update follows it. Two independent
    evaluation axes vary target-to-query lag and the number of prepended old
    writes. This is a semi-synthetic task, not an NLU benchmark.

    Each synthetic fragment is tokenized separately on purpose. No special
    vocabulary, record-position embeddings, or key labels enter the model.
    """

    names = (" Ada", " Ben", " Eve", " Finn", " Hugo", " Jack", " Rose", " Sam")

    def __init__(self, corpus, tokenizer):
        self.corpus = corpus
        self.enc = tokenizer
        answer_tokens = [self.enc.encode(" " + str(digit)) for digit in range(10)]
        if not all(len(tokens) == 1 for tokens in answer_tokens):
            raise ValueError("The probe requires one-token space-prefixed digits")
        self.answers = [tokens[0] for tokens in answer_tokens]

    def record(self, key: int, value: int, style: int = 0) -> list[int]:
        """Tokenize one update, whose last token contains the assigned value."""
        if style == 0:
            prefix = f"\nUpdate:{self.names[key]} code ="
        else:
            prefix = f"\nThe current code for{self.names[key]} is"
        return self.enc.encode(prefix) + [self.answers[value]]

    def query(self, key: int, style: int = 0) -> list[int]:
        """Tokenize the answer prompt; the answer is the next predicted token."""
        suffix = " =" if style == 0 else " is"
        return self.enc.encode(f"\nLatest code for{self.names[key]}" + suffix)

    def example(
        self,
        split: str,
        seed: int,
        lag: int,
        base_length: int,
        prefix_copies: int = 0,
        kind: str = "conflict",
        style: int = 0,
    ) -> dict:
        """Build a reproducible probe with exact target-to-final-query lag.

        ``base_length`` excludes prepended stale copies. Adding copies keeps
        the complete base suffix identical, including the latest target and
        its distance from the query. A singleton contains no stale focal write.
        """
        if kind not in ("conflict", "singleton"):
            raise ValueError("kind must be 'conflict' or 'singleton'")
        if prefix_copies < 0 or lag < 0 or base_length < 1:
            raise ValueError("Require nonnegative lag/copies and positive length")
        if style not in (0, 1):
            raise ValueError("style must be 0 or 1")
        if kind == "singleton" and prefix_copies:
            raise ValueError("Singleton requires no old focal writes")

        rng = np.random.default_rng(seed)
        key = int(rng.integers(len(self.names)))
        other = (key + int(rng.integers(1, len(self.names)))) % len(self.names)
        answer = int(rng.integers(10))
        stale_answer = (answer + 5) % 10
        other_answer = int(rng.integers(10))

        latest_write = self.record(key, answer, style)
        old_write = self.record(key, stale_answer, style)
        distractor = self.record(other, other_answer, style)
        query = self.query(key, style)
        before_length = base_length - lag - len(latest_write)
        after_length = lag - len(query)
        if before_length < len(old_write) or after_length < len(distractor):
            raise ValueError(
                f"Lag {lag}, length {base_length} cannot fit the probe grammar"
            )

        background, doc_id, offset = self.corpus.sample(split, base_length, rng)
        before_target = background[:before_length]
        old_position = None
        if kind == "conflict":
            # The base old write stays close to the latest target. More old
            # writes go at the beginning, leaving this geometry unchanged.
            before_target[-len(old_write) :] = old_write
            old_position = before_length - 1

        after_target = background[before_length : before_length + after_length]
        after_target[-len(distractor) :] = distractor
        extra_prefix = old_write * prefix_copies
        tokens = extra_prefix + before_target + latest_write + after_target + query

        target_position = len(extra_prefix) + before_length + len(latest_write) - 1
        assert len(tokens) - 1 - target_position == lag
        assert len(tokens) == base_length + len(extra_prefix)
        stale_positions = [
            len(old_write) * (index + 1) - 1 for index in range(prefix_copies)
        ]
        if old_position is not None:
            stale_positions.append(len(extra_prefix) + old_position)

        return {
            "tokens": tokens,
            "answer": self.answers[answer],
            "answer_digit": answer,
            "key": key,
            "other": other,
            "lag": lag,
            "prefix_copies": prefix_copies,
            "target_position": target_position,
            "stale_positions": stale_positions,
            "irrelevant_position": len(tokens) - len(query) - 1,
            "doc_id": doc_id,
            "offset": offset,
            "base_length": base_length,
            "kind": kind,
            "seed": seed,
            "style": style,
        }

    def edits(self, example: dict) -> list[dict]:
        """Make causal edits: latest value, stale values, or unrelated value.

        Only changing the latest focal value changes the correct answer. The
        other edits diagnose whether the model depends on irrelevant tokens.
        """
        result = []
        for variant in ("base", "latest_flip", "stale_flip", "irrelevant_flip"):
            edited = copy.deepcopy(example)
            if variant == "latest_flip":
                edited["answer"] = self.answers[(edited["answer_digit"] + 1) % 10]
                edited["tokens"][edited["target_position"]] = edited["answer"]
            elif variant == "stale_flip":
                if not edited["stale_positions"]:
                    continue
                for position in edited["stale_positions"]:
                    current_digit = self.answers.index(edited["tokens"][position])
                    edited["tokens"][position] = self.answers[(current_digit + 1) % 10]
            elif variant == "irrelevant_flip":
                position = edited["irrelevant_position"]
                current_digit = self.answers.index(edited["tokens"][position])
                edited["tokens"][position] = self.answers[(current_digit + 1) % 10]
            edited["variant"] = variant
            result.append(edited)
        return result

    def batch(
        self,
        config: ProbeBatchConfig,
        step: int,
        micro: int,
        seed: int,
        split: str = "train",
        task_probability: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return shifted input/target tokens and per-token loss weights.

        RNG state depends only on run seed, update, and accumulation index.
        Replaying a checkpoint therefore reproduces the original data stream.
        Natural-text examples use unit weights. Probe examples upweight only
        the final answer; all earlier tokens still receive next-token loss.
        """
        rng = np.random.default_rng(seed * 1000003 + step * 131 + micro * 17)
        probability = (
            config.mixture_probability if task_probability is None else task_probability
        )
        if not 0 <= probability <= 1:
            raise ValueError("task_probability must lie in [0, 1]")

        inputs, targets, weights = [], [], []
        for _ in range(config.batch_size):
            token_weights = np.ones(config.train_length, dtype=np.float32)
            if rng.random() < probability:
                lag = int(rng.choice(config.train_lags))
                probe = self.example(
                    split,
                    int(rng.integers(2**31)),
                    lag,
                    config.train_length,
                    kind="conflict" if rng.random() < 0.8 else "singleton",
                    style=int(rng.integers(2)),
                )
                sequence = probe["tokens"] + [probe["answer"]]
                token_weights[-1] = config.answer_weight
            else:
                sequence, _, _ = self.corpus.sample(split, config.train_length + 1, rng)
            inputs.append(sequence[:-1])
            targets.append(sequence[1:])
            weights.append(token_weights)

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(np.asarray(weights)),
        )
