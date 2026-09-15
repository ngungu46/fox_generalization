"""Data integrity, heldout separation, and causal probe geometry checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import tiktoken
import torch

from fox_experiments.data import Corpus, LatestWriteData, PROTOCOL_VERSION, load_corpus


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These procedural rows validate software only. Scientific experiments
        # use the separately downloaded, provenance-checked native corpus.
        arrays = np.tile(np.arange(1000, 5096, dtype=np.uint16), (4, 1))
        cached = (
            Path(__file__).resolve().parents[1]
            / "data/longcrawl64_pilot/tokenizer_cache"
        )
        if cached.is_dir():
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cached))
        cls.data = LatestWriteData(
            Corpus(arrays, arrays, arrays), tiktoken.get_encoding("gpt2")
        )

    def test_exact_independent_axes_and_causal_edits(self):
        for lag in (24, 32, 64, 128):
            base = self.data.example("test", 56, lag, 256, 0)
            prefixed = self.data.example("test", 56, lag, 256, 7)
            for example in (base, prefixed):
                self.assertEqual(
                    len(example["tokens"]) - 1 - example["target_position"], lag
                )
            self.assertEqual(prefixed["tokens"][-256:], base["tokens"])
            self.assertNotEqual(base["key"], base["other"])
            edits = self.data.edits(prefixed)
            for edited in edits:
                self.assertEqual(
                    edited["tokens"][edited["target_position"]], edited["answer"]
                )
                self.assertEqual(len(edited["tokens"]), len(prefixed["tokens"]))
            self.assertNotEqual(edits[1]["answer"], prefixed["answer"])
            self.assertEqual(edits[2]["answer"], prefixed["answer"])
            self.assertEqual(edits[3]["answer"], prefixed["answer"])

    def test_deterministic_data_replay_and_shifted_targets(self):
        config = SimpleNamespace(
            mixture_probability=0.5,
            batch_size=2,
            train_lags=(24, 32),
            train_length=96,
            answer_weight=128.0,
        )
        first = self.data.batch(config, 2, 0, 3, task_probability=1.0)
        replay = self.data.batch(config, 2, 0, 3, task_probability=1.0)
        for actual, expected in zip(first, replay):
            torch.testing.assert_close(actual, expected)
        inputs, targets, weights = first
        torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
        torch.testing.assert_close(
            weights[:, -1], torch.full((2,), config.answer_weight)
        )
        torch.testing.assert_close(weights[:, :-1], torch.ones_like(weights[:, :-1]))

    def test_stale_value_cannot_deterministically_reveal_latest_value(self):
        """A single stale digit must be compatible with multiple correct answers."""
        answers_by_stale = {digit: set() for digit in range(10)}
        stale_by_answer = {digit: set() for digit in range(10)}
        for seed in range(512):
            example = self.data.example("test", seed, 32, 128)
            self.assertEqual(example["protocol_version"], PROTOCOL_VERSION)
            position = example["stale_positions"][-1]
            stale = self.data.answers.index(example["tokens"][position])
            answer = example["answer_digit"]
            self.assertNotEqual(stale, answer)
            answers_by_stale[stale].add(answer)
            stale_by_answer[answer].add(stale)
        # The v1 (+5) shortcut has exactly one entry in each of these sets.
        self.assertTrue(all(len(values) >= 3 for values in answers_by_stale.values()))
        self.assertTrue(all(len(values) >= 3 for values in stale_by_answer.values()))

    def test_sampling_never_crosses_document_and_validation_is_disjoint(self):
        rows = np.stack([np.full(16, value, dtype=np.uint16) for value in range(4)])
        corpus = Corpus(rows, rows, rows)
        self.assertFalse(np.isin(corpus.arrays["tune"], corpus.arrays["confirm"]).any())
        rng = np.random.default_rng(19)
        for _ in range(40):
            tokens, row, start = corpus.sample("train", 13, rng)
            self.assertEqual(tokens, [row] * 13)
            self.assertLessEqual(start + len(tokens), 16)
        with self.assertRaises(ValueError):
            corpus.sample("train", 17, rng)

    def test_payload_checksum_is_checked_before_loading(self):
        rows = np.arange(4 * 16, dtype="<u2").reshape(4, 16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "tokens.u16"
            rows.tofile(payload)
            manifest = {
                "file": payload.name,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "dtype": "<u2",
                "shape": [4, 16],
                "splits": {
                    "train": {"row_start": 0, "row_stop_exclusive": 1},
                    "validation": {"row_start": 1, "row_stop_exclusive": 3},
                    "test": {"row_start": 3, "row_stop_exclusive": 4},
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            files = load_corpus(root)
            np.testing.assert_array_equal(files.validation, rows[1:3])
            self.assertFalse(files.train.flags.writeable)
            del files
            corrupted = bytearray(payload.read_bytes())
            corrupted[0] ^= 1
            payload.write_bytes(corrupted)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                load_corpus(root)

    def test_singleton_has_no_stale_focal_write(self):
        example = self.data.example("test", 56, 32, 128, kind="singleton")
        self.assertEqual(example["stale_positions"], [])
        self.assertEqual(
            [edited["variant"] for edited in self.data.edits(example)],
            ["base", "latest_flip", "irrelevant_flip"],
        )
        with self.assertRaises(ValueError):
            self.data.example("test", 56, 32, 128, 1, kind="singleton")


if __name__ == "__main__":
    unittest.main()
