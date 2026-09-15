"""Paper comparison: paired input, schedule fidelity, targets and exact resume."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from fox_experiments.data import Corpus
from fox_experiments.paper_baseline import (
    PaperBaselineConfig,
    run_paper_comparison,
    summarize_paper_comparison,
)
from fox_experiments.paper_baseline.runner import natural_batch, schedule_optimizer
from fox_experiments.training.optimizers import make_optimizer


class PaperBaselineTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # These token arrays validate software, not language-model performance.
        rows = np.tile(np.arange(1000, 1256, dtype=np.uint16), (4, 1))
        cls.corpus = Corpus(rows, rows + 1, rows + 2)

    def test_config_and_eot_shift(self):
        cfg = PaperBaselineConfig.from_dict({"profile": "smoke"})
        self.assertEqual(len(cfg.arms), 4)
        self.assertEqual(
            PaperBaselineConfig().tokens_per_update * PaperBaselineConfig().steps,
            4096000,
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            PaperBaselineConfig.from_dict({"paper_learning_rate": 0.1})
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            PaperBaselineConfig.for_profile("full")
        x, y = natural_batch(self.corpus, cfg, 7, 0)
        self.assertTrue(torch.all(x[:, 0] == 50256))
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        replay = natural_batch(self.corpus, cfg, 7, 0)
        torch.testing.assert_close(x, replay[0])
        torch.testing.assert_close(y, replay[1])

    def test_paper_schedule_is_warmup_then_cosine(self):
        from fox_experiments.paper_baseline.model import build_paper_fox

        cfg = replace(
            PaperBaselineConfig.for_profile("smoke"), steps=10, paper_warmup_steps=2
        )
        model = build_paper_fox(16, 2, 2, seed=0)
        optimizer = make_optimizer(model, cfg, "paper_adamw", 0.002)
        rates = [
            schedule_optimizer(optimizer, cfg, "paper_adamw", step)[0]
            for step in (0, 1, 2, 6, 10)
        ]
        np.testing.assert_allclose(
            rates, (0, 0.001, 0.002, 0.001, 0), rtol=1e-12, atol=1e-15
        )
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertEqual(
            {group["weight_decay"] for group in optimizer.param_groups}, {0, 0.1}
        )
        own = make_optimizer(model, cfg, "adam_annealed", 0.001)
        initial_eps = schedule_optimizer(own, cfg, "adam_annealed", 0)[1]
        final_eps = schedule_optimizer(own, cfg, "adam_annealed", 9)[1]
        self.assertLess(final_eps, initial_eps)
        self.assertEqual(own.defaults["betas"], (0.1, 0.1))

    def test_four_arms_and_matched_targets_and_resume(self):
        cfg = PaperBaselineConfig.for_profile("smoke")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_paper_comparison(cfg, ".", output, corpus=self.corpus)
            summary = pd.read_csv(output / "summary.csv")
            self.assertEqual(set(summary.arm), set(cfg.arms))
            self.assertEqual(summary.initial_state_hash.nunique(), 1)
            self.assertTrue(summary.software_only.all())
            specification = json.loads((output / "run_specification.json").read_text())
            self.assertEqual(
                specification["runtime_versions"]["torch"], str(torch.__version__)
            )
            raw = pd.read_csv(output / "lm_raw.csv")
            self.assertIn("shared_initial", set(raw.arm))
            suffix = raw[raw.metric == "same_target_suffix"]
            self.assertEqual(suffix.target_token_sha256.nunique(), 1)
            self.assertEqual(suffix.target_start_offset.nunique(), 1)
            at_train = suffix[suffix.context_length == cfg.train_length]
            np.testing.assert_allclose(at_train.nll_gain_vs_train_context, 0)
            state_before = torch.load(
                output / "adam_fixed/last.pt", weights_only=False
            )["model"]
            with self.assertRaises(FileExistsError):
                run_paper_comparison(cfg, ".", output, corpus=self.corpus)
            with self.assertRaisesRegex(ValueError, "specification changed"):
                run_paper_comparison(
                    replace(cfg, epsilon_decay=0.02),
                    ".",
                    output,
                    resume=True,
                    corpus=self.corpus,
                )
            run_paper_comparison(cfg, ".", output, resume=True, corpus=self.corpus)
            state_after = torch.load(output / "adam_fixed/last.pt", weights_only=False)[
                "model"
            ]
            for name in state_before:
                torch.testing.assert_close(
                    state_before[name], state_after[name], rtol=0, atol=0
                )
            report = summarize_paper_comparison(output)
            self.assertTrue(Path(report["context_summary"]).exists())
            self.assertTrue(all(Path(path).exists() for path in report["figures"]))

    def test_interrupted_adam_matches_uninterrupted(self):
        cfg = replace(
            PaperBaselineConfig.for_profile("smoke"), arms=("adam_annealed",), steps=3
        )
        with tempfile.TemporaryDirectory() as temporary:
            reference, resumed = (
                Path(temporary) / "reference",
                Path(temporary) / "resumed",
            )
            run_paper_comparison(cfg, ".", reference, corpus=self.corpus)
            original = natural_batch

            def interrupted(corpus, config, step, microbatch):
                if step == 1:
                    raise RuntimeError("simulated interruption")
                return original(corpus, config, step, microbatch)

            with patch(
                "fox_experiments.paper_baseline.runner.natural_batch", interrupted
            ):
                run_paper_comparison(cfg, ".", resumed, corpus=self.corpus)
            status = json.loads((resumed / "run_status.json").read_text())
            self.assertEqual(status["status"], "complete_with_failures")
            self.assertEqual(status["failed_arms"], 1)
            run_paper_comparison(cfg, ".", resumed, resume=True, corpus=self.corpus)
            left = torch.load(reference / "adam_annealed/last.pt", weights_only=False)
            right = torch.load(resumed / "adam_annealed/last.pt", weights_only=False)
            for name in left["model"]:
                torch.testing.assert_close(
                    left["model"][name], right["model"][name], rtol=0, atol=0
                )
            self.assertEqual(
                [row["loss"] for row in left["history"]],
                [row["loss"] for row in right["history"]],
            )
            self.assertEqual(json.loads((resumed / "failures.json").read_text()), [])
            saved_spec = json.loads((resumed / "run_specification.json").read_text())
            saved_spec["device_type"] = "cuda"
            (resumed / "run_specification.json").write_text(json.dumps(saved_spec))
            with self.assertRaisesRegex(ValueError, "specification changed"):
                run_paper_comparison(cfg, ".", resumed, resume=True, corpus=self.corpus)


if __name__ == "__main__":
    unittest.main()
