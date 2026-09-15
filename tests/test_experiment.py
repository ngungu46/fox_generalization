"""Scientific plumbing: weighting, optimizer state, replay, and immutable run identity."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import tiktoken
import torch

from fox_experiments.data import Corpus, LatestWriteData
from fox_experiments.evaluation import summarize_results
from fox_experiments.models import FoXLM, ModelConfig
from fox_experiments.training import (
    ExperimentConfig,
    make_optimizer,
    run_experiment,
    train_segment,
)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cache = (
            Path(__file__).resolve().parents[1]
            / "data/longcrawl64_pilot/tokenizer_cache"
        )
        if cache.is_dir():
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
        # Procedural arrays test software, not generalization on natural text.
        arrays = np.tile(np.arange(1000, 5096, dtype=np.uint16), (4, 1))
        cls.data = LatestWriteData(
            Corpus(arrays, arrays, arrays), tiktoken.get_encoding("gpt2")
        )

    def test_json_profiles_normalize_grids_and_reject_typos(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(
                json.dumps(
                    {"profile": "smoke", "train_lags": [24], "acquisition_steps": 0}
                )
            )
            config = ExperimentConfig.from_json(path)
            self.assertEqual(config.train_lags, (24,))
            self.assertEqual(config.d_model, 16)
            self.assertEqual(config.acquisition_steps, 0)
            config.to_json(path)
            self.assertEqual(ExperimentConfig.from_json(path), config)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            ExperimentConfig.from_dict({"adam_lr": 0.01})
        with self.assertRaises(ValueError):
            replace(config, beta2=1.0)

    def test_weighted_loss_and_parameter_groups(self):
        config = ExperimentConfig.for_profile("smoke")
        model = FoXLM(
            ModelConfig(
                d_model=16, n_heads=2, n_layers=2, gate_mode="direct_constant", seed=4
            )
        )
        inputs, targets, weights = self.data.batch(
            config, 0, 0, 0, task_probability=1.0
        )
        logits = model(inputs)
        token_losses = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).reshape_as(weights)
        expected = (token_losses * weights).sum() / weights.sum()
        torch.testing.assert_close(
            model.loss(inputs, targets, loss_mask=weights), expected
        )
        optimizer = make_optimizer(model, config, "adam_annealed", 0.001)
        groups = {group["label"]: group for group in optimizer.param_groups}
        for label in ("first_gate", "retrieval_gate", "output", "representation"):
            self.assertIn(label, groups)
        self.assertEqual(
            groups["first_gate"]["lr"], 0.001 * config.first_gate_multiplier
        )
        self.assertEqual(
            groups["retrieval_gate"]["lr"], 0.001 * config.retrieval_gate_multiplier
        )
        self.assertEqual(groups["output"]["lr"], 0.001 * config.output_multiplier)

    def test_interrupted_adam_resume_matches_uninterrupted_trajectory(self):
        config = replace(ExperimentConfig.for_profile("smoke"), save_every=1)
        initial = FoXLM(
            ModelConfig(
                d_model=16, n_heads=2, n_layers=2, gate_mode="direct_constant", seed=4
            )
        )
        uninterrupted, interrupted = copy.deepcopy(initial), copy.deepcopy(initial)
        data = self.data

        class InterruptedData:
            def batch(self, config, step, microbatch, seed, **kwargs):
                if step == 1:
                    raise RuntimeError("simulated interruption")
                return data.batch(config, step, microbatch, seed, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_history, _ = train_segment(
                uninterrupted,
                data,
                config,
                "adam_annealed",
                0.001,
                3,
                91,
                root / "uninterrupted",
                task_probability=1.0,
                checkpoint_steps=(1, 3),
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_segment(
                    interrupted,
                    InterruptedData(),
                    config,
                    "adam_annealed",
                    0.001,
                    3,
                    91,
                    root / "resumed",
                    task_probability=1.0,
                    checkpoint_steps=(1, 3),
                )
            resumed = copy.deepcopy(initial)
            history, status = train_segment(
                resumed,
                data,
                config,
                "adam_annealed",
                0.001,
                3,
                91,
                root / "resumed",
                task_probability=1.0,
                checkpoint_steps=(1, 3),
            )
            self.assertEqual(status, "complete")
            self.assertEqual(
                [row["loss"] for row in history],
                [row["loss"] for row in expected_history],
            )
            self.assertEqual(
                [row["S"] for row in history], [row["S"] for row in expected_history]
            )
            for name, value in resumed.state_dict().items():
                torch.testing.assert_close(
                    value, uninterrupted.state_dict()[name], rtol=0, atol=0
                )
            self.assertLess(history[-1]["epsilon"], history[0]["epsilon"])
            saved = torch.load(root / "resumed/resume.pt", weights_only=False)
            for state in saved["optimizer"]["state"].values():
                self.assertEqual(float(state["step"]), 3)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                train_segment(
                    resumed,
                    data,
                    config,
                    "adam_annealed",
                    0.002,
                    3,
                    91,
                    root / "resumed",
                    task_probability=1.0,
                    checkpoint_steps=(1, 3),
                )

    def test_run_collision_resume_and_report(self):
        config = replace(
            ExperimentConfig.for_profile("smoke"),
            d_model=8,
            train_length=64,
            pretrain_steps=0,
            acquisition_steps=0,
            trial_steps=1,
            continuation_steps=1,
            train_lags=(24,),
            test_lags=(24,),
            prefix_copies=(0,),
            validation_histories=1,
            lm_eval_length=32,
            gate_modes=("direct_constant",),
            optimizer_arms=("sgd",),
            include_paper_control=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = run_experiment(self.data, config, output)
            summary_before = Path(result["summary"]).read_bytes()
            with self.assertRaisesRegex(FileExistsError, "resume=True"):
                run_experiment(self.data, config, output)
            with self.assertRaisesRegex(ValueError, "differs"):
                run_experiment(
                    self.data, replace(config, beta1=0.2), output, resume=True
                )
            run_experiment(self.data, config, output, resume=True)
            self.assertEqual(Path(result["summary"]).read_bytes(), summary_before)
            panels, ranges, contrasts, figures = summarize_results(output)
            self.assertEqual(len(pd.read_csv(result["summary"])), 1)
            self.assertFalse(panels.empty)
            self.assertEqual(len(ranges), 1)
            self.assertTrue(contrasts.empty)
            self.assertTrue(all(path.exists() for path in figures))
            altered = copy.deepcopy(self.data)
            altered.corpus.arrays["train"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "dataset"):
                run_experiment(altered, config, output, resume=True)


if __name__ == "__main__":
    unittest.main()
