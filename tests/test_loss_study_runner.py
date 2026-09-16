"""Objective identities, truthful qualification, deterministic pairing and resume."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from fox_experiments.loss_study import (
    LossStudyConfig,
    run_loss_study,
    summarize_loss_study,
)
from fox_experiments.loss_study.config import branches
from fox_experiments.loss_study.data import StudyData
from fox_experiments.loss_study.model import build_model
from fox_experiments.loss_study.runner import (
    compute_losses,
    gradient_diagnostics,
    make_optimizer,
    schedule_optimizer,
    summarize_evaluations,
)


class LossStudyRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny(self, **overrides):
        cfg = replace(
            LossStudyConfig.for_profile("smoke"),
            gate_modes=("factorized_constant",),
            optimizers=("adam_fixed",),
            objectives=("answer_only",),
            eval_histories=1,
            eval_prefixes=(0,),
            short_lags=(1,),
            eval_lags=(1, 4),
            train_max_prefix=0,
            steps=2,
            eval_every=1,
            diagnostic_every=1,
        )
        return replace(cfg, **overrides)

    def test_full_vocab_loss_weighting_and_answer_only_gradient_mask(self):
        cfg = self.tiny(answer_weight=7)
        logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 0.0, 1.0]]],
            requires_grad=True,
        )
        batch = {
            "targets": torch.tensor([[0, 1, 2, -100]]),
            "answer_mask": torch.tensor([[False, False, True, False]]),
        }
        losses = compute_losses(logits, batch, cfg)
        ce = F.cross_entropy(logits[0, :3], batch["targets"][0, :3], reduction="none")
        torch.testing.assert_close(losses["answer_only"], ce[2])
        torch.testing.assert_close(
            losses["all_tokens"], (ce[0] + ce[1] + 7 * ce[2]) / 9
        )
        torch.testing.assert_close(losses["answer_coefficient"], torch.tensor(7 / 9))
        torch.testing.assert_close(losses["answer_fraction"], torch.tensor(1 / 3))
        losses["answer_only"].backward(retain_graph=True)
        self.assertEqual(float(logits.grad[0, :2].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 2].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0)
        logits.grad = None
        losses["all_tokens"].backward()
        self.assertGreater(float(logits.grad[0, :2].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0)
        vanilla = compute_losses(logits, batch, replace(cfg, answer_weight=1))
        torch.testing.assert_close(vanilla["all_tokens"], ce.mean())

    def test_gradient_decomposition_and_summable_tail(self):
        cfg = self.tiny(representation_schedule="summable", answer_weight=3)
        data = StudyData(cfg)
        model = build_model(cfg, data, "factorized_constant", 0)
        batch = data.batch(0, 0)
        losses = compute_losses(model(batch["input_ids"]), batch, cfg)
        diagnostic = {
            r["parameter_group"]: r
            for r in gradient_diagnostics(model, losses, "all_tokens")
        }
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        losses["all_tokens"].backward()
        actual = (
            torch.stack(
                [
                    p.grad.double().square().sum()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
            .sum()
            .sqrt()
        )
        self.assertAlmostEqual(
            diagnostic["all"]["combined_gradient_norm"], float(actual), places=6
        )
        self.assertGreater(diagnostic["all"]["weighted_other_gradient_norm"], 0)
        opt = make_optimizer(model, cfg, "adam_annealed", 0.1)
        schedule_optimizer(opt, cfg, "adam_annealed", 0.1, 1000)
        reps = [g for g in opt.param_groups if g["category"] == "representation"]
        self.assertAlmostEqual(
            reps[0]["lr"],
            0.1 * cfg.representation_multiplier * 2**-cfg.representation_power,
        )
        self.assertAlmostEqual(opt.param_groups[0]["eps"], cfg.eps0 * np.exp(-10))

    def test_short_failure_means_missing_radius_and_ceiling_is_censored(self):
        cfg = self.tiny()
        spec = branches(cfg)[0]
        rows = []
        for split, lags in (("short", cfg.short_lags), ("long", cfg.eval_lags)):
            for lag in lags:
                for edit in ("base", "latest"):
                    rows.append(
                        {
                            **spec,
                            "step": 0,
                            "split": split,
                            "lag": lag,
                            "prefix": 0,
                            "history_id": 0,
                            "edit": edit,
                            "correct": int(not (split == "short" and edit == "latest")),
                            "probability": 0.99,
                            "nll": 0.01,
                        }
                    )
        failed = summarize_evaluations(rows, cfg)[0]
        self.assertFalse(failed["short_qualified"])
        self.assertTrue(np.isnan(failed["tested_grid_radius"]))
        self.assertEqual(failed["short_base_accuracy"], 1)
        for r in rows:
            r["correct"] = 1
        passed = summarize_evaluations(rows, cfg)[0]
        self.assertTrue(passed["right_censored"])
        self.assertEqual(passed["tested_grid_radius"], 4)
        self.assertEqual(passed["radius_status"], "right_censored_at_grid_ceiling")
        for r in rows:
            if r["split"] == "long" and r["lag"] == 4:
                r["correct"] = 0
        bounded = summarize_evaluations(rows, cfg)[0]
        self.assertEqual(bounded["tested_grid_radius"], 1)
        self.assertEqual(bounded["next_failed_tested_lag"], 4)
        self.assertFalse(bounded["right_censored"])

    def test_paired_inputs_initial_functions_and_resume_without_duplicates(self):
        cfg = self.tiny(
            gate_modes=("factorized_constant", "direct_constant"),
            objectives=("answer_only", "all_tokens"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "paired"
            result = run_loss_study(cfg, out)
            self.assertEqual(result["status"], "complete", result["failures"])
            training = pd.read_csv(out / "training.csv")
            self.assertEqual(len(training), 8)
            self.assertTrue(
                (training.groupby("step").batch_sha256.nunique() == 1).all()
            )
            branches_csv = pd.read_csv(out / "branches.csv")
            self.assertEqual(branches_csv.shared_non_gate_initial_hash.nunique(), 1)
            self.assertEqual(
                branches_csv.groupby("gate_mode").initial_state_hash.nunique().max(), 1
            )
            raw_before = pd.read_csv(out / "eval.csv")
            run_loss_study(cfg, out, resume=True)
            pd.testing.assert_frame_equal(raw_before, pd.read_csv(out / "eval.csv"))
            with self.assertRaises(FileExistsError):
                run_loss_study(cfg, out)
            with self.assertRaisesRegex(ValueError, "specification changed"):
                run_loss_study(replace(cfg, answer_weight=2), out, resume=True)
            report = summarize_loss_study(out)
            self.assertEqual(len(report["figures"]), 4)
            self.assertTrue(report["software_only"])

    def test_interrupted_checkpoint_replays_exactly(self):
        cfg = self.tiny(steps=3)
        original = StudyData.batch

        def interrupt(data, step, seed, device="cpu"):
            if step == 1:
                raise KeyboardInterrupt()
            return original(data, step, seed, device)

        with tempfile.TemporaryDirectory() as temporary:
            clean, stopped = Path(temporary) / "clean", Path(temporary) / "stopped"
            run_loss_study(cfg, clean)
            with patch.object(StudyData, "batch", interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_loss_study(cfg, stopped)
            self.assertEqual(
                json.loads((stopped / "status.json").read_text())["status"],
                "interrupted",
            )
            self.assertEqual(
                run_loss_study(cfg, stopped, resume=True)["status"], "complete"
            )
            branch = branches(cfg)[0]["branch"]
            a = torch.load(clean / branch / "last.pt", weights_only=False)
            b = torch.load(stopped / branch / "last.pt", weights_only=False)
            self.assertEqual(a["step"], b["step"])
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], atol=0, rtol=0
                )
            pd.testing.assert_frame_equal(
                pd.DataFrame(a["evaluation"]),
                pd.DataFrame(b["evaluation"]),
                check_exact=True,
            )
            self.assertEqual(
                [r["loss"] for r in a["training"]], [r["loss"] for r in b["training"]]
            )

    def test_failed_evaluation_resumes_before_next_update(self):
        from fox_experiments.loss_study import runner

        cfg = self.tiny(steps=3)
        original = runner.evaluate

        def fail_once(model, data, config, branch, step, split="short", device="cpu"):
            if step == 1 and split == "long":
                raise RuntimeError("deliberate evaluation failure")
            return original(model, data, config, branch, step, split, device)

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with patch.object(runner, "evaluate", fail_once):
                result = run_loss_study(cfg, out)
            self.assertEqual(result["status"], "failed")
            self.assertIn(
                "deliberate evaluation failure", result["failures"][0]["failure"]
            )
            self.assertTrue(Path(result["failures"][0]["traceback_file"]).exists())
            self.assertEqual(
                run_loss_study(cfg, out, resume=True)["status"], "complete"
            )
            raw = pd.read_csv(out / "eval.csv")
            self.assertEqual(set(raw.step), {0, 1, 2, 3})
            self.assertTrue((raw.groupby("step").split.nunique() == 2).all())
            self.assertFalse(
                raw.duplicated(
                    ["branch", "step", "split", "lag", "prefix", "history_id", "edit"]
                ).any()
            )


if __name__ == "__main__":
    unittest.main()
