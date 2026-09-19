"""Portable deterministic checkpoint/resume and hardware-benchmark QA."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

try:
    from . import a100_runner as large
    from .core import Config
except ImportError:
    import a100_runner as large
    from core import Config


def options():
    return dict(seeds=(0,), eval_lags=(1, 2, 4, 8), prefixes=(0, 8),
                log_every=4, checkpoint_every=4, check_every=2, progress=False,
                fixed_lags=(1, 2, 4), theta_values=(.5,), clock_coefficients=(.01,), error_targets=(.1, .01))


def cfg(steps=12):
    return Config(n=4, d=32, R=2, steps=steps, device="cpu", pair_batch=16,
                  beta1=.1, beta2=.1, max_seconds=1000.)


def checkpoint(directory, mode, optimizer):
    return torch.load(Path(directory) / "checkpoints" / f"seed0_{mode}_{optimizer}.pt", map_location="cpu", weights_only=False)


def assert_same_state(test, left, right):
    for a, b in zip(left["model"], right["model"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in ("step", "S", "examples", "acquired", "reference_gap"):
        test.assertEqual(left["state"][key], right["state"][key])
    test.assertEqual(left["rng_state"], right["rng_state"])
    test.assertEqual(left["initial_hash"], right["initial_hash"])
    if left["optimizer"] is not None:
        test.assertEqual(left["optimizer"]["t"], right["optimizer"]["t"])
        for name in ("ms", "ml", "vl"):
            for a, b in zip(left["optimizer"][name], right["optimizer"][name]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


class LargeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_extended_budget_exactly_matches_uninterrupted_all_four_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = large.run_large_suite(cfg(12), root / "full", **options())
            extended = large.run_large_suite(cfg(7), root / "extended", **options())
            large.run_large_suite(cfg(12), extended, **options())
            for mode in ("learned", "retrieval_frozen"):
                for optimizer in ("sgd", "adam"):
                    left, right = checkpoint(full, mode, optimizer), checkpoint(extended, mode, optimizer)
                    assert_same_state(self, left, right)
                    self.assertEqual(right["state"]["status"], "finite_budget_complete")
                    self.assertEqual(right["state"]["resume_count"], 1)
                    self.assertTrue(all(audit["passed"] for audit in right["audits"]))
                    self.assertEqual(len(right["audits"]), 2)
                    self.assertTrue(right["errors"])
                    self.assertTrue(right["radii"])
            before = (Path(extended) / "checkpoints" / "seed0_learned_adam.pt").read_bytes()
            large.run_large_suite(cfg(12), extended, **options())
            self.assertEqual(before, (Path(extended) / "checkpoints" / "seed0_learned_adam.pt").read_bytes())
            self.assertTrue((Path(extended) / "asymptotic_errors.csv").exists())
            self.assertTrue((Path(extended) / "certified_radii.csv").exists())

    def test_operational_pause_preserves_rng_and_adam_moments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setting = {**options(), "gate_modes": ("retrieval_frozen",), "optimizers": ("adam",)}
            full = large.run_large_suite(cfg(), root / "full", **setting)
            resumed = large.run_large_suite(cfg(), root / "paused", max_updates=5, **setting)
            saved = checkpoint(resumed, "retrieval_frozen", "adam")
            self.assertEqual(saved["state"]["status"], "paused")
            self.assertEqual(saved["state"]["step"], 5)
            self.assertEqual(saved["optimizer"]["t"], 5)
            large.run_large_suite(cfg(), resumed, **setting)
            assert_same_state(self, checkpoint(full, "retrieval_frozen", "adam"), checkpoint(resumed, "retrieval_frozen", "adam"))

    def test_pause_during_acquisition_preserves_whole_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setting = {**options(), "gate_modes": ("learned",), "optimizers": ("adam",)}
            full = large.run_large_suite(cfg(), root / "full", **setting)
            resumed = large.run_large_suite(cfg(), root / "paused", max_updates=1, **setting)
            large.run_large_suite(cfg(), resumed, **setting)
            assert_same_state(self, checkpoint(full, "learned", "adam"), checkpoint(resumed, "learned", "adam"))

    def test_scientific_config_source_and_budget_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            setting = {**options(), "gate_modes": ("learned",), "optimizers": ("sgd",)}
            out = large.run_large_suite(cfg(7), Path(temporary) / "run", **setting)
            with self.assertRaisesRegex(ValueError, "Scientific configuration"):
                large.run_large_suite(replace(cfg(12), lr_sgd=4.), out, **setting)
            with self.assertRaisesRegex(ValueError, "shrinking"):
                large.run_large_suite(cfg(6), out, **setting)
            changed = large._source_hashes()
            changed["core.py"] = "tampered"
            with patch.object(large, "_source_hashes", return_value=changed):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    large.run_large_suite(cfg(12), out, **setting)
            with self.assertRaises(FileExistsError):
                large.run_large_suite(cfg(12), out, resume=False, **setting)

    def test_numerical_failure_rolls_back_all_state_to_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setting = {**options(), "gate_modes": ("learned",), "optimizers": ("adam",)}
            expected = large.run_large_suite(cfg(4), root / "expected", **setting)
            real_step = large._optimizer_step

            def corrupt(model, optimizer, gradients, rates, step):
                real_step(model, optimizer, gradients, rates, step)
                if step == 5:
                    model.theta[0] = float("nan")

            with patch.object(large, "_optimizer_step", side_effect=corrupt):
                out = large.run_large_suite(cfg(12), root / "corrupted", **setting)
            recovered = checkpoint(out, "learned", "adam")
            self.assertEqual(recovered["state"]["status"], "numerical_stop")
            self.assertEqual(recovered["state"]["failed_at_step"], 6)
            assert_same_state(self, checkpoint(expected, "learned", "adam"), recovered)
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in recovered["model"]))
            self.assertTrue(all(row["step"] <= 4 for row in recovered["history"]))

    def test_benchmark_is_fresh_and_reports_measured_local_runtime(self):
        result = large.benchmark_config(cfg(), steps=2, warmup=1, num_seeds=2)
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(len(result["rows"]), 4)
        self.assertGreater(result["estimated_matrix_seconds"], 0.)
        for row in result["rows"]:
            self.assertEqual(row["measured_steps"], 2)
            self.assertGreater(row["seconds_per_step"], 0.)
            self.assertTrue(row["acquired_positive_gaps"])
        json.dumps(result, allow_nan=False)

    def test_compiled_gradient_audit_has_no_silent_fallback(self):
        model = large.Model(cfg())
        model.theta[-1] = .2
        backend = large._GradientBackend(model, compiled=True)
        with patch.object(torch, "compile", side_effect=RuntimeError("test compile failure")):
            with self.assertRaises(RuntimeError):
                backend()


if __name__ == "__main__":
    unittest.main(verbosity=2)
