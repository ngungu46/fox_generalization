"""Independent stream, derivative, optimizer, initialization, and dataset QA."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

try:
    from .core import (Config, Model, LogAdam, acquisition_rates, boundary_log_odds,
                       dense_objective, dense_stream, export_dataset, gradient_step,
                       native_gradients, native_loss, pair_cases, rates_at, slog)
except ImportError:
    from core import (Config, Model, LogAdam, acquisition_rates, boundary_log_odds,
                      dense_objective, dense_stream, export_dataset, gradient_step,
                      native_gradients, native_loss, pair_cases, rates_at, slog)


def moderate_model(R=2, gate_mode="learned", w=.8):
    c = Config(n=3, d=5, R=R, gate_mode=gate_mode, D=.3, c0=1.2)
    model = Model(c)
    gen = torch.Generator().manual_seed(812)
    model.Q = torch.randn(c.n, c.d, generator=gen, dtype=torch.float64) * .3
    model.K = torch.randn(c.n, c.d, generator=gen, dtype=torch.float64) * .3
    model.theta = torch.tensor([.8, .9, .65, .8, .4, w], dtype=torch.float64)
    return model


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_stream_loss_and_gradients(self):
        for R in (2, 3, 4, 7):
            for gate_mode in ("learned", "retrieval_frozen", "both_frozen"):
                for w in (-.6, 0., .8):
                    for sampled in (False, True):
                        with self.subTest(R=R, gate=gate_mode, w=w, sampled=sampled):
                            model = moderate_model(R, gate_mode, w)
                            counts = torch.tensor([0., 3., 1., 2., 0., 4.]) if sampled else None
                            params = [p.clone().requires_grad_(True) for p in model.params()]
                            loss_dense = dense_objective(model, counts, params)
                            expected = torch.autograd.grad(loss_dense, params)
                            native, info = native_gradients(model, counts)
                            signed, slog_info = model.gradients(counts)
                            torch.testing.assert_close(info["loss"], loss_dense, rtol=2e-12, atol=2e-14)
                            torch.testing.assert_close(slog_info["logloss"].exp(), loss_dense, rtol=2e-12, atol=2e-14)
                            for target, actual, (sign, logabs) in zip(expected, native, signed):
                                self.assertFalse(bool(torch.isnan(logabs).any()))
                                torch.testing.assert_close(actual, target, rtol=2e-10, atol=2e-13)
                                torch.testing.assert_close(sign * logabs.exp(), target, rtol=2e-10, atol=2e-13)

    def test_boundary_odds_match_independent_attention(self):
        for gate_mode in ("learned", "retrieval_frozen", "both_frozen"):
            model = moderate_model(4, gate_mode)
            q, p, u, v, x, w, m, g, h, rho, lrho = model.quantities()
            for lag in (1, 2, 3, 6, 20):
                for prefix in (0, 1, 9):
                    for a, b in ((0, 1), (1, 2)):
                        keys = [a] * (prefix + 1) + [b] * (lag - 1)
                        values = [-1] * prefix + [1] + [-1] * (lag - 1)
                        loss, mass, logit = dense_stream(model.params(), model.cfg, keys, values, a)
                        lo = boundary_log_odds(model.gaps()[a, b], m, h, rho, lag, prefix)
                        torch.testing.assert_close(mass[prefix], torch.sigmoid(-lo), rtol=2e-12, atol=2e-14)
                        torch.testing.assert_close(logit, -w * torch.tanh(lo / 2), rtol=2e-12, atol=2e-14)

    def test_trusted_counts_path_matches_checked_path(self):
        model = moderate_model(4)
        counts = torch.tensor([0., 3., 1., 2., 0., 4.], dtype=torch.float64)
        checked, checked_info = model.gradients(counts)
        trusted, trusted_info = model.gradients(counts, validate_counts=False)
        for expected, actual in zip(checked, trusted):
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        torch.testing.assert_close(checked_info["table_log_span"], trusted_info["table_log_span"])

    def test_log_adam_matches_torch_adam_complete_history(self):
        for b1, b2 in ((.9, .999), (0., .9)):
            left = [torch.tensor([.2, -.7, .4], dtype=torch.float64)]
            right = [left[0].clone().requires_grad_(True)]
            log_adam = LogAdam(left, b1, b2)
            reference = torch.optim.Adam(right, lr=.01, betas=(b1, b2), eps=1e-8)
            for t in range(30):
                grad = torch.tensor([math.sin(t), math.cos(t) * .001, 0.], dtype=torch.float64)
                right[0].grad = grad.clone()
                reference.step()
                log_adam.step(left, [slog(grad)], [.01], math.log(1e-8))
                torch.testing.assert_close(left[0], right[0], rtol=2e-12, atol=3e-14)
            self.assertEqual(log_adam.t, 30)

    def test_annealed_adam_matches_torch(self):
        left = [torch.tensor([.1, .3], dtype=torch.float64)]
        right = [left[0].clone().requires_grad_(True)]
        optimizer = LogAdam(left, .9, .999)
        ref = torch.optim.Adam(right, lr=.02, betas=(.9, .999), eps=.001)
        for t in range(20):
            eps = .001 * math.exp(-.3 * t)
            grad = torch.tensor([(-1.) ** t * .01, math.exp(-t)], dtype=torch.float64)
            ref.param_groups[0]["eps"] = eps
            right[0].grad = grad
            ref.step()
            optimizer.step(left, [slog(grad)], [.02], math.log(eps))
        torch.testing.assert_close(left[0], right[0], rtol=2e-12, atol=3e-14)

    def test_frozen_coordinates_and_buffers(self):
        for gate_mode in ("retrieval_frozen", "both_frozen"):
            for kind in ("sgd", "adam"):
                model = moderate_model(4, gate_mode)
                before = model.theta.clone()
                optimizer = LogAdam(model.params(), .9, .999)
                frozen = list(model.frozen_scalar_indices)
                for _ in range(5):
                    gradients, _ = model.gradients()
                    self.assertTrue(bool((gradients[-1][0][frozen] == 0).all()))
                    rates = [.001, .001, .01]
                    if kind == "adam":
                        optimizer.step(model.params(), gradients, rates, math.log(1e-8))
                    else:
                        gradient_step(model, gradients, rates)
                torch.testing.assert_close(before[frozen], model.theta[frozen], rtol=0, atol=0)
                if kind == "adam":
                    self.assertTrue(bool(torch.isneginf(optimizer.vl[-1][frozen]).all()))
                    self.assertTrue(bool((optimizer.ms[-1][frozen] == 0).all()))
                _, rates = rates_at(model.cfg, kind, 12)
                self.assertTrue(bool((rates[-1][frozen] == 0).all()))

    def test_initialization_and_R_adjusted_acquisition(self):
        for R in (2, 4, 7):
            c = Config(n=4, d=256, R=R)
            model = Model(c)
            self.assertFalse(torch.equal(model.Q, model.K))
            self.assertEqual(float(model.theta[-1]), 0.)
            self.assertAlmostEqual(float(model.quantities()[8]), c.h0)
            self.assertEqual(float(model.theta[0]), float(model.theta[1]))
            self.assertEqual(float(model.theta[2]), float(model.theta[3]))
            gradients, _ = model.gradients()
            self.assertTrue(bool((gradients[0][0] == 0).all()))
            self.assertTrue(bool((gradients[1][0] == 0).all()))
            self.assertTrue(bool((gradients[-1][0][:-1] == 0).all()))
            rates = acquisition_rates(c, "sgd")
            probe = Model(c)
            probe.Q.zero_(); probe.K.zero_(); probe.theta[-1] = .2
            z = torch.zeros((), dtype=torch.float64, requires_grad=True)
            _, _, _, so, sc = probe.rows(z)
            phi = c.wo * torch.nn.functional.softplus(-.2 * so) + c.wc * torch.nn.functional.softplus(-.2 * sc)
            astar = -float(torch.autograd.grad(phi, z)[0]) / (c.n - 1)
            self.assertAlmostEqual(rates[2][0], 1 / astar)
            for rate in rates:
                grads, _ = model.gradients()
                gradient_step(model, grads, rate)
            self.assertGreater(float(model.gaps()[model.mask].min()), 0.)
            for actual, (sign, logabs) in zip(native_gradients(model)[0], model.gradients()[0]):
                torch.testing.assert_close(actual, sign * logabs.exp(), rtol=2e-9, atol=1e-13)

    def test_dataset_pair_count_and_weighted_loss(self):
        model = moderate_model(5)
        cases = pair_cases(model.cfg)
        self.assertEqual(len(cases), model.cfg.n * (model.cfg.n - 1))
        for case in cases:
            self.assertNotEqual(case["query_key"], case["distractor_key"])
            self.assertEqual({row["target_lag"] for row in case["recall"]}, {5})
            self.assertEqual({row["target_lag"] for row in case["overwrite"]}, {1})
        with tempfile.TemporaryDirectory() as path:
            manifest = export_dataset(model.cfg, path)
            expanded = [json.loads(line) for line in (Path(path) / "dataset.jsonl").read_text().splitlines()]
            recall = (Path(path) / "recall_at_R.jsonl").read_text().splitlines()
            self.assertEqual(len(recall), len(cases))
            self.assertEqual(len(expanded), 4 * len(cases) + 2 * model.cfg.n)
            self.assertAlmostEqual(sum(row["population_weight"] for row in expanded), 1.)
            loss = sum(row["population_weight"] * dense_stream(model.params(), model.cfg, row["keys"],
                                                               row["values"], row["query"], row["answer"])[0]
                       for row in expanded)
            torch.testing.assert_close(loss, native_loss(model), rtol=2e-12, atol=2e-14)

    def test_log_backend_retains_tiny_binder_gradients(self):
        model = moderate_model(4)
        model.theta[2:4] = 30.
        log_gradients, _ = model.gradients()
        native, _ = native_gradients(model)
        self.assertEqual(float(native[-1][2]), 0.)
        self.assertTrue(bool(torch.isfinite(log_gradients[-1][1][2])))
        self.assertLess(float(log_gradients[-1][1][2]), -1000.)
        self.assertNotEqual(float(log_gradients[-1][0][2]), 0.)

    def test_frozen_stale_tail_threshold(self):
        for h in (.5, math.log(2), 1.):
            model = Model(Config(n=3, d=5, gate_mode="retrieval_frozen", h0=h))
            q, p, u, v, x, w, m, g, h, rho, lrho = model.quantities()
            lo = boundary_log_odds(model.gaps()[model.mask], m, h, rho, 1, math.inf)
            torch.testing.assert_close(torch.sigmoid(-lo), torch.full_like(lo, 1 - math.exp(-float(h))))
            if float(h) < math.log(2):
                self.assertTrue(bool((lo > 0).all()))

    def test_config_and_counts_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Config(beta1=.99, beta2=.9)
        with self.assertRaises(ValueError):
            Config(R=1)
        with self.assertRaises(ValueError):
            Config(h0=1., x0=-1.)
        with self.assertRaises(ValueError):
            Config(kcal=.4, wo=.3, wc=.3)
        model = moderate_model()
        for counts in (torch.zeros(6), torch.ones(3), -torch.ones(6)):
            with self.assertRaises(ValueError):
                model.gradients(counts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
