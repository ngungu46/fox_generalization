"""Independent finite-stream and numerical checks for E_t(r) envelopes."""
import itertools
import math
import unittest

import torch
from torch.nn import functional as F

try:
    from .asymptotics import (certified_radius, error_bounds, evaluate_asymptotics,
                             evaluate_snapshot, scalar_error_bounds, scalar_snapshot, snapshot)
    from .core import Config, Model, dense_stream
except ImportError:
    from asymptotics import (certified_radius, error_bounds, evaluate_asymptotics,
                            evaluate_snapshot, scalar_error_bounds, scalar_snapshot, snapshot)
    from core import Config, Model, dense_stream


def known_model(gate_mode="learned", m=20., h=1.2, w=4.):
    model = Model(Config(n=2, d=2, R=2, gate_mode=gate_mode, h0=h))
    model.Q = torch.tensor([[.5, .1], [0., .6]], dtype=torch.float64)
    model.K = torch.tensor([[.5, 0.], [.05, .5]], dtype=torch.float64)
    model.theta[:2] = math.sqrt(m)
    model.theta[2:4] = 2.
    model.theta[-1] = w
    return model


class AsymptoticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_witness_limit_matches_literal_long_streams(self):
        model = known_model()
        state = snapshot(model)
        for lag in (1, 2, 3, 8):
            with self.subTest(lag=lag):
                finite_errors = []
                for a, b in ((0, 1), (1, 0)):
                    prefix = 60  # omitted geometric tail < exp(-72)
                    keys = [a] * (prefix + 1) + [b] * (lag - 1)
                    values = [-1] * prefix + [1] + [-1] * (lag - 1)
                    logit = dense_stream(model.params(), model.cfg, keys, values, a)[2]
                    finite_errors.append(float(F.logsigmoid(-logit)))
                self.assertAlmostEqual(error_bounds(state, lag)["log_lower_error"], max(finite_errors), places=12)

    def test_upper_bounds_all_short_streams(self):
        # Enumerate arbitrary old keys/values and both labels, not just the
        # witness family, to independently audit the uniform transfer formula.
        model = known_model(m=6., h=1.1)
        state = snapshot(model)
        upper = {r: error_bounds(state, r)["upper_error"] for r in range(1, 6)}
        alphabet = [(key, value) for key in range(2) for value in (-1., 1.)]
        checked = 0
        for length in range(1, 6):
            for records in itertools.product(alphabet, repeat=length):
                keys, values = zip(*records)
                for query in set(keys):
                    latest = max(index for index, key in enumerate(keys) if key == query)
                    lag = length - latest
                    logit = dense_stream(model.params(), model.cfg, keys, values, query, values[latest])[2]
                    self.assertLessEqual(float(torch.sigmoid(-logit)), upper[lag] + 3e-14)
                    checked += 1
        self.assertGreater(checked, 2000)

    def test_bounds_monotone_and_scalar_history_matches(self):
        model = known_model()
        state = snapshot(model)
        previous_lower = previous_upper = -1.
        for lag in range(1, 80):
            row = error_bounds(state, lag)
            scalar = scalar_error_bounds(state.delta_min, float(state.m), float(state.h),
                                         float(state.rho), float(state.w), lag, state.max_row_norm)
            self.assertAlmostEqual(row["log_lower_error"], scalar["log_lower_error"], places=14)
            self.assertAlmostEqual(row["log_upper_error"], scalar["log_upper_error"], places=14)
            self.assertLessEqual(previous_lower, row["lower_error"])
            self.assertLessEqual(previous_upper, row["upper_error"])
            self.assertLessEqual(row["lower_error"], row["upper_error"] + 1e-14)
            previous_lower, previous_upper = row["lower_error"], row["upper_error"]

    def test_log_errors_retain_extreme_confidence(self):
        model = known_model(m=1e5, h=20., w=1e4)
        model.theta[2:4] = 10.
        row = error_bounds(snapshot(model), 1)
        self.assertEqual(row["lower_error"], 0.)  # legitimate float64 underflow
        self.assertEqual(row["upper_error"], 0.)
        self.assertTrue(math.isfinite(row["log_lower_error"]))
        self.assertLess(row["log_lower_error"], -9000.)
        self.assertLessEqual(row["log_lower_error"], row["log_upper_error"])

    def test_huge_lags_do_not_allocate_sequences_or_overflow_int64(self):
        state = snapshot(known_model())
        for lag in (10 ** 12, 10 ** 30, 10 ** 300):
            row = error_bounds(state, lag)
            self.assertEqual(row["lag"], lag)
            self.assertAlmostEqual(row["log_lower_error"], float(F.logsigmoid(state.w)), places=14)
            self.assertAlmostEqual(row["log_upper_error"], float(F.logsigmoid(state.w)), places=14)

    def test_certified_radius_is_maximal_for_the_bound(self):
        model = known_model(m=200., h=3., w=20.)
        model.theta[2:4] = 4.
        state = snapshot(model)
        for target in (.1, .01, .001):
            row = certified_radius(state, target)
            radius = row["certified_radius"]
            self.assertGreater(radius, 1)
            self.assertLessEqual(error_bounds(state, radius)["upper_error"], target)
            self.assertGreater(error_bounds(state, radius + 1)["upper_error"], target)
        model.theta[-1] = 0.
        self.assertEqual(certified_radius(snapshot(model), .1)["certified_radius"], 0)

    def test_invalid_conditions_never_emit_a_zero_upper_bound(self):
        for condition in ("norm", "gap", "decoder", "scale"):
            model = known_model()
            if condition == "norm":
                model.Q *= 10
            elif condition == "gap":
                model.K = model.K.flip(0)
            elif condition == "decoder":
                model.theta[-1] = -2
            else:
                model.theta[0] *= -1
            state = snapshot(model)
            row = error_bounds(state, 2)
            self.assertFalse(row["bound_valid"])
            self.assertIsNone(row["upper_error"])
            self.assertIsNone(row["log_upper_error"])
            self.assertIsNone(certified_radius(state, .1)["certified_radius"])
            self.assertTrue(math.isfinite(row["log_lower_error"]))

    def test_predeclared_clocks_and_reference_gap(self):
        for gate in ("learned", "retrieval_frozen"):
            model = known_model(gate, m=10000.)
            state = snapshot(model)
            rows, radii = evaluate_asymptotics(model, {"seed": 8}, 900, 1e6,
                                               reference_gap=state.delta_min / 2,
                                               fixed_lags=(1,), theta_values=(.5,),
                                               clock_coefficients=(.01,), error_targets=(.1,))
            by_probe = {row["probe"]: row for row in rows}
            power = 1 if gate == "learned" else 2
            self.assertEqual(by_probe["clock"]["lag"], math.floor(1 + .01 * (1e6) ** power))
            self.assertGreater(by_probe["clock"]["lag"], model.cfg.eval_max_lag)
            self.assertTrue(by_probe["reference_theta"]["reference_gap_retained"])
            self.assertLess(by_probe["reference_theta"]["lag"], by_probe["adaptive_theta"]["lag"])
            self.assertEqual({row["lag"] for row in rows if row["probe"] == "fixed"}, {1, 2, 3, 4, 5})
            self.assertEqual(radii[0]["seed"], 8)

    def test_saved_scalar_history_preserves_every_probe(self):
        model = known_model()
        state = snapshot(model)
        recovered = scalar_snapshot(state.delta_min, float(state.m), float(state.h),
                                    float(state.rho), float(state.w), state.max_row_norm)
        expected = evaluate_asymptotics(model, {"seed": 0}, 100, 200.)
        actual = evaluate_snapshot(recovered, model.cfg.R, model.cfg.gate_mode, {"seed": 0}, 100, 200.)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
