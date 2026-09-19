"""Literal-stream derivatives and sampler invariants for random training."""
import copy
from dataclasses import replace
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "library"))

from fox_restricted.legacy.core import Config, Model, LogAdam, gradient_step, rates_at
from fox_restricted.data.random_streams import RandomBatch, RandomStreamConfig, batch_from_streams, sample_stream_batch
from fox_restricted.training.random_backend import audit_random_gradients, native_random_loss, random_gradients


def small_model(gate_mode="learned"):
    cfg = Config(n=5, d=7, R=3, sigma=.2, gate_mode=gate_mode, beta1=.1, beta2=.1)
    model = Model(cfg)
    model.theta[:] = torch.tensor([1.4, .9, .3, -.4, .4, 1.2], dtype=torch.float64)
    return model


@pytest.mark.parametrize("distribution", ["geometric", "bounded_uniform"])
def test_sampler_latest_query_and_mixed_key_invariants(distribution):
    settings = RandomStreamConfig(batch_size=1200, prefix_distribution=distribution, prefix_mean=4, max_prefix=8)
    batch = sample_stream_batch(5, 4, settings, rng=np.random.default_rng(82)).validate(5, 4)
    assert set(batch.lags.tolist()) == {1, 2, 3, 4}
    assert set(batch.labels.tolist()) == {-1., 1.}
    assert set(batch.queries.tolist()) == set(range(5))
    assert any(len(torch.unique(batch.keys[index, :batch.lengths[index]])) > 2 for index in range(batch.size))
    prefixes = batch.lengths - batch.lags
    assert bool((prefixes == 0).any())
    if distribution == "bounded_uniform":
        assert int(prefixes.max()) <= settings.max_prefix
    else:
        # max_prefix is not an implicit truncation of the full-support sampler.
        assert int(prefixes.max()) > settings.max_prefix


def test_sampler_resume_and_dataset_tensor_roundtrip():
    rng = np.random.default_rng(15)
    settings = RandomStreamConfig(batch_size=19, prefix_mean=5)
    sample_stream_batch(4, 3, settings, rng=rng)
    state = copy.deepcopy(rng.bit_generator.state)
    expected = sample_stream_batch(4, 3, settings, rng=rng)
    rng.bit_generator.state = state
    actual = sample_stream_batch(4, 3, settings, rng=rng)
    restored = RandomBatch.from_state_dict(actual.state_dict())
    for name in expected.__dataclass_fields__:
        assert torch.equal(getattr(expected, name), getattr(actual, name))
        assert torch.equal(getattr(expected, name), getattr(restored, name))


@pytest.mark.parametrize("gate_mode", ["learned", "retrieval_frozen", "both_frozen"])
@pytest.mark.parametrize("counts", [None, [0., 3., 1., 2., 1., 0.]])
def test_general_stream_signed_log_derivatives_match_literal_attention(gate_mode, counts):
    model = small_model(gate_mode)
    batch = batch_from_streams([
        {"keys": [0], "values": [1], "query": 0},
        {"keys": [0, 1, 2, 0, 3, 4], "values": [-1, 1, -1, 1, 1, -1], "query": 0},
        {"keys": [1, 1, 2, 3, 1], "values": [1, -1, 1, -1, -1], "query": 1},
        {"keys": [4, 1, 2, 3], "values": [1, 1, 1, 1], "query": 1},
        {"keys": [2, 2, 2], "values": [1, -1, 1], "query": 2},
        {"keys": [3, 0, 2, 0], "values": [-1, -1, 1, 1], "query": 3},
    ]).validate(model.cfg.n)
    audit = audit_random_gradients(model, batch, counts, tolerance=2e-12)
    assert audit["passed"], audit
    gradients, info = random_gradients(model, batch, counts)
    assert math.exp(float(info["logloss"])) == pytest.approx(float(native_random_loss(model, batch, counts, literal=True)), rel=2e-13)
    assert float(native_random_loss(model, batch, counts)) == pytest.approx(math.exp(float(info["logloss"])), rel=2e-13)
    for index in model.frozen_scalar_indices:
        assert gradients[-1][0][index] == 0
        assert torch.isneginf(gradients[-1][1][index])


def test_weighted_dataset_matches_duplicated_minibatch():
    model = small_model()
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=8, prefix_mean=3), rng=np.random.default_rng(1))
    counts = torch.tensor([0., 2., 1., 0., 3., 0., 0., 1.], dtype=torch.float64)
    selection = [1, 1, 2, 4, 4, 4, 7]
    weighted, info = random_gradients(model, batch, counts)
    duplicated, other = random_gradients(model, batch.select(selection))
    for (s1, l1), (s2, l2) in zip(weighted, duplicated):
        torch.testing.assert_close(s1 * l1.exp(), s2 * l2.exp(), rtol=2e-12, atol=1e-16)
    torch.testing.assert_close(info["logloss"], other["logloss"], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("w", [0., -1.7, 2.3])
def test_signed_log_handles_zero_and_negative_decoder(w):
    model = small_model()
    model.theta[-1] = w
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=13, prefix_mean=2), rng=np.random.default_rng(84))
    audit = audit_random_gradients(model, batch, tolerance=3e-12)
    assert audit["passed"], audit


def test_underflowed_probability_keeps_finite_signed_log_derivatives():
    model = Model(Config(n=2, d=2, R=2))
    model.Q[:] = torch.eye(2, dtype=torch.float64)
    model.K[:] = torch.eye(2, dtype=torch.float64)
    model.theta[:] = torch.tensor([30., 30., 4., 4., 1., 1200.], dtype=torch.float64)
    batch = batch_from_streams([{"keys": [0, 1], "values": [1, -1], "query": 0}])
    gradients, info = random_gradients(model, batch)
    assert float(info["logloss"]) < -1000
    assert math.exp(float(info["logloss"])) == 0.
    assert gradients[-1][0][0] != 0
    assert torch.isfinite(gradients[-1][1][0])
    assert float(gradients[-1][1][0]) < -1000


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_optimizer_steps_agree_with_native_torch(optimizer):
    model = small_model()
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=11, prefix_mean=3), rng=np.random.default_rng(20))
    params = [p.detach().clone().requires_grad_() for p in model.params()]
    rate = .002
    if optimizer == "adam":
        native = torch.optim.Adam(params, lr=rate, betas=(.1, .1), eps=1e-18)
        log_adam = LogAdam(model.params(), .1, .1)
    else:
        native = torch.optim.SGD(params, lr=rate)
    for _ in range(5):
        native.zero_grad()
        loss = native_random_loss(model, batch, params=params, literal=True)
        loss.backward()
        native.step()
        gradients = random_gradients(model, batch)[0]
        if optimizer == "adam":
            log_adam.step(model.params(), gradients, [rate] * 3, math.log(1e-18))
        else:
            gradient_step(model, gradients, [rate] * 3)
    for expected, actual in zip(params, model.params()):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)


def test_invalid_data_and_memory_guard_fail_explicitly():
    settings = RandomStreamConfig(batch_size=10, prefix_mean=100, max_batch_records=10)
    with pytest.raises(MemoryError, match="not truncated or resampled"):
        sample_stream_batch(4, 3, settings, rng=np.random.default_rng(1))
    model = small_model()
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=3), rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="counts"):
        random_gradients(model, batch, counts=[0, 0, 0])
    malformed = copy.deepcopy(batch)
    malformed.labels[:] = 0
    with pytest.raises(ValueError, match="Values and labels"):
        random_gradients(model, malformed)


def test_long_irrelevant_prefix_does_not_trigger_table_underflow_guard():
    from fox_restricted.legacy.a100_runner import _check_health
    model = Model(Config(n=2, d=2, R=2))
    model.Q[:] = torch.eye(2, dtype=torch.float64)
    model.K[:] = torch.eye(2, dtype=torch.float64)
    model.theta[:] = torch.tensor([2., 2., 2., 2., 1., 3.], dtype=torch.float64)
    batch = batch_from_streams([
        {"keys": [0, 1] * 1000 + [0, 1], "values": [-1] * 2000 + [1, -1], "query": 0}
    ])
    gradients, info = random_gradients(model, batch)
    assert float(info["record_log_span"]) > 2500
    assert float(info["table_log_span"]) < 10
    assert _check_health(model, gradients, info, model.theta.clone()) is None
    audit = audit_random_gradients(model, batch, tolerance=2e-11)
    assert audit["passed"], audit


def test_independent_rare_query_derivatives_survive_2000_log_unit_difference():
    from fox_restricted.legacy.a100_runner import _check_health
    model = Model(Config(n=4, d=4, R=2))
    model.Q[:] = torch.eye(4, dtype=torch.float64)
    model.K[:] = torch.diag(torch.tensor([.01, 1., 2., 1.], dtype=torch.float64))
    model.theta[:] = torch.tensor([math.sqrt(1000), math.sqrt(1000), 2., 2., 1., 3.], dtype=torch.float64)
    batch = batch_from_streams([
        {"keys": [0, 1], "values": [1, -1], "query": 0},
        {"keys": [2, 3], "values": [1, -1], "query": 2},
    ])
    gradients, info = random_gradients(model, batch)
    assert float(info["record_log_span"]) > 1900
    assert float(info["table_log_span"]) < 10
    assert _check_health(model, gradients, info, model.theta.clone()) is None
    # Independent two-record closed form. The second stream has target score
    # gap 2m(1-rho)-h; its ordinary derivative is too small for float64 exp.
    _, _, _, _, _, w, m, _, h, rho, _ = model.quantities()
    odds = -2 * m * (1 - rho) + h
    alignment = -torch.tanh(odds / 2)
    log_pressure = (torch.log(2 * w) + torch.nn.functional.logsigmoid(-w * alignment)
                    + odds - 2 * torch.nn.functional.softplus(odds) - math.log(2))
    expected_q_log = log_pressure + (2 * m * (1 - rho)).log()
    expected_k_log = log_pressure + (m * (1 - rho)).log()
    assert float(expected_q_log) < -1900
    assert gradients[0][0][2, 2] == -1
    assert gradients[1][0][2, 2] == -1
    assert float(gradients[0][1][2, 2]) == pytest.approx(float(expected_q_log), abs=1e-10)
    assert float(gradients[1][1][2, 2]) == pytest.approx(float(expected_k_log), abs=1e-10)


def test_guard_still_rejects_unsafe_range_inside_a_final_table_product():
    from fox_restricted.legacy.a100_runner import _check_health
    model = Model(Config(n=3, d=3, R=2))
    model.Q[:] = torch.eye(3, dtype=torch.float64)
    model.K[:] = torch.tensor([[1., 0., 0.], [.99, 1., 0.], [-1., 0., 1.]], dtype=torch.float64)
    model.theta[:] = torch.tensor([math.sqrt(1000), math.sqrt(1000), 2., 2., 1., 3.], dtype=torch.float64)
    batch = batch_from_streams([
        {"keys": [0, 1], "values": [1, -1], "query": 0},
        {"keys": [0, 2], "values": [1, -1], "query": 0},
    ])
    gradients, info = random_gradients(model, batch)
    assert float(info["table_log_span"]) > 1900
    assert "650 log units" in _check_health(model, gradients, info, model.theta.clone())


def test_grouped_gradients_respect_deterministic_algorithms_mode():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = small_model()
        batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=13), rng=np.random.default_rng(52))
        left = random_gradients(model, batch)[0]
        right = random_gradients(model, batch)[0]
        for (ls, ll), (rs, rl) in zip(left, right):
            assert torch.equal(ls, rs)
            assert torch.equal(ll, rl)
    finally:
        torch.use_deterministic_algorithms(previous)
