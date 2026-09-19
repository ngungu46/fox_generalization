"""Packing preserves the exact random objective, including extreme log scales."""
import math

import numpy as np
import pytest
import torch

from fox_restricted.data.random_streams import RandomStreamConfig, batch_from_streams, sample_stream_batch
from fox_restricted.legacy.core import Config, Model, LogAdam, gradient_step
from fox_restricted.training.packed_random import pack_random_batch, packed_random_gradients
from fox_restricted.training.random_backend import native_random_loss, random_gradients


def _model(gate="learned"):
    model = Model(Config(n=5, d=7, R=3, sigma=.2, gate_mode=gate))
    model.theta[:] = torch.tensor([1.4, .9, .3, -.4, .4, 1.2], dtype=torch.float64)
    return model


def _assert_padded_parity(model, batch, counts=None):
    expected, expected_info = random_gradients(model, batch, counts)
    packed = pack_random_batch(batch, model.cfg.n)
    actual, actual_info = packed_random_gradients(model, packed, counts)
    for (es, el), (s, l) in zip(expected, actual):
        torch.testing.assert_close(s, es, atol=0, rtol=0)
        torch.testing.assert_close(l, el, atol=3e-10, rtol=3e-12)
    for key in expected_info:
        torch.testing.assert_close(actual_info[key], expected_info[key], atol=3e-10, rtol=3e-12)
    return actual, actual_info


@pytest.mark.parametrize("gate", ["learned", "retrieval_frozen", "both_frozen"])
@pytest.mark.parametrize("weighted", [False, True])
def test_packed_matches_padded_and_literal_stream_autodiff(gate, weighted):
    model = _model(gate)
    batch = batch_from_streams([
        {"keys": [0], "values": [1], "query": 0},
        {"keys": [0, 1, 2, 0, 3, 4], "values": [-1, 1, -1, 1, 1, -1], "query": 0},
        {"keys": [1, 1, 2, 3, 1], "values": [1, -1, 1, -1, -1], "query": 1},
        {"keys": [4, 1, 2, 3], "values": [1, 1, 1, 1], "query": 1},
        {"keys": [2, 2, 2], "values": [1, -1, 1], "query": 2},
        {"keys": [3, 0, 2, 0], "values": [-1, -1, 1, 1], "query": 3},
    ])
    counts = [0., 3., 1., 2., 1., 0.] if weighted else None
    actual, info = _assert_padded_parity(model, batch, counts)
    params = [p.detach().clone().requires_grad_(True) for p in model.params()]
    loss = native_random_loss(model, batch, counts, params, literal=True)
    expected = torch.autograd.grad(loss, params)
    for target, (sign, logabs) in zip(expected, actual):
        torch.testing.assert_close(sign * logabs.exp(), target, rtol=3e-12, atol=1e-16)
    assert float(info["logloss"]) == pytest.approx(float(loss.detach().log()), abs=1e-13)


def test_packing_has_no_padding_and_preserves_each_record():
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=53, prefix_mean=6),
                                rng=np.random.default_rng(22))
    packed = pack_random_batch(batch, 5)
    assert packed.record_count == int(batch.lengths.sum())
    assert packed.record_count < packed.padded_records
    assert packed.size == batch.size
    torch.testing.assert_close(packed.keys, batch.keys[batch.mask])
    torch.testing.assert_close(packed.values, batch.values[batch.mask])
    assert int(packed.first.sum()) == batch.size
    torch.testing.assert_close(packed.current_pairs[packed.first], packed.previous_pairs[packed.first])
    _assert_padded_parity(_model(), batch)


@pytest.mark.parametrize("w", [0., -1.7, 2.3])
def test_packed_zero_negative_and_positive_decoder(w):
    model = _model()
    model.theta[-1] = w
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=13, prefix_mean=2),
                                rng=np.random.default_rng(84))
    _assert_padded_parity(model, batch)


def test_all_same_values_have_no_attention_gradient():
    model = _model()
    batch = batch_from_streams([
        {"keys": [0, 1, 2], "values": [1, 1, 1], "query": 0},
        {"keys": [3], "values": [-1], "query": 3},
    ])
    gradients, info = _assert_padded_parity(model, batch)
    for sign, logabs in gradients[:2]:
        assert not bool(sign.any())
        assert bool(torch.isneginf(logabs).all())
    assert not bool(gradients[-1][0][:-1].any())
    assert torch.isfinite(info["logloss"])


def test_packed_underflowed_loss_stays_finite_in_log_space():
    model = Model(Config(n=2, d=2, R=2))
    model.Q[:] = torch.eye(2, dtype=torch.float64)
    model.K[:] = torch.eye(2, dtype=torch.float64)
    model.theta[:] = torch.tensor([30., 30., 4., 4., 1., 1200.], dtype=torch.float64)
    batch = batch_from_streams([{"keys": [0, 1], "values": [1, -1], "query": 0}])
    gradients, info = _assert_padded_parity(model, batch)
    assert float(info["logloss"]) < -1000
    assert torch.isfinite(gradients[-1][1][0])
    assert float(gradients[-1][1][0]) < -1000


def test_packed_preserves_independent_rare_query_derivatives():
    model = Model(Config(n=4, d=4, R=2))
    model.Q[:] = torch.eye(4, dtype=torch.float64)
    model.K[:] = torch.diag(torch.tensor([.01, 1., 2., 1.], dtype=torch.float64))
    model.theta[:] = torch.tensor([math.sqrt(1000), math.sqrt(1000), 2., 2., 1., 3.], dtype=torch.float64)
    batch = batch_from_streams([
        {"keys": [0, 1], "values": [1, -1], "query": 0},
        {"keys": [2, 3], "values": [1, -1], "query": 2},
    ])
    gradients, info = _assert_padded_parity(model, batch)
    assert float(info["record_log_span"]) > 1900
    assert float(info["table_log_span"]) < 10
    assert gradients[0][0][2, 2] == -1
    assert float(gradients[0][1][2, 2]) < -1900


def test_packed_long_prefix_and_zero_count_match_reference():
    model = Model(Config(n=2, d=2, R=2))
    model.Q[:] = torch.eye(2, dtype=torch.float64)
    model.K[:] = torch.eye(2, dtype=torch.float64)
    model.theta[:] = torch.tensor([2., 2., 2., 2., 1., 3.], dtype=torch.float64)
    batch = batch_from_streams([
        {"keys": [0, 1] * 1000 + [0, 1], "values": [-1] * 2000 + [1, -1], "query": 0},
        {"keys": [1], "values": [1], "query": 1},
    ])
    _, info = _assert_padded_parity(model, batch, [2., 0.])
    assert float(info["record_log_span"]) > 2500
    assert float(info["table_log_span"]) < 10


def test_packed_counts_validation_and_vocabulary_check():
    model = _model()
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=3), rng=np.random.default_rng(2))
    packed = pack_random_batch(batch, 5)
    for counts in ([0., 0., 0.], [-1., 1., 1.], [1., 1.], [1., 1., math.nan]):
        with pytest.raises(ValueError, match="counts"):
            packed_random_gradients(model, packed, counts)
    with pytest.raises(ValueError, match="vocabulary"):
        packed.validate(4)
    checked = packed_random_gradients(model, packed, [0., 2., 3.])
    trusted = packed_random_gradients(model, packed, [0., 2., 3.], validate=False)
    for left, right in zip(checked[0], trusted[0]):
        for a, b in zip(left, right):
            assert torch.equal(a, b)


def test_packed_respects_deterministic_algorithms_mode():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = _model()
        batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=13), rng=np.random.default_rng(52))
        packed = pack_random_batch(batch, 5)
        left = packed_random_gradients(model, packed)[0]
        right = packed_random_gradients(model, packed)[0]
        for (ls, ll), (rs, rl) in zip(left, right):
            assert torch.equal(ls, rs)
            assert torch.equal(ll, rl)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_packed_optimizer_matches_literal_ordinary_loss(optimizer):
    model = _model()
    batch = sample_stream_batch(5, 3, RandomStreamConfig(batch_size=11, prefix_mean=3),
                                rng=np.random.default_rng(20))
    packed = pack_random_batch(batch, 5)
    params = [p.detach().clone().requires_grad_(True) for p in model.params()]
    rate = .002
    native = (torch.optim.Adam(params, lr=rate, betas=(.1, .1), eps=1e-18)
              if optimizer == "adam" else torch.optim.SGD(params, lr=rate))
    adam = LogAdam(model.params(), .1, .1) if optimizer == "adam" else None
    for _ in range(5):
        native.zero_grad()
        native_random_loss(model, batch, params=params, literal=True).backward()
        native.step()
        gradients = packed_random_gradients(model, packed, validate=False)[0]
        if adam is not None:
            adam.step(model.params(), gradients, [rate] * 3, math.log(1e-18))
        else:
            gradient_step(model, gradients, [rate] * 3)
    for expected, actual in zip(params, model.params()):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)
