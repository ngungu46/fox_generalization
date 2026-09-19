"""Full-update capture, complete-state parity, and failed-audit rollback."""
import pytest
import torch

from fox_restricted.data.random_streams import batch_from_streams
from fox_restricted.legacy.core import Config, Model, LogAdam, rates_at
from fox_restricted.legacy.a100_runner import _optimizer_step
from fox_restricted.training.acceleration import CompiledUpdate
from fox_restricted.training.random_backend import random_gradients


def make_model(gate):
    cfg = Config(n=3, d=8, R=3, gate_mode=gate, beta1=.1, beta2=.1)
    model = Model(cfg)
    model.Q.mul_(1e5)
    model.K.mul_(1e5)
    model.theta[-1] = .7
    return model


@pytest.mark.parametrize("gate", ["learned", "retrieval_frozen"])
@pytest.mark.parametrize("kind", ["sgd", "adam"])
@pytest.mark.parametrize("dataset", ["pair", "random"])
def test_compiled_complete_update_matches_eager_without_step_recompilation(gate, kind, dataset):
    torch.set_num_threads(1)
    torch._dynamo.reset()
    model, reference = make_model(gate), make_model(gate)
    adam = LogAdam(model.params(), .1, .1) if kind == "adam" else None
    ref_adam = LogAdam(reference.params(), .1, .1) if kind == "adam" else None
    batch = batch_from_streams([
        dict(keys=[0, 1, 0, 2], values=[-1, 1, 1, -1], query=0),
        dict(keys=[2, 1, 0], values=[1, -1, 1], query=1),
        dict(keys=[2, 2], values=[-1, 1], query=2),
        dict(keys=[1, 0, 1, 2, 2], values=[1, -1, -1, 1, -1], query=1),
    ]) if dataset == "random" else None
    def objective(target, counts):
        if batch is None:
            return target.gradients(counts, validate_counts=False)
        return random_gradients(target, batch, counts, validate=False)
    captures = []
    def capture(graph_module, inputs):
        captures.append(graph_module)
        return graph_module.forward
    gradient_function = None if batch is None else lambda counts, actual_batch: random_gradients(model, actual_batch, counts, validate=False)
    update = CompiledUpdate(model, adam, gradient_function, backend=capture)
    size = 6 if batch is None else batch.size
    counts = torch.arange(1, size + 1, dtype=torch.float64) if kind == "sgd" else None
    frozen_initial = model.theta[4].clone()
    for step in range(1, 7):
        _, rates = rates_at(model.cfg, kind, step - 1)
        expected, expected_info = objective(reference, counts)
        _optimizer_step(reference, ref_adam, expected, rates, step)
        actual, info = update(rates, step, counts=counts, batch=batch)
        torch.testing.assert_close(info["logloss"], expected_info["logloss"], rtol=0, atol=0)
        for left, right in zip(model.params(), reference.params()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        if adam is not None:
            assert adam.t == ref_adam.t == step
            for name in ("ms", "ml", "vl"):
                for left, right in zip(getattr(adam, name), getattr(ref_adam, name)):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        if gate == "retrieval_frozen":
            torch.testing.assert_close(model.theta[4], frozen_initial, rtol=0, atol=0)
    assert len(captures) == 1
    assert len(update.audits) == 1
    assert update.audits[0]["parameter_and_optimizer_state_audit"]


def test_compile_failure_restores_parameters_moments_and_step():
    model = make_model("learned")
    adam = LogAdam(model.params(), .1, .1)
    before = [x.clone() for x in (*model.params(), *adam.ms, *adam.ml, *adam.vl)]
    def failed_backend(graph_module, inputs):
        raise RuntimeError("intentional backend failure")
    update = CompiledUpdate(model, adam, backend=failed_backend)
    _, rates = rates_at(model.cfg, "adam", 0)
    with pytest.raises(RuntimeError, match="original model and optimizer buffers were restored"):
        update(rates, 1)
    for left, right in zip(before, (*model.params(), *adam.ms, *adam.ml, *adam.vl)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert adam.t == 0
    assert update.audits == []


def test_audit_failure_after_compiled_mutation_rolls_back():
    torch._dynamo.reset()
    model = make_model("learned")
    adam = LogAdam(model.params(), .1, .1)
    before = [x.clone() for x in (*model.params(), *adam.ms, *adam.ml, *adam.vl)]
    update = CompiledUpdate(model, adam, backend="eager")
    update._initialize()
    original = update._compiled
    def corrupt(*args):
        result = original(*args)
        model.Q.add_(1.)
        return result
    update._compiled = corrupt
    _, rates = rates_at(model.cfg, "adam", 0)
    with pytest.raises(RuntimeError, match="original model and optimizer buffers were restored"):
        update(rates, 1)
    for left, right in zip(before, (*model.params(), *adam.ms, *adam.ml, *adam.vl)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert adam.t == 0


def test_reconstructed_backend_continues_nonzero_adam_moments():
    torch._dynamo.reset()
    model = make_model("retrieval_frozen")
    reference = make_model("retrieval_frozen")
    adam, ref_adam = [LogAdam(target.params(), .9, .999) for target in (model, reference)]
    # A branch may construct its backend before the three eager acquisition
    # steps; capture must see the replaced moment buffers at first actual use.
    update = CompiledUpdate(model, adam, backend="eager")
    for step in range(1, 4):
        _, rates = rates_at(model.cfg, "adam", step - 1)
        for target, opt in ((model, adam), (reference, ref_adam)):
            gradients, _ = target.gradients()
            _optimizer_step(target, opt, gradients, rates, step)
    _, rates = rates_at(model.cfg, "adam", 3)
    gradients, _ = reference.gradients()
    _optimizer_step(reference, ref_adam, gradients, rates, 4)
    update(rates, 4)
    assert adam.t == ref_adam.t == 4
    for left, right in zip((*model.params(), *adam.ml, *adam.vl),
                           (*reference.params(), *ref_adam.ml, *ref_adam.vl)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("beta1,beta2", [(0., .9), (.9, .999)])
def test_compiled_adam_retains_gradients_below_float64_range(beta1, beta2):
    torch._dynamo.reset()
    model, reference = make_model("learned"), make_model("learned")
    model.cfg.eps_decay = reference.cfg.eps_decay = 20000.
    adam = LogAdam(model.params(), beta1, beta2)
    ref_adam = LogAdam(reference.params(), beta1, beta2)
    def gradients(target):
        result = [(torch.ones_like(p), torch.full_like(p, -10000.)) for p in target.params()]
        return result, {"logloss": torch.tensor(-10000., dtype=torch.float64)}
    update = CompiledUpdate(model, adam, lambda counts, batch: gradients(model), backend="eager")
    for step in (1, 2, 3):
        _, rates = rates_at(model.cfg, "adam", step - 1)
        values, _ = gradients(reference)
        _optimizer_step(reference, ref_adam, values, rates, step)
        update(rates, step)
    for left, right in zip((*model.params(), *adam.ml, *adam.vl),
                           (*reference.params(), *ref_adam.ml, *ref_adam.vl)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert bool(torch.isfinite(adam.ml[0]).all())
    assert bool((adam.ml[0] < -9999).all())
