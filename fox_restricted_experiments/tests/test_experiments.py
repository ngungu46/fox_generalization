"""Eight-arm protocol, true random objectives, and exact continuation checks."""
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fox_restricted.data import RandomBatch, RandomStreamConfig
from fox_restricted.evaluation.probes import convergence_probes
from fox_restricted.legacy import a100_runner
from fox_restricted.legacy.core import Config, Model, LogAdam, acquisition_rates
from fox_restricted.legacy.runner import fingerprint
from fox_restricted.study import make_experiments
from fox_restricted.training.experiment import train, summarize
from fox_restricted.training.random_backend import random_gradients


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_experiments(steps=12, sampling="fixed", dataset_size=12):
    cfg = Config(n=4, d=64, R=2, steps=steps, device="cpu", pair_batch=16,
                 beta1=.1, beta2=.1, max_seconds=1000.)
    return make_experiments(
        cfg, seeds=(0,), random_sampling=sampling, random_dataset_size=dataset_size,
        random_data=RandomStreamConfig(batch_size=16, prefix_mean=3., max_batch_records=100_000),
        log_every=4, checkpoint_every=4, check_every=2)


def load_checkpoint(run, experiment):
    filename = "seed0.pt" if experiment.dataset == "random" else f"seed0_{experiment.gate_mode}_{experiment.optimizer}.pt"
    return torch.load(run.path / "checkpoints" / filename, map_location="cpu", weights_only=False)


def assert_same_state(left, right):
    for a, b in zip(left["model"], right["model"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in ("step", "S", "examples", "acquired", "reference_gap"):
        assert left["state"][key] == right["state"][key]
    assert left["rng_state"] == right["rng_state"]
    assert left["initial_hash"] == right["initial_hash"]
    if left["optimizer"] is None:
        assert right["optimizer"] is None
    else:
        for key in ("t", "b1", "b2"):
            assert left["optimizer"][key] == right["optimizer"][key]
        for key in ("ms", "ml", "vl"):
            for a, b in zip(left["optimizer"][key], right["optimizer"][key]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_all_eight_arms_complete_with_paired_initialization_and_data(tmp_path):
    experiments = tiny_experiments()
    assert list(experiments) == [
        "pairs_learned_sgd", "pairs_learned_adam", "pairs_alibi_sgd", "pairs_alibi_adam",
        "random_learned_sgd", "random_learned_adam", "random_alibi_sgd", "random_alibi_adam"]
    runs = [train(experiment, tmp_path, progress=False) for experiment in experiments.values()]
    summaries = summarize(runs)
    assert len(summaries) == 8
    assert {row["status"] for row in summaries} == {"finite_budget_complete"}
    assert {int(row["completed_steps"]) for row in summaries} == {12}
    assert len({row["initial_hash"] for row in summaries}) == 1
    random_rows = [row for row in summaries if row["experiment"].startswith("random_")]
    assert len({row["dataset_hash"] for row in random_rows}) == 1
    for run, experiment in zip(runs, experiments.values()):
        checkpoint = load_checkpoint(run, experiment)
        if experiment.gate == "alibi":
            assert checkpoint["model"][-1][4] == checkpoint["frozen_initial"][4]
        with (run.path / "asymptotic_errors.csv").open() as handle:
            errors = list(csv.DictReader(handle))
        fixed_lags = {int(row["lag"]) for row in errors if row["probe"] == "fixed"}
        assert {3, 4, 5} <= fixed_lags
        if experiment.dataset == "random":
            description = json.loads((run.path / "dataset" / "seed0.json").read_text())
            assert description["size"] == 12
            assert description["role"] == "training_population"
            assert sum(description["lag_counts"].values()) == 12


@pytest.mark.parametrize("dataset,sampling,optimizer", [
    ("pairs", "fixed", "sgd"), ("pairs", "fixed", "adam"),
    ("random", "fixed", "sgd"), ("random", "fixed", "adam"),
    ("random", "online", "sgd"), ("random", "online", "adam"),
])
def test_pause_resume_preserves_optimizer_sampler_and_parameters(tmp_path, dataset, sampling, optimizer):
    experiment = tiny_experiments(sampling=sampling)[f"{dataset}_learned_{optimizer}"]
    whole = train(experiment, tmp_path / "whole", progress=False)
    partial = train(experiment, tmp_path / "continued", progress=False, max_updates=5)
    assert load_checkpoint(partial, experiment)["state"]["step"] == 5
    continued = train(experiment, tmp_path / "continued", progress=False)
    assert_same_state(load_checkpoint(whole, experiment), load_checkpoint(continued, experiment))
    assert load_checkpoint(continued, experiment)["state"]["resume_count"] == 1


@pytest.mark.parametrize("dataset", ["pairs", "random"])
def test_declared_step_budget_can_extend_from_seven_to_twelve(tmp_path, dataset):
    short = tiny_experiments(steps=7)[f"{dataset}_alibi_adam"]
    long = replace(short, config=replace(short.config, steps=12))
    train(short, tmp_path / "extended", progress=False)
    extended = train(long, tmp_path / "extended", progress=False)
    whole = train(long, tmp_path / "whole", progress=False)
    assert_same_state(load_checkpoint(extended, long), load_checkpoint(whole, long))


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_pair_orchestration_preserves_original_training_trajectory(tmp_path, optimizer):
    experiment = tiny_experiments()[f"pairs_learned_{optimizer}"]
    actual = train(experiment, tmp_path / "new", progress=False)
    original = tmp_path / "original"
    a100_runner.run_large_suite(
        experiment.config, original, seeds=(0,), gate_modes=(experiment.gate_mode,),
        optimizers=(optimizer,), eval_lags=(3, 4, 5), prefixes=(0, 64),
        log_every=4, checkpoint_every=4, check_every=2, progress=False,
        fixed_lags=(3, 4, 5), theta_values=(.5,), clock_coefficients=(.01,))
    from fox_restricted.study import Run
    assert_same_state(load_checkpoint(actual, experiment), load_checkpoint(Run(original, "original"), experiment))


@pytest.mark.parametrize("dataset,field", [
    ("pairs", "theta"), ("pairs", "config"),
    ("random", "theta"), ("random", "config"), ("random", "random_data"),
])
def test_resume_rejects_changed_scientific_settings(tmp_path, dataset, field):
    experiment = tiny_experiments()[f"{dataset}_learned_adam"]
    train(experiment, tmp_path, progress=False, max_updates=3)
    changes = {
        "theta": {"theta": .75},
        "config": {"config": replace(experiment.config, R=3)},
        "random_data": {"random_data": replace(experiment.random_data, prefix_mean=4.)},
    }[field]
    with pytest.raises(ValueError, match="changed|configuration|settings"):
        train(replace(experiment, **changes), tmp_path, progress=False)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_random_acquisition_is_exact_random_objective_without_pair_warm_start(tmp_path, optimizer):
    experiment = tiny_experiments(steps=3)[f"random_learned_{optimizer}"]
    # The inherited SGD rate formula evaluates a deterministic zero-table pair
    # probe. Once those fixed constants exist, no training update may use pairs.
    rates_by_step = acquisition_rates(experiment.config, optimizer)
    with patch.object(Model, "gradients", side_effect=AssertionError("pair loss must not be called")), \
         patch("fox_restricted.training.random_training.acquisition_rates", return_value=rates_by_step):
        run = train(experiment, tmp_path, progress=False)
    actual = load_checkpoint(run, experiment)
    state = torch.load(run.path / "dataset" / "seed0.pt", map_location="cpu", weights_only=True)
    population = RandomBatch.from_state_dict(state)
    model = Model(experiment.config)
    adam = LogAdam(model.params(), .1, .1) if optimizer == "adam" else None
    for step, rates in enumerate(rates_by_step, 1):
        gradients, _ = random_gradients(model, population)
        a100_runner._optimizer_step(model, adam, gradients, rates, step)
    assert fingerprint(model.params()) == actual["final_hash"]
    assert actual["state"]["examples"] == 3 * experiment.dataset_size
    assert all(row["passed"] for row in actual["audits"])


def test_theory_ratio_probe_fixes_coefficient_and_keeps_unavailable_gap_explicit():
    model = Model(tiny_experiments()["pairs_learned_adam"].config)
    tags = dict(seed=0, gate_mode="learned", optimizer="adam")
    model.theta[0] = model.theta[1] = 100.
    reference = .1
    rows, _ = convergence_probes(model, tags, 10, 2., reference, .5)
    row = next(row for row in rows if row["probe"] == "theory_ratio")
    m, h = float(model.quantities()[6]), float(model.quantities()[8])
    assert row["coefficient"] == .5 * reference
    assert row["lag"] == max(1, math.floor(.5 * reference * m / h))
    assert row["reference_gap_retained"] is False
    assert row["bound_valid"] is False
    unavailable, _ = convergence_probes(model, tags, 0, 0., None, .5)
    row = next(row for row in unavailable if row["probe"] == "theory_ratio")
    assert row["lag"] is None
    assert row["lower_error"] is None and row["upper_error"] is None
    assert row["bound_invalid_reason"] == "positive_acquired_gap_unavailable"
