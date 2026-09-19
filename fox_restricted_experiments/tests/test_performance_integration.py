"""Real-run acceleration, exact continuation, and known-release provenance upgrades."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from fox_restricted import Config, RandomStreamConfig, make_experiments, train
from fox_restricted.training.performance import PerformanceConfig
from fox_restricted.training import checkpoint_upgrade as upgrade


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    torch.set_num_threads(before)


def tiny_experiments():
    return make_experiments(
        Config(n=4, d=32, R=2, steps=9, device="cpu", pair_batch=16,
               beta1=.1, beta2=.1, max_seconds=1000.),
        seeds=(0,), random_dataset_size=12,
        random_data=RandomStreamConfig(batch_size=8, prefix_mean=3., max_batch_records=10000),
        log_every=3, checkpoint_every=3, check_every=3)


def checkpoint_file(run, experiment):
    name = ("seed0.pt" if experiment.dataset == "random" else
            f"seed0_{experiment.gate_mode}_{experiment.optimizer}.pt")
    return run.path / "checkpoints" / name


def load(run, experiment):
    return torch.load(checkpoint_file(run, experiment), map_location="cpu", weights_only=False)


def assert_recursive_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_recursive_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_recursive_equal(a, b)
    else:
        assert left == right


def assert_training_state(left, right, *, exact=True):
    for a, b in zip(left["model"], right["model"]):
        torch.testing.assert_close(a, b, rtol=0 if exact else 1e-9, atol=0 if exact else 1e-12)
    assert_recursive_equal(left["rng_state"], right["rng_state"])
    assert left["initial_hash"] == right["initial_hash"]
    for field in ("step", "S", "examples", "acquired", "reference_gap"):
        assert left["state"][field] == right["state"][field]
    if left["optimizer"] is None:
        assert right["optimizer"] is None
    else:
        for field in ("t", "b1", "b2"):
            assert left["optimizer"][field] == right["optimizer"][field]
        for block in ("ms", "ml", "vl"):
            for a, b in zip(left["optimizer"][block], right["optimizer"][block]):
                torch.testing.assert_close(a, b, rtol=0 if exact else 1e-9, atol=0 if exact else 1e-10)


def test_all_eight_accelerated_arms_pause_and_resume_in_one_runtime(tmp_path):
    """Exercise the same eight-arm process and full optimizer restoration as Colab."""
    performance = PerformanceConfig(pack_random=True, compile_updates=True, compile_backend="eager")
    for experiment in tiny_experiments().values():
        whole_run = train(experiment, tmp_path / "whole", performance=performance, progress=False)
        split_run = train(experiment, tmp_path / "split", performance=performance,
                          progress=False, max_updates=5)
        partial = load(split_run, experiment)
        assert partial["state"]["step"] == 5
        split_run = train(experiment, tmp_path / "split", performance=performance, progress=False)
        whole, split = load(whole_run, experiment), load(split_run, experiment)
        assert_training_state(whole, split)
        assert split["state"]["resume_count"] == 1
        assert split["state"]["status"] == "finite_budget_complete"
        assert len(split["performance_sessions"]) == 2
        latest = split["performance_sessions"][-1]
        assert latest["last_step"] == experiment.config.steps
        assert latest["compile_audits"][0]["parameter_and_optimizer_state_audit"]
        if experiment.dataset == "random":
            assert latest["packing_audit"]["absolute_log_loss_error"] < 1e-9
        if experiment.optimizer == "adam":
            assert partial["optimizer"]["t"] == 5
            assert split["optimizer"]["t"] == 9
        if experiment.gate == "alibi":
            assert split["model"][-1][4] == partial["frozen_initial"][4]
        assert any((split_run.path / "performance_source").glob("*/sha256.json"))


@pytest.mark.parametrize("dataset,optimizer", [("pairs", "sgd"), ("pairs", "adam"),
                                             ("random", "sgd"), ("random", "adam")])
def test_enable_packing_after_reference_checkpoint_retains_training_state(tmp_path, dataset, optimizer):
    experiment = tiny_experiments()[f"{dataset}_learned_{optimizer}"]
    expected_run = train(experiment, tmp_path / "reference", progress=False)
    run = train(experiment, tmp_path / "switch", progress=False, max_updates=5)
    before = load(run, experiment)
    saved_rows = copy.deepcopy(before["history"])
    run = train(experiment, tmp_path / "switch", progress=False,
                performance=PerformanceConfig(pack_random=True))
    after = load(run, experiment)
    assert_training_state(load(expected_run, experiment), after, exact=False)
    assert after["state"]["resume_count"] == 1
    assert after["history"][:len(saved_rows)] == saved_rows
    assert after["performance_sessions"][-1]["last_step"] == 9
    if optimizer == "adam":
        assert before["optimizer"]["t"] == 5 and after["optimizer"]["t"] == 9


def make_old_release_fixture(tmp_path, monkeypatch, dataset):
    """Recognize only test-fixture orchestration hashes; never loosen kernel guards."""
    experiment = tiny_experiments()[f"{dataset}_learned_adam"]
    run = train(experiment, tmp_path, progress=False, max_updates=5)
    manifest_path = run.path / "config.json"
    manifest = json.loads(manifest_path.read_text())
    names = (["a100_runner.py"] if dataset == "pairs" else
             ["legacy/a100_runner.py", "training/random_training.py"])
    for name in names:
        source = run.path / "source" / name
        source.write_text(source.read_text() + "\n# previous supported orchestration release fixture\n")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest["source_sha256"][name] = digest
        attribute = "OLD_RANDOM_RUNNER_HASH" if name.endswith("random_training.py") else "OLD_RUNNER_HASH"
        monkeypatch.setattr(upgrade, attribute, digest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    checkpoint = load(run, experiment)
    if "source_sha256" in checkpoint:
        checkpoint["source_sha256"] = copy.deepcopy(manifest["source_sha256"])
        torch.save(checkpoint, checkpoint_file(run, experiment))
    return experiment, run, manifest, load(run, experiment)


def state_without_provenance(payload):
    payload = copy.deepcopy(payload)
    payload.pop("source_sha256", None)
    return payload


@pytest.mark.parametrize("dataset", ["pairs", "random"])
def test_known_runner_upgrade_preserves_every_training_field_and_is_idempotent(tmp_path, monkeypatch, dataset):
    experiment, run, old_manifest, before = make_old_release_fixture(tmp_path, monkeypatch, dataset)
    old_source = {name: (run.path / "source" / name).read_bytes() for name in old_manifest["source_sha256"]}
    tables = {path.name: path.read_bytes() for path in run.path.glob("*.csv")}
    assert upgrade.upgrade_known_runner(run.path, dataset)
    assert_recursive_equal(state_without_provenance(before), state_without_provenance(load(run, experiment)))
    assert tables == {path.name: path.read_bytes() for path in run.path.glob("*.csv")}
    for name, content in old_source.items():
        assert (run.path / "source_before_performance_upgrade" / name).read_bytes() == content
    archived = json.loads((run.path / "source_before_performance_upgrade" / "original_config.json").read_text())
    assert archived == old_manifest
    manifest = json.loads((run.path / "config.json").read_text())
    assert len(manifest["implementation_history"]) == 1
    assert json.loads((run.path / "performance_upgrade.json").read_text())["complete"]
    checkpoint_bytes = checkpoint_file(run, experiment).read_bytes()
    assert not upgrade.upgrade_known_runner(run.path, dataset)
    assert checkpoint_file(run, experiment).read_bytes() == checkpoint_bytes
    resumed = train(experiment, tmp_path, progress=False,
                    performance=PerformanceConfig(pack_random=True))
    after = load(resumed, experiment)
    assert after["state"]["step"] == 9 and after["optimizer"]["t"] == 9
    assert after["history"][:len(before["history"])] == before["history"]
    assert after["initial_hash"] == before["initial_hash"]


@pytest.mark.parametrize("dataset", ["pairs", "random"])
@pytest.mark.parametrize("interruption", ["journal", "manifest"])
def test_known_upgrade_recovers_interrupted_journal_without_state_loss(tmp_path, monkeypatch, dataset, interruption):
    experiment, run, old_manifest, before = make_old_release_fixture(tmp_path, monkeypatch, dataset)
    actual_atomic_json = upgrade._atomic_json
    interrupted = False

    def disconnect_after_commit(path, value):
        nonlocal interrupted
        actual_atomic_json(path, value)
        target = "performance_upgrade.json" if interruption == "journal" else "config.json"
        if not interrupted and Path(path).name == target:
            interrupted = True
            raise OSError("simulated disconnection after atomic metadata commit")

    with monkeypatch.context() as context:
        context.setattr(upgrade, "_atomic_json", disconnect_after_commit)
        with pytest.raises(OSError, match="simulated disconnection"):
            upgrade.upgrade_known_runner(run.path, dataset)
    assert not json.loads((run.path / "performance_upgrade.json").read_text())["complete"]
    assert_recursive_equal(state_without_provenance(before), state_without_provenance(load(run, experiment)))
    assert upgrade.upgrade_known_runner(run.path, dataset)
    assert_recursive_equal(state_without_provenance(before), state_without_provenance(load(run, experiment)))
    assert len(json.loads((run.path / "config.json").read_text())["implementation_history"]) == 1
    assert json.loads((run.path / "source_before_performance_upgrade" / "original_config.json").read_text()) == old_manifest
    assert not upgrade.upgrade_known_runner(run.path, dataset)


@pytest.mark.parametrize("dataset", ["pairs", "random"])
def test_unknown_scientific_source_change_is_rejected_without_upgrading(tmp_path, monkeypatch, dataset):
    experiment, run, manifest, before = make_old_release_fixture(tmp_path, monkeypatch, dataset)
    kernel = "core.py" if dataset == "pairs" else "legacy/core.py"
    source = run.path / "source" / kernel
    source.write_text(source.read_text() + "\n# unrelated scientific-kernel modification\n")
    manifest["source_sha256"][kernel] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run.path / "config.json").write_text(json.dumps(manifest))
    checkpoint_bytes = checkpoint_file(run, experiment).read_bytes()
    assert not upgrade.upgrade_known_runner(run.path, dataset)
    assert not (run.path / "performance_upgrade.json").exists()
    assert checkpoint_file(run, experiment).read_bytes() == checkpoint_bytes
    with pytest.raises(ValueError, match="source|sources"):
        train(experiment, tmp_path, progress=False, performance=PerformanceConfig())
    assert_recursive_equal(before, load(run, experiment))


@pytest.mark.parametrize("dataset", ["pairs", "random"])
def test_known_release_requires_intact_original_source_snapshot(tmp_path, monkeypatch, dataset):
    experiment, run, _, before = make_old_release_fixture(tmp_path, monkeypatch, dataset)
    kernel = "core.py" if dataset == "pairs" else "legacy/core.py"
    (run.path / "source" / kernel).write_text("tampered archive")
    with pytest.raises(ValueError, match="verify the original checkpoint source"):
        upgrade.upgrade_known_runner(run.path, dataset)
    assert not (run.path / "performance_upgrade.json").exists()
    assert_recursive_equal(before, load(run, experiment))
