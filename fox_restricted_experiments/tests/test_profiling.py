"""Profiling must measure real trajectories without changing training artifacts."""
from dataclasses import replace
import json

import pytest
import torch

from fox_restricted import Config, RandomStreamConfig, make_experiments, train
from fox_restricted.legacy.runner import fingerprint
from fox_restricted.profiling import profile_experiment, format_profile


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny(dataset="pairs", optimizer="sgd", sampling="fixed"):
    return make_experiments(
        Config(n=4, d=32, R=2, steps=10, device="cpu", pair_batch=16,
               beta1=.1, beta2=.1, max_seconds=1000.),
        seeds=(0,), random_sampling=sampling, random_dataset_size=12,
        random_data=RandomStreamConfig(batch_size=8, prefix_mean=3., max_batch_records=10000),
        log_every=5, checkpoint_every=5, check_every=5,
    )[f"{dataset}_learned_{optimizer}"]


@pytest.mark.parametrize("dataset,optimizer,sampling", [
    ("pairs", "sgd", "fixed"), ("pairs", "adam", "fixed"),
    ("random", "sgd", "fixed"), ("random", "adam", "fixed"),
    ("random", "sgd", "online"), ("random", "adam", "online"),
])
def test_profile_matches_same_seed_data_acquisition_and_training_updates(tmp_path, dataset, optimizer, sampling):
    experiment = tiny(dataset, optimizer, sampling)
    result = profile_experiment(experiment, warmup=0, steps=1, diagnostics=False, verbose=False)
    assert result["measured_final_step"] == 5
    assert result["total_disposable_updates"] == 5
    run = train(experiment, tmp_path, max_updates=5, progress=False)
    filename = "seed0.pt" if dataset == "random" else f"seed0_learned_{optimizer}.pt"
    checkpoint = torch.load(run.path / "checkpoints" / filename, map_location="cpu", weights_only=False)
    assert result["acquisition"]["initial_hash"] == checkpoint["initial_hash"]
    assert result["measured_final_hash"] == fingerprint(checkpoint["model"])
    assert result["hardware"]["device"] == "cpu"
    assert "gpu_name" not in result["hardware"]
    assert result["phase_mean_ms"]["fused_update"] is None
    assert result["steady_state"]["updates_per_second"] > 0
    json.dumps(result, allow_nan=False)


def test_profile_reports_real_padding_and_diagnostic_health():
    result = profile_experiment(tiny("random", "adam"), warmup=1, steps=1, verbose=False)
    data = result["data"]
    assert data["sequences"] == 12
    assert data["padded_records"] == 12 * data["maximum_records"]
    assert data["padded_records"] >= data["valid_records"]
    assert data["padding_fraction"] == pytest.approx(1 - data["valid_records"] / data["padded_records"])
    assert result["diagnostics"]["health_issue"] is None
    assert result["diagnostics"]["probe_rows"] > 0
    assert result["diagnostics"]["seconds"] > 0
    assert "padding" in format_profile(result)


def test_io_and_trace_use_only_disposable_files_and_separate_report(tmp_path):
    io = tmp_path / "io"
    io.mkdir()
    checkpoint = io / "existing_training.pt"
    checkpoint.write_bytes(b"untouched")
    out = tmp_path / "reports"
    result = profile_experiment(tiny(), out, warmup=0, steps=1, verbose=False,
                                io_directories={"local": io}, trace=True, trace_steps=1)
    assert checkpoint.read_bytes() == b"untouched"
    assert list(io.iterdir()) == [checkpoint]
    assert len(result["checkpoint_io"]) == 1
    assert result["checkpoint_io"][0]["payload_bytes"] > 0
    assert result["checkpoint_io"][0]["atomic_write_fsync_seconds"] > 0
    assert result["total_disposable_updates"] == 6
    assert result["measured_final_step"] == 5
    assert (out / "profile_pairs_learned_sgd_seed0.json").exists()
    assert (out / "profile_pairs_learned_sgd_seed0.trace.json").exists()
    assert "traceEvents" in json.loads((out / "profile_pairs_learned_sgd_seed0.trace.json").read_text())
    assert "aten::" in result["operator_table"]


@pytest.mark.parametrize("kwargs", [dict(steps=0), dict(warmup=-1), dict(steps=True),
                                    dict(seed=-1), dict(trace=True), dict(io_directories={})])
def test_profile_rejects_invalid_requests_before_training(kwargs):
    with pytest.raises(ValueError):
        profile_experiment(tiny(), **kwargs)


def test_requested_cuda_is_not_silently_profiled_on_cpu(monkeypatch):
    experiment = tiny()
    experiment = replace(experiment, config=replace(experiment.config, device="cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        profile_experiment(experiment, steps=1, warmup=0, verbose=False)


@pytest.mark.parametrize("dataset,compiled", [("pairs", False), ("random", False),
                                             ("pairs", True), ("random", True)])
def test_operational_backend_profiles_same_configured_compute_path(dataset, compiled):
    from fox_restricted.training.performance import PerformanceConfig
    performance = PerformanceConfig(pack_random=True, compile_updates=compiled, compile_backend="eager")
    result = profile_experiment(tiny(dataset, "adam"), performance=performance,
                                warmup=1, steps=1, verbose=False)
    assert result["backend"]["compile_updates"] == compiled
    assert result["diagnostics"]["health_issue"] is None
    if dataset == "random":
        assert result["engine_stats"]["packing_audit"]["absolute_log_loss_error"] < 1e-9
    if compiled:
        assert result["phase_mean_ms"]["fused_update"] > 0
        assert result["phase_mean_ms"]["gradient"] is None
        assert result["phase_mean_ms"]["optimizer"] is None
        assert len(result["engine_stats"]["compile_audits"]) == 1
    else:
        assert result["phase_mean_ms"]["gradient"] > 0
        assert result["phase_mean_ms"]["optimizer"] > 0
        assert result["phase_mean_ms"]["fused_update"] is None


def test_performance_backend_requires_warmup_and_respects_online_shape_restriction():
    from fox_restricted.training.performance import PerformanceConfig
    with pytest.raises(ValueError, match="warmup"):
        profile_experiment(tiny(), performance=PerformanceConfig(), warmup=0, steps=1)
    with pytest.raises(ValueError, match="fixed dataset"):
        profile_experiment(tiny("random", "adam", "online"),
                           performance=PerformanceConfig(compile_updates=True), warmup=1, steps=1)
