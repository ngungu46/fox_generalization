"""Disposable, hardware-measured profiling of the exact selected experiment.

A profile starts from the declared Gaussian initialization and runs the same
three acquisition updates and data distribution as training. It never opens a
training checkpoint. Phase timings synchronize CUDA and include Python launch
cost; a separate block measures steady throughput without per-phase barriers.
No GPU performance is inferred from a CPU profile.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from .data.random_streams import sample_stream_batch
from .evaluation.probes import convergence_probes
from .legacy.core import Model, LogAdam, acquisition_rates, rates_at
from .legacy.runner import _validate, clean, fingerprint
from .legacy.a100_runner import (
    _atomic_torch_save, _cpu_tensors, _optimizer_state, _optimizer_step,
    _step_inputs, _check_health,
)
from .training.random_backend import random_gradients


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device, operation):
    _sync(device)
    started = time.perf_counter()
    value = operation()
    _sync(device)
    return value, time.perf_counter() - started


def _hardware(device):
    device = torch.device(device)
    details = dict(device=str(device), torch_version=str(torch.__version__),
                   python_version=platform.python_version(), platform=platform.platform(),
                   dtype="float64", cpu_threads=torch.get_num_threads(),
                   cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        details.update(gpu_name=properties.name,
                       gpu_compute_capability=list(torch.cuda.get_device_capability(device)),
                       gpu_total_memory_bytes=properties.total_memory,
                       gpu_multiprocessors=properties.multi_processor_count,
                       matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    return details


def _batch_description(batch, role):
    padded = batch.keys.numel()
    valid = int(batch.lengths.sum())
    return dict(role=role, sequences=batch.size, maximum_records=batch.keys.shape[1],
                valid_records=valid, padded_records=padded,
                padded_to_valid_ratio=padded / valid,
                padding_fraction=1 - valid / padded,
                mean_records=valid / batch.size)


def _inputs(experiment, cfg, population, rng, step, acquisition):
    """Use the same sample sizes, RNGs and frequency counts as the runners."""
    if experiment.dataset == "pairs":
        base, rates, counts, examples = _step_inputs(cfg, experiment.optimizer, step, acquisition, rng)
        return base, rates, counts, None, examples
    base, rates = ((0., acquisition[step - 1]) if step <= 3 else
                   rates_at(cfg, experiment.optimizer, step - 4))
    if experiment.random_sampling == "online":
        batch = sample_stream_batch(cfg.n, cfg.R, experiment.random_data,
                                    rng=rng, device=cfg.device)
        return base, rates, None, batch, batch.size
    counts = None
    size = population.size
    if experiment.optimizer == "sgd" and step > 3:
        size = math.ceil(experiment.random_data.batch_size *
                         (1 + (step - 4) / cfg.offset) ** cfg.batch_growth)
        probabilities = np.full(population.size, 1 / population.size)
        counts = torch.as_tensor(rng.multinomial(size, probabilities),
                                 device=cfg.device, dtype=torch.float64)
    return base, rates, counts, None, size


def _checkpoint_io(model, optimizer, rng, initial_tables, directories):
    """Measure new temporary files only, deleting every I/O probe afterwards."""
    payload, transfer_seconds = _timed(model.cfg.device, lambda: dict(
        model=_cpu_tensors(model.params()), optimizer=_optimizer_state(optimizer),
        initial_tables=initial_tables, rng_state=rng.bit_generator.state,
        purpose="disposable profiling only; not a resumable training checkpoint"))
    rows = []
    for label, directory in directories.items():
        directory = Path(directory).expanduser().resolve() if directory is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fox-profile-io-", dir=directory) as temporary:
            destination = Path(temporary) / "disposable.pt"
            _, write_seconds = _timed("cpu", lambda: _atomic_torch_save(destination, payload))
            size = destination.stat().st_size
            _, read_seconds = _timed("cpu", lambda: torch.load(destination, map_location="cpu", weights_only=False))
        rows.append(dict(label=str(label), directory=str(directory) if directory else "system temporary directory",
                         payload_bytes=size, device_to_cpu_seconds=transfer_seconds,
                         atomic_write_fsync_seconds=write_seconds, cached_read_seconds=read_seconds,
                         note="New temporary model/moment payload; deleted after measurement. History/CSV growth and remote durability are not measured."))
    return rows


def profile_experiment(experiment, out_dir=None, *, warmup=5, steps=20, seed=None,
                       performance=None, diagnostics=True, io_directories=None,
                       trace=False, trace_steps=3, verbose=True):
    """Profile one actual arm on its configured device without changing a run.

    ``performance=None`` measures the original eager, padded implementation.
    Pass a ``PerformanceConfig`` to measure the current training backend. A
    compiled update is timed as one fused phase, not assigned fictional
    gradient/optimizer splits. First compilation belongs to setup/warmup;
    timed throughput is measured after warmup on a fresh disposable model.

    ``io_directories={"local": "/content", "drive": "/content/drive/MyDrive/..."}``
    additionally compares atomic checkpoint writes in disposable subfolders.
    Omit it to skip I/O. ``trace=True`` writes a Chrome/PyTorch profiler trace
    under ``out_dir`` and requires that directory explicitly.

    This runs 3 acquisition + warmup + 2*steps updates (plus trace_steps when
    tracing). Timed phases and throughput use consecutive states. A short
    early-time estimate excludes periodic logging/checkpoints and cannot
    promise identical throughput later in an online or growing-batch run.
    """
    for name, value, minimum in (("warmup", warmup, 0), ("steps", steps, 1), ("trace_steps", trace_steps, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if performance is not None and warmup < 1:
        raise ValueError("A performance backend requires warmup >= 1 to exclude first-use audits and compilation from throughput")
    if trace and out_dir is None:
        raise ValueError("trace=True requires out_dir for the trace artifact")
    if io_directories is not None and (not isinstance(io_directories, dict) or not io_directories):
        raise ValueError("io_directories must be a nonempty label-to-directory mapping or None")
    seed = experiment.seeds[0] if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cfg = replace(experiment.config, seed=seed, gate_mode=experiment.gate_mode)
    _validate(cfg)
    device = cfg.device
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This profile requests CUDA, but this runtime has no CUDA GPU")
    if out_dir is not None:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = Model(cfg)
    optimizer = LogAdam(model.params(), cfg.beta1, cfg.beta2) if experiment.optimizer == "adam" else None
    initial_hash = fingerprint(model.params())
    initial_tables = _cpu_tensors(model.params()[:2]) if io_directories and experiment.dataset == "pairs" else None
    frozen_initial = model.theta.detach().clone()
    _sync(device)
    initialization_seconds = time.perf_counter() - started
    rng = np.random.default_rng(seed + 104729)
    population = None
    data = dict(role="exact finite pair population", ordered_pairs=cfg.n * (cfg.n - 1),
                physical_sequence_rows=4 * cfg.n * (cfg.n - 1) + 2 * cfg.n,
                implementation="closed-form pair loss; physical streams are not materialized per update")
    started = time.perf_counter()
    if experiment.dataset == "random":
        population = sample_stream_batch(cfg.n, cfg.R, experiment.random_data,
                                        rng=np.random.default_rng(seed + 130363),
                                        device=device, batch_size=experiment.dataset_size)
        population.validate(cfg.n, cfg.R)
        role = "fixed training population" if experiment.random_sampling == "fixed" else "fixed diagnostic sample; online batch lengths vary"
        data = _batch_description(population, role)
        data["gradient_layout"] = ("packed valid records" if performance is not None and performance.pack_random
                                   else "rectangular padded records")
    _sync(device)
    dataset_seconds = time.perf_counter() - started
    engine = None
    if performance is not None:
        from .training.performance import ComputeEngine
        engine = ComputeEngine(model, optimizer, performance, population=population,
                               online=experiment.dataset == "random" and experiment.random_sampling == "online")
    fused = bool(performance is not None and performance.compile_updates)

    def gradients(counts, batch):
        if engine is not None:
            return engine.gradients(counts=counts, batch=batch)
        if experiment.dataset == "pairs":
            return model.gradients(counts, validate_counts=False)
        return random_gradients(model, population if batch is None else batch, counts, validate=False)

    acquisition = acquisition_rates(cfg, experiment.optimizer)
    S = 0.
    current_step = 0

    def update():
        nonlocal current_step, S
        current_step += 1
        base, rates, counts, batch, _ = _inputs(experiment, cfg, population, rng, current_step, acquisition)
        if engine is None:
            grads, info = gradients(counts, batch)
            _optimizer_step(model, optimizer, grads, rates, current_step)
        else:
            grads, info = engine.update(rates, current_step, counts=counts, batch=batch)
        S += base
        return grads, info

    def repeat(count):
        for _ in range(count):
            update()

    _, acquisition_seconds = _timed(device, lambda: repeat(3))
    acquired_gap = float(model.gaps()[model.mask].min())
    acquisition_hash = fingerprint(model.params())
    _, warmup_seconds = _timed(device, lambda: repeat(warmup))
    timings = {name: [] for name in ("sampling_and_schedule", "gradient", "optimizer", "fused_update")}
    for _ in range(steps):
        current_step += 1
        inputs, duration = _timed(device, lambda: _inputs(experiment, cfg, population, rng, current_step, acquisition))
        timings["sampling_and_schedule"].append(duration)
        base, rates, counts, batch, _ = inputs
        if fused:
            _, duration = _timed(device, lambda: engine.update(rates, current_step, counts=counts, batch=batch))
            timings["fused_update"].append(duration)
        else:
            (grads, _), duration = _timed(device, lambda: gradients(counts, batch))
            timings["gradient"].append(duration)
            _, duration = _timed(device, lambda: _optimizer_step(model, optimizer, grads, rates, current_step))
            timings["optimizer"].append(duration)
        S += base
    _, steady_seconds = _timed(device, lambda: repeat(steps))
    final_hash = fingerprint(model.params())
    diagnostics_result = None
    if diagnostics:
        def diagnose():
            counts = None
            grads, info = gradients(counts, None)
            health = _check_health(model, grads, info, frozen_initial)
            errors, radii = convergence_probes(
                model, dict(seed=seed, gate_mode=cfg.gate_mode, optimizer=experiment.optimizer),
                current_step, S, acquired_gap, experiment.theta)
            return dict(health_issue=health, probe_rows=len(errors), radius_rows=len(radii),
                        log_loss=float(info["logloss"]))
        diagnostics_result, duration = _timed(device, diagnose)
        diagnostics_result["seconds"] = duration
        diagnostics_result["scope"] = "Full-population gradient, health check and convergence probes; excludes CSV/plot output and pair moment diagnostics."
    io = _checkpoint_io(model, optimizer, rng, initial_tables, io_directories) if io_directories else []
    trace_path = None
    operator_table = None
    if trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
            for _ in range(trace_steps):
                update()
            _sync(device)
        trace_path = out_dir / f"profile_{experiment.name}_seed{seed}.trace.json"
        profiler.export_chrome_trace(str(trace_path))
        sort_by = "self_cuda_time_total" if torch.device(device).type == "cuda" else "self_cpu_time_total"
        operator_table = profiler.key_averages().table(sort_by=sort_by, row_limit=20)
    phase_ms = {name: (1000 * sum(values) / len(values) if values else None) for name, values in timings.items()}
    updates_per_second = steps / steady_seconds
    result = dict(schema_version=1, experiment=experiment.name, seed=seed,
                  config=asdict(cfg), random_settings=asdict(experiment.random_data) if experiment.dataset == "random" else None,
                  hardware=_hardware(device), data=data,
                  backend="original_eager_padded" if performance is None else asdict(performance),
                  engine_stats=clean(engine.describe()) if engine is not None else None,
                  setup_seconds=dict(initialization=initialization_seconds, dataset=dataset_seconds,
                                     acquisition=acquisition_seconds, warmup=warmup_seconds),
                  acquisition=dict(updates=3, minimum_gap=acquired_gap, initial_hash=initial_hash, final_hash=acquisition_hash),
                  phase_mean_ms=phase_ms, phase_timing_method="Wall time with CUDA synchronization before and after every phase; includes host launches and synchronization overhead.",
                  steady_state=dict(measured_updates=steps, seconds=steady_seconds,
                                    updates_per_second=updates_per_second, milliseconds_per_update=1000 / updates_per_second,
                                    requested_updates_per_seed=cfg.steps,
                                    estimated_training_seconds_per_seed=cfg.steps / updates_per_second,
                                    estimated_training_seconds_all_seeds=cfg.steps * len(experiment.seeds) / updates_per_second,
                                    excludes="Compilation, acquisition, health/probe/log/checkpoint overhead; measured early trajectory and one seed only."),
                  diagnostics=diagnostics_result, checkpoint_io=io,
                  measured_final_hash=final_hash, measured_final_step=current_step - (trace_steps if trace else 0),
                  total_disposable_updates=current_step,
                  trace_path=str(trace_path) if trace_path else None, operator_table=operator_table)
    if torch.device(device).type == "cuda":
        result["hardware"]["process_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    result = clean(result)
    if out_dir is not None:
        destination = out_dir / f"profile_{experiment.name}_seed{seed}.json"
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        result["report_path"] = str(destination)
    if verbose:
        print(format_profile(result), flush=True)
    return result


def format_profile(result):
    """Compact notebook-readable timing summary, with measurement limits."""
    hardware = result["hardware"]
    steady = result["steady_state"]
    lines = [f"{result['experiment']} / seed {result['seed']} / {hardware.get('gpu_name', hardware['device'])} / FP64",
             f"Measured: {steady['updates_per_second']:.2f} updates/s ({steady['milliseconds_per_update']:.2f} ms/update).",
             "Synchronized phases: " + ", ".join(f"{name} {value:.2f} ms" for name, value in result["phase_mean_ms"].items() if value is not None)]
    if "padded_to_valid_ratio" in result["data"]:
        data = result["data"]
        lines.append(f"Data: {data['valid_records']:,} valid / {data['padded_records']:,} padded records "
                     f"({data['padding_fraction']:.1%} reference padding). "
                     f"Gradient layout: {data['gradient_layout']}.")
    lines.append(f"Early-rate estimate for {steady['requested_updates_per_seed']:,} updates: "
                 f"{steady['estimated_training_seconds_per_seed']/3600:.2f} hours/seed; "
                 "excludes setup, diagnostics and checkpoint I/O.")
    if result["trace_path"]:
        lines.append(f"Profiler trace: {result['trace_path']}")
    return "\n".join(lines)
