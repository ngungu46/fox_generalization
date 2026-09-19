"""Small Colab helpers and measured benchmarks, separate from scientific code."""
from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np
import torch

from .legacy.core import Model, LogAdam, acquisition_rates, rates_at
from .legacy.a100_runner import _optimizer_step, _step_inputs, _sync, _check_health
from .data.random_streams import sample_stream_batch
from .training.random_training import _draw_gradients, _draw_inputs


def check_runtime(device="cuda", *, require_a100=True):
    """Check actual hardware and retain the numerical settings of the theory experiment."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Select a CUDA GPU runtime in Colab; use SMOKE=True for the CPU check.")
        name = torch.cuda.get_device_name(device)
        if require_a100 and "A100" not in name.upper():
            raise RuntimeError(f"Received {name}. Select A100 or explicitly set require_a100=False.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        name = "CPU"
        torch.set_num_threads(1)
    return dict(device=device, name=name, torch_version=torch.__version__,
                parameter_dtype="float64", gradient_storage="signed-log", mixed_precision=False)


def prepare_output(run_tag, *, use_drive=True, local_root="fox_results"):
    """Mount Drive in Colab and return a persistent output directory."""
    if not run_tag or Path(run_tag).name != run_tag or run_tag in (".", ".."):
        raise ValueError("run_tag must be a single nonempty directory name")
    if use_drive:
        from google.colab import drive
        drive.mount("/content/drive")
        root = Path("/content/drive/MyDrive/fox_restricted_runs")
    else:
        root = Path(local_root).expanduser().resolve()
    out = root / run_tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def benchmark(experiments, *, measured_steps=10, warmup=3, performance=None):
    """Time each requested objective at its real shape on the current hardware.

    All benchmark states are discarded. These are initial throughput estimates,
    excluding dataset export, reports, Drive writes, and later state changes.
    """
    if type(measured_steps) is not int or measured_steps < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("Use positive measured_steps and nonnegative warmup")
    rows = []
    values = experiments.values() if isinstance(experiments, dict) else experiments
    for experiment in values:
        c = replace(experiment.config, seed=experiment.seeds[0], gate_mode=experiment.gate_mode)
        model = Model(c)
        adam = LogAdam(model.params(), c.beta1, c.beta2) if experiment.optimizer == "adam" else None
        rng = np.random.default_rng(c.seed + 104729)
        population = None
        if experiment.dataset == "random":
            population = sample_stream_batch(c.n, c.R, experiment.random_data,
                rng=np.random.default_rng(c.seed + 130363), device=c.device, batch_size=experiment.dataset_size)
        acquisition = acquisition_rates(c, experiment.optimizer)
        frozen = model.theta.clone()
        compute = None
        if performance is not None:
            from .training.performance import ComputeEngine
            compute = ComputeEngine(model, adam, performance, population,
                                    online=experiment.dataset == "random" and experiment.random_sampling == "online")

        def update(step):
            if experiment.dataset == "pairs":
                _, rates, counts, _ = _step_inputs(c, experiment.optimizer, step, acquisition, rng)
                if compute is not None:
                    return compute.update(rates, step, counts=counts)
                gradients, info = model.gradients(counts, validate_counts=False)
            else:
                _, rates = (0., acquisition[step - 1]) if step <= 3 else rates_at(c, experiment.optimizer, step - 4)
                if compute is not None:
                    counts, batch, _ = _draw_inputs(model, experiment, population, rng, step)
                    return compute.update(rates, step, counts=counts, batch=batch)
                gradients, info, _ = _draw_gradients(model, experiment, population, rng, step)
            _optimizer_step(model, adam, gradients, rates, step)
            return gradients, info

        if compute is not None and performance.compile_updates:
            print(f"{experiment.name}: compiling and auditing the update; first use can take several minutes", flush=True)
        warmup = max(1, warmup) if compute is not None and performance.compile_updates else warmup
        for step in range(1, 4 + warmup):
            update(step)
        _sync(c.device)
        started = time.monotonic()
        for step in range(4 + warmup, 4 + warmup + measured_steps):
            gradients, info = update(step)
        _sync(c.device)
        seconds = time.monotonic() - started
        issue = _check_health(model, gradients, info, frozen)
        if issue:
            raise FloatingPointError(f"Benchmark {experiment.name}: {issue}")
        per_step = seconds / measured_steps
        row = dict(experiment=experiment.name, seconds_per_update=per_step,
                   estimated_hours_all_seeds=per_step*c.steps*len(experiment.seeds)/3600,
                   measured_updates=measured_steps, device=c.device,
                   acquired_gap_positive=bool((model.gaps()[model.mask] > 0).all()))
        if compute is not None:
            row["performance"] = compute.describe()
        rows.append(row)
        print(f"{experiment.name}: {per_step:.4g} s/update; "
              f"rough estimate {row['estimated_hours_all_seeds']:.2f} hours for all seeds", flush=True)
    print("Estimates exclude evaluation/checkpoint I/O and later changes in runtime.", flush=True)
    return rows
