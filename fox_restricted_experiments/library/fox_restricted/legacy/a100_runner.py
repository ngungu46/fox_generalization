"""Long-run restricted experiments with exact optimizer-state continuation.

Checkpoints commit validated states by writing a temporary sibling and replacing
the destination. A process killed between commits resumes the last commit.
The configurable health interval reduces CPU/GPU synchronization; a failure
rolls back every parameter, moment, RNG, clock, and diagnostic row to that commit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

try:
    from .core import Config, Model, LogAdam, acquisition_rates, gradient_step, rates_at, export_dataset
    from .runner import _validate, audit_native, clean, evaluate, fingerprint, snapshot, write_csv
    from .asymptotics import evaluate_asymptotics
except ImportError:
    from core import Config, Model, LogAdam, acquisition_rates, gradient_step, rates_at, export_dataset
    from runner import _validate, audit_native, clean, evaluate, fingerprint, snapshot, write_csv
    from asymptotics import evaluate_asymptotics


SCHEMA_VERSION = 1
TRAINING_SOURCES = ("core.py", "runner.py", "a100_runner.py", "asymptotics.py")
RESUMABLE_STATUSES = {"running", "finite_budget_complete", "time_budget_stop", "paused", "interrupted"}


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(clean(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    write_csv(temporary, rows)
    os.replace(temporary, path)


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in TRAINING_SOURCES}


def _scientific_config(config):
    return {key: value for key, value in asdict(config).items() if key not in ("steps", "max_seconds")}


def _cpu_tensors(values):
    return [value.detach().cpu().clone() for value in values]


def _optimizer_state(optimizer):
    if optimizer is None:
        return None
    return dict(t=optimizer.t, b1=optimizer.b1, b2=optimizer.b2,
                ms=_cpu_tensors(optimizer.ms), ml=_cpu_tensors(optimizer.ml), vl=_cpu_tensors(optimizer.vl))


def _restore(payload, model, optimizer, rng):
    with torch.no_grad():
        for parameter, saved in zip(model.params(), payload["model"]):
            parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
    state = payload["optimizer"]
    if optimizer is not None:
        if state is None or state["b1"] != optimizer.b1 or state["b2"] != optimizer.b2:
            raise ValueError("Checkpoint optimizer does not match the configured Adam")
        optimizer.t = state["t"]
        for name in ("ms", "ml", "vl"):
            setattr(optimizer, name, [saved.to(device=parameter.device, dtype=parameter.dtype).clone()
                                     for saved, parameter in zip(state[name], model.params())])
    elif state is not None:
        raise ValueError("SGD checkpoint unexpectedly contains Adam moments")
    rng.bit_generator.state = payload["rng_state"]


def _check_health(model, gradients, info, frozen_initial):
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.params()):
        return "nonfinite parameter"
    if not math.isfinite(float(info["logloss"])):
        return "nonfinite signed-log objective"
    if not all(bool(torch.isfinite(sign).all()) and
               bool(torch.where(sign != 0, torch.isfinite(logabs), torch.ones_like(sign, dtype=torch.bool)).all())
               for sign, logabs in gradients):
        return "nonfinite active signed-log gradient"
    if float(info["table_log_span"]) > 650:
        return "raw-table gradient aggregation exceeds 650 log units"
    frozen = list(model.frozen_scalar_indices)
    if frozen and not torch.equal(model.theta[frozen], frozen_initial[frozen]):
        return "frozen gate coordinate changed"
    return None


def _gradient_equivalence(eager, compiled, tolerance=1e-7):
    """Compare signs and logarithmic magnitudes without exponent underflow."""
    diagnostics = []
    for (es, el), (cs, cl) in zip(eager, compiled):
        active = es != 0
        signs_equal = bool(torch.equal(es, cs))
        discrepancy = torch.where(active, torch.abs(el - cl), torch.zeros_like(el))
        max_error = float(discrepancy.max())
        diagnostics.append(dict(signs_equal=signs_equal, max_absolute_log_gradient_error=max_error))
    if not all(row["signs_equal"] and math.isfinite(row["max_absolute_log_gradient_error"])
               and row["max_absolute_log_gradient_error"] <= tolerance for row in diagnostics):
        raise RuntimeError(f"Compiled gradient equivalence audit failed: {diagnostics}")
    return diagnostics


class _GradientBackend:
    def __init__(self, model, compiled=False):
        self.model = model
        self.compiled = compiled
        self.function = None
        self.audits = []
        self.compile_seconds = 0.
        self.shapes = set()

    def __call__(self, counts=None):
        if not self.compiled:
            return self.model.gradients(counts, validate_counts=False)
        if self.function is None:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_gradients=True requires torch.compile; use False explicitly for eager execution")
            self.function = torch.compile(lambda counts: self.model.gradients(counts, validate_counts=False),
                                          fullgraph=True, mode="default")
        key = "full_batch" if counts is None else (tuple(counts.shape), str(counts.dtype))
        if key in self.shapes:
            return self.function(counts)
        # Compilation is deliberately opt-in, and must agree at the actual state.
        expected, expected_info = self.model.gradients(counts, validate_counts=False)
        _sync(self.model.cfg.device)
        start = time.monotonic()
        try:
            actual, actual_info = self.function(counts)
            _sync(self.model.cfg.device)
        except Exception as exc:
            raise RuntimeError("Compiled gradients failed. Eager fallback is not automatic; restart with compile_gradients=False in a new run directory.") from exc
        self.compile_seconds += time.monotonic() - start
        audit = _gradient_equivalence(expected, actual)
        loss_error = abs(float(expected_info["logloss"] - actual_info["logloss"]))
        if not loss_error <= 1e-9:
            raise RuntimeError(f"Compiled objective differs from eager objective: log-loss discrepancy {loss_error}")
        self.audits.append(dict(input_kind=str(key), blocks=audit, absolute_log_loss_error=loss_error))
        self.shapes.add(key)
        return actual, actual_info


def _step_inputs(config, kind, step, acquisition, rng):
    if step <= 3:
        return 0., acquisition[step - 1], None, config.n * (config.n - 1)
    tail = step - 4
    base, rates = rates_at(config, kind, tail)
    counts, batch = None, config.n * (config.n - 1)
    if kind == "sgd":
        batch = math.ceil(config.pair_batch * (1 + tail / config.offset) ** config.batch_growth)
        # Aggregated iid sampling is exact, and the RNG state is checkpointed.
        probs = np.full(config.n * (config.n - 1), 1 / (config.n * (config.n - 1)))
        counts = torch.as_tensor(rng.multinomial(batch, probs), device=config.device, dtype=torch.float64)
    return base, rates, counts, batch


def _optimizer_step(model, optimizer, gradients, rates, step):
    if optimizer is None:
        gradient_step(model, gradients, rates)
    else:
        c = model.cfg
        optimizer.step(model.params(), gradients, rates, 3 * math.log(c.sigma) - c.eps_decay * (step - 1))


def _moment_diagnostics(model, optimizer, step, S):
    q, p, u, v, x, w, m, g, h, rho, lrho = [float(value) for value in model.quantities()]
    row = {name: value / S if S > 0 else None
           for name, value in (("q_over_S", q), ("p_over_S", p), ("u_over_S", u),
                               ("v_over_S", v), ("x_over_S", x), ("h_over_S", h), ("w_over_S", w))}
    row.update(m_over_S_squared=m / S ** 2 if S > 0 else None,
               g_over_S_squared=g / S ** 2 if S > 0 else None)
    if optimizer is None or optimizer.t == 0:
        return row
    logeps = 3 * math.log(model.cfg.sigma) - model.cfg.eps_decay * (step - 1)
    corrm = math.log1p(-optimizer.b1 ** optimizer.t)
    corrv = math.log1p(-optimizer.b2 ** optimizer.t)
    log_rms = .5 * (optimizer.vl[-1] - corrv)
    denom = torch.logaddexp(log_rms, torch.full_like(log_rms, logeps))
    direction = torch.where(optimizer.ms[-1] == 0, 0.,
                            optimizer.ms[-1] * torch.exp(optimizer.ml[-1] - corrm - denom))
    row.update(adam_t=optimizer.t, adam_log_epsilon=logeps)
    gradient_signs, gradient_logs = model.gradients(validate_counts=False)[0][-1]
    active = [i for i in range(6) if i not in model.frozen_scalar_indices]
    factors = [i for i in range(4) if i not in model.frozen_scalar_indices]
    row["adam_scalar_epsilon_dominated_fraction"] = float((logeps > log_rms[active]).double().mean())
    row["adam_factor_epsilon_dominated_fraction"] = float((logeps > log_rms[factors]).double().mean())
    for i, name in enumerate(("q", "p", "u", "v", "x", "w")):
        row[f"adam_log_rms_{name}"] = float(log_rms[i])
        row[f"adam_log_epsilon_over_rms_{name}"] = logeps - float(log_rms[i])
        row[f"adam_direction_{name}"] = float(direction[i])
        row[f"adam_log_abs_mhat_over_sqrt_v_{name}"] = (
            float(optimizer.ml[-1][i] - corrm - log_rms[i]) if float(optimizer.ms[-1][i]) != 0 else None)
        row[f"adam_log_grad_over_sqrt_v_{name}"] = (
            float(gradient_logs[i] - log_rms[i]) if float(gradient_signs[i]) != 0 else None)
        row[f"adam_population_gradient_sign_{name}"] = float(gradient_signs[i])
    return row


def _branch_summary(payload, requested_steps):
    state = payload["state"]
    return dict(**payload["tags"], completed_steps=state["step"], requested_steps=requested_steps,
                status=state["status"], reason=state["reason"], initial_hash=payload["initial_hash"],
                final_hash=fingerprint(payload["model"]), acquired_positive_gaps=state["acquired"],
                train_R=payload["config"]["R"], S=state["S"],
                elapsed_seconds=state["elapsed_seconds"], examples=state["examples"],
                updates_per_second=state["step"] / state["elapsed_seconds"] if state["elapsed_seconds"] > 0 else None,
                acquired_reference_gap=state["reference_gap"], resume_count=state["resume_count"],
                failed_at_step=state.get("failed_at_step"), checkpoint_schema=SCHEMA_VERSION,
                compile_seconds=state.get("compile_seconds", 0.))


def _branch_record(payload, requested_steps):
    """Lightweight derived ledger; model/optimizer tensors stay in the .pt file."""
    return dict(summary=_branch_summary(payload, requested_steps),
                history=payload["history"], probabilities=payload["probabilities"],
                errors=payload["errors"], radii=payload["radii"], audits=payload["audits"],
                compile_audits=payload.get("compile_audits", []))


def _aggregate(out, requested_steps, records_cache):
    names = {"history.csv": "history", "lag_probabilities.csv": "probabilities",
             "asymptotic_errors.csv": "errors", "certified_radii.csv": "radii"}
    collected = {name: [] for name in names}
    runs, audits, compile_audits = [], [], []
    for label in sorted(records_cache):
        payload = records_cache[label]
        runs.append({**payload["summary"], "requested_steps": requested_steps})
        for filename, field in names.items():
            collected[filename].extend(payload[field])
        audits.extend(payload["audits"])
        compile_audits.extend(payload.get("compile_audits", []))
    for filename, rows in collected.items():
        _atomic_csv(out / filename, rows)
    _atomic_csv(out / "runs.csv", runs)
    _atomic_json(out / "native_gradient_audits.json", audits)
    _atomic_json(out / "compiled_gradient_audits.json", compile_audits)


def run_large_suite(cfg, out_dir, seeds=(0, 1, 2), gate_modes=("learned", "retrieval_frozen"),
                    optimizers=("sgd", "adam"),
                    eval_lags=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                    prefixes=(0, 64), log_every=1000, checkpoint_every=1000,
                    resume=True, progress=True, fixed_lags=(1, 2, 3, 4, 5, 8, 16, 32, 64),
                    theta_values=(.25, .5, .75), clock_coefficients=(.001, .003, .01),
                    error_targets=(.1, .01, .001), check_every=100,
                    compile_gradients=False, max_updates=None, compute_factory=None):
    """Run or resume a complete matrix without resetting any training state.

    Only ``cfg.steps`` and ``cfg.max_seconds`` may increase on resume. Reporting
    source changes are allowed; core, training, and asymptotic-evaluator sources
    must match their original hashes. ``max_updates`` is an optional operational
    pause budget for this invocation, useful for managed sessions and QA.
    """
    _validate(cfg)
    if any(type(value) is not int or value < 1 for value in (log_every, checkpoint_every, check_every)):
        raise ValueError("Logging, checkpoint, and health intervals must be positive integers")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer or None")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct nonnegative integer seeds")
    if not gate_modes or not set(gate_modes) <= {"learned", "retrieval_frozen", "both_frozen"}:
        raise ValueError("Unknown or empty gate modes")
    if not optimizers or not set(optimizers) <= {"sgd", "adam"}:
        raise ValueError("Use sgd and/or adam")
    if len(set(gate_modes)) != len(gate_modes) or len(set(optimizers)) != len(optimizers):
        raise ValueError("Do not repeat matrix branches")
    if not eval_lags or any(type(lag) is not int or lag < 1 for lag in eval_lags):
        raise ValueError("Use positive integer evaluation lags")
    if not prefixes or any(type(prefix) is not int or prefix < 0 for prefix in prefixes):
        raise ValueError("Use nonnegative integer finite prefixes")
    for coefficients in (theta_values, clock_coefficients, error_targets):
        if not coefficients or any(not math.isfinite(value) or value <= 0 for value in coefficients):
            raise ValueError("Asymptotic coefficients and error targets must be positive and finite")
    out = Path(out_dir).resolve()
    hashes = _source_hashes()
    options = clean(dict(seeds=seeds, gate_modes=gate_modes, optimizers=optimizers,
                         eval_lags=eval_lags, prefixes=prefixes, log_every=log_every,
                         checkpoint_every=checkpoint_every, check_every=check_every,
                         fixed_lags=fixed_lags, theta_values=theta_values,
                         clock_coefficients=clock_coefficients, error_targets=error_targets,
                         compile_gradients=compile_gradients))
    manifest_path = out / "config.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Existing run requires resume=True or a new output directory")
        saved = json.loads(manifest_path.read_text())
        if saved["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema")
        if saved["scientific_config"] != clean(_scientific_config(cfg)) or saved["options"] != options:
            raise ValueError("Scientific configuration or diagnostic protocol changed; start a new run")
        if saved["source_sha256"] != hashes:
            changed = [name for name, digest in hashes.items() if saved["source_sha256"].get(name) != digest]
            raise ValueError(f"Training source changed since checkpoint: {changed}; use the saved source snapshot or start a new run")
        if cfg.steps < saved["config"]["steps"] or cfg.max_seconds < saved["config"]["max_seconds"]:
            raise ValueError("Resume permits extending steps/max_seconds, not shrinking their declared budgets")
        saved["config"] = asdict(cfg)
        saved["budget_history"].append(dict(steps=cfg.steps, max_seconds=cfg.max_seconds))
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Nonempty output directory has no compatible long-run manifest")
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints").mkdir()
        (out / "branch_records").mkdir()
        (out / "source").mkdir()
        for name in TRAINING_SOURCES:
            (out / "source" / name).write_bytes((Path(__file__).resolve().parent / name).read_bytes())
        export_dataset(cfg, out / "dataset")
        saved = dict(schema_version=SCHEMA_VERSION, config=asdict(cfg), options=options,
                     scientific_config=_scientific_config(cfg), source_sha256=hashes,
                     budget_history=[dict(steps=cfg.steps, max_seconds=cfg.max_seconds)],
                     torch_version=torch.__version__, numpy_version=np.__version__,
                     python_version=platform.python_version(), cuda_available=torch.cuda.is_available(),
                     device_name=torch.cuda.get_device_name(cfg.device) if str(cfg.device).startswith("cuda") else "CPU",
                     numerical_backend="float64 signed-log exact gradients and Adam moments; fullgraph compilation opt-in",
                     protocol="three ordinary-answer acquisition steps, practical polynomial continuation, no moment reset",
                     resume_policy="All scientific settings and training source hashes fixed; only steps/max_seconds may increase",
                     recovery_policy="A numerical failure or interrupted update rolls back to the last atomic validated checkpoint",
                     source_policy="core.py, runner.py, a100_runner.py, asymptotics.py pinned; reporting/notebook source excluded",
                     timing_policy="Cumulative branch compute/evaluation time across resumes; checkpoint I/O separately reported",
                     limits=["Finite runs cannot prove convergence as t tends to infinity.",
                             "Polynomial SGD continuation is practical, not the proof's existential envelope schedule.",
                             "Frozen answer theorem covers R=2; larger R is an empirical extension.",
                             "Finite-prefix probes do not exhaust all streams; certified bounds are separate."])
    _atomic_json(manifest_path, saved)
    (out / "branch_records").mkdir(exist_ok=True)
    invocation_updates = 0
    initial_hash_by_seed = {}
    records_cache = {}
    for path in (out / "checkpoints").glob("*.pt"):
        existing = torch.load(path, map_location="cpu", weights_only=False)
        seed = existing["tags"]["seed"]
        if seed in initial_hash_by_seed and initial_hash_by_seed[seed] != existing["initial_hash"]:
            raise ValueError("Saved paired initializations disagree")
        initial_hash_by_seed[seed] = existing["initial_hash"]
        # Once per invocation, repair derived records from authoritative state.
        records_cache[path.stem] = _branch_record(existing, cfg.steps)
        _atomic_json(out / "branch_records" / f"{path.stem}.json", records_cache[path.stem])

    for seed in seeds:
        for gate_mode in gate_modes:
            for kind in optimizers:
                c = replace(cfg, seed=seed, gate_mode=gate_mode)
                tags = dict(seed=seed, gate_mode=gate_mode, optimizer=kind)
                label = f"seed{seed}_{gate_mode}_{kind}"
                checkpoint_path = out / "checkpoints" / f"{label}.pt"
                model = Model(c)
                initial_hash = fingerprint(model.params())
                initial_hash_by_seed.setdefault(seed, initial_hash)
                if initial_hash_by_seed[seed] != initial_hash:
                    raise AssertionError("Initial tensors differ between matched branches")
                optimizer = LogAdam(model.params(), c.beta1, c.beta2) if kind == "adam" else None
                rng = np.random.default_rng(seed + 104729)
                acquisition = acquisition_rates(c, kind)
                if checkpoint_path.exists():
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if payload["initial_hash"] != initial_hash or payload["source_sha256"] != hashes:
                        raise ValueError("Checkpoint provenance does not match this branch")
                    if payload["state"]["status"] not in RESUMABLE_STATUSES:
                        if progress:
                            print(f"{label}: retained {payload['state']['status']} (not resumed)", flush=True)
                        continue
                    if payload["state"]["step"] >= c.steps:
                        if progress:
                            print(f"{label}: already completed {payload['state']['step']} steps", flush=True)
                        continue
                    if payload["state"]["elapsed_seconds"] >= c.max_seconds:
                        if progress:
                            print(f"{label}: cumulative time budget exhausted; increase max_seconds to continue", flush=True)
                        continue
                    _restore(payload, model, optimizer, rng)
                    payload["state"]["resume_count"] += 1
                    payload["state"]["status"], payload["state"]["reason"] = "running", ""
                else:
                    payload = dict(schema_version=SCHEMA_VERSION, config=asdict(c), tags=tags,
                                   source_sha256=hashes, initial_hash=initial_hash,
                                   initial_tables=_cpu_tensors(model.params()[:2]),
                                   frozen_initial=model.theta.detach().cpu().clone(),
                                   history=[], probabilities=[], errors=[], radii=[], audits=[], compile_audits=[],
                                   state=dict(step=0, S=0., examples=0, elapsed_seconds=0., status="running", reason="",
                                              acquired=None, reference_gap=None, resume_count=0, compile_seconds=0.,
                                              checkpoint_write_seconds=0.))
                    audit = audit_native(model)
                    payload["audits"].append(dict(**tags, step=0, **audit))
                    if not audit["passed"]:
                        raise AssertionError(f"Initial ordinary-loss audit failed: {label}: {audit}")
                state = payload["state"]
                initial_tables = [value.to(c.device) for value in payload["initial_tables"]]
                frozen_initial = payload["frozen_initial"].to(c.device)
                elapsed_before = state["elapsed_seconds"]
                session_start = time.monotonic()
                backend = _GradientBackend(model, compiled=compile_gradients)
                compute = compute_factory(model, optimizer) if compute_factory is not None else None
                if compute is not None:
                    payload.setdefault("performance_sessions", []).append({})
                paused = False

                def elapsed():
                    return elapsed_before + time.monotonic() - session_start

                def record():
                    # Replace rows for this exact step if a budget boundary is revisited.
                    for name in ("history", "probabilities", "errors", "radii"):
                        payload[name] = [row for row in payload[name] if row["step"] != state["step"]]
                    row = snapshot(model, tags, state["step"], state["S"], initial_tables, state["examples"], elapsed())
                    row.update(_moment_diagnostics(model, optimizer, state["step"], state["S"]))
                    row.update(acquired_reference_gap=state["reference_gap"],
                               updates_per_second=state["step"] / elapsed() if elapsed() > 0 else None,
                               compile_gradients=compile_gradients)
                    payload["history"].append(row)
                    payload["probabilities"].extend(evaluate(model, tags, state["step"], state["S"], eval_lags, prefixes))
                    errors, radii = evaluate_asymptotics(model, tags, state["step"], state["S"],
                        reference_gap=state["reference_gap"], fixed_lags=fixed_lags, theta_values=theta_values,
                        clock_coefficients=clock_coefficients, error_targets=error_targets)
                    payload["errors"].extend(errors)
                    payload["radii"].extend(radii)
                    if progress:
                        print(f"{label}: {state['step']}/{c.steps}, loss={row['loss']:.4g}, "
                              f"S={state['S']:.4g}, content horizon={row['content_horizon']:.3g}", flush=True)

                def commit():
                    state["elapsed_seconds"] = elapsed()
                    state["compile_seconds"] += backend.compile_seconds
                    backend.compile_seconds = 0.
                    payload["compile_audits"].extend(dict(**tags, step=state["step"], **row) for row in backend.audits)
                    backend.audits.clear()
                    if compute is not None:
                        payload["performance_sessions"][-1] = dict(last_step=state["step"], **compute.describe())
                    payload.update(model=_cpu_tensors(model.params()), optimizer=_optimizer_state(optimizer),
                                   rng_state=rng.bit_generator.state, config=asdict(c))
                    start_write = time.monotonic()
                    _atomic_torch_save(checkpoint_path, payload)
                    state["checkpoint_write_seconds"] += time.monotonic() - start_write
                    records_cache[label] = _branch_record(payload, c.steps)
                    _atomic_json(out / "branch_records" / f"{label}.json", records_cache[label])
                    _aggregate(out, c.steps, records_cache)

                def rollback(reason, failed_step, status="numerical_stop"):
                    nonlocal payload, state
                    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    _restore(payload, model, optimizer, rng)
                    state = payload["state"]
                    state.update(status=status, reason=reason, failed_at_step=failed_step)
                    # The recovered model is the last committed validated state.
                    payload["config"] = asdict(c)
                    _atomic_torch_save(checkpoint_path, payload)
                    records_cache[label] = _branch_record(payload, c.steps)
                    _atomic_json(out / "branch_records" / f"{label}.json", records_cache[label])
                    _aggregate(out, c.steps, records_cache)

                if state["step"] == 0:
                    record()
                    commit()
                try:
                    for step in range(state["step"] + 1, c.steps + 1):
                        if elapsed() >= c.max_seconds:
                            gradients, info = model.gradients(validate_counts=False)
                            reason = _check_health(model, gradients, info, frozen_initial)
                            if reason:
                                rollback(reason, step - 1)
                            else:
                                state.update(status="time_budget_stop", reason="cumulative per-branch time budget")
                                record()
                                commit()
                            break
                        base, rates, counts, batch = _step_inputs(c, kind, step, acquisition, rng)
                        # Keep acquisition native; compile at the acquired state.
                        if compute is None:
                            gradients, info = (model.gradients(counts, validate_counts=False) if step <= 3 else backend(counts))
                            _optimizer_step(model, optimizer, gradients, rates, step)
                        else:
                            gradients, info = compute.update(rates, step, counts=counts)
                        state["step"], state["S"] = step, state["S"] + base
                        state["examples"] += 4 * batch + 2 * c.n
                        invocation_updates += 1
                        should_pause = max_updates is not None and invocation_updates >= max_updates
                        is_checkpoint = step <= 3 or step % checkpoint_every == 0 or step == c.steps or should_pause
                        is_log = step == 3 or step % log_every == 0 or step == c.steps or should_pause
                        if step % check_every == 0 or is_checkpoint or is_log:
                            health_gradients, health_info = model.gradients(validate_counts=False)
                            reason = _check_health(model, health_gradients, health_info, frozen_initial)
                            if reason:
                                rollback(reason, step)
                                break
                        if step == 3:
                            state["reference_gap"] = float(model.gaps()[model.mask].min())
                            state["acquired"] = state["reference_gap"] > 0 and float(model.theta[-1]) > 0
                            audit = audit_native(model)
                            payload["audits"].append(dict(**tags, step=3, **audit))
                            if not audit["passed"]:
                                rollback(f"Acquisition native audit failed: {audit}", step)
                                break
                            if not state["acquired"]:
                                state.update(status="acquisition_failed", reason="three ordinary updates did not acquire every positive gap and decoder")
                                record()
                                commit()
                                break
                        if step == c.steps:
                            state.update(status="finite_budget_complete", reason="")
                        elif should_pause:
                            state.update(status="paused", reason="max_updates operational pause")
                            paused = True
                        if is_log:
                            record()
                        if is_checkpoint:
                            commit()
                        if should_pause:
                            break
                except KeyboardInterrupt:
                    rollback("Interrupted between atomic checkpoints; restored committed state", state["step"], "interrupted")
                    raise
                except Exception:
                    # Do not commit a possibly half-updated state after an exception.
                    rollback("Exception between atomic checkpoints; restored committed state", state["step"], "interrupted")
                    raise
                if progress:
                    print(f"{label}: {state['status']} at committed step {state['step']}", flush=True)
                if paused or (max_updates is not None and invocation_updates >= max_updates):
                    return str(out)
    _aggregate(out, cfg.steps, records_cache)
    return str(out)


def benchmark_config(cfg, steps=30, warmup=5, num_seeds=3, compile_gradients=False,
                     gate_modes=("learned", "retrieval_frozen"), optimizers=("sgd", "adam")):
    """Measure local hardware throughput using fresh models, never run state.

    Excludes compilation, warmup, evaluation, and checkpoint I/O from steady
    training timing; reports compilation separately. Matrix estimates are
    estimates, not promises, and exclude diagnostic and storage overhead.
    """
    _validate(cfg)
    if type(steps) is not int or steps < 1 or type(warmup) is not int or warmup < 0 or num_seeds < 1:
        raise ValueError("Use positive benchmark steps/seed count and nonnegative warmup")
    rows = []
    for gate in gate_modes:
        for kind in optimizers:
            c = replace(cfg, seed=0, gate_mode=gate)
            model = Model(c)
            optimizer = LogAdam(model.params(), c.beta1, c.beta2) if kind == "adam" else None
            rng = np.random.default_rng(104729)
            acquisition = acquisition_rates(c, kind)
            start_acquisition = time.monotonic()
            for step in range(1, 4):
                _, rates, counts, _ = _step_inputs(c, kind, step, acquisition, rng)
                _optimizer_step(model, optimizer, model.gradients(counts, validate_counts=False)[0], rates, step)
            _sync(c.device)
            acquisition_seconds = time.monotonic() - start_acquisition
            native = audit_native(model)
            if not native["passed"]:
                raise AssertionError(f"Benchmark acquired native audit failed: {native}")
            acquired_gap = float(model.gaps()[model.mask].min())
            backend = _GradientBackend(model, compiled=compile_gradients)
            # At least one untimed call compiles the exact backend shape.
            for step in range(4, 4 + max(1, warmup)):
                _, rates, counts, _ = _step_inputs(c, kind, step, acquisition, rng)
                _optimizer_step(model, optimizer, backend(counts)[0], rates, step)
            _sync(c.device)
            if str(c.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(c.device)
            start = time.monotonic()
            for step in range(4 + max(1, warmup), 4 + max(1, warmup) + steps):
                _, rates, counts, _ = _step_inputs(c, kind, step, acquisition, rng)
                _optimizer_step(model, optimizer, backend(counts)[0], rates, step)
            _sync(c.device)
            seconds = time.monotonic() - start
            gradients, info = model.gradients(validate_counts=False)
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.params()):
                raise FloatingPointError("Benchmark produced nonfinite parameters")
            per_step = seconds / steps
            rows.append(dict(gate_mode=gate, optimizer=kind, measured_steps=steps,
                             measured_training_seconds=seconds, seconds_per_step=per_step,
                             updates_per_second=1 / per_step, acquisition_seconds=acquisition_seconds,
                             compile_seconds=backend.compile_seconds, compiled_audits=backend.audits,
                             acquired_minimum_gap=acquired_gap, acquired_positive_gaps=acquired_gap > 0,
                             estimated_branch_seconds=acquisition_seconds + max(0, cfg.steps - 3) * per_step,
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(c.device) if str(c.device).startswith("cuda") else None))
    estimated = num_seeds * sum(row["estimated_branch_seconds"] for row in rows)
    return clean(dict(device=cfg.device,
                      device_name=torch.cuda.get_device_name(cfg.device) if str(cfg.device).startswith("cuda") else "CPU",
                      n=cfg.n, d=cfg.d, R=cfg.R, requested_steps=cfg.steps, num_seeds=num_seeds,
                      compile_gradients=compile_gradients, rows=rows,
                      estimated_matrix_seconds=estimated, estimated_matrix_hours=estimated / 3600,
                      caveat="Measured on this runtime; estimates exclude evaluations/checkpoints and long-run changes in sampling batch cost."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--d", type=int, default=4096)
    parser.add_argument("--R", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=86400.)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--compile-gradients", action="store_true")
    parser.add_argument("--moment-preset", choices=("paper_short_memory", "standard_memory"), default="paper_short_memory")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    beta1, beta2 = (.1, .1) if args.moment_preset == "paper_short_memory" else (.9, .999)
    cfg = Config(n=args.n, d=args.d, R=args.R, steps=args.steps, device=args.device,
                 max_seconds=args.max_seconds, beta1=beta1, beta2=beta2)
    if args.benchmark_only:
        print(json.dumps(benchmark_config(cfg, num_seeds=len(args.seeds), compile_gradients=args.compile_gradients), indent=2))
    else:
        run_large_suite(cfg, args.out, seeds=tuple(args.seeds), compile_gradients=args.compile_gradients)


if __name__ == "__main__":
    main()
