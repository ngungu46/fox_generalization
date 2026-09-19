"""Resumable training on genuinely random structured streams.

The fixed-data default samples the population once, uses full-dataset Adam,
and uses iid frequency-weighted SGD batches. Online sampling is explicit.
The original deterministic acquisition rates are applied to this objective,
without adding calibration or overwrite examples from the pair experiment.
"""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from ..legacy.core import Model, LogAdam, acquisition_rates, rates_at
from ..legacy.runner import clean, fingerprint, _validate
from ..legacy.a100_runner import (
    _atomic_json, _atomic_csv, _atomic_torch_save, _cpu_tensors,
    _optimizer_state, _restore, _optimizer_step, _check_health,
)
from ..data.random_streams import RandomBatch, sample_stream_batch
from ..evaluation.probes import convergence_probes
from .random_backend import random_gradients, audit_random_gradients


def _sources():
    root = Path(__file__).resolve().parents[1]
    names = ("legacy/core.py", "legacy/runner.py", "legacy/a100_runner.py", "legacy/asymptotics.py",
             "data/random_streams.py", "training/random_backend.py", "training/random_training.py",
             "evaluation/probes.py", "study.py")
    return root, {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _manifest(experiment):
    settings = asdict(experiment)
    # Only declared computational budgets may grow on resumption.
    for field in ("steps", "max_seconds"):
        settings["config"].pop(field)
    return clean(settings)


def _dataset(experiment, cfg, out):
    """Generate once from an independent RNG, then retain exact tensors."""
    path = out / "dataset" / f"seed{cfg.seed}.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        rng = np.random.default_rng(cfg.seed + 130363)
        batch = sample_stream_batch(cfg.n, cfg.R, experiment.random_data, rng=rng,
                                    batch_size=experiment.dataset_size)
        state = batch.state_dict()
        _atomic_torch_save(path, state)
    batch = RandomBatch.from_state_dict(state, device=cfg.device).validate(cfg.n, cfg.R)
    digest = fingerprint([state[key] for key in sorted(state)])
    description = dict(seed=cfg.seed, fingerprint=digest, size=batch.size,
                       mean_records=float(batch.lengths.double().mean()),
                       maximum_records=int(batch.lengths.max()),
                       lag_counts={str(lag): int((batch.lags == lag).sum()) for lag in range(1, cfg.R + 1)},
                       settings=asdict(experiment.random_data),
                       role="training_population" if experiment.random_sampling == "fixed" else "fixed_diagnostic_sample")
    _atomic_json(path.with_suffix(".json"), description)
    return batch, digest


def _draw_gradients(model, experiment, population, rng, step):
    cfg = model.cfg
    if experiment.random_sampling == "online":
        batch = sample_stream_batch(cfg.n, cfg.R, experiment.random_data, rng=rng, device=cfg.device)
        return (*random_gradients(model, batch, validate=False), batch.size)
    counts = None
    size = population.size
    if experiment.optimizer == "sgd" and step > 3:
        size = math.ceil(experiment.random_data.batch_size *
                         (1 + (step - 4) / cfg.offset) ** cfg.batch_growth)
        probabilities = np.full(population.size, 1 / population.size)
        counts = torch.as_tensor(rng.multinomial(size, probabilities), device=cfg.device, dtype=torch.float64)
    return (*random_gradients(model, population, counts, validate=False), size)


def _diagnostics(model, tags, step, S, examples, elapsed, reference_gap, info):
    q, p, u, v, x, w, m, g, h, rho, lrho = [float(v) for v in model.quantities()]
    gaps = model.gaps()[model.mask]
    norms = max(float(table.norm(dim=1).max()) for table in (model.Q, model.K))
    return dict(tags, step=step, S=S, examples=examples, elapsed_seconds=elapsed,
                loss=math.exp(float(info["logloss"])), log_loss=float(info["logloss"]),
                q=q, p=p, u=u, v=v, x=x, w=w, m=m, g=g, h=h, rho=rho, log_rho=lrho,
                m_rho=m*rho, delta_min=float(gaps.min()), delta_max=float(gaps.max()),
                row_norm_max=norms, content_horizon=1 + m*float(gaps.min())/h,
                acquired_reference_gap=reference_gap, train_R=model.cfg.R,
                mean_correct_probability=float(info["mean_correct_probability"]),
                diagnostic_objective="fixed_random_population_or_online_diagnostic_sample")


def _write_tables(out, records, requested_steps):
    for filename, key in (("history.csv", "history"), ("asymptotic_errors.csv", "errors"),
                          ("certified_radii.csv", "radii")):
        _atomic_csv(out / filename, [row for record in records.values() for row in record[key]])
    summaries = []
    for record in records.values():
        state = record["state"]
        summaries.append(dict(record["tags"], requested_steps=requested_steps,
                              completed_steps=state["step"], status=state["status"], reason=state["reason"],
                              S=state["S"], acquired_positive_gaps=state["acquired"],
                              acquired_reference_gap=state["reference_gap"],
                              initial_hash=record["initial_hash"], final_hash=record["final_hash"],
                              dataset_hash=record["dataset_hash"], resume_count=state["resume_count"]))
    _atomic_csv(out / "runs.csv", summaries)


def run_random_experiment(experiment, out, *, resume=True, progress=True, max_updates=None):
    """Run one random-data arm across its seeds, committing validated state."""
    cfg = experiment.config
    _validate(cfg)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive integer or None")
    out = Path(out)
    source_root, hashes = _sources()
    settings = _manifest(experiment)
    manifest_path = out / "config.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Existing experiment requires resume=True")
        manifest = json.loads(manifest_path.read_text())
        if manifest["settings"] != settings or manifest["source_sha256"] != hashes:
            raise ValueError("Scientific settings or training sources changed; use a new run folder")
        if cfg.steps < manifest["config"]["steps"] or cfg.max_seconds < manifest["config"]["max_seconds"]:
            raise ValueError("Resume may extend, but cannot shrink, declared budgets")
        manifest["config"] = asdict(cfg)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Nonempty folder has no compatible experiment manifest")
        for directory in (out, out / "checkpoints", out / "dataset", out / "source"):
            directory.mkdir(parents=True, exist_ok=True)
        for name in hashes:
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source_root / name).read_bytes())
        manifest = dict(schema_version=1, config=asdict(cfg), settings=settings, source_sha256=hashes,
                        torch_version=torch.__version__, device=cfg.device,
                        dataset="random mixed-key mixed-value structured streams, latest target lag <= R",
                        acquisition="same deterministic three-update rates, applied only to random-data loss",
                        objective="ordinary binary answer logistic loss; no added calibration or pair templates",
                        scope="Random-data behavior is an extension experiment; the finite-pair cutoff theorem does not apply.")
    _atomic_json(manifest_path, manifest)
    records = {}
    for path in (out / "checkpoints").glob("*.pt"):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        records[path.stem] = {key: saved[key] for key in
                             ("tags", "state", "history", "errors", "radii", "initial_hash", "dataset_hash", "final_hash")}
    updates = 0
    for seed in experiment.seeds:
        c = replace(cfg, seed=seed, gate_mode=experiment.gate_mode)
        model = Model(c)
        adam = LogAdam(model.params(), c.beta1, c.beta2) if experiment.optimizer == "adam" else None
        rng = np.random.default_rng(seed + 104729)
        population, dataset_hash = _dataset(experiment, c, out)
        audit_batch = population.select(torch.arange(min(8, population.size), device=c.device))
        tags = dict(seed=seed, gate_mode=c.gate_mode, optimizer=experiment.optimizer, dataset="random")
        checkpoint = out / "checkpoints" / f"seed{seed}.pt"
        initial_hash = fingerprint(model.params())
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["initial_hash"] != initial_hash or payload["dataset_hash"] != dataset_hash:
                raise ValueError("Initial parameters or stored random dataset changed")
            state = payload["state"]
            if state["status"] == "numerical_stop" or state["step"] >= cfg.steps:
                if progress:
                    print(f"{experiment.name} seed {seed}: retained {state['status']} at {state['step']}", flush=True)
                continue
            if state["elapsed_seconds"] >= c.max_seconds:
                continue
            _restore(payload, model, adam, rng)
            state["resume_count"] += 1
            state.update(status="running", reason="")
        else:
            payload = dict(tags=tags, initial_hash=initial_hash, dataset_hash=dataset_hash,
                           frozen_initial=model.theta.detach().cpu().clone(),
                           history=[], errors=[], radii=[], audits=[],
                           state=dict(step=0, S=0., examples=0, elapsed_seconds=0.,
                                      status="running", reason="", acquired=None, reference_gap=None, resume_count=0))
            state = payload["state"]
            audit = audit_random_gradients(model, audit_batch)
            payload["audits"].append(dict(step=0, **audit))
            if not audit["passed"]:
                raise AssertionError(f"Random objective native audit failed: {audit}")
        acquisition = acquisition_rates(c, experiment.optimizer)
        frozen_initial = payload["frozen_initial"].to(c.device)
        before, started = state["elapsed_seconds"], time.monotonic()

        def elapsed():
            return before + time.monotonic() - started

        def record(info=None):
            if info is None:
                _, info = random_gradients(model, population, validate=False)
            for key in ("history", "errors", "radii"):
                payload[key] = [row for row in payload[key] if row["step"] != state["step"]]
            payload["history"].append(_diagnostics(model, tags, state["step"], state["S"],
                                                   state["examples"], elapsed(), state["reference_gap"], info))
            errors, radii = convergence_probes(model, tags, state["step"], state["S"],
                                               state["reference_gap"], experiment.theta)
            payload["errors"].extend(errors)
            payload["radii"].extend(radii)
            if progress:
                print(f"{experiment.name} seed {seed}: {state['step']}/{c.steps}, "
                      f"loss={math.exp(float(info['logloss'])):.5g}, "
                      f"gap={float(model.gaps()[model.mask].min()):.4g}", flush=True)

        def commit():
            state["elapsed_seconds"] = elapsed()
            payload.update(model=_cpu_tensors(model.params()), optimizer=_optimizer_state(adam),
                           rng_state=rng.bit_generator.state, final_hash=fingerprint(model.params()))
            _atomic_torch_save(checkpoint, payload)
            records[checkpoint.stem] = {key: payload[key] for key in
                                      ("tags", "state", "history", "errors", "radii", "initial_hash", "dataset_hash", "final_hash")}
            _write_tables(out, records, cfg.steps)

        def rollback(reason, attempted_step, status="numerical_stop"):
            nonlocal payload, state
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            _restore(payload, model, adam, rng)
            state = payload["state"]
            state.update(status=status, reason=reason, failed_at_step=attempted_step)
            _atomic_torch_save(checkpoint, payload)
            records[checkpoint.stem] = {key: payload[key] for key in
                                      ("tags", "state", "history", "errors", "radii", "initial_hash", "dataset_hash", "final_hash")}
            _write_tables(out, records, cfg.steps)

        if state["step"] == 0:
            record()
            commit()
        try:
            for step in range(state["step"] + 1, c.steps + 1):
                if elapsed() >= c.max_seconds:
                    gradients, info = random_gradients(model, population, validate=False)
                    issue = _check_health(model, gradients, info, frozen_initial)
                    if issue:
                        rollback(issue, step - 1)
                    else:
                        state.update(status="time_budget_stop", reason="cumulative per-seed time limit")
                        record(info)
                        commit()
                    break
                base, rates = (0., acquisition[step - 1]) if step <= 3 else rates_at(c, experiment.optimizer, step - 4)
                gradients, info, count = _draw_gradients(model, experiment, population, rng, step)
                _optimizer_step(model, adam, gradients, rates, step)
                state.update(step=step, S=state["S"] + base, examples=state["examples"] + count)
                updates += 1
                pause = max_updates is not None and updates >= max_updates
                log = step == 3 or step % experiment.log_every == 0 or step == c.steps or pause
                save = step <= 3 or step % experiment.checkpoint_every == 0 or step == c.steps or pause
                if log or save or step % experiment.check_every == 0:
                    gradients, info = random_gradients(model, population, validate=False)
                    issue = _check_health(model, gradients, info, frozen_initial)
                    if issue:
                        rollback(issue, step)
                        break
                if step == 3:
                    state["reference_gap"] = float(model.gaps()[model.mask].min())
                    state["acquired"] = state["reference_gap"] > 0 and float(model.theta[-1]) > 0
                    audit = audit_random_gradients(model, audit_batch)
                    payload["audits"].append(dict(step=step, **audit))
                    if not audit["passed"]:
                        rollback(f"Random objective acquired audit failed: {audit}", step)
                        break
                    # An empirical extension must retain unsuccessful acquisition
                    # rather than inventing a positive gap or replacing its data.
                if step == c.steps:
                    state.update(status="finite_budget_complete", reason="")
                elif pause:
                    state.update(status="paused", reason="max_updates operational pause")
                if log:
                    record(info)
                if save:
                    commit()
                if pause:
                    return
        except KeyboardInterrupt:
            rollback("Interrupted; restored last atomic checkpoint", state["step"], "interrupted")
            raise
        except Exception:
            rollback("Exception; restored last atomic checkpoint", state["step"], "interrupted")
            raise
    _write_tables(out, records, cfg.steps)
