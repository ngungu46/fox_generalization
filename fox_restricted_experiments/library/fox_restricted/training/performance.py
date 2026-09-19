"""Operational compute choices; data, objective, rates and precision stay fixed."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time

from ..legacy.a100_runner import _gradient_equivalence, _optimizer_step, _sync


@dataclass(frozen=True)
class PerformanceConfig:
    """Use packing first; compilation is explicit and audited on this runtime."""
    pack_random: bool = True
    compile_updates: bool = False
    compile_backend: str = "inductor"

    def __post_init__(self):
        if type(self.pack_random) is not bool or type(self.compile_updates) is not bool:
            raise ValueError("pack_random and compile_updates must be boolean")
        if self.compile_backend not in ("inductor", "eager"):
            raise ValueError("compile_backend must be inductor (execution) or eager (capture tests)")


def implementation_hashes():
    root = Path(__file__).resolve().parent
    names = ("performance.py", "acceleration.py", "packed_random.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def save_performance_source(path):
    """Retain the exact accelerated implementation alongside its checkpoint ledger."""
    from ..legacy.a100_runner import _atomic_json
    hashes = implementation_hashes()
    digest = hashlib.sha256(str(sorted(hashes.items())).encode()).hexdigest()[:16]
    target = Path(path) / "performance_source" / digest
    target.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    for name in hashes:
        (target / name).write_bytes((root / name).read_bytes())
    _atomic_json(target / "sha256.json", hashes)


class ComputeEngine:
    """One persistent compute object per model, with optional fused updates.

    Static random populations are packed once. A packed-versus-reference audit
    runs at the acquired/restored state before its first accelerated update.
    Online populations may be packed per batch but are not compiled: changing
    sampled shapes would otherwise cause repeated expensive compilation.
    """
    def __init__(self, model, optimizer, performance=None, population=None, *, online=False):
        self.model, self.optimizer = model, optimizer
        self.performance = performance or PerformanceConfig(pack_random=False)
        self.population, self.online = population, online
        if online and self.performance.compile_updates:
            raise ValueError("Compiled updates require a fixed dataset shape; use compile_updates=False for online sampling")
        self.packed = None
        self.pack_seconds = 0.
        self.packing_audit = None
        self.compiled = None
        if population is not None and self.performance.pack_random:
            from .packed_random import pack_random_batch
            started = time.monotonic()
            self.packed = pack_random_batch(population, model.cfg.n)
            _sync(model.cfg.device)
            self.pack_seconds = time.monotonic() - started

    def _gradient_function(self, counts=None, batch=None):
        if self.population is None:
            return self.model.gradients(counts, validate_counts=False)
        if self.performance.pack_random:
            from .packed_random import packed_random_gradients, pack_random_batch
            packed = self.packed if batch is None else pack_random_batch(batch, self.model.cfg.n)
            return packed_random_gradients(self.model, packed, counts, validate=False)
        from .random_backend import random_gradients
        return random_gradients(self.model, self.population if batch is None else batch, counts, validate=False)

    def gradients(self, counts=None, batch=None):
        return self._gradient_function(counts, batch)

    def _audit_packing(self, counts, batch):
        if self.packed is None or self.packing_audit is not None:
            return
        from .random_backend import random_gradients
        expected, expected_info = random_gradients(
            self.model, self.population if batch is None else batch, counts, validate=False)
        actual, actual_info = self._gradient_function(counts, batch)
        blocks = _gradient_equivalence(expected, actual)
        discrepancy = abs(float(expected_info["logloss"] - actual_info["logloss"]))
        if not math.isfinite(discrepancy) or discrepancy > 1e-9:
            raise RuntimeError(f"Packed objective audit failed: log-loss discrepancy {discrepancy}")
        self.packing_audit = dict(blocks=blocks, absolute_log_loss_error=discrepancy)

    def update(self, rates, step, counts=None, batch=None):
        # Retain the original acquisition arithmetic. Acceleration starts from
        # the resulting actual state and never resets parameters or moments.
        if step <= 3:
            if self.population is None:
                gradients, info = self.model.gradients(counts, validate_counts=False)
            else:
                from .random_backend import random_gradients
                gradients, info = random_gradients(self.model, self.population if batch is None else batch,
                                                   counts, validate=False)
            _optimizer_step(self.model, self.optimizer, gradients, rates, step)
            return gradients, info
        self._audit_packing(counts, batch)
        if self.performance.compile_updates:
            if self.compiled is None:
                from .acceleration import CompiledUpdate
                self.compiled = CompiledUpdate(self.model, self.optimizer,
                    gradient_function=self._gradient_function, backend=self.performance.compile_backend)
            return self.compiled(rates, step, counts=counts, batch=batch)
        gradients, info = self._gradient_function(counts, batch)
        _optimizer_step(self.model, self.optimizer, gradients, rates, step)
        return gradients, info

    def describe(self):
        return dict(settings=asdict(self.performance), source_sha256=implementation_hashes(),
                    packing_seconds=self.pack_seconds, packing_audit=self.packing_audit,
                    compile_seconds=self.compiled.compile_seconds if self.compiled is not None else 0.,
                    compile_audits=self.compiled.audits if self.compiled is not None else [])
