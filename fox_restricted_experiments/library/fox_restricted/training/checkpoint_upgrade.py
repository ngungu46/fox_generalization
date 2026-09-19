"""Recognize the previous runner release without resetting a running study.

Only the two known orchestration files may change in this upgrade. Model,
objective, sampler, rates, probes and configuration must retain their hashes.
The original manifest/source are archived and every checkpoint's tensor state
is retained. Unknown source changes remain errors in the normal resume guard.
"""
import hashlib
import json
from pathlib import Path
import shutil

import torch

from ..legacy.a100_runner import _atomic_json, _atomic_torch_save, _source_hashes


OLD_RUNNER_HASH = "0b998714da486bcfbc3e09f1ee87f6098b46a3943048117564818b48e0b42961"
OLD_RANDOM_RUNNER_HASH = "9b57e8456aad2610b6c8b5b1d60b063970bf8e7e5c77cdbece1e593abcd973dd"


def upgrade_known_runner(path, dataset):
    """Idempotently finish one known performance-only provenance transition.

    A journal is written before mutations, so a disconnection between metadata
    writes is repairable. This never changes model tensors, moments, RNG,
    clocks, accumulated results, or the scientific settings.
    """
    path = Path(path)
    manifest_path = path / "config.json"
    if not manifest_path.exists():
        return False
    if dataset == "pairs":
        root = Path(__file__).resolve().parents[1] / "legacy"
        current = _source_hashes()
        allowed_old = {"a100_runner.py": OLD_RUNNER_HASH}
    else:
        from .random_training import _sources
        root, current = _sources()
        allowed_old = {"legacy/a100_runner.py": OLD_RUNNER_HASH,
                       "training/random_training.py": OLD_RANDOM_RUNNER_HASH}
    saved = json.loads(manifest_path.read_text())
    old = saved.get("source_sha256", {})
    journal_path = path / "performance_upgrade.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else None
    if old == current and (journal is None or journal.get("complete")):
        return False
    if journal and not journal.get("complete"):
        if journal["new_source_sha256"] != current or old not in (journal["old_source_sha256"], current):
            raise ValueError("An interrupted performance upgrade belongs to different source; restore that revision first")
        old = journal["old_source_sha256"]
    else:
        if set(old) != set(current):
            return False
        changed = {name for name in old if old[name] != current[name]}
        if not changed or any(name not in allowed_old or old[name] != allowed_old[name] for name in changed):
            return False
        # Verify the old executable snapshot before recognizing this release.
        for name, digest in old.items():
            source = path / "source" / name
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Cannot verify the original checkpoint source: {name}")
        journal = dict(version=1, complete=False, old_source_sha256=old,
                       new_source_sha256=current,
                       reason="optional packed data and compiled update orchestration; scientific kernel unchanged",
                       state_policy="parameters, moments, RNG, steps, clocks and measurements retained")
        _atomic_json(journal_path, journal)
    archive = path / "source_before_performance_upgrade"
    if not archive.exists():
        temporary_archive = path / ".source_before_performance_upgrade.tmp"
        if temporary_archive.exists():
            shutil.rmtree(temporary_archive)
        shutil.copytree(path / "source", temporary_archive)
        _atomic_json(temporary_archive / "original_config.json", saved)
        temporary_archive.replace(archive)
    for checkpoint in (path / "checkpoints").glob("*.pt"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "source_sha256" in payload:
            if payload["source_sha256"] not in (old, current):
                raise ValueError(f"Checkpoint source does not match the recognized release: {checkpoint.name}")
            if payload["source_sha256"] != current:
                payload["source_sha256"] = current
                _atomic_torch_save(checkpoint, payload)
    for name in current:
        destination = path / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes((root / name).read_bytes())
        temporary.replace(destination)
    saved["source_sha256"] = current
    history = saved.setdefault("implementation_history", [])
    if not any(row.get("old_source_sha256") == old and row.get("new_source_sha256") == current for row in history):
        history.append(dict(journal, complete=True))
    _atomic_json(manifest_path, saved)
    journal["complete"] = True
    _atomic_json(journal_path, journal)
    return True
