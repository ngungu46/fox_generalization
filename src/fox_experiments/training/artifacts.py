"""Atomic run artifacts and stable identities for safe checkpoint reuse."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn


def dump_json(path: str | Path, value: Any) -> None:
    """Replace a JSON artifact only after its complete contents are on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous CSV records without exposing a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as stream:
        if keys:
            writer = csv.DictWriter(stream, keys)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Read cached CSV records for reconstructing aggregate outputs."""
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    """Keep the previous checkpoint valid if writing the new one is interrupted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def state_hash(model: nn.Module) -> str:
    """Hash named parameter/buffer contents to audit paired initial models."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def dataset_identity(data: Any) -> dict[str, Any]:
    """Fingerprint the local corpus splits and answer vocabulary before resuming."""
    result = {"answer_token_ids": data.answers, "splits": {}}
    for split in ("train", "validation", "test"):
        array = data.corpus.arrays[split]
        digest = hashlib.sha256()
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
        result["splits"][split] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest.hexdigest(),
        }
    return result


def finite_json_row(row: dict[str, Any]) -> dict[str, Any]:
    """Represent nonfinite diagnostics explicitly as missing values in JSON."""
    return {
        key: (
            None
            if isinstance(value, (float, np.floating)) and not math.isfinite(value)
            else value
        )
        for key, value in row.items()
    }


@contextmanager
def exclusive_run_lock(directory: Path) -> Iterator[None]:
    """Prevent two local notebook/process writers from sharing one output directory.

    Colab/Linux and macOS release this advisory lock automatically on a crash.
    The lock protects local processes; do not run the same Google Drive output
    from multiple machines, because network filesystem locking is not guaranteed.
    """
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another process is writing this run: {directory}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
