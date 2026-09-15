"""Verified corpus files and document-local sampling for the pilot.

CorpusFiles holds the three memory-mapped source splits. Corpus supplies the
training sampler and divides validation documents again into independent
learning-rate tuning and confirmation subsets.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .integrity import sha256_file as _sha256


@dataclass
class CorpusFiles:
    """Read-only token arrays plus the downloaded subset's provenance."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    manifest: dict


def load_corpus(data_dir: str | Path = "data") -> CorpusFiles:
    """Load read-only [number_of_documents,65536] arrays without network access."""
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / manifest["file"]
    if _sha256(path) != manifest["sha256"]:
        raise RuntimeError("LongCrawl64 data checksum does not match manifest")
    expected_bytes = (
        int(np.prod(manifest["shape"])) * np.dtype(manifest["dtype"]).itemsize
    )
    if path.stat().st_size != expected_bytes:
        raise RuntimeError("Token payload size does not match manifest shape/dtype")
    arr = np.memmap(
        path, dtype=manifest["dtype"], mode="r", shape=tuple(manifest["shape"])
    )
    portions = {
        name: arr[spec["row_start"] : spec["row_stop_exclusive"]]
        for name, spec in manifest["splits"].items()
    }
    return CorpusFiles(**portions, manifest=manifest)


class Corpus:
    """Document-local sampler for arrays small enough to validate at startup.

    Splits contain complete source documents. Validation documents are divided
    once into independent ``tune`` and ``confirm`` slices. Full training uses
    the separate lazy native-Zarr reader instead of materializing a large array.
    """

    def __init__(self, train, validation, test):
        self.arrays = {
            "train": np.asarray(train),
            "validation": np.asarray(validation),
            "test": np.asarray(test),
        }
        for split, array in self.arrays.items():
            min_rows = 2 if split == "validation" else 1
            if array.ndim != 2 or array.shape[0] < min_rows or array.shape[1] < 1:
                raise ValueError(
                    f"{split}: expected 2D token rows; validation needs two rows"
                )
            if array.min() < 0 or array.max() >= 50257:
                raise ValueError(f"{split}: token id out of GPT-2 range")

        midpoint = len(self.arrays["validation"]) // 2
        self.arrays["tune"] = self.arrays["validation"][:midpoint]
        self.arrays["confirm"] = self.arrays["validation"][midpoint:]

    def sample(
        self, split: str, length: int, rng: np.random.Generator
    ) -> tuple[list[int], int, int]:
        """Return tokens, source row, and start; the window stays within a row."""
        array = self.arrays[split]
        if not 1 <= length <= array.shape[1]:
            raise ValueError(
                f"Requested {length} tokens; source row supports 1..{array.shape[1]}"
            )
        row = int(rng.integers(len(array)))
        start = int(rng.integers(array.shape[1] - length + 1))
        tokens = array[row, start : start + length].astype(np.int64).tolist()
        return tokens, row, start
