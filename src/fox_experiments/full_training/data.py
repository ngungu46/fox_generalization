"""Read the real training split of LongCrawl64 without materializing it in RAM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REPOSITORY = "clankur/longcrawl64"
REVISION = "5e2f38e601ca7c16fa304f30536d9e2945bb2a32"
SHAPES = {"train": (6609334, 65536), "heldout": (52131, 65536)}


def metadata(path, split, strict=True):
    path = Path(path)
    info = json.loads((path / ".zarray").read_text())
    if info.get("zarr_format") != 2 or len(info["shape"]) != 2:
        raise ValueError("Expected a native Zarr v2 token matrix")
    if strict and tuple(info["shape"]) != SHAPES[split]:
        raise ValueError(
            f"{split} must have native shape {SHAPES[split]}; got {info['shape']}"
        )
    if np.dtype(info["dtype"]) != np.dtype("<u2") or info.get("order") != "C":
        raise ValueError("Expected native little-endian uint16 tokens in C order")
    return info


class NativeRows:
    """A lazy row partition that fails on missing compressed chunks.

    Zarr normally substitutes zero for missing chunks. That is unacceptable for
    incomplete downloads, so every sampled window checks its backing files.
    """

    def __init__(self, path, start, stop, info):
        import zarr

        self.path, self.start, self.stop = Path(path), start, stop
        self.info = info
        self.shape = (stop - start, info["shape"][1])
        self.array = zarr.open_array(str(self.path), mode="r")

    def __len__(self):
        return self.shape[0]

    def window(self, row, start, length):
        absolute = row + self.start
        if not (0 <= row < len(self) and 0 <= start <= self.shape[1] - length):
            raise IndexError("Requested window is outside this document partition")
        rows, columns = self.info["chunks"]
        separator = self.info.get("dimension_separator", ".")
        for col_chunk in range(start // columns, (start + length - 1) // columns + 1):
            key = separator.join((str(absolute // rows), str(col_chunk)))
            if not (self.path / key).is_file():
                raise FileNotFoundError(f"Missing native chunk: {self.path / key}")
        tokens = np.asarray(
            self.array[absolute, start : start + length], dtype=np.int64
        )
        if len(tokens) != length or (tokens >= 50257).any():
            raise ValueError("Invalid or truncated GPT-2 token window")
        return tokens.tolist()


class NativeCorpus:
    """Native train rows; heldout rows partitioned into tune/confirm/test.

    Every heldout row remains excluded from training. No pilot subset is used.
    The first 4096 heldout rows are development data and the remainder is test.
    """

    def __init__(self, root, strict=True):
        root = Path(root)
        train_info = metadata(root / "train.zarr", "train", strict)
        heldout_info = metadata(root / "heldout.zarr", "heldout", strict)
        n = heldout_info["shape"][0]
        split = min(2048, n // 4)
        if split < 1:
            raise ValueError("Need at least four heldout documents")
        train = NativeRows(root / "train.zarr", 0, train_info["shape"][0], train_info)

        def heldout(lo, hi):
            return NativeRows(root / "heldout.zarr", lo, hi, heldout_info)

        self.arrays = {
            "train": train,
            "validation": heldout(0, 2 * split),
            "tune": heldout(0, split),
            "confirm": heldout(split, 2 * split),
            "test": heldout(2 * split, n),
        }

    def sample(self, split, length, rng):
        rows = self.arrays[split]
        if length < 1 or length > rows.shape[1]:
            raise ValueError(f"Length must be between 1 and {rows.shape[1]}")
        row = int(rng.integers(len(rows)))
        offset = int(rng.integers(rows.shape[1] - length + 1))
        return rows.window(row, offset, length), row + rows.start, offset


def download_plan(root):
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "destination": str(Path(root).resolve()),
        "patterns": ["train.zarr/**", "heldout.zarr/**"],
        "expected_shapes": SHAPES,
        "storage_note": "Upstream estimates an approximately 800 GB dataset; provision at least 1 TB. Actual compressed download size varies.",
        "splits": "Train uses train.zarr only; heldout is partitioned for tuning, confirmation and test.",
    }


def download(root):
    """Explicit opt-in full download. Existing completed files are reused."""
    from huggingface_hub import snapshot_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=["train.zarr/**", "heldout.zarr/**"],
        local_dir=root,
    )
    (root / "source.json").write_text(json.dumps(download_plan(root), indent=2))


def inspect_data(root, check_all_chunks=False):
    report = download_plan(root)
    report["stores"] = {}
    for split in SHAPES:
        path = Path(root) / f"{split}.zarr"
        info = metadata(path, split)
        details = {
            "shape": info["shape"],
            "chunks": info["chunks"],
            "dtype": info["dtype"],
        }
        if check_all_chunks:
            separator = info.get("dimension_separator", ".")
            grid = [math.ceil(n / c) for n, c in zip(info["shape"], info["chunks"])]
            missing = []
            missing_count = 0
            for i in range(grid[0]):
                for j in range(grid[1]):
                    key = separator.join((str(i), str(j)))
                    if not (path / key).is_file():
                        missing_count += 1
                        if len(missing) < 5:
                            missing.append(key)
            details.update(missing_count=missing_count, missing_examples=missing)
            if missing_count:
                raise FileNotFoundError(
                    f"{split}: {missing_count} missing chunks, e.g. {missing}"
                )
        report["stores"][split] = details
    corpus = NativeCorpus(root)
    for split in ("train", "tune", "confirm", "test"):
        corpus.sample(split, 128, np.random.default_rng(0))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/longcrawl64_full")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the full corpus (upstream estimates roughly 800 GB)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check metadata and all compressed chunk files",
    )
    args = parser.parse_args()
    if args.download:
        download(args.data_root)
    result = (
        inspect_data(args.data_root, True)
        if args.verify
        else download_plan(args.data_root)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
