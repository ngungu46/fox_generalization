"""Download and reconstruct document-aligned LongCrawl64 pilot rows.

The original public bucket was unavailable when this pilot was prepared. A
pinned public mirror supplies native Zarr token arrays. The convenience pilot
uses disjoint source-heldout documents; these are not the benchmark splits.
First reconstruction fetches about 248 MB; the resulting subset is 18 MiB.
"""

from __future__ import annotations
import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .constants import (
    DOCUMENT_LENGTH,
    REVISION,
    SOURCE_BASE,
    DATA_FILENAME,
    DEFAULT_SUBSET_SHA256,
    SOURCE_SHAPE,
    CHUNK_SHAPE,
)
from .integrity import sha256_file as _sha256


def _fetch(path, suffix, max_bytes):
    if path.exists():
        return path
    req = urllib.request.Request(
        SOURCE_BASE + suffix,
        headers={"Accept-Encoding": "identity", "User-Agent": "FoX-Colab-pilot/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected HTTP status: {response.status}")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError(f"Refusing oversized Zarr object {suffix}")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return path


def prepare_longcrawl64(
    data_dir="data", train_docs=128, validation_docs=8, test_docs=8
):
    """Reconstruct/cache genuine native rows; return a provenance manifest.

    Default: 8,388,608 train tokens + 524,288 validation + 524,288 test.
    Splits are disjoint source documents. No contiguous flattening assumption is
    made: each row is reconstructed from all 32 Zarr sequence-axis chunks.
    Requires numpy and numcodecs only. First run downloads ~248MB; later runs
    reuse the SHA256-verified 18MiB native token payload.
    """
    counts = {
        "train": int(train_docs),
        "validation": int(validation_docs),
        "test": int(test_docs),
    }
    if any(n < 1 for n in counts.values()):
        raise ValueError("Each split needs at least one complete document")
    n_rows = sum(counts.values())
    if n_rows > CHUNK_SHAPE[0]:
        raise ValueError(
            "Pilot downloader supports at most 2048 source documents; use the original Zarr loader for larger runs"
        )
    n_bytes = n_rows * DOCUMENT_LENGTH * 2
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    path, manifest_path = root / DATA_FILENAME, root / "manifest.json"
    if manifest_path.exists() and path.exists():
        saved = json.loads(manifest_path.read_text())
        if (
            saved.get("file") == DATA_FILENAME
            and saved.get("split_counts") == counts
            and path.stat().st_size == n_bytes
        ):
            actual = _sha256(path)
            if actual != saved["sha256"]:
                raise RuntimeError(
                    "Cached LongCrawl64 checksum mismatch; explicitly remove the corrupted cache"
                )
            if (
                counts == {"train": 128, "validation": 8, "test": 8}
                and DEFAULT_SUBSET_SHA256
                and actual != DEFAULT_SUBSET_SHA256
            ):
                raise RuntimeError(
                    "Default native subset does not match the reference checksum"
                )
            return saved
    import numcodecs

    cache = root / "zarr_cache"
    cache.mkdir(exist_ok=True)
    metadata_path = _fetch(cache / ".zarray", ".zarray", 10000)
    metadata = json.loads(metadata_path.read_text())
    required = {
        "shape": SOURCE_SHAPE,
        "chunks": CHUNK_SHAPE,
        "dtype": "<u2",
        "order": "C",
        "zarr_format": 2,
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Unexpected native Zarr layout: {metadata}")
    compressor = numcodecs.get_codec(metadata["compressor"])

    def fetch_chunk(j):
        return _fetch(cache / f"0.{j}", f"0.{j}", 16 * 1024 * 1024)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        chunk_paths = list(
            pool.map(fetch_chunk, range(DOCUMENT_LENGTH // CHUNK_SHAPE[1]))
        )
    array = np.empty((n_rows, DOCUMENT_LENGTH), dtype="<u2")
    source_chunks = []
    for j, chunk_path in enumerate(chunk_paths):
        chunk_bytes = chunk_path.read_bytes()
        decoded = compressor.decode(chunk_bytes)
        if len(decoded) != int(np.prod(CHUNK_SHAPE)) * 2:
            raise ValueError(f"Invalid decompressed chunk size: {chunk_path.name}")
        chunk = np.frombuffer(decoded, dtype="<u2").reshape(CHUNK_SHAPE, order="C")
        array[:, j * CHUNK_SHAPE[1] : (j + 1) * CHUNK_SHAPE[1]] = chunk[:n_rows]
        source_chunks.append(
            {
                "key": chunk_path.name,
                "bytes": len(chunk_bytes),
                "sha256": hashlib.sha256(chunk_bytes).hexdigest(),
            }
        )
    if int(array.max()) > 50256:
        raise ValueError("Payload contains out-of-range GPT-2 token IDs")
    row_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in array]
    if len(set(row_hashes)) != len(row_hashes):
        raise ValueError(
            "Duplicate complete source documents in requested pilot subset"
        )
    temporary = path.with_suffix(".part")
    array.tofile(temporary)
    actual_sha = _sha256(temporary)
    if (
        counts == {"train": 128, "validation": 8, "test": 8}
        and DEFAULT_SUBSET_SHA256
        and actual_sha != DEFAULT_SUBSET_SHA256
    ):
        raise RuntimeError(
            "Freshly reconstructed native subset does not match the reference checksum"
        )
    temporary.replace(path)
    splits, offset = {}, 0
    for name, count in counts.items():
        splits[name] = {
            "row_start": offset,
            "row_stop_exclusive": offset + count,
            "rows": count,
            "tokens": count * DOCUMENT_LENGTH,
        }
        offset += count
    manifest = {
        "dataset": "LongCrawl64 native-token pilot subset",
        "source_url": SOURCE_BASE,
        "source_repository": "clankur/longcrawl64 (third-party mirror)",
        "source_revision": REVISION,
        "dataset_description_url": "https://manifestai.com/articles/longcrawl64/index.html",
        "license_url": f"https://huggingface.co/datasets/clankur/longcrawl64/blob/{REVISION}/README.md",
        "license_note": "Mirror card declares MIT. Underlying web text originates in RedPajama-v2/Common Crawl; this is not a blanket copyright grant for source documents.",
        "original_source_status": "Official gs://longcrawl64 public bucket returned HTTP404 on 2026-09-15; pinned HF mirror used.",
        "source_split_note": "Local train/validation/test repurpose disjoint rows from source heldout.zarr. This is a pilot-specific split, not the original benchmark split. H200 experiments must use genuine source train rows and fresh reserved heldout documents.",
        "document_boundary_note": "Each native source row is reconstructed from all 32 Zarr sequence-axis chunks. No token stream flattening or packing crosses document boundaries.",
        "boundary_verification": "verified native Zarr layout: 52131x65536 rows; C-order little-endian uint16; 2048x2048 compressed chunks; all 32 sequence-axis chunks reconstructed independently",
        "source_zarr_metadata": metadata,
        "source_metadata_sha256": _sha256(metadata_path),
        "source_chunks": source_chunks,
        "file": DATA_FILENAME,
        "dtype": "<u2",
        "shape": [n_rows, DOCUMENT_LENGTH],
        "vocab_size": 50257,
        "eot_token_id": 50256,
        "tokenizer": "tiktoken.get_encoding('gpt2')",
        "upstream_preprocessing": "Long documents were truncated to 65536 tokens and randomly circularly rolled by the dataset authors; no new tokenization or rolling here.",
        "download_bytes": sum(item["bytes"] for item in source_chunks)
        + metadata_path.stat().st_size,
        "subset_bytes": n_bytes,
        "sha256": actual_sha,
        "split_counts": counts,
        "splits": splits,
        "row_sha256": row_hashes,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Small convenience prefix, not a random sample of the full corpus.",
            "No near-duplicate detection beyond exact full-document hashes.",
            "Maximum within-document evaluation length is 65536 tokens.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
