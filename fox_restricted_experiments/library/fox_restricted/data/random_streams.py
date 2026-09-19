"""Sample arbitrary structured streams whose latest queried key has lag ≤ R.

There is no uniform distribution over all finite streams.  The default makes
the sampling measure explicit: uniform query, lag, and binary values, with an
independent geometric older-prefix length. Every legal finite stream then has
positive probability. The optional bounded-uniform prefix distribution is a
different, finite-support experiment, and is named as such in saved settings.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class RandomStreamConfig:
    batch_size: int = 4096
    prefix_distribution: str = "geometric"
    prefix_mean: float = 16.0
    max_prefix: int = 64
    max_batch_records: int = 2_000_000

    def __post_init__(self):
        for name, minimum in (("batch_size", 1), ("max_prefix", 0), ("max_batch_records", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.prefix_distribution not in ("geometric", "bounded_uniform"):
            raise ValueError("prefix_distribution must be geometric or bounded_uniform")
        if not math.isfinite(self.prefix_mean) or self.prefix_mean <= 0:
            raise ValueError("prefix_mean must be positive and finite for full support")


@dataclass
class RandomBatch:
    """Right-padded record arrays. Padding is excluded by ``mask`` everywhere."""

    keys: torch.Tensor          # (batch, records), key IDs in [0, N)
    values: torch.Tensor        # (batch, records), binary values -1 or +1
    mask: torch.Tensor          # (batch, records), True exactly on valid records
    queries: torch.Tensor       # (batch,)
    labels: torch.Tensor        # (batch,), value of the latest queried key
    lengths: torch.Tensor       # (batch,)
    lags: torch.Tensor          # (batch,), newest record has lag 1

    @property
    def size(self):
        return self.keys.shape[0]

    def select(self, indices):
        """Select rows (duplicates allowed), retaining the common padded width."""
        indices = torch.as_tensor(indices, device=self.keys.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("indices must be a nonempty one-dimensional vector")
        return RandomBatch(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})

    def state_dict(self):
        """CPU tensors suitable for an ordinary trusted local torch checkpoint."""
        return {name: getattr(self, name).detach().cpu().clone() for name in self.__dataclass_fields__}

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        return cls(**{name: state[name].to(device) for name in cls.__dataclass_fields__})

    def validate(self, n, max_lag=None):
        if self.keys.ndim != 2 or self.keys.numel() == 0:
            raise ValueError("keys must be a nonempty batch-by-record array")
        if self.values.shape != self.keys.shape or self.mask.shape != self.keys.shape:
            raise ValueError("keys, values and mask must have identical shapes")
        if any(x.shape != (self.size,) for x in (self.queries, self.labels, self.lengths, self.lags)):
            raise ValueError("queries, labels, lengths and lags must have one entry per stream")
        if self.keys.dtype != torch.long or self.queries.dtype != torch.long or self.mask.dtype != torch.bool:
            raise ValueError("keys/queries must be int64 and mask must be bool")
        if any(x.device != self.keys.device for x in (self.values, self.mask, self.queries,
                                                    self.labels, self.lengths, self.lags)):
            raise ValueError("All batch arrays must be on the same device")
        if bool(((self.keys < 0) | (self.keys >= n)).any()) or bool(((self.queries < 0) | (self.queries >= n)).any()):
            raise ValueError("Key IDs are outside the vocabulary")
        if not bool(((self.values == 1) | (self.values == -1)).all()) or not bool(((self.labels == 1) | (self.labels == -1)).all()):
            raise ValueError("Values and labels must be -1 or +1")
        positions = torch.arange(self.keys.shape[1], device=self.keys.device)[None, :]
        if bool(((self.lengths < 1) | (self.lengths > self.keys.shape[1])).any()):
            raise ValueError("Invalid stream lengths")
        if not torch.equal(self.mask, positions < self.lengths[:, None]):
            raise ValueError("mask must describe contiguous right-padded streams")
        if bool(((self.lags < 1) | (self.lags > self.lengths)).any()):
            raise ValueError("lags must lie between 1 and the stream length")
        if max_lag is not None and bool((self.lags > max_lag).any()):
            raise ValueError("A sampled target exceeds the training lag bound")
        target = self.lengths - self.lags
        row = torch.arange(self.size, device=self.keys.device)
        if not torch.equal(self.keys[row, target], self.queries) or not torch.equal(self.values[row, target], self.labels):
            raise ValueError("The target must match the query key and answer label")
        if bool(((self.keys == self.queries[:, None]) & self.mask & (positions > target[:, None])).any()):
            raise ValueError("The queried key may not recur after the designated target")
        return self


def sample_stream_batch(n, R, settings=None, *, rng=None, device="cpu", batch_size=None):
    """Draw an iid minibatch without truncating or rejecting long prefixes.

    ``rng`` is a NumPy Generator. Save and restore ``rng.bit_generator.state``
    together with the optimizer checkpoint for exact sampler continuation.
    ``max_batch_records`` is a memory guard, not a distributional truncation:
    exceeding it raises an error rather than silently replacing the batch.
    """
    settings = settings or RandomStreamConfig()
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if isinstance(R, bool) or not isinstance(R, int) or R < 1:
        raise ValueError("R must be an integer >= 1")
    size = settings.batch_size if batch_size is None else batch_size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    rng = np.random.default_rng() if rng is None else rng
    if settings.prefix_distribution == "geometric":
        prefix = rng.geometric(1.0 / (1.0 + settings.prefix_mean), size=size) - 1
    else:
        prefix = rng.integers(0, settings.max_prefix + 1, size=size)
    lag = rng.integers(1, R + 1, size=size)
    lengths = prefix + lag
    width = int(lengths.max())
    if size * width > settings.max_batch_records:
        raise MemoryError(f"Draw requires {size * width:,} padded records, exceeding max_batch_records="
                          f"{settings.max_batch_records:,}. Increase the guard or reduce batch_size; "
                          "the draw was not truncated or resampled.")
    queries = rng.integers(0, n, size=size)
    labels = 2 * rng.integers(0, 2, size=size) - 1
    keys = rng.integers(0, n, size=(size, width))
    values = 2 * rng.integers(0, 2, size=(size, width)) - 1
    positions = np.arange(width)[None, :]
    # Draw suffix keys from the N-1 non-query keys using a gap-free mapping.
    suffix_keys = rng.integers(0, n - 1, size=(size, width))
    suffix_keys += suffix_keys >= queries[:, None]
    keys = np.where(positions > prefix[:, None], suffix_keys, keys)
    keys[np.arange(size), prefix] = queries
    values[np.arange(size), prefix] = labels
    mask = positions < lengths[:, None]
    keys = np.where(mask, keys, 0)
    values = np.where(mask, values, 1)
    tensor = lambda array, dtype: torch.as_tensor(array, device=device, dtype=dtype)
    return RandomBatch(tensor(keys, torch.long), tensor(values, torch.float64),
                       tensor(mask, torch.bool), tensor(queries, torch.long),
                       tensor(labels, torch.float64), tensor(lengths, torch.long),
                       tensor(lag, torch.long))


def batch_from_streams(streams, *, device="cpu"):
    """Build a validated batch from dictionaries with keys/values/query/label.

    This small convenience API is useful for exact model audits and hand-made
    counterexamples; training uses the vectorized sampler above.
    """
    streams = list(streams)
    if not streams:
        raise ValueError("At least one stream is required")
    width = max(len(row["keys"]) for row in streams)
    keys = torch.zeros((len(streams), width), dtype=torch.long, device=device)
    values = torch.ones((len(streams), width), dtype=torch.float64, device=device)
    queries, labels, lengths, lags = [], [], [], []
    for index, row in enumerate(streams):
        size = len(row["keys"])
        if size < 1 or len(row["values"]) != size:
            raise ValueError("Each stream needs equally sized nonempty key/value lists")
        query = row["query"]
        matching = [position for position, key in enumerate(row["keys"]) if key == query]
        if not matching:
            raise ValueError("The query key must appear in the stream")
        keys[index, :size] = torch.as_tensor(row["keys"], device=device)
        values[index, :size] = torch.as_tensor(row["values"], device=device)
        queries.append(query)
        labels.append(row.get("label", row["values"][matching[-1]]))
        lengths.append(size)
        lags.append(size - matching[-1])
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    return RandomBatch(keys, values, torch.arange(width, device=device)[None] < lengths[:, None],
                       torch.tensor(queries, dtype=torch.long, device=device),
                       torch.tensor(labels, dtype=torch.float64, device=device), lengths,
                       torch.tensor(lags, dtype=torch.long, device=device))
