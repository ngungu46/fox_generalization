"""Exact random-stream training using only the records that actually exist.

Packing is a one-time layout change for a fixed sampled population. It does
not change the data, weights, ordinary answer loss, floating-point precision,
or signed-log gradient representation. Streams remain contiguous and static
grouping plans replace work on a batch's globally padded rectangular array.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from ..legacy.core import logsoftplus, sadd, slog, ssum
from .random_backend import _weights


@dataclass(frozen=True)
class PackedRandomBatch:
    """Precomputed record and reduction indices for one unchanged population.

    Construct with :func:`pack_random_batch`; retain the original ``RandomBatch``
    for dataset checkpoints and independent literal-stream audits. A packed
    object is derived state and can be reconstructed after resumption.
    """

    n: int
    keys: torch.Tensor
    values: torch.Tensor
    queries: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor
    lags: torch.Tensor
    stream_ids: torch.Tensor
    first: torch.Tensor
    distances: torch.Tensor
    correct: torch.Tensor
    current_pairs: torch.Tensor
    previous_pairs: torch.Tensor
    current_order: torch.Tensor
    current_sorted_pairs: torch.Tensor
    current_pair_lengths: torch.Tensor
    previous_order: torch.Tensor
    previous_sorted_pairs: torch.Tensor
    previous_pair_lengths: torch.Tensor
    padded_records: int

    @property
    def size(self):
        return self.lengths.numel()

    @property
    def record_count(self):
        return self.keys.numel()

    def validate(self, n):
        # Construction checks all stream invariants. These inexpensive checks
        # catch mixing a packed population with a different model vocabulary.
        if n != self.n:
            raise ValueError("Packed population and model vocabulary sizes differ")
        return self


@torch.no_grad()
def pack_random_batch(batch, n, *, validate=True):
    """Remove padding once, preserving every sampled valid record and its order."""
    if validate:
        batch.validate(n)
    keys, values = batch.keys[batch.mask], batch.values[batch.mask]
    size, total = batch.size, keys.numel()
    stream_ids = torch.repeat_interleave(
        torch.arange(size, device=keys.device), batch.lengths, output_size=total)
    starts = batch.lengths.cumsum(0) - batch.lengths
    positions = torch.arange(total, device=keys.device) - starts[stream_ids]
    first = positions == 0
    queries = batch.queries[stream_ids]
    current_pairs = queries * n + keys
    # The first record has one binder candidate. Its dummy previous entry is
    # identical and receives log weight -inf when derivatives are assembled.
    previous_keys = torch.cat((keys[:1], keys[:-1]))
    previous_keys = torch.where(first, keys, previous_keys)
    previous_pairs = queries * n + previous_keys
    current_order = torch.argsort(current_pairs, stable=True)
    previous_order = torch.argsort(previous_pairs, stable=True)
    return PackedRandomBatch(
        n=n, keys=keys, values=values, queries=batch.queries,
        labels=batch.labels, lengths=batch.lengths, lags=batch.lags,
        stream_ids=stream_ids, first=first,
        distances=(batch.lengths[stream_ids] - positions).to(values.dtype),
        correct=batch.labels[stream_ids] * values > 0,
        current_pairs=current_pairs, previous_pairs=previous_pairs,
        current_order=current_order, current_sorted_pairs=current_pairs[current_order],
        current_pair_lengths=torch.bincount(current_pairs, minlength=n * n),
        previous_order=previous_order, previous_sorted_pairs=previous_pairs[previous_order],
        previous_pair_lengths=torch.bincount(previous_pairs, minlength=n * n),
        padded_records=batch.keys.numel())


def _segment_logsumexp(logs, lengths, segment_ids):
    """Stable contiguous segmented logsumexp, including empty/all--inf sums.

    Trailing dimensions permit computing positive/negative (or correct/wrong)
    sums together. Scaling happens separately inside *each* segment, so an
    e^-2000 derivative cannot be erased by another key's order-one derivative.
    """
    # Lengths are immutable grouping plans checked at packing time. Repeating
    # the operator's host-side length validation would synchronize every CUDA
    # reduction; unsafe=True skips that redundant check, not numerical guards.
    maximum = torch.segment_reduce(logs, "max", lengths=lengths, unsafe=True)
    shifted = torch.where(torch.isfinite(logs), (logs - maximum[segment_ids]).exp(), 0.)
    return maximum + torch.segment_reduce(shifted, "sum", lengths=lengths, unsafe=True).log()


def _table_gradients(model, batch, score_signs, table_logs, m, rho, lrho):
    signs = score_signs * m.sign()
    current = table_logs + torch.where(batch.first, 0., torch.log1p(-rho))
    previous = torch.where(batch.first, -torch.inf, table_logs + lrho)
    def pool(logs, order, lengths, indices):
        signed_logs = torch.stack((torch.where(signs > 0, logs, -torch.inf),
                                   torch.where(signs < 0, logs, -torch.inf)), dim=1)
        return _segment_logsumexp(signed_logs[order], lengths, indices)
    # Keep the current/previous sums separate, as in the padded reference.
    # This also preserves exact cancellations for constant-key streams.
    pools = torch.logaddexp(
        pool(current, batch.current_order, batch.current_pair_lengths, batch.current_sorted_pairs),
        pool(previous, batch.previous_order, batch.previous_pair_lengths, batch.previous_sorted_pairs))
    positive, negative = pools.unbind(1)
    matrix_signs, matrix_logs = sadd(torch.ones_like(positive), positive,
                                    -torch.ones_like(negative), negative)
    n = model.cfg.n
    matrix_signs, matrix_logs = matrix_signs.reshape(n, n), matrix_logs.reshape(n, n)
    finite = torch.isfinite(matrix_logs)
    high = torch.where(finite, matrix_logs, -torch.inf)
    low = torch.where(finite, matrix_logs, torch.inf)
    row_high, column_high = high.amax(dim=1), high.amax(dim=0)
    row_shift = torch.where(torch.isfinite(row_high), row_high, 0.)[:, None]
    column_shift = torch.where(torch.isfinite(column_high), column_high, 0.)[None, :]
    row_matrix = matrix_signs * (matrix_logs - row_shift).exp()
    column_matrix = matrix_signs * (matrix_logs - column_shift).exp()
    sq, lq = slog(row_matrix @ model.K)
    sk, lk = slog(column_matrix.T @ model.Q)
    row_span = torch.where(finite.any(dim=1), row_high - low.amin(dim=1), 0.).max()
    column_span = torch.where(finite.any(dim=0), column_high - low.amin(dim=0), 0.).max()
    return [(sq, lq + row_shift), (sk, lk + column_shift.T)], torch.maximum(row_span, column_span)


@torch.no_grad()
def packed_random_gradients(model, batch, counts=None, *, validate=True):
    """Same gradients/info as ``random_gradients``, with no padded-record work.

    ``counts`` weights streams, not records; zero-count streams have exactly
    zero loss weight. All signed-log arithmetic and the final per-row/column
    scaling guard are retained. Pass ``validate=False`` only for trusted
    internally constructed populations and frequency counts.
    """
    if validate:
        batch.validate(model.cfg.n)
    lp = _weights(batch, counts, validate)
    q, p, u, v, x, w, m, g, h, rho, lrho = model.quantities()
    table = (model.Q @ model.K.T).reshape(-1)
    dots, previous = table[batch.current_pairs], table[batch.previous_pairs]
    bound_dots = torch.where(batch.first, dots, (1 - rho) * dots + rho * previous)
    scores = m * bound_dots - h * batch.distances
    pools = _segment_logsumexp(
        torch.stack((torch.where(batch.correct, scores, -torch.inf),
                     torch.where(~batch.correct, scores, -torch.inf)), dim=1),
        batch.lengths, batch.stream_ids)
    lc, lw = pools.unbind(1)
    lz = torch.logaddexp(lc, lw)
    alignment = -torch.tanh((lw - lc) / 2)
    stream = batch.stream_ids
    opposite = torch.where(batch.correct, lw[stream], lc[stream])
    score_logs = (math.log(2) + w.abs().log() + F.logsigmoid(-w * alignment)[stream]
                  + (scores - lz[stream]) + (opposite - lz[stream]) + lp[stream])
    score_signs = torch.where(batch.correct, -w.sign(), w.sign())
    score_signs = torch.where(torch.isfinite(score_logs), score_signs, 0.)
    sm, lm = ssum(score_signs * bound_dots.sign(), score_logs + bound_dots.abs().log())
    sh, lh = ssum(-score_signs, score_logs + batch.distances.log())
    delta = previous - dots
    sg, lg = ssum(-score_signs * (m * delta).sign(),
                  score_logs + m.abs().log() + delta.abs().log()
                  + math.log(2) + lrho + torch.log1p(-rho))
    sw, lw_grad = ssum(-alignment.sign(), lp + alignment.abs().log() + F.logsigmoid(-w * alignment))
    scalar_signs = torch.stack((sm * p.sign(), sm * q.sign(), sg * v.sign(), sg * u.sign(), sh, sw))
    scalar_logs = torch.stack((lm + (model.cfg.c0 * p).abs().log(),
                               lm + (model.cfg.c0 * q).abs().log(),
                               lg + v.abs().log() + F.logsigmoid(u * v),
                               lg + u.abs().log() + F.logsigmoid(u * v),
                               lh + F.logsigmoid(x), lw_grad))
    if model.frozen_scalar_indices:
        index = list(model.frozen_scalar_indices)
        scalar_signs[index], scalar_logs[index] = 0., -torch.inf
    table_logs = score_logs + m.abs().log()
    finite = torch.isfinite(table_logs)
    high = torch.where(finite, table_logs, -torch.inf).max()
    low = torch.where(finite, table_logs, torch.inf).min()
    gradients, span = _table_gradients(model, batch, score_signs, table_logs, m, rho, lrho)
    gradients.append((scalar_signs, scalar_logs))
    info = {"logloss": torch.logsumexp(lp + logsoftplus(-w * alignment), dim=0),
            "table_log_span": span,
            "record_log_span": torch.where(finite.any(), high - low, torch.zeros_like(high)),
            "mean_correct_probability": (lp.exp() * torch.sigmoid(w * alignment)).sum(),
            "mean_target_lag": (lp.exp() * batch.lags).sum(),
            "mean_stream_records": (lp.exp() * batch.lengths).sum()}
    return gradients, info
