"""Exact ordinary-answer gradients for arbitrary structured-token streams.

Both attention heads are the same as ``legacy.core.dense_stream``. Only the
training data changes. Correct- and incorrect-value attention pools provide
signed-log derivatives that survive very small probabilities and gradients.
No calibration term, pointer loss, or pair-only surrogate is added here.
"""
import math

import torch
from torch.nn import functional as F

from ..legacy.core import dense_stream, logsoftplus, quantities_from_theta, sadd, slog, ssum


def _weights(batch, counts, validate):
    if counts is None:
        return torch.full((batch.size,), -math.log(batch.size), dtype=batch.values.dtype,
                          device=batch.values.device)
    counts = torch.as_tensor(counts, device=batch.values.device, dtype=batch.values.dtype)
    if validate and (counts.shape != (batch.size,) or not bool(torch.isfinite(counts).all())
                     or bool((counts < 0).any()) or not bool(counts.sum() > 0)):
        raise ValueError("counts must be finite nonnegative weights with positive total, one per stream")
    return counts.log() - counts.sum().log()


def _forward(model, batch, params=None, detach_frozen=False):
    Q, K, theta = model.params() if params is None else params
    quantities = quantities_from_theta(model.cfg, theta, detach_frozen=detach_frozen)
    q, p, u, v, x, w, m, g, h, rho, lrho = quantities
    table = Q @ K.T
    dots = table[batch.queries[:, None], batch.keys]
    previous = torch.cat((dots[:, :1], dots[:, :-1]), dim=1)
    # The first record has a single binder candidate; its content is exact.
    bound_dots = torch.cat((dots[:, :1], (1 - rho) * dots[:, 1:] + rho * previous[:, 1:]), dim=1)
    positions = torch.arange(batch.keys.shape[1], device=Q.device, dtype=Q.dtype)[None, :]
    distances = batch.lengths[:, None] - positions
    scores = torch.where(batch.mask, m * bound_dots - h * distances, -torch.inf)
    correct = batch.labels[:, None] * batch.values > 0
    log_correct = torch.logsumexp(torch.where(correct, scores, -torch.inf), dim=1)
    log_wrong = torch.logsumexp(torch.where(~correct, scores, -torch.inf), dim=1)
    log_total = torch.logaddexp(log_correct, log_wrong)
    alignment = -torch.tanh((log_wrong - log_correct) / 2)
    return quantities, dots, previous, bound_dots, distances, scores, correct, log_correct, log_wrong, log_total, alignment


def _scatter_logsumexp(logs, indices, size):
    """Log-sum positive contributions separately for every queried-key/key pair.

    Scaling within each pair keeps, for example, an e^-2000 derivative for an
    otherwise rare key even when another pair has an order-one derivative.
    Contributions far below the maximum *within the same positive sum* may
    round away, as in ordinary logsumexp; they cannot underflow another pair.
    """
    logs, indices = logs.reshape(-1), indices.reshape(-1)
    maximum = torch.full((size,), -torch.inf, device=logs.device, dtype=logs.dtype)
    maximum.scatter_reduce_(0, indices, logs, reduce="amax", include_self=True)
    shifted = torch.where(torch.isfinite(logs), torch.exp(logs - maximum[indices]), 0.)
    total = torch.zeros_like(maximum)
    total.scatter_add_(0, indices, shifted)
    return maximum + total.log()


def _table_derivatives(model, batch, score_signs, table_logs, m, rho, lrho):
    """Aggregate in signed logs, then scale each final matrix-product row.

    The score derivative matrix A has shape N×N. Q receives A@K and K receives
    A.T@Q. These products need separate row and column shifts, since a global
    shift can erase all derivatives of a rare queried key or candidate key.
    ``span`` measures the remaining range inside these final products; the
    runner retains its conservative 650-log-unit guard on this range.
    """
    n = model.cfg.n
    signs = score_signs * m.sign()
    current = table_logs + torch.log1p(-rho)
    current[:, 0] = table_logs[:, 0]  # first record has no preceding binder key
    indices = batch.queries[:, None] * n + batch.keys
    positive = _scatter_logsumexp(torch.where(signs > 0, current, -torch.inf), indices, n*n)
    negative = _scatter_logsumexp(torch.where(signs < 0, current, -torch.inf), indices, n*n)
    if batch.keys.shape[1] > 1:
        previous = table_logs[:, 1:] + lrho
        previous_indices = batch.queries[:, None] * n + batch.keys[:, :-1]
        positive = torch.logaddexp(positive, _scatter_logsumexp(
            torch.where(signs[:, 1:] > 0, previous, -torch.inf), previous_indices, n*n))
        negative = torch.logaddexp(negative, _scatter_logsumexp(
            torch.where(signs[:, 1:] < 0, previous, -torch.inf), previous_indices, n*n))
    matrix_signs, matrix_logs = sadd(torch.ones_like(positive), positive,
                                    -torch.ones_like(negative), negative)
    matrix_signs, matrix_logs = matrix_signs.reshape(n, n), matrix_logs.reshape(n, n)
    finite = torch.isfinite(matrix_logs)
    high = torch.where(finite, matrix_logs, -torch.inf)
    low = torch.where(finite, matrix_logs, torch.inf)
    row_high, column_high = high.amax(dim=1), high.amax(dim=0)
    row_shift = torch.where(torch.isfinite(row_high), row_high, 0.)[:, None]
    column_shift = torch.where(torch.isfinite(column_high), column_high, 0.)[None, :]
    row_matrix = matrix_signs * torch.exp(matrix_logs - row_shift)
    column_matrix = matrix_signs * torch.exp(matrix_logs - column_shift)
    sq, lq = slog(row_matrix @ model.K)
    sk, lk = slog(column_matrix.T @ model.Q)
    row_span = torch.where(finite.any(dim=1), row_high - low.amin(dim=1), 0.).max()
    column_span = torch.where(finite.any(dim=0), column_high - low.amin(dim=0), 0.).max()
    return [(sq, lq + row_shift), (sk, lk + column_shift.T)], torch.maximum(row_span, column_span)


@torch.no_grad()
def random_gradients(model, batch, counts=None, *, validate=True):
    """Return exact empirical-loss gradients as ``(sign, log_abs)`` tensors.

    The objective is the uniform mean over ``batch`` when counts is None.
    Frequency counts instead give sum(counts * loss) / sum(counts), as needed
    for SGD on a fixed random dataset. ``validate=False`` is reserved for
    batches/counts generated internally and already validated at construction.
    """
    if validate:
        batch.validate(model.cfg.n)
    lp = _weights(batch, counts, validate)
    (quantities, dots, previous, bound_dots, distances, scores, correct,
     lc, lw, lz, alignment) = _forward(model, batch)
    q, p, u, v, x, w, m, g, h, rho, lrho = quantities
    opposite_mass = torch.where(correct, lw[:, None], lc[:, None])
    # ∂L/∂score_j = -2*w*sigmoid(-w*s)*b_j*α_j*(opposite value mass).
    score_logs = (math.log(2) + w.abs().log() + F.logsigmoid(-w * alignment)[:, None]
                  + (scores - lz[:, None]) + (opposite_mass - lz[:, None]) + lp[:, None])
    score_signs = torch.where(correct, -torch.sign(w), torch.sign(w)).expand_as(scores)
    score_signs = torch.where(torch.isfinite(score_logs), score_signs, 0.)

    sm, lm = ssum(score_signs * bound_dots.sign(), score_logs + bound_dots.abs().log())
    sh, lh = ssum(-score_signs, score_logs + distances.clamp_min(1).log())
    delta = previous - dots
    sg, lg = ssum(-score_signs * torch.sign(m * delta),
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
    table_gradients, table_span = _table_derivatives(model, batch, score_signs, table_logs, m, rho, lrho)
    gradients = [*table_gradients, (scalar_signs, scalar_logs)]
    logloss = torch.logsumexp(lp + logsoftplus(-w * alignment), dim=0)
    info = {"logloss": logloss,
            "table_log_span": table_span,
            "record_log_span": torch.where(finite.any(), high - low, torch.zeros_like(high)),
            "mean_correct_probability": (lp.exp() * torch.sigmoid(w * alignment)).sum(),
            "mean_target_lag": (lp.exp() * batch.lags).sum(),
            "mean_stream_records": (lp.exp() * batch.lengths).sum()}
    return gradients, info


def native_random_loss(model, batch, counts=None, params=None, *, literal=False):
    """Ordinary PyTorch loss for independent autodiff checks at finite scales.

    Set literal=True to execute each stream through the original explicit
    two-head model, including the binder's actual attention softmax.
    """
    lp = _weights(batch, counts, True)
    if literal:
        params = model.params() if params is None else params
        losses = torch.stack([
            dense_stream(params, model.cfg, batch.keys[i, :int(batch.lengths[i])],
                         batch.values[i, :int(batch.lengths[i])], int(batch.queries[i]),
                         float(batch.labels[i]))[0]
            for i in range(batch.size)
        ])
    else:
        result = _forward(model, batch, params=params, detach_frozen=True)
        # Conventional softmax avoids an all--inf logsumexp backward on a
        # stream whose values all agree. The analytic path handles it in logs.
        w, scores = result[0][5], result[5]
        alignment = (scores.softmax(dim=1) * batch.labels[:, None] * batch.values).sum(dim=1)
        losses = F.softplus(-w * alignment)
    return (lp.exp() * losses).sum()


def audit_random_gradients(model, batch, counts=None, *, tolerance=1e-8):
    """Check every parameter derivative against literal-stream autograd."""
    with torch.enable_grad():
        params = [parameter.detach().clone().requires_grad_(True) for parameter in model.params()]
        loss = native_random_loss(model, batch, counts, params, literal=True)
        expected = torch.autograd.grad(loss, params)
    actual, info = random_gradients(model, batch, counts)
    errors = []
    for target, (sign, logabs) in zip(expected, actual):
        error = (target - sign * logabs.exp()).abs().max()
        scale = target.abs().max().clamp_min(1e-300)
        errors.append(float(error / scale))
    return {"passed": all(math.isfinite(error) and error <= tolerance for error in errors),
            "relative_max_gradient_error_by_block": errors,
            "loss": float(loss.detach()), "logloss": float(info["logloss"]),
            "comparison": "literal per-stream two-head softmax autodiff"}
