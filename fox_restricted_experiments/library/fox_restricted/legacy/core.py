"""Restricted ordinary-answer FoX experiment, implemented in float64.

The analytic backend evaluates the exact finite data objective. Signed-log
gradients and Adam buffers retain very small derivatives without substituting
pointer supervision, sign descent, gradient clipping, or an epsilon floor.
``dense_stream`` independently materializes both attention heads for auditing.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


@dataclass
class Config:
    n: int = 8
    d: int = 2048
    R: int = 2
    steps: int = 12000
    seed: int = 0
    device: str = "cpu"
    sigma: float = 1e-6
    gamma: float = 1 / 16
    c0: float = 1.
    D: float = 0.
    kcal: float = .6
    wo: float = .2
    wc: float = .2
    q0: float = .5
    u0: float = 1.7
    h0: float = 1.
    x0: float | None = None
    gate_mode: str = "learned"
    am: float = 1.
    ag: float = 1.5
    ax: float = .05
    aw: float = .1
    beta1: float = .9
    beta2: float = .999
    eps_decay: float = .1
    lr_adam: float = .015
    lr_sgd: float = 5.
    offset: float = 1000.
    power: float = .75
    table_lr: float = 1e-9
    table_power: float = 2.
    pair_batch: int = 4096
    batch_growth: float = .5
    log_points: int = 80
    eval_max_lag: int = 128
    max_seconds: float = 3600.

    def __post_init__(self):
        for name, minimum in (("n", 2), ("d", 1), ("R", 2), ("steps", 0),
                              ("pair_batch", 1), ("log_points", 2), ("eval_max_lag", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.gate_mode not in ("learned", "retrieval_frozen", "both_frozen"):
            raise ValueError("gate_mode must be learned, retrieval_frozen, or both_frozen")
        positive = ("sigma", "gamma", "c0", "q0", "u0", "h0", "kcal", "wo", "wc",
                    "am", "ag", "ax", "aw", "lr_adam", "lr_sgd", "offset", "table_lr")
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in positive):
            raise ValueError("Initialization scales, mixture weights, and learning rates must be positive and finite")
        if not self.kcal > .5 or not math.isclose(self.kcal + self.wo + self.wc, 1., abs_tol=1e-12):
            raise ValueError("Require kcal > 1/2 and kcal + wo + wc = 1")
        if not 0 <= self.beta1 < 1 or not self.beta1 ** 2 < self.beta2 < 1:
            raise ValueError("Adam requires 0 <= beta1 < 1 and beta1**2 < beta2 < 1")
        if not 2 / 3 < self.power <= 1 or not self.table_power > 1:
            raise ValueError("Use scalar power in (2/3, 1] and summable table power > 1")
        if self.eps_decay < 0 or self.batch_growth < 0:
            raise ValueError("epsilon decay and batch growth must be nonnegative")
        if self.x0 is None:
            self.x0 = self.h0 + math.log(-math.expm1(-self.h0))
        if not math.isfinite(self.x0) or not math.isfinite(self.D):
            raise ValueError("x0 and D must be finite")
        effective_h = max(self.x0, 0.) + math.log1p(math.exp(-abs(self.x0)))
        if not math.isclose(effective_h, self.h0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("x0 must equal inverse_softplus(h0); use replace(cfg, h0=new_h, x0=None) to change the slope")


def frozen_scalar_indices(config):
    return {"learned": (), "retrieval_frozen": (4,), "both_frozen": (2, 3, 4)}[config.gate_mode]


def active_parameter_count(config):
    return 2 * config.n * config.d + 6 - len(frozen_scalar_indices(config))


def slog(x):
    return torch.sign(x), torch.log(torch.abs(x))


def sadd(s1, l1, s2, l2):
    """Signed log addition; exact cancellation has sign zero and log -inf."""
    same = s1 == s2
    hi = torch.maximum(l1, l2)
    diff = torch.abs(l1 - l2)
    difference = hi + torch.log(-torch.expm1(-diff))
    logabs = torch.where(same, torch.logaddexp(l1, l2), difference)
    sign = torch.where(same, s1, torch.where(l1 > l2, s1, s2))
    zero = ((l1 == l2) & ~same) | (torch.isneginf(l1) & torch.isneginf(l2))
    return torch.where(zero, 0., sign), torch.where(zero, -torch.inf, logabs)


def ssum(signs, logs, dim=None):
    if dim is None:
        signs, logs, dim = signs.reshape(-1), logs.reshape(-1), 0
    pos = torch.logsumexp(torch.where(signs > 0, logs, -torch.inf), dim)
    neg = torch.logsumexp(torch.where(signs < 0, logs, -torch.inf), dim)
    return sadd(torch.ones_like(pos), pos, -torch.ones_like(neg), neg)


def logsoftplus(x):
    # For x < -30, the relative difference from exp(x) is < 5e-14.
    return torch.where(x < -30, x, torch.log(F.softplus(x)))


def quantities_from_theta(config, theta, detach_frozen=False):
    q, p, u, v, x, w = theta.unbind()
    if detach_frozen:
        if config.gate_mode == "both_frozen":
            u, v = u.detach(), v.detach()
        if config.gate_mode != "learned":
            x = x.detach()
    m, g, h = config.c0 * q * p, F.softplus(u * v), F.softplus(x)
    lrho = F.logsigmoid(config.D - 2 * g)
    rho = torch.exp(lrho)
    return q, p, u, v, x, w, m, g, h, rho, lrho


def recall_terms(z, m, h, rho, lag):
    """One term per wrong record, in target-relative log-weight units."""
    if lag < 2:
        raise ValueError("Recall training lag must be >= 2")
    first = -z * m * (1 - rho) + h
    if lag == 2:
        return first.unsqueeze(-1)
    offsets = torch.arange(2, lag, dtype=z.dtype, device=z.device)
    pure = (-z * m).unsqueeze(-1) + offsets * h
    return torch.cat((first.unsqueeze(-1), pure), dim=-1)


def _pair_log_mass(z, counts, validate=True):
    if counts is None:
        return torch.full_like(z, -math.log(z.numel()))
    counts = torch.as_tensor(counts, device=z.device, dtype=z.dtype)
    if validate:
        if counts.shape != z.shape or bool((counts < 0).any()) or not bool(counts.sum() > 0):
            raise ValueError("counts must be a nonnegative vector with positive total, one entry per ordered pair")
        if not bool(torch.isfinite(counts).all()):
            raise ValueError("counts must be finite")
    return torch.log(counts) - torch.log(counts.sum())


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        self.Q = torch.randn(cfg.n, cfg.d, generator=generator, device=cfg.device, dtype=torch.float64) * cfg.sigma
        self.K = torch.randn(cfg.n, cfg.d, generator=generator, device=cfg.device, dtype=torch.float64) * cfg.sigma
        self.theta = torch.tensor([cfg.q0, cfg.q0, cfg.u0, cfg.u0, cfg.x0, 0.],
                                  device=cfg.device, dtype=torch.float64)
        self.mask = ~torch.eye(cfg.n, device=cfg.device, dtype=torch.bool)
        self.aa, self.bb = torch.where(self.mask)
        self.frozen_scalar_indices = frozen_scalar_indices(cfg)

    def params(self):
        return [self.Q, self.K, self.theta]

    def gaps(self):
        scores = self.Q @ self.K.T
        return scores.diag()[:, None] - scores

    def quantities(self):
        return quantities_from_theta(self.cfg, self.theta)

    @property
    def active_parameter_count(self):
        return active_parameter_count(self.cfg)

    def rows(self, z):
        q, p, u, v, x, w, m, g, h, rho, lrho = self.quantities()
        terms = torch.stack((z * m * rho - 2 * h, z * m * rho - 3 * h,
                             -z * m * (1 - 2 * rho) - h), dim=-1)
        lo = torch.logsumexp(terms, dim=-1)
        lc = torch.logsumexp(recall_terms(z, m, h, rho, self.cfg.R), dim=-1)
        return terms, lo, lc, -torch.tanh(lo / 2), -torch.tanh(lc / 2)

    @torch.no_grad()
    def gradients(self, counts=None, validate_counts=True):
        """Exact ordinary-loss derivatives in sign/log-absolute representation.

        Pair counts give a stratified estimator: calibration is exact and both
        overwrite and recall losses are evaluated for every sampled pair.
        ``validate_counts=False`` is only for internally generated multinomial
        counts. It removes host synchronizations from the trusted GPU path.
        """
        c = self.cfg
        q, p, u, v, x, w, m, g, h, rho, lrho = self.quantities()
        z = self.gaps()[self.aa, self.bb]
        terms, lo, lc, so, sc = self.rows(z)
        rterms = recall_terms(z, m, h, rho, c.R)
        lp = _pair_log_mass(z, counts, validate=validate_counts)
        ltw = torch.log(2 * torch.abs(w))
        common_o = lp + math.log(c.wo) + ltw + F.logsigmoid(-w * so) - 2 * F.softplus(lo)
        common_c = lp + math.log(c.wc) + ltw + F.logsigmoid(-w * sc) - 2 * F.softplus(lc)
        la, lb, le = (common_o[:, None] + terms).unbind(-1)
        recall_pressure = common_c[:, None] + rterms
        lf = recall_pressure[:, 0]
        # P is the derivative with respect to M=z*m, divided by sign(w).
        logs_p = torch.cat((torch.stack((la + lrho, lb + lrho,
                                        le + torch.log(torch.abs(1 - 2 * rho)),
                                        lf + torch.log1p(-rho)), dim=-1),
                            recall_pressure[:, 1:]), dim=-1)
        signs_p = -torch.ones_like(logs_p)
        signs_p[:, :2] = 1.
        signs_p[:, 2] = -torch.sign(1 - 2 * rho)
        ps, pl = ssum(signs_p, logs_p, dim=-1)
        ps = ps * torch.sign(w)
        dzs, dzl = ps * torch.sign(m), pl + torch.log(torch.abs(m))
        shift = torch.max(dzl)
        shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
        weights = torch.zeros(c.n, c.n, device=z.device, dtype=z.dtype)
        weights[self.aa, self.bb] = dzs * torch.exp(dzl - shift)
        score_grad = -weights
        score_grad.diagonal().add_(weights.sum(dim=1))
        sq, lq = slog(score_grad @ self.K)
        sk, lk = slog(score_grad.T @ self.Q)
        sm, lm = ssum(ps * torch.sign(z), pl + torch.log(torch.abs(z)))
        rec_offsets = torch.arange(1, c.R, dtype=z.dtype, device=z.device)
        logs_h = torch.cat((torch.stack((la + math.log(2), lb + math.log(3), le), dim=-1),
                            recall_pressure + rec_offsets.log()), dim=-1)
        signs_h = torch.ones_like(logs_h)
        signs_h[:, :3] = -1.
        sh, lh = ssum(signs_h * torch.sign(w), logs_h)
        # Only the first recall distractor's binder leaks the queried key.
        binder_pressure = torch.logsumexp(torch.stack((la, lb, le + math.log(2), lf), dim=-1), dim=-1)
        sg, lg = ssum(-torch.sign(z * m * w), binder_pressure + torch.log(torch.abs(z * m))
                     + math.log(2) + lrho + torch.log1p(-rho))
        sw, lw = ssum(torch.cat((-torch.ones(1, device=z.device, dtype=z.dtype), -torch.sign(so), -torch.sign(sc))),
                       torch.cat(((math.log(c.kcal) + F.logsigmoid(-w)).reshape(1),
                                  lp + math.log(c.wo) + torch.log(torch.abs(so)) + F.logsigmoid(-w * so),
                                  lp + math.log(c.wc) + torch.log(torch.abs(sc)) + F.logsigmoid(-w * sc))))
        st = torch.stack((sm * torch.sign(p), sm * torch.sign(q), sg * torch.sign(v),
                          sg * torch.sign(u), sh, sw))
        lt = torch.stack((lm + torch.log(torch.abs(c.c0 * p)), lm + torch.log(torch.abs(c.c0 * q)),
                          lg + torch.log(torch.abs(v)) + F.logsigmoid(u * v),
                          lg + torch.log(torch.abs(u)) + F.logsigmoid(u * v),
                          lh + F.logsigmoid(x), lw))
        if self.frozen_scalar_indices:
            indices = list(self.frozen_scalar_indices)
            st[indices], lt[indices] = 0., -torch.inf
        logloss = torch.logsumexp(torch.cat(((math.log(c.kcal) + logsoftplus(-w)).reshape(1),
                                             lp + math.log(c.wo) + logsoftplus(-w * so),
                                             lp + math.log(c.wc) + logsoftplus(-w * sc))), dim=0)
        finite = torch.isfinite(dzl)
        high = torch.where(finite, dzl, -torch.inf).max()
        low = torch.where(finite, dzl, torch.inf).min()
        span = torch.where(finite.any(), high - low, torch.zeros_like(w))
        return [(sq, lq + shift), (sk, lk + shift), (st, lt)], {"logloss": logloss, "table_log_span": span}


def native_loss(model, counts=None, params=None):
    """Conventional PyTorch autograd loss, used as an independent derivative check.

    This path may underflow at extreme checkpoints; it never substitutes a
    log-gradient result when native arithmetic loses a derivative.
    """
    Q, K, theta = model.params() if params is None else params
    c = model.cfg
    q, p, u, v, x, w, m, g, h, rho, lrho = quantities_from_theta(c, theta, detach_frozen=True)
    scores = Q @ K.T
    z = (scores.diag()[:, None] - scores)[model.mask]
    log_o = torch.logsumexp(torch.stack((z * m * rho - 2 * h, z * m * rho - 3 * h,
                                       -z * m * (1 - 2 * rho) - h), dim=-1), dim=-1)
    log_c = torch.logsumexp(recall_terms(z, m, h, rho, c.R), dim=-1)
    loss = c.wo * F.softplus(w * torch.tanh(log_o / 2)) + c.wc * F.softplus(w * torch.tanh(log_c / 2))
    return c.kcal * F.softplus(-w) + (loss * _pair_log_mass(z, counts).exp()).sum()


def native_gradients(model, counts=None):
    with torch.enable_grad():
        params = [p.detach().clone().requires_grad_(True) for p in model.params()]
        loss = native_loss(model, counts, params)
        gradients = [p.detach() for p in torch.autograd.grad(loss, params)]
    return gradients, {"loss": loss.detach(), "logloss": loss.detach().log(),
                       "native_loss_underflow": bool(loss.detach() == 0)}


def dense_stream(params, config, keys, values, query, label=1.):
    """Literal two-head forward pass on serialized key/value record inputs.

    Returns ordinary answer loss, record attention, and correct-answer logit.
    The binder scores raw-token offsets 1 and 3. Retrieval uses slope h/2 per
    raw token, equivalent to h per record; the query is never a candidate.
    """
    Q, K, theta = params
    q, p, u, v, x, w = theta.unbind()
    if config.gate_mode == "both_frozen":
        u, v = u.detach(), v.detach()
    if config.gate_mode != "learned":
        x = x.detach()
    g, h = F.softplus(u * v), F.softplus(x)
    keys = torch.as_tensor(keys, device=Q.device, dtype=torch.long)
    values = torch.as_tensor(values, device=Q.device, dtype=Q.dtype)
    if keys.ndim != 1 or values.shape != keys.shape or keys.numel() < 1:
        raise ValueError("A stream needs equally sized nonempty key/value vectors")
    if bool(((keys < 0) | (keys >= config.n)).any()) or not 0 <= int(query) < config.n:
        raise ValueError("Key IDs must be in the configured vocabulary")
    if not bool(((values == -1) | (values == 1)).all()) or float(label) not in (-1., 1.):
        raise ValueError("Values and answer label must be -1 or +1")
    vectors = K[keys]
    binding_weights = torch.stack((-g, config.D - 3 * g)).softmax(dim=0)
    bound = torch.cat((vectors[:1], binding_weights[0] * vectors[1:] + binding_weights[1] * vectors[:-1]), dim=0)
    raw_distances = 2 * torch.arange(keys.numel(), 0, -1, device=Q.device, dtype=Q.dtype)
    scores = config.c0 * q * p * (bound @ Q[int(query)]) - (h / 2) * raw_distances
    attention = scores.softmax(dim=0)
    correct_logit = label * w * torch.sum(attention * values)
    return F.softplus(-correct_logit), attention, correct_logit


def dense_objective(model, counts=None, params=None):
    """Slow literal-stream objective, including both label complements."""
    params = model.params() if params is None else params
    c = model.cfg
    rows = []
    for a in range(c.n):
        for b in range(c.n):
            if a == b:
                continue
            o = sum(dense_stream(params, c, [a, a, b, a], [-y, -y, -y, y], a, y)[0] for y in (-1, 1)) / 2
            rec = sum(dense_stream(params, c, [a] + [b] * (c.R - 1), [y] + [-y] * (c.R - 1), a, y)[0] for y in (-1, 1)) / 2
            rows.append(c.wo * o + c.wc * rec)
    rows = torch.stack(rows)
    calibration = torch.stack([dense_stream(params, c, [a], [y], a, y)[0]
                               for a in range(c.n) for y in (-1, 1)]).mean()
    return c.kcal * calibration + (rows * _pair_log_mass(rows, counts).exp()).sum()


class LogAdam:
    """Bias-corrected Adam with signed-log first and log second moments.

    All buffers start at zero. Moment histories persist across acquisition and
    continuation. Only storage arithmetic differs from conventional Adam.
    """
    def __init__(self, params, b1, b2):
        if not 0 <= b1 < 1 or not b1 ** 2 < b2 < 1:
            raise ValueError("Require 0 <= beta1 < 1 and beta1**2 < beta2 < 1")
        self.b1, self.b2, self.t = b1, b2, 0
        self.ms = [torch.zeros_like(p) for p in params]
        self.ml = [torch.full_like(p, -torch.inf) for p in params]
        self.vl = [torch.full_like(p, -torch.inf) for p in params]

    @torch.no_grad()
    def step(self, params, grads, rates, logeps):
        self.t += 1
        correction_m = math.log1p(-self.b1 ** self.t)
        correction_v = math.log1p(-self.b2 ** self.t)
        directions = []
        for i, (parameter, (sign, logabs), rate) in enumerate(zip(params, grads, rates)):
            self.ms[i], self.ml[i] = sadd(self.ms[i], self.ml[i] + (math.log(self.b1) if self.b1 else -math.inf),
                                          sign, logabs + math.log1p(-self.b1))
            self.vl[i] = torch.logaddexp(self.vl[i] + math.log(self.b2), 2 * logabs + math.log1p(-self.b2))
            denominator = torch.logaddexp(.5 * (self.vl[i] - correction_v), torch.full_like(parameter, logeps))
            direction = torch.where(self.ms[i] == 0, 0., self.ms[i] * torch.exp(self.ml[i] - correction_m - denominator))
            parameter.add_(-rate * direction)
            directions.append(direction)
        return directions


@torch.no_grad()
def gradient_step(model, grads, rates):
    for parameter, (sign, logabs), rate in zip(model.params(), grads, rates):
        parameter.add_(-rate * sign * torch.exp(logabs))


def _mask_rates(config, scalar_rates):
    scalar_rates[list(frozen_scalar_indices(config))] = 0.
    return scalar_rates


def rates_at(c, kind, t):
    if kind not in ("sgd", "adam", "adam_annealed", "adam_fixed"):
        raise ValueError("kind must be sgd, adam, adam_annealed, or adam_fixed")
    base = (c.lr_adam if kind.startswith("adam") else c.lr_sgd) * (1 + t / c.offset) ** (-c.power)
    table = c.table_lr * (1 + t / c.offset) ** (-c.table_power)
    multipliers = _mask_rates(c, torch.tensor([c.am, c.am, c.ag, c.ag, c.ax, c.aw], device=c.device, dtype=torch.float64))
    return base, [table, table, base * multipliers]


def acquisition_rates(c, kind):
    """Three ordinary-loss updates with deterministic rates fixed before draws.

    SGD rates use the exact zero-gap objective at this R and initialization.
    These practical finite constants do not instantiate the proof's unspecified
    sufficiently-small bounds or the extra R>2 basin preparation protocol.
    """
    if kind.startswith("adam"):
        theta = _mask_rates(c, torch.tensor([1e-5] * 5 + [.005], device=c.device, dtype=torch.float64))
        return [[c.gamma / math.sqrt(c.d)] * 2 + [theta.clone()] for _ in range(3)]
    if kind != "sgd":
        raise ValueError("Acquisition supports sgd or adam")
    probe = Model(c)
    probe.Q.zero_()
    probe.K.zero_()
    sw, lw = probe.gradients()[0][-1]
    fw = float(sw[-1] * lw[-1].exp())
    if not math.isfinite(fw) or fw >= 0:
        raise ValueError(f"Initial decoder derivative must be negative; got {fw}")
    probe.theta[-1] = .2
    with torch.enable_grad():
        z = torch.zeros((), device=c.device, dtype=torch.float64, requires_grad=True)
        _, _, _, so, sc = probe.rows(z)
        phi = c.wo * F.softplus(-.2 * so) + c.wc * F.softplus(-.2 * sc)
        dstar = -float(torch.autograd.grad(phi, z)[0])
    if not math.isfinite(dstar) or dstar <= 0:
        raise ValueError(f"Acquisition requires a positive zero-gap matching signal; dstar={dstar}")
    astar = dstar / (c.n - 1)
    theta = _mask_rates(c, torch.tensor([1e-7] * 5 + [1e-4], device=c.device, dtype=torch.float64))
    first = theta.clone()
    first[-1] = .2 / (-fw)
    return [[1e-7, 1e-7, first],
            [c.gamma / (astar * c.sigma * math.sqrt(c.d))] * 2 + [theta.clone()],
            [1 / astar] * 2 + [theta.clone()]]


def _geom_log(count, h):
    if count == 0:
        return torch.full_like(h, -torch.inf)
    if float(h) == 0:
        return torch.full_like(h, math.log(count))
    return torch.log(-torch.expm1(-count * h)) - torch.log(-torch.expm1(-h))


def boundary_log_odds(gaps, m, h, rho, lag, prefix_count):
    """Exact witness odds: stale a-prefix, correct a, then lag-1 wrong b's.

    ``prefix_count=math.inf`` gives the infinite-prefix limit analytically.
    """
    if not isinstance(lag, int) or lag < 1 or prefix_count < 0:
        raise ValueError("lag must be a positive integer and prefix_count nonnegative")
    result = (-h + _geom_log(prefix_count, h)).expand_as(gaps)
    if lag >= 2:
        result = torch.logaddexp(result, -m * (1 - rho) * gaps + h)
    if lag >= 3:
        result = torch.logaddexp(result, -m * gaps + (lag - 1) * h + _geom_log(lag - 2, h))
    return result


def _sequence(keys, values, query, answer, family, pair_id=None):
    tokens = []
    for key, value in zip(keys, values):
        tokens.extend((f"key_{key}", f"value_{value:+d}"))
    tokens.extend(("?", f"key_{query}"))
    matching = [i for i, key in enumerate(keys) if key == query]
    target = matching[-1]
    return {"family": family, "pair_id": pair_id, "keys": keys, "values": values,
            "query": query, "answer": answer, "target_lag": len(keys) - target,
            "tokens": tokens}


def pair_cases(config):
    """Exactly N(N-1) pair cases; each contains four supervised sequences."""
    result = []
    for a in range(config.n):
        for b in range(config.n):
            if a == b:
                continue
            pair_id = len(result)
            overwrite = [_sequence([a, a, b, a], [-y, -y, -y, y], a, y, "overwrite", pair_id) for y in (-1, 1)]
            recall = [_sequence([a] + [b] * (config.R - 1), [y] + [-y] * (config.R - 1),
                                a, y, "recall", pair_id) for y in (-1, 1)]
            result.append({"pair_id": pair_id, "query_key": a, "distractor_key": b,
                           "overwrite": overwrite, "recall": recall})
    return result


def export_dataset(config, directory):
    """Export the pair-indexed data and its fully expanded weighted sequences."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cases = pair_cases(config)
    calibration = [_sequence([a], [y], a, y, "calibration") for a in range(config.n) for y in (-1, 1)]
    sequences = []
    for case in cases:
        for family, weight in (("overwrite", config.wo), ("recall", config.wc)):
            for row in case[family]:
                sequences.append({**row, "population_weight": weight / (2 * len(cases))})
    sequences.extend({**row, "population_weight": config.kcal / len(calibration)} for row in calibration)
    manifest = {"key_vocabulary_size": config.n, "pair_cases": len(cases), "recall_lag": config.R,
                "base_recall_sequences": len(cases), "base_overwrite_sequences": len(cases),
                "paired_sequences_with_complements": 4 * len(cases),
                "calibration_sequences": len(calibration), "total_expanded_sequences": len(sequences),
                "mixture_weights": {"calibration": config.kcal, "overwrite": config.wo, "recall": config.wc},
                "lag_units": "records, newest record at lag 1", "query_is_attention_candidate": False,
                "dataset_items": "N(N-1) pair cases; each evaluates both families and complements",
                "files": {"pair_cases": "pairs.jsonl", "expanded_sequences": "dataset.jsonl",
                          "base_recall_at_R": "recall_at_R.jsonl", "pair_index": "pair_index.csv"}}
    (directory / "pairs.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    (directory / "dataset.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sequences))
    (directory / "recall_at_R.jsonl").write_text("".join(json.dumps(case["recall"][1]) + "\n" for case in cases))
    (directory / "pair_index.csv").write_text("pair_id,query_key,distractor_key\n" + "".join(
        f'{case["pair_id"]},{case["query_key"]},{case["distractor_key"]}\n' for case in cases))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
