"""Compile the complete FP64 gradient-and-optimizer update.

The eager implementation remains the numerical reference. Compilation fuses
pointwise signed-log arithmetic, including Adam's large moment arrays, while
retaining the objective, precision, schedule, and all checkpoint state. No
CUDA graphs, approximate dtype, optimizer substitution, or fallback is used.

Only computation belongs in this graph: sampling, logging, checkpointing,
health checks, and the optimizer's Python step counter stay outside. Changing
rates, bias corrections and epsilon enter as tensors, so a new training step
does not specialize/recompile the graph. Fixed-shape populations are the
intended use; repeatedly changing online batch shapes may compile new graphs.
"""
import math
import time

import torch

from ..legacy.core import sadd


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _signature(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, (int, float, bool, str)):
        return value
    # RandomBatch is intentionally passed as tensors through its dataclass;
    # its content must not become a compiler guard or audit cache key.
    return tuple((name, _signature(getattr(value, name))) for name in value.__dataclass_fields__)


def _compare_gradients(expected, actual, tolerance):
    blocks = []
    for (left_sign, left_log), (right_sign, right_log) in zip(expected, actual):
        same_sign = torch.equal(left_sign, right_sign)
        delta = torch.where(left_sign != 0, (left_log - right_log).abs(), 0.)
        maximum = float(delta.max())
        blocks.append(dict(signs_equal=bool(same_sign), max_absolute_log_gradient_error=maximum))
        if not same_sign or not math.isfinite(maximum) or maximum > tolerance:
            raise RuntimeError(f"Compiled signed-log gradients differ from eager: {blocks}")
    return blocks


class CompiledUpdate:
    """Audited fullgraph update for one model and its persistent optimizer.

    ``gradient_function(counts, batch)`` returns ``(gradients, info)``. If it is
    omitted, use the ordered-pair objective. For random data, pass a closure
    calling ``random_gradients(model, batch, counts, validate=False)``.

    ``update(rates, step, counts=None, batch=None)`` updates parameters and Adam
    moments exactly once, and returns pre-update gradients and diagnostics.
    Parameters and moment buffers are captured lazily on first call, allowing
    eager acquisition or checkpoint restoration before compiled continuation.
    The first call at each input shape compares gradients, objective, model,
    and complete Adam moments; it restores the original state on failure.
    The initial audit/compilation duration is exposed separately from runtime.

    ``backend`` is primarily for testing; production uses PyTorch Inductor.
    CUDA compilation and speed must be checked on the actual GPU. CPU graph
    capture alone is not evidence of A100 speed or numerical parity.
    """

    def __init__(self, model, optimizer=None, gradient_function=None, *, backend="inductor", tolerance=1e-7):
        if not hasattr(torch, "compile"):
            raise RuntimeError("Compiled updates require torch.compile")
        if any(parameter.dtype != torch.float64 for parameter in model.params()):
            raise ValueError("Restricted experiments require float64 model parameters")
        self.model, self.optimizer = model, optimizer
        self.compile_seconds, self.audits = 0., []
        self.tolerance = tolerance
        self._audited = set()
        self._gradient_function = gradient_function or (lambda counts, batch: model.gradients(counts, validate_counts=False))
        self._backend = backend
        self._compiled = None

    def _initialize(self):
        if self._compiled is not None:
            return
        optimizer, gradient_function = self.optimizer, self._gradient_function
        self._parameters = self.model.params()
        self._moments = [] if optimizer is None else [*optimizer.ms, *optimizer.ml, *optimizer.vl]
        if optimizer is not None:
            b1, b2 = optimizer.b1, optimizer.b2
            log_b1 = math.log(b1) if b1 else -math.inf
            log_b2, log_one_minus_b1, log_one_minus_b2 = math.log(b2), math.log1p(-b1), math.log1p(-b2)
            ms, ml, vl = optimizer.ms, optimizer.ml, optimizer.vl

        @torch.no_grad()
        def update(control, scalar_rates, counts, batch):
            gradients, info = gradient_function(counts, batch)
            rates = (control[0], control[1], scalar_rates)
            for index, (parameter, (sign, logabs), rate) in enumerate(zip(self._parameters, gradients, rates)):
                if optimizer is None:
                    parameter.add_(-rate * sign * torch.exp(logabs))
                else:
                    first_sign, first_log = sadd(ms[index], ml[index] + log_b1,
                                                 sign, logabs + log_one_minus_b1)
                    second_log = torch.logaddexp(vl[index] + log_b2, 2 * logabs + log_one_minus_b2)
                    denominator = torch.logaddexp(.5 * (second_log - control[3]), control[4])
                    direction = torch.where(first_sign == 0, 0.,
                                            first_sign * torch.exp(first_log - control[2] - denominator))
                    parameter.add_(-rate * direction)
                    ms[index].copy_(first_sign)
                    ml[index].copy_(first_log)
                    vl[index].copy_(second_log)
            return gradients, info

        self._eager = update
        self._compiled = torch.compile(update, backend=self._backend, mode="default", fullgraph=True, dynamic=False)

    def _state(self):
        return [value.detach().clone() for value in (*self._parameters, *self._moments)]

    @torch.no_grad()
    def _restore(self, state):
        for value, saved in zip((*self._parameters, *self._moments), state):
            value.copy_(saved)

    def _inputs(self, rates, step):
        # Rates[0:2] are host scalars produced by the established schedule.
        # The six scalar-coordinate rates are already a device tensor.
        correction_m = correction_v = 0.
        if self.optimizer is not None:
            next_t = self.optimizer.t + 1
            correction_m = math.log1p(-self.optimizer.b1 ** next_t)
            correction_v = math.log1p(-self.optimizer.b2 ** next_t)
        logeps = 3 * math.log(self.model.cfg.sigma) - self.model.cfg.eps_decay * (step - 1)
        control = torch.tensor([rates[0], rates[1], correction_m, correction_v, logeps],
                               dtype=torch.float64, device=self.model.Q.device)
        scalar_rates = torch.as_tensor(rates[2], dtype=torch.float64, device=self.model.Q.device)
        return control, scalar_rates

    def __call__(self, rates, step, counts=None, batch=None):
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("step must be a positive integer")
        self._initialize()
        if self.optimizer is not None and any(
                current is not captured for current, captured in
                zip((*self.optimizer.ms, *self.optimizer.ml, *self.optimizer.vl), self._moments)):
            raise RuntimeError("Adam buffers were replaced after compiled training began; reconstruct CompiledUpdate after restoring or running an eager optimizer step")
        control, scalar_rates = self._inputs(rates, step)
        key = (_signature(counts), _signature(batch))
        if key in self._audited:
            try:
                result = self._compiled(control, scalar_rates, counts, batch)
            except Exception as exc:
                raise RuntimeError("Compiled update failed; no eager fallback was taken. Resume the last validated checkpoint with compilation disabled.") from exc
        else:
            original = self._state()
            _sync(self.model.Q.device)
            started = time.monotonic()
            try:
                expected, expected_info = self._eager(control, scalar_rates, counts, batch)
                expected_state = self._state()
                self._restore(original)
                actual, actual_info = self._compiled(control, scalar_rates, counts, batch)
                _sync(self.model.Q.device)
                blocks = _compare_gradients(expected, actual, self.tolerance)
                loss_error = abs(float(expected_info["logloss"] - actual_info["logloss"]))
                if not math.isfinite(loss_error) or loss_error > 1e-9:
                    raise RuntimeError(f"Compiled objective differs from eager: log-loss error {loss_error}")
                for left, right in zip(expected_state, (*self._parameters, *self._moments)):
                    torch.testing.assert_close(right, left, rtol=1e-8, atol=1e-12, equal_nan=False)
                self.audits.append(dict(input_kind=str(key), blocks=blocks,
                                        absolute_log_loss_error=loss_error,
                                        parameter_and_optimizer_state_audit=True,
                                        compilation="full gradient and optimizer update; float64"))
                self._audited.add(key)
                result = actual, actual_info
            except Exception as exc:
                self._restore(original)
                raise RuntimeError("Compiled update or numerical audit failed; original model and optimizer buffers were restored. No eager fallback was taken.") from exc
            finally:
                self.compile_seconds += time.monotonic() - started
        if self.optimizer is not None:
            self.optimizer.t += 1
        return result
