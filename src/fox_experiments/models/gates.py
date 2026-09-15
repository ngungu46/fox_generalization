"""First-layer gate interventions and ordinary data-dependent FoX gates.

The upstream FoX implementation writes log retention as logsigmoid(z). Here
we parameterize a positive decay, softplus(raw), with raw = -z. Thus
log retention = -decay; the score is otherwise identical.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    """Stable inverse, including the high-decay initialization used in theory."""
    if value <= 0:
        raise ValueError("softplus output must be strictly positive")
    return value + math.log(-math.expm1(-value))


class ForgetGate(nn.Module):
    """Decay a >= 0. Parameters and prefix sums remain FP32 under autocast.

    The constant modes intervene on block zero only. The data modes have
    a_t,h = softplus(w_h @ x_t + b_h); factorized_data replaces its argument
    by u_h * (w_h @ x_t + v_h), an empirical extension beyond the constant
    gate theorem. Its u=1 initialization exactly matches original_data.
    """

    def __init__(
        self,
        width: int,
        heads: int,
        mode: str,
        g0: float,
        seed: int,
        weight_std: float = 0.02,
    ):
        super().__init__()
        self.mode, self.heads, self.width = mode, heads, width
        raw = inverse_softplus(g0)
        if mode == "direct_constant":
            self.b = nn.Parameter(torch.full((heads,), raw))
        elif mode == "factorized_constant":
            # Positive raw decay supports the manuscript's balanced u=v>0
            # initialization. At weak decay (g0 <= log 2), real equal factors
            # cannot have a negative product, so use an explicitly asymmetric
            # function-matched sensitivity arm instead of a locked zero pair.
            factor_u, factor_v = (
                (math.sqrt(raw), math.sqrt(raw)) if raw > 0 else (1.0, raw)
            )
            self.u = nn.Parameter(torch.full((heads,), factor_u))
            self.v = nn.Parameter(torch.full((heads,), factor_v))
        elif mode in ("original_data", "factorized_data"):
            # Private RNG means gate parameter count cannot perturb core weights.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.weight = nn.Parameter(
                torch.randn(heads, width, generator=generator) * weight_std
            )
            if mode == "factorized_data":
                self.u = nn.Parameter(torch.ones(heads))
                self.v = nn.Parameter(torch.full((heads,), raw))
            else:
                self.b = nn.Parameter(torch.full((heads,), raw))
        else:
            raise ValueError(f"Unknown gate mode {mode}")

    def forward(self, x: Tensor) -> Tensor:
        """Return [batch, heads, time] decay rates."""
        # Explicitly disabling autocast is necessary: F.linear otherwise lowers
        # precision even when all input tensors have first been cast to float.
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.mode == "direct_constant":
                raw = self.b.float()[None, :, None].expand(x.shape[0], -1, x.shape[1])
            elif self.mode == "factorized_constant":
                raw = (self.u.float() * self.v.float())[None, :, None].expand(
                    x.shape[0], -1, x.shape[1]
                )
            elif self.mode == "original_data":
                raw = F.linear(
                    x.float(), self.weight.float(), self.b.float()
                ).transpose(1, 2)
            else:
                raw = (
                    F.linear(x.float(), self.weight.float(), self.v.float())
                    * self.u.float()
                ).transpose(1, 2)
            return F.softplus(raw)
