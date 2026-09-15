"""Learned two-stage binding model and its function-matched gate ablation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .config import MechanismConfig


class BindingFoX(nn.Module):
    """Independent learned key tables, a two-key local binder, record-only readout."""

    def __init__(self, config: MechanismConfig, gate="factorized", device="cpu"):
        super().__init__()
        if gate not in ("factorized", "direct"):
            raise ValueError("gate must be factorized or direct")
        self.config, self.gate = config, gate
        dtype = getattr(torch, config.dtype)
        generator = torch.Generator().manual_seed(config.seed)
        for name in ("Q", "K"):
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.randn(
                        config.n_keys, config.width, generator=generator, dtype=dtype
                    )
                    * config.sigma
                ),
            )
        for name, value in (("q", 0.5), ("p", 0.5), ("x", -1.0), ("w", 0.0)):
            setattr(self, name, nn.Parameter(torch.tensor(value, dtype=dtype)))
        if gate == "factorized":
            self.u = nn.Parameter(torch.tensor(1.7, dtype=dtype))
            self.v = nn.Parameter(torch.tensor(1.7, dtype=dtype))
        else:
            self.z = nn.Parameter(torch.tensor(1.7**2, dtype=dtype))
        self.to(device)

    def quantities(self):
        g = F.softplus(self.u * self.v if self.gate == "factorized" else self.z)
        return (
            self.config.c0 * self.q * self.p,
            g,
            F.softplus(self.x),
            torch.sigmoid(self.config.D - 2 * g),
        )

    def forward(self, keys, values, query, return_attention=False):
        m, _, h, rho = self.quantities()
        raw = self.K[keys]
        previous = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
        bound = (1 - rho) * raw + rho * previous
        content = torch.einsum("bd,bld->bl", self.Q[query], bound) * m
        lags = torch.arange(keys.shape[1], 0, -1, dtype=raw.dtype, device=raw.device)
        attention = (content - h * lags).softmax(-1)
        logits = self.w * (attention * values).sum(-1)
        return (logits, attention) if return_attention else logits

    def gaps(self):
        scores = self.Q @ self.K.T
        return scores.diagonal()[:, None] - scores

    @torch.no_grad()
    def converted(self, gate):
        """Preserve the complete effective function when replacing u*v by z."""
        other = BindingFoX(self.config, gate, str(self.Q.device))
        for name in ("Q", "K", "q", "p", "x", "w"):
            getattr(other, name).copy_(getattr(self, name))
        latent = self.u * self.v if self.gate == "factorized" else self.z
        if gate == "direct":
            other.z.copy_(latent)
        elif self.gate == "factorized":
            other.u.copy_(self.u)
            other.v.copy_(self.v)
        else:
            if latent <= 0:
                raise ValueError(
                    "Positive balanced factors require positive gate latent."
                )
            other.u.copy_(latent.sqrt())
            other.v.copy_(latent.sqrt())
        return other
