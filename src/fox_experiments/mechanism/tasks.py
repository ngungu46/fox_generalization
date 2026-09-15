"""Overwrite and lag-R recall training objectives."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def templates(model, recall_lag=2):
    """All ordered pairs, original O and a fixed R-record recall template.

    Value complements have the identical signed-logit loss by linearity.
    R=2 is the manuscript objective; R>2 is an exploratory change of objective.
    """
    if recall_lag < 2:
        raise ValueError("recall_lag must be at least 2")
    a, b = torch.where(
        ~torch.eye(model.config.n_keys, dtype=torch.bool, device=model.Q.device)
    )
    overwrite = torch.stack([a, a, b, a], 1)
    ov = torch.tensor(
        [-1.0, -1.0, -1.0, 1.0], dtype=model.Q.dtype, device=model.Q.device
    ).expand(len(a), -1)
    recall = torch.cat([a[:, None], b[:, None].expand(-1, recall_lag - 1)], 1)
    rv = -torch.ones_like(recall, dtype=model.Q.dtype)
    rv[:, 0] = 1.0
    return (overwrite, ov, a), (recall, rv, a)


def loss_function(model, data):
    cfg = model.config
    return (
        cfg.calibration_weight * F.softplus(-model.w)
        + cfg.overwrite_weight * F.softplus(-model(*data[0])).mean()
        + cfg.recall_weight * F.softplus(-model(*data[1])).mean()
    )
