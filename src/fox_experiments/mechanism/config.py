"""Configuration for the controlled two-stage binding experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MechanismConfig:
    n_keys: int = 8
    width: int = 32
    sigma: float = 0.01
    seed: int = 0
    acquisition_steps: int = 300
    acquisition_lr: float = 0.01
    continuation_steps: int = 1000
    adam_lr: float = 0.015
    sgd_lr: float = 5.0
    tau: float = 1000.0
    power: float = 0.75
    table_lr: float = 1e-9
    table_power: float = 2.0
    eps0: float = 1e-8
    eps_decay: float = 0.01
    beta1: float = 0.1
    beta2: float = 0.1
    alpha_m: float = 1.0
    alpha_g: float = 1.5
    alpha_x: float = 0.05
    alpha_w: float = 0.1
    c0: float = 1.0
    D: float = 0.0
    calibration_weight: float = 0.6
    overwrite_weight: float = 0.2
    recall_weight: float = 0.2
    log_every: int = 100
    radius_cap: int = 256
    error_threshold: float = 0.01
    dtype: str = "float64"
