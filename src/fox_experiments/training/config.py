"""Configuration for the small-scale, real-text optimizer experiment.

JSON files are the public experiment interface. Optimizer interventions, training
lags, and evaluation grids belong here rather than in notebook implementation cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """One acquisition / optimizer-continuation study.

    ``smoke`` verifies software only. ``pilot`` is the Colab study. ``replicate``
    increases replication and evaluation; it is not the original paper's full run.
    The full-training backend has its own configuration.
    """

    profile: str = "pilot"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    train_length: int = 256
    batch_size: int = 2
    accumulation: int = 1
    pretrain_steps: int = 100
    acquisition_steps: int = 400
    trial_steps: int = 40
    continuation_steps: int = 400
    save_every: int = 100
    seeds: tuple[int, ...] = (0,)
    train_lags: tuple[int, ...] = (32, 64, 96)
    test_lags: tuple[int, ...] = (32, 64, 96, 128, 256, 512)
    prefix_copies: tuple[int, ...] = (0, 16, 64)
    test_histories: int = 4
    validation_histories: int = 12
    lm_eval_length: int = 1024
    lm_eval_rows: int = 2
    answer_weight: float = 128.0
    mixture_probability: float = 0.5
    acquisition_lr: float = 0.001
    adam_lrs: tuple[float, ...] = (0.0003, 0.001, 0.003)
    sgd_lrs: tuple[float, ...] = (0.01, 0.05, 0.2)
    beta1: float = 0.1
    beta2: float = 0.1
    eps0: float = 1e-8
    epsilon_decay: float = 0.01
    lr_offset: float = 1000.0
    lr_power: float = 0.75
    first_gate_multiplier: float = 1.5
    retrieval_gate_multiplier: float = 0.05
    output_multiplier: float = 0.1
    g0: float = 2.944102615
    other_g0: float = 0.1
    short_threshold: float = 0.9
    query_chunk: int = 128
    gate_modes: tuple[str, ...] = (
        "factorized_constant",
        "direct_constant",
        "original_data",
    )
    optimizer_arms: tuple[str, ...] = ("sgd", "adam_fixed", "adam_annealed")
    include_paper_control: bool = True

    def __post_init__(self) -> None:
        tuple_fields = (
            "seeds",
            "train_lags",
            "test_lags",
            "prefix_copies",
            "adam_lrs",
            "sgd_lrs",
            "gate_modes",
            "optimizer_arms",
        )
        for name in tuple_fields:
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{name} must be a nonempty list or tuple")
            value = tuple(value)
            if len(value) != len(set(value)):
                raise ValueError(f"{name} must not contain duplicates")
            setattr(self, name, value)
        positive_integers = (
            "d_model",
            "n_heads",
            "n_layers",
            "train_length",
            "batch_size",
            "accumulation",
            "trial_steps",
            "continuation_steps",
            "save_every",
            "test_histories",
            "validation_histories",
            "lm_eval_length",
            "lm_eval_rows",
            "query_chunk",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("pretrain_steps", "acquisition_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_layers < 2:
            raise ValueError(
                "This binding/retrieval experiment needs at least two layers"
            )
        if not 0 <= self.mixture_probability <= 1:
            raise ValueError("mixture_probability must be in [0, 1]")
        if not 0 < self.short_threshold <= 1:
            raise ValueError("short_threshold must be in (0, 1]")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam beta1 and beta2 must be in [0, 1)")
        positive_numbers = (
            "answer_weight",
            "acquisition_lr",
            "eps0",
            "lr_offset",
            "g0",
            "other_g0",
            "first_gate_multiplier",
            "retrieval_gate_multiplier",
            "output_multiplier",
        )
        for name in positive_numbers:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.epsilon_decay < 0 or self.lr_power < 0:
            raise ValueError("epsilon_decay and lr_power must be nonnegative")
        if any(lr <= 0 for lr in self.adam_lrs + self.sgd_lrs):
            raise ValueError("Learning rate candidates must be positive")
        if any(
            not isinstance(lag, int) or lag < 1
            for lag in self.train_lags + self.test_lags
        ):
            raise ValueError("Lags must be positive integers")
        if max(self.train_lags) >= self.train_length:
            raise ValueError("Training lags must be smaller than train_length")
        if any(
            not isinstance(n, int) or n < 0 for n in self.prefix_copies + self.seeds
        ):
            raise ValueError("prefix_copies and seeds must be nonnegative integers")
        valid_gates = {
            "factorized_constant",
            "direct_constant",
            "original_data",
            "factorized_data",
        }
        if not set(self.gate_modes) <= valid_gates:
            raise ValueError(f"Unknown gate mode; choose from {sorted(valid_gates)}")
        if not set(self.optimizer_arms) <= {"sgd", "adam_fixed", "adam_annealed"}:
            raise ValueError(
                "optimizer_arms supports sgd, adam_fixed, and adam_annealed"
            )

    @classmethod
    def for_profile(cls, profile: str) -> "ExperimentConfig":
        """Return the original notebook's named experiment budgets."""
        if profile == "smoke":
            return cls(
                profile="smoke",
                d_model=16,
                n_heads=2,
                n_layers=2,
                train_length=96,
                batch_size=1,
                pretrain_steps=1,
                acquisition_steps=2,
                trial_steps=1,
                continuation_steps=2,
                save_every=1,
                train_lags=(24, 32),
                test_lags=(24, 32, 48),
                prefix_copies=(0, 2),
                test_histories=1,
                validation_histories=2,
                lm_eval_length=64,
                lm_eval_rows=1,
                adam_lrs=(0.001,),
                sgd_lrs=(0.05,),
                query_chunk=32,
            )
        if profile in ("pilot", "downscale"):
            return cls(profile=profile)
        if profile == "replicate":
            return cls(
                profile="replicate",
                seeds=(0, 1, 2),
                pretrain_steps=500,
                acquisition_steps=1500,
                trial_steps=100,
                continuation_steps=2000,
                test_histories=32,
                validation_histories=64,
                test_lags=(32, 64, 96, 128, 256, 512, 1024),
                lm_eval_length=4096,
                lm_eval_rows=8,
            )
        raise ValueError("profile must be smoke, pilot, downscale, or replicate")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        """Load explicit overrides on the selected profile; reject misspelled keys."""
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"Unknown experiment configuration fields: {sorted(unknown)}"
            )
        defaults = asdict(cls.for_profile(values.get("profile", "pilot")))
        return cls(**{**defaults, **values})

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Read a human-editable JSON configuration, including list-valued grids."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable copy of the configuration."""
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        """Write every resolved setting so runs are reviewable."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )


def source_gate_for(gate_mode: str) -> str:
    """Pair factorized/direct interventions by a common acquired function."""
    return (
        "original_data"
        if gate_mode in ("original_data", "factorized_data")
        else "direct_constant"
    )


def branch_specs(config: ExperimentConfig) -> list[tuple[str, str]]:
    """Enumerate gate × optimizer interventions and the optional paper control."""
    specs = [
        (gate, optimizer)
        for gate in config.gate_modes
        for optimizer in config.optimizer_arms
    ]
    if config.include_paper_control:
        specs.append(("original_data", "paper_adamw"))
    return specs


def budget(config: ExperimentConfig) -> dict[str, Any]:
    """Count training work; GPU runtime and evaluation overhead must be measured."""
    arms = branch_specs(config)
    sources = {source_gate_for(gate) for gate, _ in arms}
    trials = sum(
        len(config.sgd_lrs if arm == "sgd" else config.adam_lrs) for _, arm in arms
    )
    updates_per_seed = (
        len(sources) * (config.pretrain_steps + config.acquisition_steps)
        + trials * config.trial_steps
        + len(arms) * config.continuation_steps
    )
    updates = len(config.seeds) * updates_per_seed
    return {
        "training_updates": updates,
        "training_input_tokens": updates
        * config.batch_size
        * config.accumulation
        * config.train_length,
        "arms": len(arms),
        "seeds": len(config.seeds),
        "full_attention": True,
        "largest_probe_tokens_upper_bound": (
            max(config.train_length, max(config.test_lags) + 64)
            + max(config.prefix_copies) * 20
        ),
        "note": "Validation/evaluation add compute. Measure time on your GPU; no runtime guarantee.",
    }
