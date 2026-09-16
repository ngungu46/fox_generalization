"""Small, paired objective study. All changes are explicit in saved configuration."""

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path


@dataclass
class LossStudyConfig:
    profile: str = "colab"
    objectives: tuple = ("answer_only", "all_tokens")
    optimizers: tuple = ("sgd", "adam_fixed", "adam_annealed")
    gate_modes: tuple = ("factorized_constant", "direct_constant")
    seeds: tuple = (0,)
    steps: int = 2000
    batch_size: int = 32
    eval_every: int = 250
    checkpoint_every: int = 100
    diagnostic_every: int = 250
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    n_keys: int = 8
    n_values: int = 2
    data_mode: str = "records"
    train_support: str = "diverse"
    train_max_lag: int = 4
    train_max_prefix: int = 4
    older_stale_fraction: float = 0.5
    short_lags: tuple = (1, 2, 4)
    eval_lags: tuple = (1, 2, 4, 8, 16, 32)
    eval_prefixes: tuple = (0, 4, 16)
    eval_histories: int = 16
    qualification_threshold: float = 0.9
    probability_threshold: float = 0.9
    routing: str = "record"
    first_g0: float = 2.944102615
    retrieval_g0: float = 0.5
    retrieval_gate: str = "constant"
    qk_norm: bool = False
    output_gate: bool = False
    representation_schedule: str = "shared"
    representation_multiplier: float = 1.0
    representation_power: float = 2.0
    first_gate_multiplier: float = 1.5
    retrieval_gate_multiplier: float = 0.05
    output_multiplier: float = 1.0
    learning_rates: dict = field(
        default_factory=lambda: {
            "sgd": 0.1,
            "adam_fixed": 0.003,
            "adam_annealed": 0.003,
            "paper_adamw": 0.003,
        }
    )
    branch_learning_rates: dict = field(default_factory=dict)
    beta1: float = 0.1
    beta2: float = 0.1
    eps0: float = 1e-8
    epsilon_decay: float = 0.01
    lr_offset: float = 1000.0
    lr_power: float = 0.75
    weight_decay: float = 0.0
    clip_grad: float | None = None
    answer_weight: float = 1.0
    query_chunk: int = 128

    def __post_init__(self):
        for name in (
            "objectives",
            "optimizers",
            "gate_modes",
            "seeds",
            "short_lags",
            "eval_lags",
            "eval_prefixes",
        ):
            setattr(self, name, tuple(getattr(self, name)))
        if self.profile not in ("smoke", "colab", "custom"):
            raise ValueError("Unknown profile")
        allowed = {
            "objectives": {"answer_only", "all_tokens"},
            "optimizers": {"sgd", "adam_fixed", "adam_annealed", "paper_adamw"},
            "gate_modes": {"factorized_constant", "direct_constant", "original_data"},
        }
        for name, options in allowed.items():
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or not set(values) <= options
            ):
                raise ValueError(f"Invalid {name}: {values}")
        for name in (
            "steps",
            "batch_size",
            "eval_every",
            "checkpoint_every",
            "diagnostic_every",
            "d_model",
            "n_heads",
            "n_layers",
            "n_keys",
            "n_values",
            "train_max_lag",
            "eval_histories",
            "query_chunk",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_keys < 2 or self.n_values < 2 or self.n_layers < 2:
            raise ValueError("At least two keys, values, and layers are needed")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.data_mode not in ("records", "text_background"):
            raise ValueError("Unknown data_mode")
        if self.routing not in ("record", "causal"):
            raise ValueError("Unknown routing")
        if self.data_mode == "text_background" and self.routing != "causal":
            raise ValueError(
                "Text uses causal routing; record masks would supply a parser"
            )
        if self.train_support not in ("diverse", "theory_templates"):
            raise ValueError("Unknown train_support")
        if self.train_support == "theory_templates" and self.n_values != 2:
            raise ValueError("Theory templates use binary values")
        if self.train_support == "theory_templates" and (
            self.train_max_lag != 2 or self.train_max_prefix != 3
        ):
            raise ValueError("Literal theory templates have maximum lag 2 and prefix 3")
        if self.retrieval_gate not in ("constant", "data"):
            raise ValueError("Unknown retrieval_gate")
        if self.representation_schedule not in ("shared", "summable"):
            raise ValueError("Unknown representation_schedule")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Need distinct seeds")
        if not self.short_lags or max(self.short_lags) > self.train_max_lag:
            raise ValueError("Short lags must lie within the training range")
        if not set(self.short_lags) <= set(self.eval_lags) or 1 not in self.eval_lags:
            raise ValueError("Evaluation must include lag 1 and all short lags")
        if (
            tuple(sorted(set(self.eval_lags))) != self.eval_lags
            or min(self.eval_lags) < 1
        ):
            raise ValueError("eval_lags must be positive, sorted and unique")
        if (
            not self.eval_prefixes
            or 0 not in self.eval_prefixes
            or min(self.eval_prefixes) < 0
        ):
            raise ValueError("eval_prefixes must include zero and be nonnegative")
        if self.train_max_prefix < 0:
            raise ValueError("train_max_prefix must be nonnegative")
        if not 0 <= self.older_stale_fraction <= 1:
            raise ValueError("older_stale_fraction must lie in [0,1]")
        for value in (
            self.first_g0,
            self.retrieval_g0,
            self.eps0,
            self.answer_weight,
            self.lr_offset,
            self.representation_multiplier,
            self.first_gate_multiplier,
            self.retrieval_gate_multiplier,
            self.output_multiplier,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Decays, rates, epsilon and loss weights must be positive"
                )
        if "factorized_constant" in self.gate_modes and self.first_g0 <= math.log(2):
            raise ValueError(
                "Balanced positive binding factors require first_g0 > log(2)"
            )
        if not 0 <= self.beta1 < 1 or not self.beta1**2 < self.beta2 < 1:
            raise ValueError("Require beta1^2 < beta2 < 1")
        if self.epsilon_decay < 0 or self.weight_decay < 0 or self.lr_power < 0:
            raise ValueError("Schedule/decay values must be nonnegative")
        if self.representation_power <= 1:
            raise ValueError("Summable representation tail needs power > 1")
        if self.clip_grad is not None and self.clip_grad <= 0:
            raise ValueError("clip_grad must be positive or None")
        if any(
            not 0 < t <= 1
            for t in (self.qualification_threshold, self.probability_threshold)
        ):
            raise ValueError("Thresholds must be in (0,1]")
        for arm in self.optimizers:
            if (
                arm not in self.learning_rates
                or not math.isfinite(self.learning_rates[arm])
                or self.learning_rates[arm] <= 0
            ):
                raise ValueError(f"Need positive learning rate for {arm}")
        for value in self.branch_learning_rates.values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Branch learning rates must be finite and positive")

    @classmethod
    def for_profile(cls, profile="colab"):
        if profile == "smoke":
            return cls(
                profile="smoke",
                steps=2,
                batch_size=4,
                eval_every=2,
                checkpoint_every=1,
                diagnostic_every=2,
                d_model=16,
                n_heads=2,
                n_keys=4,
                train_max_lag=2,
                train_max_prefix=1,
                short_lags=(1, 2),
                eval_lags=(1, 2, 4),
                eval_prefixes=(0, 2),
                eval_histories=2,
            )
        return cls(profile=profile)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def branches(config):
    return [
        dict(
            branch=f"seed{seed}__{gate}__{optimizer}__{objective}",
            seed=seed,
            gate_mode=gate,
            optimizer=optimizer,
            objective=objective,
        )
        for seed in config.seeds
        for gate in config.gate_modes
        for optimizer in config.optimizers
        for objective in config.objectives
    ]


def intervention(config, name):
    """Change one named condition. Run each in a separate output directory."""
    changes = {
        "baseline": {},
        "unrestricted_routing": {"routing": "causal"},
        "data_dependent_retrieval": {"retrieval_gate": "data"},
        "qk_normalization": {"qk_norm": True},
        "learned_output_gate": {"output_gate": True},
        "slow_representations": {
            "representation_schedule": "summable",
            "representation_multiplier": 0.1,
        },
        "weight_decay": {"weight_decay": 0.01},
        "gradient_clipping": {"clip_grad": 1.0},
        "larger_epsilon": {"eps0": 1e-4},
        "answer_upweight": {"answer_weight": 16.0},
        "theory_template_support": {
            "train_support": "theory_templates",
            "train_max_lag": 2,
            "train_max_prefix": 3,
            "short_lags": (1, 2),
        },
        "longer_training_lag": {"train_max_lag": 8},
        "dense_stale_prefix": {"older_stale_fraction": 1.0},
        "weak_retrieval_init": {"retrieval_g0": 0.1},
        "faster_retrieval_updates": {"retrieval_gate_multiplier": 0.5},
    }
    if name not in changes:
        raise ValueError(f"Unknown intervention; choose one of {tuple(changes)}")
    return replace(config, **changes[name])
