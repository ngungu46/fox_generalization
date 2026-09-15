"""From-initialization, natural-text-only default-FoX recipe comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass
class PaperBaselineConfig:
    """Scaled paper recipe with optimizer and optional paired gate interventions.

    These finite pilot settings are not the paper's original token/model budget.
    Rates are declared candidates, not selected or validated hyperparameters.
    """

    profile: str = "pilot"
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 2
    train_length: int = 256
    batch_size: int = 4
    accumulation: int = 2
    steps: int = 2000
    save_every: int = 100
    seed: int = 0
    arms: tuple[str, ...] = ("paper_adamw", "adam_fixed", "adam_annealed", "sgd")
    model_variants: tuple[str, ...] = ("original_data",)
    comparison_first_g0: float = 2.944102615
    comparison_other_g0: float = 0.1
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "paper_adamw": 0.002,
            "adam_fixed": 0.001,
            "adam_annealed": 0.001,
            "sgd": 0.01,
        }
    )
    paper_warmup_steps: int | None = None
    beta1: float = 0.1
    beta2: float = 0.1
    eps0: float = 1e-8
    epsilon_decay: float = 0.01
    lr_offset: float = 1000.0
    lr_power: float = 0.75
    first_gate_multiplier: float = 1.5
    retrieval_gate_multiplier: float = 0.05
    output_multiplier: float = 0.1
    eval_contexts: tuple[int, ...] = (256, 512, 1024)
    eval_rows: int = 4
    eval_target_tokens: int = 32
    query_chunk: int = 128
    logit_chunk: int = 64
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = False
    precision: str = "fp32"
    short_retrieval_diagnostic: bool = False
    short_lags: tuple[int, ...] = (32, 64, 96)
    short_histories: int = 12
    short_threshold: float = 0.9

    def __post_init__(self):
        for name in ("arms", "model_variants", "eval_contexts", "short_lags"):
            value = tuple(getattr(self, name))
            if not value or len(value) != len(set(value)):
                raise ValueError(f"{name} must be nonempty, without duplicates")
            setattr(self, name, value)
        for name in (
            "d_model",
            "n_layers",
            "n_heads",
            "train_length",
            "batch_size",
            "accumulation",
            "steps",
            "save_every",
            "eval_rows",
            "eval_target_tokens",
            "query_chunk",
            "logit_chunk",
            "short_histories",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads or self.n_layers < 2:
            raise ValueError("Width must divide into heads, with at least two layers")
        if self.seed < 0 or self.profile not in ("pilot", "colab", "smoke"):
            raise ValueError(
                "Use a nonnegative seed and profile pilot, colab, or smoke"
            )
        if not set(self.arms) <= {"paper_adamw", "adam_fixed", "adam_annealed", "sgd"}:
            raise ValueError("Unknown optimizer arm")
        if not set(self.model_variants) <= {
            "original_data",
            "factorized_constant",
            "direct_constant",
        }:
            raise ValueError("Unknown model variant")
        if len(self.model_variants) > 1 and (
            "original_data" not in self.model_variants or "paper_adamw" not in self.arms
        ):
            raise ValueError(
                "A model comparison requires the original_data paper_adamw control"
            )
        if (
            not math.isfinite(self.comparison_first_g0)
            or not math.isfinite(self.comparison_other_g0)
            or min(self.comparison_first_g0, self.comparison_other_g0) <= 0
        ):
            raise ValueError(
                "Comparison gate initial decays must be finite and positive"
            )
        if any(
            variant != "original_data" for variant in self.model_variants
        ) and self.comparison_first_g0 <= math.log(2):
            raise ValueError(
                "Constant comparisons require the balanced-factor initialization: first_g0 > log(2)"
            )
        if not any(
            variant == "original_data" or arm != "paper_adamw"
            for variant in self.model_variants
            for arm in self.arms
        ):
            raise ValueError("No supported model/optimizer branches were selected")
        if any(self.learning_rates.get(arm, 0) <= 0 for arm in self.arms):
            raise ValueError("Every selected arm needs a positive learning rate")
        if any(
            not isinstance(n, int) or n < 1
            for n in self.eval_contexts + self.short_lags
        ):
            raise ValueError(
                "Evaluation contexts and short lags must be positive integers"
            )
        if self.eval_target_tokens > min((*self.eval_contexts, self.train_length)):
            raise ValueError("Matched target suffix must fit every evaluated context")
        if max((*self.eval_contexts, self.train_length)) > 65536:
            raise ValueError("LongCrawl64 rows support at most 65536 tokens")
        if (
            self.paper_warmup_steps is not None
            and not 0 <= self.paper_warmup_steps < self.steps
        ):
            raise ValueError("paper_warmup_steps must lie in [0, steps)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must lie in [0, 1)")
        if (
            self.eps0 <= 0
            or self.epsilon_decay < 0
            or self.lr_offset <= 0
            or self.lr_power < 0
        ):
            raise ValueError("Invalid epsilon or learning-rate schedule")
        if any(
            getattr(self, name) <= 0
            for name in (
                "first_gate_multiplier",
                "retrieval_gate_multiplier",
                "output_multiplier",
            )
        ):
            raise ValueError("Parameter-group multipliers must be positive")
        if not 0 < self.short_threshold <= 1:
            raise ValueError("short_threshold must lie in (0, 1]")
        if (
            self.short_retrieval_diagnostic
            and max(self.short_lags) + 32 > self.train_length
        ):
            raise ValueError("Short retrieval grammar needs room within train_length")
        if self.precision not in ("fp32", "bf16") or self.attention_backend not in (
            "sdpa",
            "upstream",
        ):
            raise ValueError("Use fp32/bf16 precision and sdpa/upstream attention")
        if self.attention_backend == "upstream" and self.precision != "bf16":
            raise ValueError("The upstream backend requires BF16 autocast")

    @property
    def warmup_steps(self):
        if self.paper_warmup_steps is not None:
            return self.paper_warmup_steps
        return min(self.steps - 1, max(1, round(self.steps * 0.1)))

    @property
    def tokens_per_update(self):
        return self.train_length * self.batch_size * self.accumulation

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def for_profile(cls, profile):
        if profile == "smoke":
            return cls(
                profile="smoke",
                d_model=16,
                n_layers=2,
                n_heads=2,
                train_length=32,
                batch_size=1,
                accumulation=1,
                steps=2,
                save_every=1,
                eval_contexts=(32, 64),
                eval_rows=1,
                eval_target_tokens=8,
                query_chunk=32,
                logit_chunk=32,
            )
        if profile in ("pilot", "colab"):
            return cls(profile=profile)
        raise ValueError("Unknown profile")

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown paper comparison settings: {sorted(unknown)}")
        defaults = cls.for_profile(values.get("profile", "pilot")).to_dict()
        return cls(**{**defaults, **values})

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def branch_specs(config):
    """Default-only studies retain old arm IDs; model matrices use unique IDs.

    Paper AdamW is the original-data-gate reference. Constant gates are compared
    with the native SGD/Adam recipes and are never called paper FoX.
    """
    legacy_ids = config.model_variants == ("original_data",)
    return [
        {
            "branch": optimizer if legacy_ids else f"{variant}__{optimizer}",
            "gate_mode": variant,
            "optimizer": optimizer,
        }
        for variant in config.model_variants
        for optimizer in config.arms
        if variant == "original_data" or optimizer != "paper_adamw"
    ]
