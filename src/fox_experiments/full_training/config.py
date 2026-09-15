"""Readable configuration shared by CLI commands and the full-training notebook."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class FullConfig:
    name: str = "h200_124m"
    data_root: str = "data/longcrawl64_full"
    output_root: str = "runs/full"
    seed: int = 0
    model: dict = field(
        default_factory=lambda: {
            "d_model": 640,
            "n_heads": 10,
            "n_layers": 12,
            "ff_hidden": 1728,
            "g0": 2.944102615,
            "other_g0": 0.1,
            "attention_backend": "upstream",
            "gradient_checkpointing": True,
        }
    )
    task: dict = field(
        default_factory=lambda: {
            "profile": "full",
            "train_length": 2048,
            "batch_size": 1,
            "accumulation": 16,
            "train_lags": [128, 256, 512, 1024],
            "test_lags": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            "prefix_copies": [0, 128, 512, 2048],
            "test_histories": 32,
            "validation_histories": 64,
            "lm_eval_length": 16384,
            "lm_eval_rows": 16,
            "answer_weight": 128.0,
            "mixture_probability": 0.5,
            "beta1": 0.1,
            "beta2": 0.1,
            "eps0": 1e-8,
            "epsilon_decay": 0.01,
            "lr_offset": 1000.0,
            "lr_power": 0.75,
            "first_gate_multiplier": 1.5,
            "retrieval_gate_multiplier": 0.05,
            "output_multiplier": 0.1,
            "short_threshold": 0.9,
            "query_chunk": 256,
        }
    )
    pretrain_steps: int = 2048
    acquisition_steps: int = 2048
    continuation_steps: int = 4096
    learning_rates: dict = field(
        default_factory=lambda: {
            "source": 0.001,
            "sgd": 0.05,
            "adam_fixed": 0.001,
            "adam_annealed": 0.001,
            "paper_adamw": 0.001,
        }
    )
    warmup_steps: int = 256
    checkpoint_every: int = 256
    log_every: int = 10
    logit_chunk: int = 128
    precision: str = "bf16"
    expected_world_size: int = 4
    strict_data: bool = True

    def validate(self):
        if self.precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        for name in (
            "checkpoint_every",
            "log_every",
            "logit_chunk",
            "expected_world_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("pretrain_steps", "acquisition_steps", "continuation_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            self.pretrain_steps + self.acquisition_steps < 1
            or self.continuation_steps < 1
        ):
            raise ValueError("Source and continuation stages need at least one update")
        for name in ("batch_size", "accumulation", "train_length"):
            if self.task.get(name, 1) < 1:
                raise ValueError(f"task.{name} must be positive")
        if self.task["train_length"] >= 65536:
            raise ValueError("Natural text batches require train_length + 1 <= 65536")
        if self.task.get("eps0", 1e-8) <= 0 or self.task.get("epsilon_decay", 0.01) < 0:
            raise ValueError(
                "Adam epsilon must be positive; epsilon decay must be nonnegative"
            )
        if any(lr <= 0 for lr in self.learning_rates.values()):
            raise ValueError("Learning rates must be positive")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    return FullConfig(**json.loads(Path(path).read_text())).validate()


def task_config(config):
    from fox_experiments.training.config import ExperimentConfig

    values = dict(config.task)
    for name in ("d_model", "n_heads", "n_layers", "g0", "other_g0"):
        if name in config.model:
            values[name] = config.model[name]
    return ExperimentConfig(**values)


def budget(config, world_size=None):
    """Input-token counts per stage; evaluation and LR trials are additional."""
    world = world_size or config.expected_world_size
    tokens = (
        config.task["train_length"]
        * config.task["batch_size"]
        * config.task["accumulation"]
        * world
    )
    source = config.pretrain_steps + config.acquisition_steps
    return {
        "world_size": world,
        "global_tokens_per_update": tokens,
        "source_tokens_each": source * tokens,
        "continuation_tokens_each": config.continuation_steps * tokens,
        "two_sources_nine_arms_tokens": (2 * source + 9 * config.continuation_steps)
        * tokens,
        "learning_rate_note": "Rates are initial candidates; calibrate on heldout tune rows before fixing full runs.",
        "claim": "Finite empirical experiment; no infinite-length claim or exact paper reproduction.",
    }
