"""The eight named experiments used by the Colab notebook."""
from dataclasses import dataclass, replace
import math
from pathlib import Path

from .legacy.core import Config
from .data.random_streams import RandomStreamConfig


@dataclass(frozen=True)
class Experiment:
    name: str
    config: Config
    dataset: str
    optimizer: str
    gate: str
    seeds: tuple[int, ...] = (0, 1, 2)
    theta: float = 0.5
    random_data: RandomStreamConfig = RandomStreamConfig(max_batch_records=8_000_000)
    random_sampling: str = "fixed"
    random_dataset_size: int | None = None
    log_every: int = 1000
    checkpoint_every: int = 5000
    check_every: int = 100

    def __post_init__(self):
        if self.dataset not in ("pairs", "random"):
            raise ValueError("dataset must be pairs or random")
        if self.optimizer not in ("sgd", "adam") or self.gate not in ("learned", "alibi"):
            raise ValueError("Use SGD/Adam and learned/ALiBi")
        if self.random_sampling not in ("fixed", "online"):
            raise ValueError("random_sampling must be fixed or online")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("seeds must contain distinct nonnegative integers")
        if not math.isfinite(self.theta) or not 0 < self.theta < 1:
            raise ValueError("theta must lie strictly between zero and one")
        for name in ("log_every", "checkpoint_every", "check_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.random_dataset_size is not None and (type(self.random_dataset_size) is not int or self.random_dataset_size < 1):
            raise ValueError("random_dataset_size must be a positive integer or None")

    @property
    def gate_mode(self):
        return "learned" if self.gate == "learned" else "retrieval_frozen"

    @property
    def dataset_size(self):
        return self.random_dataset_size or self.config.n * (self.config.n - 1)


@dataclass(frozen=True)
class Run:
    path: Path
    name: str


def a100_config(**overrides):
    """Paper initialization and practical continuation, with the larger budget.

    The short-memory Adam preset is explicit. Change beta1/beta2 here to use
    the previous (0.9, 0.999) control; use a new output folder when doing so.
    """
    defaults = dict(n=128, d=4096, R=2, steps=200_000, device="cuda",
                    beta1=0.1, beta2=0.1, max_seconds=24 * 3600)
    defaults.update(overrides)
    return Config(**defaults)


def make_experiments(config, *, seeds=(0, 1, 2), random_data=None,
                     random_sampling="fixed", random_dataset_size=None,
                     theta=0.5, log_every=1000, checkpoint_every=5000, check_every=100):
    """Build the requested 2 datasets × 2 gates × 2 optimizers, in notebook order."""
    random_data = random_data or RandomStreamConfig(max_batch_records=8_000_000)
    experiments = {}
    for dataset in ("pairs", "random"):
        for gate in ("learned", "alibi"):
            for optimizer in ("sgd", "adam"):
                name = f"{dataset}_{gate}_{optimizer}"
                experiments[name] = Experiment(
                    name=name, config=replace(config), dataset=dataset, gate=gate,
                    optimizer=optimizer, seeds=tuple(seeds), theta=theta,
                    random_data=random_data, random_sampling=random_sampling,
                    random_dataset_size=random_dataset_size, log_every=log_every,
                    checkpoint_every=checkpoint_every, check_every=check_every)
    return experiments
