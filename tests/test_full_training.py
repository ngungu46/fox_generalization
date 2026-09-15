"""Meaningful CPU checks for the full runner; GPU kernels need cluster preflight."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fox_experiments.full_training.config import FullConfig, budget, task_config
from fox_experiments.full_training.trainer import (
    LossModule,
    load_model,
    schedule_values,
    train,
)
from fox_experiments.full_training.data import NativeCorpus, metadata
from fox_experiments.models import FoXLM, ModelConfig


@pytest.fixture
def native_data(tmp_path):
    zarr = pytest.importorskip("zarr")
    root = tmp_path / "native"
    root.mkdir()
    for split in ("train", "heldout"):
        values = np.arange(8 * 512, dtype=np.uint16).reshape(8, 512) + 1
        array = zarr.open_array(
            str(root / f"{split}.zarr"),
            mode="w",
            shape=values.shape,
            chunks=(2, 128),
            dtype="<u2",
            order="C",
        )
        array[:] = values
    return root


@pytest.fixture(autouse=True)
def local_tokenizer_cache(monkeypatch):
    cache = (
        Path(__file__).resolve().parents[1] / "data/longcrawl64_pilot/tokenizer_cache"
    )
    if cache.is_dir() and not os.environ.get("TIKTOKEN_CACHE_DIR"):
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))


def tiny_config(root):
    config = FullConfig(
        name="software_test",
        data_root=str(root),
        precision="fp32",
        expected_world_size=1,
        pretrain_steps=1,
        acquisition_steps=1,
        continuation_steps=3,
        checkpoint_every=1,
        log_every=100,
        logit_chunk=32,
        warmup_steps=1,
        strict_data=False,
        model={
            "d_model": 8,
            "n_heads": 1,
            "n_layers": 2,
            "ff_hidden": 16,
            "g0": 2.944102615,
            "other_g0": 0.1,
            "attention_backend": "sdpa",
            "gradient_checkpointing": True,
            "dropout": 0.1,
        },
        task={
            "profile": "full",
            "train_length": 64,
            "batch_size": 1,
            "accumulation": 2,
            "train_lags": [24, 32],
            "test_lags": [24, 32, 48],
            "prefix_copies": [0, 2],
            "test_histories": 1,
            "validation_histories": 1,
            "lm_eval_length": 64,
            "lm_eval_rows": 1,
            "mixture_probability": 0.5,
            "answer_weight": 8.0,
            "beta1": 0.1,
            "beta2": 0.1,
            "eps0": 1e-8,
            "epsilon_decay": 0.01,
            "query_chunk": 32,
        },
    )
    return config.validate()


def test_native_splits_and_missing_chunks_fail(native_data):
    corpus = NativeCorpus(native_data, strict=False)
    assert corpus.arrays["tune"].stop == corpus.arrays["confirm"].start
    assert corpus.arrays["confirm"].stop == corpus.arrays["test"].start
    assert corpus.arrays["train"].path != corpus.arrays["test"].path
    values = corpus.arrays["train"].window(0, 0, 16)
    assert values == list(range(1, 17))
    (native_data / "train.zarr" / "0.0").unlink()
    with pytest.raises(FileNotFoundError, match="Missing native chunk"):
        corpus.arrays["train"].window(0, 0, 16)
    with pytest.raises(ValueError, match="native shape"):
        metadata(native_data / "heldout.zarr", "heldout", strict=True)


def test_global_weighted_accumulation_matches_full_batch():
    torch.manual_seed(19)
    config = ModelConfig(vocab_size=31, d_model=8, n_heads=1, n_layers=2, ff_hidden=16)
    whole = FoXLM(config)
    pieces = copy.deepcopy(whole)
    tokens = torch.randint(0, 31, (2, 8))
    targets = torch.randint(0, 31, (2, 8))
    weights = torch.ones(2, 8)
    weights[0, -1] = 11
    whole.loss(tokens, targets, weights).backward()
    wrapper = LossModule(pieces, 8)
    denominator = weights.sum()
    for i in range(2):
        (
            wrapper(tokens[i : i + 1], targets[i : i + 1], weights[i : i + 1])
            / denominator
        ).backward()
    for actual, expected in zip(pieces.parameters(), whole.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_full_resume_restores_optimizer_schedule_rng_and_source_pairing(
    native_data, tmp_path
):
    config = tiny_config(native_data)
    source_dir = tmp_path / "source"
    train(
        config,
        stage="source",
        gate="direct_constant",
        run_dir=source_dir,
        allow_cpu=True,
    )
    source = source_dir / "last.pt"
    complete = tmp_path / "complete"
    resumed = tmp_path / "resumed"
    common = dict(
        stage="continuation",
        gate="factorized_constant",
        arm="adam_annealed",
        init_from=source,
        allow_cpu=True,
    )
    train(config, run_dir=complete, **common)
    train(config, run_dir=resumed, stop_after=1, **common)
    train(config, run_dir=resumed, **common)
    a = torch.load(complete / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    assert a["step"] == b["step"] == 3
    assert a["scheduler"] == b["scheduler"]
    assert (
        a["specification"]["source_checkpoint_sha256"]
        == b["specification"]["source_checkpoint_sha256"]
    )
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    for number, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, b["optimizer"]["state"][number][name], rtol=0, atol=0
            )
    assert (resumed / "checkpoint_000003.pt").exists()
    assert "first_gate_u_epsilon_dominated_fraction" in b["history"][-1]
    # A changed budget is a new study, never a silent optimizer restart.
    changed = copy.deepcopy(config)
    changed.continuation_steps += 1
    with pytest.raises(ValueError, match="Resume specification changed"):
        train(changed, run_dir=resumed, **common)
    # Native full checkpoints feed the same finite report pipeline as the pilot.
    from fox_experiments.full_training.evaluation import evaluate, collect_reports

    result = evaluate(
        config,
        complete / "checkpoint_000003.pt",
        complete / "evaluation" / "final",
        allow_cpu=True,
    )
    assert result["checkpoint_step"] == 3
    reports = collect_reports(complete)
    assert reports["status"] == "collected"
    assert (complete / "analysis" / "retrieval_raw.csv").exists()
    assert reports["plots"]


def test_schedules_budget_and_diagnostic_layer_count(native_data):
    config = tiny_config(native_data)
    config.model["n_layers"] = 12
    assert task_config(config).n_layers == 12
    assert budget(config)["global_tokens_per_update"] == 128
    fixed = schedule_values(config, "adam_fixed", 2, 3, 0.001)
    annealed = schedule_values(config, "adam_annealed", 2, 3, 0.001)
    assert annealed[0] == fixed[0]
    assert annealed[1] < fixed[1]
    config.task["epsilon_decay"] = 100.0
    with pytest.raises(FloatingPointError, match="FP32"):
        schedule_values(config, "adam_annealed", 2, 3, 0.001)


def test_kernel_backend_fails_explicitly_on_cpu():
    from fox_experiments.full_training.attention import upstream_forgetting_attention

    q = torch.zeros(1, 1, 8, 64)
    with pytest.raises(RuntimeError, match="CUDA"):
        upstream_forgetting_attention(q, q, q, torch.zeros(1, 1, 8))


def test_matrix_covers_sources_and_checkpoint_progression():
    from fox_experiments.full_training.__main__ import matrix_commands

    config = FullConfig()
    commands = matrix_commands(
        config, "configs/full/h200_124m.json", [0], 4, "runs/test"
    )
    training = [command for command in commands if "train" in command]
    evaluations = [command for command in commands if "evaluate" in command]
    assert len(training) == 11  # two sources plus 3 gate x 3 optimizer branches
    assert len(evaluations) == 20  # two baselines plus halfway/final for nine arms
    assert any(
        "runs/test/seed0/factorized_constant_adam_annealed/checkpoint_002048.pt" in c
        for c in evaluations
    )
