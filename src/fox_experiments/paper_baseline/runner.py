"""Pure-LM paper-recipe comparison, separate from the acquisition experiment.

All arms start with identical default-FoX weights and see identical EOT-reset
natural-text batches. Only the complete optimizer recipe changes. This is a
scaled experiment, not a replication of the original paper's compute budget.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from fox_experiments.data import Corpus, load_corpus
from fox_experiments.training.artifacts import (
    atomic_checkpoint,
    dump_json,
    exclusive_run_lock,
    state_hash,
    table,
)
from fox_experiments.training.optimizers import make_optimizer, set_optimizer_schedule
from .config import PaperBaselineConfig

POLICY_VERSION = "default_fox_pure_lm_v1"
EOT = 50256


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def _corpus_identity(corpus):
    result = {}
    for split in ("train", "validation", "test"):
        array = corpus.arrays[split]
        digest = hashlib.sha256()
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
        result[split] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest.hexdigest(),
        }
    return result


def natural_batch(corpus, config, step, microbatch):
    """Deterministic document-local chunks, with the paper's EOT input shift.

    Pilot chunks are randomly sampled from the verified local subset. Original
    paper training used a deterministic full-dataset traversal; that is an
    explicit sampling/budget difference, shared by every optimizer here.
    """
    rng = np.random.default_rng(config.seed * 1000003 + step * 131 + microbatch * 17)
    labels = [
        corpus.sample("train", config.train_length, rng)[0]
        for _ in range(config.batch_size)
    ]
    targets = torch.tensor(labels, dtype=torch.long)
    inputs = torch.cat((torch.full((len(labels), 1), EOT), targets[:, :-1]), dim=1)
    return inputs, targets


def schedule_optimizer(optimizer, config, arm, step):
    """Paper linear warmup THEN cosine; our polynomial LR/epsilon interventions."""
    rate = config.learning_rates[arm]
    if arm != "paper_adamw":
        return set_optimizer_schedule(optimizer, config, arm, rate, step, config.steps)
    warmup = config.warmup_steps
    if warmup and step < warmup:
        multiplier = step / warmup
    else:
        fraction = min(1.0, max(0.0, (step - warmup) / (config.steps - warmup)))
        multiplier = 0.5 * (1.0 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = rate * multiplier
    return rate * multiplier, 1e-8


def _autocast(config, device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if config.precision == "bf16"
        else nullcontext()
    )


def _rng_state(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(saved, device):
    torch.set_rng_state(saved["cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"].cpu(), device)


def _train_arm(model, corpus, config, arm, directory, signature, initial_hash):
    device = next(model.parameters()).device
    optimizer = make_optimizer(model, config, arm, config.learning_rates[arm])
    checkpoint = directory / "last.pt"
    history, start, cumulative_lr, prior_seconds = [], 0, 0.0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if (
            saved["signature"] != signature
            or saved["initial_state_hash"] != initial_hash
        ):
            raise ValueError(
                "Checkpoint provenance differs; choose a new output directory"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, start, cumulative_lr = saved["history"], saved["step"], saved["S"]
        prior_seconds = saved["seconds"]
        _restore_rng(saved["rng"], device)
    else:
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
    started = time.perf_counter()
    for step in range(start, config.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr, epsilon = schedule_optimizer(optimizer, config, arm, step)
        total_loss = 0.0
        for micro in range(config.accumulation):
            x, y = natural_batch(corpus, config, step, micro)
            x, y = x.to(device), y.to(device)
            with _autocast(config, device):
                loss = (
                    model.loss(
                        x,
                        y,
                        query_chunk=config.query_chunk,
                        logit_chunk=config.logit_chunk,
                    )
                    / config.accumulation
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update {step + 1}")
            loss.backward()
            total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0 if arm == "paper_adamw" else float("inf")
        )
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite gradient at update {step + 1}")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite parameter at update {step + 1}")
        cumulative_lr += lr
        elapsed = prior_seconds + time.perf_counter() - started
        history.append(
            {
                "step": step + 1,
                "tokens": (step + 1) * config.tokens_per_update,
                "loss": total_loss,
                "lr": lr,
                "epsilon": epsilon,
                "gradient_norm_before_clip": float(norm),
                "S": cumulative_lr,
                "seconds": elapsed,
            }
        )
        if (step + 1) % config.save_every == 0 or step + 1 == config.steps:
            atomic_checkpoint(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "model_config": asdict(model.config),
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "S": cumulative_lr,
                    "history": history,
                    "seconds": elapsed,
                    "rng": _rng_state(device),
                    "signature": signature,
                    "initial_state_hash": initial_hash,
                    "arm": arm,
                    "policy_version": POLICY_VERSION,
                },
            )
            table(directory / "training.csv", history)
            print(
                f"{arm}: {step + 1}/{config.steps} updates; LM loss {total_loss:.4f}",
                flush=True,
            )
    return history


def _evaluation_windows(corpus, config):
    length = max((*config.eval_contexts, config.train_length))
    array = corpus.arrays["test"]
    if length > array.shape[1] or config.eval_rows > array.shape[0]:
        raise ValueError("Requested evaluation windows exceed available test documents")
    rng = np.random.default_rng(config.seed + 800000)
    documents = rng.choice(len(array), config.eval_rows, replace=False)
    windows = []
    for sample, document in enumerate(documents):
        offset = int(rng.integers(array.shape[1] - length + 1))
        tokens = np.asarray(
            array[document, offset : offset + length], dtype=np.int64
        ).tolist()
        windows.append(
            {
                "sample": sample,
                "doc_id": int(document),
                "offset": offset,
                "tokens": tokens,
            }
        )
    return windows


@torch.no_grad()
def evaluate_language_model(model, corpus, config, arm, checkpoint_step):
    """Score identical held-out targets under different reset context windows.

    Every context predicts the same last K target tokens. `context_length` is
    the reset window length; individual target positions have their ordinary
    causal prefixes within it. Positive gain means extra context reduced NLL.
    """
    model.eval()
    device = next(model.parameters()).device
    rows = []
    contexts = sorted(set((*config.eval_contexts, config.train_length)))
    for window in _evaluation_windows(corpus, config):
        tokens = window["tokens"]
        suffix_rows = []
        for context in contexts:
            labels = tokens[-context:]
            inputs = torch.tensor([[EOT] + labels[:-1]], device=device)
            targets = torch.tensor([labels], device=device)
            with _autocast(config, device):
                hidden = model.encode(inputs, query_chunk=config.query_chunk)
                losses = []
                for start in range(0, context, config.logit_chunk):
                    logits = model.logits(
                        hidden[:, start : start + config.logit_chunk]
                    ).float()
                    losses.extend(
                        F.cross_entropy(
                            logits.flatten(0, 1),
                            targets[:, start : start + config.logit_chunk].flatten(),
                            reduction="none",
                        )
                        .cpu()
                        .tolist()
                    )
            metadata = {
                "arm": arm,
                "checkpoint_step": checkpoint_step,
                "sample": window["sample"],
                "doc_id": window["doc_id"],
                "window_offset": window["offset"],
                "context_length": context,
                "train_length": config.train_length,
                "software_only": config.profile == "smoke",
            }
            if context == contexts[-1]:
                rows.extend(
                    {
                        **metadata,
                        "metric": "per_position",
                        "position": i + 1,
                        "target_offset": window["offset"] + i,
                        "nll": nll,
                    }
                    for i, nll in enumerate(losses)
                )
            suffix_rows.append(
                {
                    **metadata,
                    "metric": "same_target_suffix",
                    "nll": float(np.mean(losses[-config.eval_target_tokens :])),
                    "target_start_offset": window["offset"]
                    + len(tokens)
                    - config.eval_target_tokens,
                    "target_count": config.eval_target_tokens,
                    "target_token_sha256": hashlib.sha256(
                        np.asarray(
                            tokens[-config.eval_target_tokens :], dtype="<i8"
                        ).tobytes()
                    ).hexdigest(),
                }
            )
        reference = next(
            row["nll"]
            for row in suffix_rows
            if row["context_length"] == config.train_length
        )
        rows.extend(
            {**row, "nll_gain_vs_train_context": reference - row["nll"]}
            for row in suffix_rows
        )
    return rows


def _short_diagnostic(model, corpus, config, data_dir):
    """Optional retrieval check, never an LM stopping/selection criterion."""
    import os
    import tiktoken
    from fox_experiments.data import LatestWriteData
    from fox_experiments.evaluation.metrics import short_score
    from fox_experiments.training.config import ExperimentConfig

    cache = Path(data_dir) / "tokenizer_cache"
    if cache.is_dir():
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
    data = LatestWriteData(corpus, tiktoken.get_encoding("gpt2"))
    task = ExperimentConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        train_length=config.train_length,
        train_lags=config.short_lags,
        test_lags=config.short_lags,
        validation_histories=config.short_histories,
        query_chunk=config.query_chunk,
    )
    with _autocast(config, next(model.parameters()).device):
        result = short_score(model, data, task, "confirm", config.seed)
    result.pop("panels")
    return {
        **result,
        "qualified": result["worst_accuracy"] >= config.short_threshold,
        "diagnostic_only": True,
        "trained_on_retrieval": False,
        "protocol_version": data.protocol_version,
        "note": "Natural-text-only training; no long retrieval range is inferred",
    }


def run_paper_comparison(
    config, data_dir, out_dir, device="cpu", resume=False, *, corpus=None
):
    """Run independent optimizer recipes from one shared random initialization.

    Resume requires identical configuration, source, dataset and device type.
    Changed recipes need a new directory. Individual failed arms remain visible.
    """
    from .model import build_paper_fox

    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cpu" and config.precision != "fp32":
        raise ValueError("CPU validation requires precision='fp32'")
    if config.attention_backend == "upstream" and device.type != "cuda":
        raise ValueError("Upstream attention requires CUDA")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; use SDPA FP32")
    if corpus is None:
        files = load_corpus(data_dir)
        corpus = Corpus(files.train, files.validation, files.test)
    if corpus.arrays["train"].shape[1] < config.train_length:
        raise ValueError("Training context exceeds source document length")
    _evaluation_windows(corpus, config)  # Validate before any training or output.
    initial = build_paper_fox(
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        seed=config.seed,
        attention_backend=config.attention_backend,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    initial_hash = state_hash(initial)
    spec = {
        "policy_version": POLICY_VERSION,
        "config": config.to_dict(),
        "model_config": asdict(initial.config),
        "initial_state_hash": initial_hash,
        "source_sha256": _source_identity(),
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "dataset": _corpus_identity(corpus),
        "device_type": device.type,
        "attention_backend": config.attention_backend,
        "objective": "natural_next_token_only",
        "starts_from": "random_initialization",
        "hyperparameters_selected_on_this_objective": False,
        "sampling": "random_document_local_chunks_with_EOT_reset",
        "software_only": config.profile == "smoke",
        "finite_pilot": True,
    }
    signature = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = Path(out_dir)
    with exclusive_run_lock(output):
        spec_path = output / "run_specification.json"
        populated = any(path.name != ".run.lock" for path in output.iterdir())
        if populated and not resume:
            raise FileExistsError(
                "Output already exists; choose a new directory or use --resume"
            )
        if populated:
            if not spec_path.exists() or json.loads(
                spec_path.read_text()
            ) != json.loads(json.dumps(spec)):
                raise ValueError(
                    "Resume specification changed: configuration/source/data/device; choose a new directory"
                )
        else:
            dump_json(spec_path, spec)
            config.to_json(output / "config.json")
            dump_json(
                output / "environment.json",
                {
                    "torch": torch.__version__,
                    "numpy": np.__version__,
                    "device": str(device),
                    "gpu": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else None
                    ),
                    "parameter_count": sum(p.numel() for p in initial.parameters()),
                    "tokens_per_arm": config.steps * config.tokens_per_update,
                },
            )
        dump_json(
            output / "run_status.json",
            {"status": "running", "software_only": config.profile == "smoke"},
        )
        if not (output / "initial_lm_raw.csv").exists():
            table(
                output / "initial_lm_raw.csv",
                evaluate_language_model(initial, corpus, config, "shared_initial", 0),
            )
        failures, summaries = [], []
        for arm in config.arms:
            directory = output / arm
            directory.mkdir(exist_ok=True)
            model = copy.deepcopy(initial)
            assert state_hash(model) == initial_hash
            try:
                history = _train_arm(
                    model, corpus, config, arm, directory, signature, initial_hash
                )
                rows = evaluate_language_model(model, corpus, config, arm, config.steps)
                table(directory / "lm_raw.csv", rows)
                suffix = [row for row in rows if row["metric"] == "same_target_suffix"]
                summary = {
                    "arm": arm,
                    "status": "complete",
                    "steps": len(history),
                    "input_tokens": len(history) * config.tokens_per_update,
                    "initial_state_hash": initial_hash,
                    "final_state_hash": state_hash(model),
                    "final_training_loss": history[-1]["loss"],
                    "matched_target_nll_at_train_context": float(
                        np.mean(
                            [
                                row["nll"]
                                for row in suffix
                                if row["context_length"] == config.train_length
                            ]
                        )
                    ),
                    "software_only": config.profile == "smoke",
                    "finite_pilot": True,
                    "rates_untuned": True,
                    "retrieval_qualified": None,
                }
                if config.short_retrieval_diagnostic:
                    diagnostic = _short_diagnostic(model, corpus, config, data_dir)
                    dump_json(directory / "short_retrieval_diagnostic.json", diagnostic)
                    summary["retrieval_qualified"] = diagnostic["qualified"]
                dump_json(directory / "status.json", summary)
                summaries.append(summary)
            except (FloatingPointError, RuntimeError) as error:
                failure = {
                    "arm": arm,
                    "status": "failed",
                    "reason": str(error),
                    "software_only": config.profile == "smoke",
                    "initial_state_hash": initial_hash,
                    "rates_untuned": True,
                }
                dump_json(directory / "status.json", failure)
                failures.append(failure)
            finally:
                del model
            table(output / "summary.csv", summaries + failures)
            dump_json(output / "failures.json", failures)
        frames = [pd.read_csv(output / "initial_lm_raw.csv")]
        frames.extend(
            pd.read_csv(output / row["arm"] / "lm_raw.csv") for row in summaries
        )
        pd.concat(frames, ignore_index=True).to_csv(output / "lm_raw.csv", index=False)
        dump_json(
            output / "run_status.json",
            {
                "status": "complete_with_failures" if failures else "complete",
                "completed_arms": len(summaries),
                "failed_arms": len(failures),
                "software_only": config.profile == "smoke",
            },
        )
    return {
        "output_dir": str(output),
        "summary": str(output / "summary.csv"),
        "lm_raw": str(output / "lm_raw.csv"),
        "failures": str(output / "failures.json"),
    }


def summarize_paper_comparison(out_dir):
    """Regenerate pure-LM plots; retrieval qualification never gates these metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(out_dir)
    config = PaperBaselineConfig.from_json(output / "config.json")
    raw = pd.read_csv(output / "lm_raw.csv")
    suffix = raw[raw.metric == "same_target_suffix"]
    means = suffix.groupby(
        ["arm", "checkpoint_step", "context_length"], as_index=False
    ).agg(
        mean_nll=("nll", "mean"),
        mean_nll_gain=("nll_gain_vs_train_context", "mean"),
        documents=("sample", "nunique"),
    )
    means.to_csv(output / "context_summary.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for arm, frame in raw.groupby("arm"):
        style = "--" if arm == "shared_initial" else "-"
        per_position = frame[frame.metric == "per_position"].copy()
        per_position["position_bin"] = ((per_position.position - 1) // 32) * 32 + 1
        positional = per_position.groupby("position_bin").agg(
            position=("position", "mean"), nll=("nll", "mean")
        )
        axes[0].plot(positional.position, positional.nll, style, label=arm)
        contexts = means[means.arm == arm].sort_values("context_length")
        axes[1].plot(contexts.context_length, contexts.mean_nll, style + "o", label=arm)
        axes[2].plot(
            contexts.context_length, contexts.mean_nll_gain, style + "o", label=arm
        )
    for axis in axes:
        axis.axvline(
            config.train_length,
            color="black",
            linestyle="--",
            alpha=0.55,
            label="Training context",
        )
        axis.grid(alpha=0.2)
    axes[0].set(
        title="Next-token loss by position",
        xlabel="Target position (32-token bin centers)",
        ylabel="NLL (nats)",
        xlim=(1, max((*config.eval_contexts, config.train_length))),
    )
    axes[1].set(
        title="Identical target suffix",
        xlabel="Reset context window (tokens)",
        ylabel="NLL (nats)",
    )
    axes[2].set(
        title="Utility of additional context",
        xlabel="Reset context window (tokens)",
        ylabel="NLL gain vs training context",
    )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xscale("log", base=2)
    axes[2].set_xscale("log", base=2)
    label = (
        "Software validation only"
        if config.profile == "smoke"
        else "Finite default-FoX pilot; optimizer rates untuned"
    )
    figure.suptitle(label)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.tight_layout(rect=(0, 0.14, 1, 0.94))
    path = output / "paper_context_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "context_summary": str(output / "context_summary.csv"),
        "figures": [str(path)],
        "summary": str(output / "summary.csv"),
    }
