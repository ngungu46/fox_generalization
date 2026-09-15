"""CLI entry points; no training or large download happens on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .config import budget, load_config


def preflight(config, check_data=False, check_kernel=False):
    import math
    from fox_experiments.models import ModelConfig
    from .data import download_plan, inspect_data

    model = ModelConfig(**config.model)
    width = model.d_model
    ff = model.ff_hidden or 64 * math.ceil((8 * width / 3) / 64)
    count = (
        2 * model.vocab_size * width
        + model.n_layers * (4 * width * width + 3 * width * ff + 2 * width)
        + width
        + (model.n_layers - 1) * model.n_heads * (width + 1)
        + 2 * model.n_heads
    )
    result = {
        "config": config.to_dict(),
        "budget": budget(config),
        "factorized_parameter_count": count,
        "adam_parameter_gradient_moment_GiB_per_rank": count * 16 / 2**30,
        "memory_note": "DDP replicates these tensors on every GPU; activations, logits and reduction buffers add memory.",
        "data": download_plan(config.data_root),
        "validation_status": "CPU software tests only until --check-kernel succeeds on your GPU",
    }
    if check_data:
        result["data"] = inspect_data(config.data_root, check_all_chunks=True)
    if check_kernel:
        from .attention import check_kernel as run_check

        result["kernel"] = run_check()
    return result


def matrix_commands(config, config_path, seeds, gpus, root):
    """Two acquired sources, paired constant interventions, three optimizer arms."""
    if gpus != config.expected_world_size:
        raise ValueError("--gpus must match expected_world_size in the config")
    root = Path(root)
    commands = []
    common = ["--config", str(config_path), "--data-root", config.data_root]

    def train_command(arguments):
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={gpus}",
            "--module",
            "fox_experiments.full_training",
            "train",
            *common,
            *arguments,
        ]

    def eval_command(checkpoint, output, seed):
        return [
            sys.executable,
            "-m",
            "fox_experiments.full_training",
            "evaluate",
            *common,
            "--seed",
            str(seed),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ]

    for seed in seeds:
        seed_root = root / f"seed{seed}"
        sources = {}
        for gate in ("direct_constant", "original_data"):
            directory = seed_root / f"source_{gate}"
            sources[gate] = directory / "last.pt"
            commands.append(
                train_command(
                    [
                        "--stage",
                        "source",
                        "--gate",
                        gate,
                        "--seed",
                        str(seed),
                        "--run-dir",
                        str(directory),
                    ]
                )
            )
            commands.append(
                eval_command(
                    directory / "last.pt", directory / "evaluation" / "source", seed
                )
            )
        for gate in ("factorized_constant", "direct_constant", "original_data"):
            source = sources[
                "original_data" if gate == "original_data" else "direct_constant"
            ]
            for arm in ("sgd", "adam_fixed", "adam_annealed"):
                directory = seed_root / f"{gate}_{arm}"
                commands.append(
                    train_command(
                        [
                            "--stage",
                            "continuation",
                            "--gate",
                            gate,
                            "--arm",
                            arm,
                            "--seed",
                            str(seed),
                            "--init-from",
                            str(source),
                            "--run-dir",
                            str(directory),
                        ]
                    )
                )
                for step in sorted(
                    {max(1, config.continuation_steps // 2), config.continuation_steps}
                ):
                    commands.append(
                        eval_command(
                            directory / f"checkpoint_{step:06d}.pt",
                            directory / "evaluation" / f"step{step:06d}",
                            seed,
                        )
                    )
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train", "evaluate", "matrix"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/full/h200_124m.json")
        sub.add_argument("--data-root")
        sub.add_argument("--seed", type=int)
        if name == "preflight":
            sub.add_argument("--check-data", action="store_true")
            sub.add_argument("--check-kernel", action="store_true")
        elif name == "train":
            sub.add_argument(
                "--stage", choices=("source", "continuation"), required=True
            )
            sub.add_argument(
                "--gate",
                choices=("factorized_constant", "direct_constant", "original_data"),
                required=True,
            )
            sub.add_argument(
                "--arm",
                choices=("sgd", "adam_fixed", "adam_annealed", "paper_adamw"),
                default="adam_annealed",
            )
            sub.add_argument("--run-dir")
            sub.add_argument("--init-from")
            sub.add_argument("--lr", type=float)
            sub.add_argument("--steps", type=int)
        elif name == "evaluate":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--output", required=True)
            sub.add_argument(
                "--split", choices=("tune", "confirm", "test"), default="test"
            )
        elif name == "matrix":
            sub.add_argument("--seeds", nargs="+", type=int, default=[0])
            sub.add_argument("--gpus", type=int, default=4)
            sub.add_argument("--run-root")
            sub.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config.data_root = args.data_root
    if args.seed is not None:
        config.seed = args.seed
    if args.command == "preflight":
        print(
            json.dumps(preflight(config, args.check_data, args.check_kernel), indent=2)
        )
    elif args.command == "train":
        from .trainer import train

        result = train(
            config,
            stage=args.stage,
            gate=args.gate,
            arm=args.arm,
            run_dir=args.run_dir,
            init_from=args.init_from,
            lr=args.lr,
            steps=args.steps,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(config, args.checkpoint, args.output, args.split)
    elif args.command == "matrix":
        root = args.run_root or str(Path(config.output_root) / config.name)
        commands = matrix_commands(config, args.config, args.seeds, args.gpus, root)
        print(
            json.dumps(
                {
                    "budget_per_seed": budget(config),
                    "seeds": args.seeds,
                    "total_input_tokens": len(args.seeds)
                    * budget(config)["two_sources_nine_arms_tokens"],
                    "run_root": root,
                    "execute": args.execute,
                },
                indent=2,
            ),
            flush=True,
        )
        for command in commands:
            print(shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
