"""Small command-line entry points; all scientific code lives in the packages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def prepare_data(data_dir: str | Path):
    """Verify/download the pilot and connect it to the latest-write task sampler."""
    import tiktoken

    from .data import Corpus, LatestWriteData, load_corpus, prepare_longcrawl64

    data_dir = Path(data_dir).resolve()
    cache = data_dir / "tokenizer_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    manifest = prepare_longcrawl64(data_dir)
    arrays = load_corpus(data_dir)
    corpus = Corpus(arrays.train, arrays.validation, arrays.test)
    return LatestWriteData(corpus, tiktoken.get_encoding("gpt2")), manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Prepare the verified pilot corpus")
    download.add_argument("--data-dir", default="data/longcrawl64_pilot")

    train = commands.add_parser("train", help="Run a configured downscale comparison")
    train.add_argument("--config", default="configs/downscale.json")
    train.add_argument("--data-dir", default="data/longcrawl64_pilot")
    train.add_argument("--out", required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default=None)
    train.add_argument("--resume", action="store_true")

    mechanism = commands.add_parser(
        "mechanism", help="Run the controlled binding model"
    )
    mechanism.add_argument("--profile", choices=("smoke", "pilot"), default="pilot")
    mechanism.add_argument("--out", required=True)
    mechanism.add_argument("--recall-lags", type=int, nargs="+", default=[2, 4, 8])
    mechanism.add_argument("--seeds", type=int, nargs="+", default=[0])

    report = commands.add_parser(
        "report", help="Regenerate plots from saved predictions"
    )
    report.add_argument("--out", required=True)
    report.add_argument("--mechanism", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "download":
        _, manifest = prepare_data(args.data_dir)
        print(json.dumps(manifest, indent=2))
    elif args.command == "train":
        import torch

        from .training import ExperimentConfig, run_experiment

        config = ExperimentConfig.from_json(args.config)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu" and config.profile != "smoke":
            parser.error(
                "Use a GPU for training; configs/smoke.json supports CPU checks."
            )
        if device == "cpu":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        data, _ = prepare_data(args.data_dir)
        run_experiment(data, config, args.out, device=device, resume=args.resume)
    elif args.command == "mechanism":
        import torch

        from .mechanism import run_mechanism

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        result = run_mechanism(
            args.out,
            args.profile,
            seeds=tuple(args.seeds),
            recall_lags=tuple(args.recall_lags),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        from .evaluation import plot_mechanism, summarize_results

        result = (
            plot_mechanism(args.out) if args.mechanism else summarize_results(args.out)
        )
        print(result)


if __name__ == "__main__":
    main()
