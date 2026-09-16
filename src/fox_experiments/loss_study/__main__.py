"""Run with python -m fox_experiments.loss_study --config ... --out ..."""

import argparse
import json

from . import LossStudyConfig, run_loss_study, summarize_loss_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if not args.analyze_only:
        if not args.config:
            parser.error("--config is required unless --analyze-only is used")
        run_loss_study(
            LossStudyConfig.from_json(args.config),
            args.out,
            args.device,
            data_dir=args.data_dir,
            resume=args.resume,
        )
    result = summarize_loss_study(args.out)
    print(json.dumps(result, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
