"""python -m fox_experiments.paper_baseline --config ... --out ..."""

import argparse
import json
from pathlib import Path

from . import PaperBaselineConfig, run_paper_comparison, summarize_paper_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Paired default-FoX pure-LM optimizer comparison"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default="data/longcrawl64_pilot")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = PaperBaselineConfig.from_json(args.config)
    if not args.analyze_only:
        result = run_paper_comparison(
            config, args.data_dir, args.out, args.device, args.resume
        )
        print(json.dumps(result, indent=2))
    print(json.dumps(summarize_paper_comparison(args.out), indent=2))
    status_path = Path(args.out) / "run_status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get(
        "failed_arms", 0
    ):
        raise SystemExit("One or more optimizer arms failed; inspect failures.json")


if __name__ == "__main__":
    main()
