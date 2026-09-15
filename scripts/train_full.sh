#!/usr/bin/env bash
# Run from the repository root in a GPU allocation. Arguments pass to the CLI.
set -euo pipefail
python -m fox_experiments.full_training "$@"
