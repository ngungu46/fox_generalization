#!/usr/bin/env bash
# Edit partition/account/GPU resource syntax for your school's scheduler.
#SBATCH --job-name=fox-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=fox-%j.log
set -euo pipefail

# Submit with: sbatch --export=ALL,FOX_REPO_DIR=/path/fox_experiments,FOX_FULL_DATA_DIR=/scratch/longcrawl64 scripts/slurm_full.sbatch
: "${FOX_REPO_DIR:?Set FOX_REPO_DIR to your checked-out repository}"
: "${FOX_FULL_DATA_DIR:?Set FOX_FULL_DATA_DIR to staged native train.zarr and heldout.zarr}"
cd "$FOX_REPO_DIR"
# Activate your CUDA Python environment here before submitting, or add its
# activation command. The notebook itself does not occupy the GPU allocation.
python -m fox_experiments.full_training preflight \
  --config configs/full/h200_124m.json --data-root "$FOX_FULL_DATA_DIR" \
  --check-data --check-kernel
python -m fox_experiments.full_training matrix \
  --config configs/full/h200_124m.json --data-root "$FOX_FULL_DATA_DIR" \
  --run-root "${FOX_FULL_RUN_DIR:-runs/full/h200_124m}" \
  --gpus 4 --seeds 0 1 2 --execute

# Matrix jobs resume their completed/partial stages when this command is rerun.
# Wall time is only a requested allocation, not an estimated completion time.
