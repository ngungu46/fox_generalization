# FoX experiments

Optimizer and length-generalization experiments for Forgetting Transformers,
organized as an installable Python project. The notebooks contain configuration,
training calls and analysis; **they do not embed or generate Python modules**.

## Start here

**The saved v1 pilot did not learn short retrieval.** Its zero all-edit scores
come from constant digit predictions; controlled-model zeros have separate
confidence/prefix causes. Read the [actual-run diagnosis](docs/downscale_v1_diagnosis.md)
before launching a longer run. No replacement acquisition recipe has yet passed
short confirmation.

| Notebook | Purpose |
|---|---|
| [01_downscale_training.ipynb](notebooks/01_downscale_training.ipynb) | CPU smoke check, Colab GPU pilot, controlled theory diagnostics and plots |
| [02_full_training.ipynb](notebooks/02_full_training.ipynb) | Full-data preparation, H200 training plans, distributed launch and saved-run analysis |
| [03_paper_baseline_comparison.ipynb](notebooks/03_paper_baseline_comparison.ipynb) | Original FoX (LLaMA) architecture and paper AdamW recipe versus our optimizers, trained from initialization |
| [04_length_generalization_comparison.ipynb](notebooks/04_length_generalization_comparison.ipynb) | Original FoX versus our factorized/direct gates and optimizers, with paired length-generalization measurements |
| [05_loss_and_constraints.ipynb](notebooks/05_loss_and_constraints.ipynb) | Latest-answer loss versus all-token NTP on identical short histories; matched gates, optimizer controls, gradient-conflict diagnostics and named constraint ablations |

**For the loss-objective and restriction study, use notebook 05.** It defaults
to the Colab training profile, starts with a compact controlled task, and
separately offers a real-text background extension. Read the
[design and troubleshooting guide](docs/loss_and_constraints_study.md).
Short acquisition failures are visible and never converted into a claimed
generalization cutoff.

**For original FoX versus our model setup, use notebook 04.** It includes the
literal `softplus(u*v)` assumption and its function-matched direct-gate control.
See [the comparison design and code](docs/length_generalization_comparison.md).

The third notebook addresses the missing original-FoX baseline. It restores the
paper's gate/weight initialization and model defaults, then trains natural text
with matched model/data/token budgets across four optimizer arms. The earlier
`original_data + paper_adamw` branch remains an adapted optimizer control. See
[the baseline comparison guide](docs/paper_baseline_comparison.md) for the
Colab run and the separate full published-scale reference.

The new source uses the supplied `forgetting-transformer-main` as its reference.
The full backend includes the authors' efficient forgetting-attention kernel.
The common model preserves the small experiment's architecture so the two scales
have comparable gate interventions. This is an optimizer study, not an exact
reproduction of the paper's published benchmark numbers.

## Where to make changes

```text
fox_experiments/
├── notebooks/                 # Short training/analysis notebooks
├── configs/
│   ├── smoke.json             # Tiny software validation
│   ├── downscale.json         # Colab-scale settings
│   ├── replicate.json         # Larger replication of the small model
│   ├── full/                  # Distributed full-training settings
│   ├── paper_baseline/        # Plain-FoX optimizer comparison and reference
│   └── length_generalization/ # Original versus factorized/direct FoX
├── src/fox_experiments/
│   ├── models/                # Model config, gates, attention, blocks, LM
│   ├── data/                  # Pilot downloading, corpus windows, task probes
│   ├── training/              # Small-run config, optimizers, trainer, experiment
│   ├── evaluation/            # Retrieval/LM metrics, tables and figures
│   ├── mechanism/             # Controlled binding model and analytic bounds
│   ├── paper_baseline/        # Pure-text gate/optimizer comparisons
│   └── full_training/         # Native full data, DDP, CUDA kernel, resume/eval
├── data/                      # Downloaded data; ignored by Git
├── outputs/                   # Checkpoints, raw predictions, reports; ignored
├── scripts/                   # Cluster launch and notebook checks
├── tests/                     # Numerical, training and resume checks
├── docs/                      # Scientific design, provenance, run instructions
└── third_party/               # Upstream notices and licenses
```

Change **gate formulas** in `models/gates.py`, **attention** in
`models/attention.py`, **task construction** in `data/probes.py`, and
**optimizer rules** in `training/optimizers.py`. Experiment budgets and grids
belong in JSON configs. The full trainer exposes its corresponding optimizer
schedule and distributed settings in `full_training/`.

## Local installation and first run

From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
fox-experiment download --data-dir data/longcrawl64_pilot
fox-experiment train --config configs/smoke.json --out outputs/smoke_v1
fox-experiment report --out outputs/smoke_v1
```

The verified 18 MiB pilot is already present in the local folder. A new clone
reconstructs it from about 248 MB of version-pinned native chunks. See
[data/README.md](data/README.md) for checksums and split provenance.

For actual downscale training use `configs/downscale.json` and a CUDA GPU. The
CPU smoke profile is not a scientific experiment. Add `--resume` to continue an
interrupted small-model run with the same configuration, code and data; choose
a fresh output directory when changing settings.

Scientific small-model runs now stop before optimizer comparisons if short
acquisition fails. The smoke profile still runs all branches to check software.
`--allow-unqualified` explicitly enables diagnostic continuation of failed
sources; its results cannot establish an acquired rule's generalization boundary.

The current generated probes use protocol `latest_write_v2_independent_stale`,
which removes v1's deterministic old-value shortcut. Use a fresh run name after
updating; retain old outputs as v1 records. The cached text corpus is unchanged.

To troubleshoot an extracted report ZIP without model weights:

```bash
fox-experiment diagnose --out PATH_TO_EXTRACTED_RESULTS/text
fox-experiment diagnose --out PATH_TO_EXTRACTED_RESULTS/mechanism --mechanism
```

The commands write explanatory tables under each run's `diagnostics/` directory
and preserve the input predictions and scientific scores.

## Git and Google Colab

This folder has its **own Git repository**. Dataset binaries, credentials,
checkpoints and outputs are ignored.
Your original notebook and supplied source folder remain available unchanged.

Create a GitHub repository, then run these commands here with its actual URL:

```bash
git add .
git commit -m "Organize FoX optimizer experiments"
git remote add origin https://github.com/YOUR_NAME/fox_experiments.git
git push -u origin main
```

In Colab, open `notebooks/01_downscale_training.ipynb` from GitHub (or upload it),
set `REPO_URL` in its first code cell, and select a GPU runtime. The notebook
clones and installs this repository. Start with `PROFILE="smoke"`, then switch
to `"downscale"`. Enable `USE_GOOGLE_DRIVE` to retain data and results between
runtime sessions. For a private repository, authenticate using Colab's normal
Git workflow; do not paste access tokens into a saved notebook.

The bootstrap reuses an existing checkout. After changing source on GitHub,
pull it into the Colab checkout and restart the Python runtime before running
again, so previously imported modules do not stay in memory. Use a recorded
commit for research runs.

## Full training

See [docs/full_training.md](docs/full_training.md) and the second notebook.
The full path uses native `train.zarr` for training and separate heldout
partitions. It provides BF16, gradient accumulation, distributed data parallel
training, activation checkpointing, saved optimizer/RNG state, and a GPU kernel
comparison before training. Data download and training require explicit launch
steps in the notebook. Full-data storage is roughly a terabyte; it does not fit
the purpose of a Colab pilot.

The default full configuration targets approximately 124M parameters. It is a
larger version of this experiment, with settings that must be assessed using
short validation before drawing generalization conclusions.

## Scientific interpretation

- Factorized and direct first-block constant gates are function-matched at the
  continuation fork. Original data-dependent FoX is a separate reference family.
- Native Adam arms share moments `(0.1, 0.1)`. Epsilon annealing is a documented
  finite-precision adaptation of the newest manuscript's schedule.
- Target lag and older-prefix length are evaluated independently. Passing the
  largest tested lag means the boundary is not yet observed.
- Only the controlled model has its model-specific arbitrary-prefix bound.
  An ordinary language model's finite test does not certify infinite context.
- A failed short acquisition or numerical stop stays in the result ledger.

Read [the experiment design](docs/experiment_design.md),
[source provenance](docs/source_provenance.json), and
[validation status](docs/validation.md) for details. The authoritative theory
source is the supplied `learning_with_cot-3.pdf`; its hash is recorded without
copying the private manuscript into this repository.
