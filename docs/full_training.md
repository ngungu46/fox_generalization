# Full training on the H200 cluster

The full notebook is a control and analysis interface. Models, data access,
optimization and evaluation live in Python modules. Run long jobs under your
scheduler; reopen the notebook to inspect their outputs.

## What this path implements

The default is a **123,889,410 parameter** FoX model with 12 blocks, width 640,
10 heads, 2,048 training tokens and untied GPT-2 embeddings/output weights.
It uses the same architecture, first-block gate interventions and latest-write
probes as the downscaled path. Attention runs through the supplied authors'
Triton kernel, without adaptive pruning. RMSNorm, SwiGLU, no positional embedding,
and data-dependent forgetting in the remaining blocks match the shared model.

The full trainer is a readable PyTorch DDP implementation, **not an exact
reproduction of the authors' Lightning/FSDP runs**. The supplied paper's original
125M configuration used width 768, 12 layers, 16K contexts and a different token
budget. The optional `h200_715m_scale.json` raises our model to approximately
715M parameters and 16K contexts. It is a scale experiment, not the paper's
760M/48B recipe. The upstream model's optional FoX-Pro changes are not used.

## Data

The pilot uses a small heldout-derived development corpus. Full training instead
requires native **train.zarr** (6,609,334 × 65,536 tokens) and **heldout.zarr**
(52,131 × 65,536 tokens). These shapes come from the supplied authors' datamodule.
The full reader never substitutes the pilot subset for the training split.

The original `gs://longcrawl64` source was unavailable during earlier data work.
The downloader therefore uses the public mirror `clankur/longcrawl64`, pinned to
revision `5e2f38e601ca7c16fa304f30536d9e2945bb2a32`. It downloads native Zarr chunks,
not a flattened `val.bin`. This is a mirror, not an assertion of official hosting.

Upstream estimates a roughly 800 GB dataset; the native logical uint16 payload is
about 873 GB, and the actual compressed download varies. Preview the source and
destination first:

```bash
python -m fox_experiments.full_training.data --data-root /scratch/longcrawl64
```

Download explicitly when your cluster storage is ready:

```bash
python -m fox_experiments.full_training.data --data-root /scratch/longcrawl64 --download --verify
```

Provision at least 1 TB and stage onto fast local/shared scratch. Zarr's native
chunks span many documents; random windows may be limited by decompression or
storage throughput. Measure data loading and GPU utilization before committing
to a large token budget. The reader keeps arrays lazy and checks every accessed
chunk, preventing incomplete downloads from silently returning Zarr fill zeros.

Heldout rows `[0,2048)` are for learning-rate tuning, `[2048,4096)` for independent
short-rule confirmation, and `[4096,52131)` for the frozen test grid. All are
excluded from training. Short `tune`/`confirm` commands never run the long test.

## Environment and hardware preflight

Use a CUDA-compatible PyTorch installation supplied by your cluster, then:

```bash
python -m pip install -e '.[full]'
python -m fox_experiments.full_training preflight \
  --config configs/full/h200_124m.json --data-root /scratch/longcrawl64 \
  --check-data --check-kernel
```

The GPT-2 tokenizer downloads its small vocabulary on first use. On offline
compute nodes, prepare it on a network-enabled login node, set
`TIKTOKEN_CACHE_DIR` to a shared directory, and run
`python -c 'import tiktoken; tiktoken.get_encoding("gpt2")'` once there. The pilot's
existing tokenizer cache is also reusable; this does not reuse its training data.

`--check-kernel` checks BF16 forward output and Q/K/V/**gate** gradients against
a dense FP32 reference on the actual GPU. Passing CPU tests does not validate
Triton compilation, CUDA performance or H200 memory use. The kernel is copied
unchanged from the supplied folder; its SHA256 and MIT/Apache notices are in
`third_party/forgetting_transformer/`.

The default has four DDP ranks, one sequence per GPU microbatch and 16 gradient
accumulation steps: **131,072 input tokens per update**. DDP replicates the model
and optimizer on each GPU. BF16 autocast affects matrix operations; parameters,
Adam moments, forget gates and gate prefix sums remain FP32. Layer activation
checkpointing and chunked/checkpointed vocabulary loss limit activation memory.

## Experiment stages and pairing

1. Acquire a `direct_constant` source and, separately, an `original_data` source.
   Each receives 2,048 natural-LM updates followed by 2,048 mixed LM/latest-write
   updates. The source uses paper-style AdamW, with a single continuous schedule
   over the two phases. This differs from the pilot's phase-local optimizer resets.
2. Copy the direct source into the factorized and direct first-gate arms.
   Conversion preserves `softplus(u*v) == softplus(b)` and every nongate weight.
   Positive raw gates use balanced factors; nonpositive raw gates use an
   explicitly asymmetric conversion outside the theorem's initialization region.
   The data-dependent gate has its own source and is an architecture control.
3. Start fresh optimizer states for SGD, fixed-epsilon Adam and annealed-epsilon
   Adam; paired branches receive the same seeded training example stream.
4. Evaluate each source baseline and each continuation halfway and at the end.
   The lag × old-prefix grid and causal edits are shared with the pilot.
   Natural-language per-position loss and same-target context comparisons are
   retained. Source baselines are saved separately in combined analysis.

Training weights are normalized over the entire accumulated global batch. This
gives the same weighted objective regardless of its partition across DDP ranks.
The pilot averages microbatch means, which can differ when weighted token counts
vary across microbatches. Keep this distinction in cross-scale comparisons.

The primary Adam moments are `(0.1,0.1)`, with first-gate/retrieval-gate/output
rate multipliers `(1.5,0.05,0.1)`. SGD uses zero momentum and zero weight decay;
the two theory-inspired Adam arms also use zero weight decay and no clipping.
The paper-style source uses AdamW `(0.9,0.95)`, weight decay 0.1 on matrix weights,
and gradient clipping 1.0. Adam epsilon is `1e-8 exp(-0.01 k)` in the 124M config;
the longer-context scale config uses decay 0.001 as a finite practical sensitivity
setting. Both use a polynomial continuation rate schedule. These are practical
adaptations, not an implementation of Appendix S's extreme signed-log arithmetic.
The run stops before epsilon leaves normal FP32 range. First-gate factors,
epsilon dominance and zero-moment fractions are logged at checkpoints.

## Calibrate before opening long-test results

The JSON learning rates are **candidates**, not measured optimal full-scale rates.
Use pilot evidence as an initial range, then run equal-budget full-size short
trials. Select using worst short-lag accuracy on `tune`, breaking ties by NLL.
Freeze the chosen rates and confirm on `confirm`; only then run the test matrix.
If a best rate hits a grid edge, expand that optimizer's grid on `tune`.

Example after acquiring the shared direct source:

```bash
torchrun --standalone --nproc_per_node=4 --module fox_experiments.full_training train \
  --config configs/full/h200_124m.json --data-root /scratch/longcrawl64 \
  --stage continuation --gate factorized_constant --arm adam_annealed \
  --init-from runs/full/source_direct/last.pt --run-dir runs/trials/adam_lr001 \
  --lr 0.001 --steps 128
python -m fox_experiments.full_training evaluate \
  --config configs/full/h200_124m.json --data-root /scratch/longcrawl64 \
  --checkpoint runs/trials/adam_lr001/last.pt --output runs/trials/adam_lr001/tune --split tune
```

Repeat the same candidate budget for SGD and both Adam arms. Keep trials outside
the final matrix's run root. Update a copied JSON config with the selected rates
and record the short-trial decisions. Resuming a trial as a full run is rejected:
full continuations restart from the common acquired source.

## Launch the frozen matrix

Preview all commands and token costs (no training):

```bash
python -m fox_experiments.full_training matrix \
  --config configs/full/h200_124m.json --data-root /scratch/longcrawl64 \
  --run-root runs/full/h200_124m --gpus 4 --seeds 0 1 2
```

Add `--execute` to run the printed stages. The primary grid is two sources plus
nine continuations: **5,905,580,032 input tokens per seed**, or about 17.72B for
three seeds. Short trials, tests and source qualification add compute. No wall
time is promised. Use one seed to establish throughput and feasibility first.
The 715M scale configuration has a separately reported, much larger budget.

`scripts/slurm_full.sbatch` is a single-node four-GPU template. Set your school's
partition/account/environment and `FOX_REPO_DIR`, `FOX_FULL_DATA_DIR` before
submitting. For multi-node jobs, launch the individual `train` command with your
cluster's `torchrun --nnodes/--node_rank/--rdzv_endpoint` setup and set
`expected_world_size` to the total GPU count. The matrix helper is single-node.

## Resume and analysis

Rerun the identical command to resume `last.pt`. It stores model, optimizer,
schedule position, integrated learning rate `S`, per-rank Python/NumPy/Torch/CUDA
RNG state, history and a configuration signature. Data sampling is deterministic
in optimizer step/global microbatch ID. Implementation files, source checkpoint
content and native data metadata are fingerprinted; changes cause a resume
mismatch. Large data chunk contents are not fully rehashed on every start.
Keep the source checkpoint and config available. Code/config edits should start
a new clearly named study. Device libraries can still affect numerical results.

Only rank zero atomically writes checkpoints. Halfway/final lightweight model
checkpoints permit finite-range growth curves. Numerical failures are reported
in `failure.json`, and the last complete checkpoint is retained.

```python
from fox_experiments.full_training.evaluation import collect_reports
report = collect_reports("runs/full/h200_124m")
```

This combines test CSVs and uses the shared report functions in `analysis/`.
It separates `source_*.csv` from continuation data so a longer source schedule
cannot hide continuation checkpoints. Individual directories retain short
qualification metrics and raw per-example results. A failure to acquire the
short rule remains an unqualified diagnostic run.

Finite tests can show a growing tested range or an SGD plateau. They cannot
establish infinite generalization, and the data-dependent gate control does not
inherit the constant-factorized theorem. Preserve unsuccessful seeds and report
both extrapolation distance and old-prefix interference separately.
