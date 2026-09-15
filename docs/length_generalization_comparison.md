# Length generalization: original FoX versus our setup

Run [notebook 04](../notebooks/04_length_generalization_comparison.ipynb) for the
joint model/optimizer comparison. Notebook 03 changes optimizers on original
data-dependent FoX only; it does not test the constant factorization assumption.

## What is compared

| Gate setup | Paper AdamW | Our fixed Adam | Our annealed Adam | Our SGD |
|---|---:|---:|---:|---:|
| Original data-dependent FoX, paper initialization | Yes | Yes | Yes | Yes |
| First gate `g=softplus(u*v)`, our initialization | — | Yes | Yes | Yes |
| First gate `g=softplus(z)`, our initialization | — | Yes | Yes | Yes |

This gives **10 branches**. Paper AdamW uses `(0.9,0.95)`, fixed epsilon,
weight decay, gradient clipping, and linear warmup followed by cosine decay.
Our Adam variants use `(0.1,0.1)`, no weight decay or clipping, polynomial
learning rates, and fixed versus exponentially decaying epsilon. Our SGD has
no momentum, clipping, or decay. Gate/output learning-rate multipliers match
the existing setup. All rates and budgets are in the JSON configuration.

### What is held fixed

Every branch uses the same plain FoX (LLaMA) backbone dimensions, RMSNorm/FFN
choices, initial non-gate weights, pure natural-text objective, input batches,
training-token budget, and held-out evaluation windows. Training starts from
initialization; there is no shared acquisition checkpoint or optimizer reset.
The backbone follows the [audited reference](paper_baseline_reference.md).

The original-data branches keep zero gate biases in every layer. Our constant
branches use first-layer decay `g0=2.944102615`, with positive balanced factors
`u=v` in the factorized case, and later-layer initial decay `0.1`. Later layers
still have trainable data-dependent gates. Direct and factorized constant pairs
start with matching whole-model functions, including the later gates. Their
parameter tensors and parameter counts need not be identical.

The original and constant branches have different initial gate functions.
Their comparison changes **gate parameterization and initialization**, including
the later-layer biases. The shared backbone removes the older experiment's
additional normalization/FFN/weight-initialization differences. This is a
controlled transfer of our gate and optimizer setup, not a replay of the old
mixed-retrieval training run.

Use the comparisons separately:

- Original-data paper AdamW versus original-data native optimizers: training
  recipe comparison on one initial model.
- Direct versus factorized at the same native optimizer: factorization effect
  from a matching initial function.
- Fixed versus annealed Adam at the same gate setup: epsilon-schedule effect.
- Original paper branch versus our factorized branch: combined setup effect.

Paper AdamW on constant gates is not included, so this is not the full
gate-by-optimizer factorial. All settings are candidate recipes, not equally
tuned optima. Do not interpret a recipe difference as the effect of one setting.

## Length-generalization measurements

The Colab preset trains on length **T=256** and tests context windows
**128, 256, 512, 1024, 2048**. Every window predicts the same final 32 target
tokens of a held-out source window, with the paper's BOS/EOT reset convention.
The 128-token case is a shorter-context control; lengths above 256 are finite
extrapolation tests, up to **8 times** the training length.

For branch `a`, define `L_a(C)` as mean negative log likelihood (NLL) on those
same target tokens with context window length `C`. The code records:

```text
nll_difference(C) = L_a(C) - L_paper(C)                 # negative favors a
context_gain_a(C) = L_a(T) - L_a(C)                    # positive: context helps
extra_context_gain_difference(C)
    = context_gain_a(C) - context_gain_paper(C)        # positive: more gain
    = nll_difference(T) - nll_difference(C)
```

**At C=T, both context-gain quantities are zero by definition.** Those zeros
are expected and do not indicate failed training. Absolute NLL is not forced
to zero. Paired differences are computed on matching document/sample/target
identities; missing or failed paper baselines and unmatched targets are marked
explicitly rather than replaced with zero.

Read absolute NLL and context gain together. A poorly trained short-context
model can gain more from an extra prefix while still having worse long-context
NLL. Training loss, held-out NLL at the training length, and initial-model
measurements help reveal that case. The per-position loss plot is another
description of the long-context behavior, not a sufficient retrieval certificate.

This is a language-model context-utilization comparison. It does not establish
that a model learned our synthetic latest-write rule. Such a retrieval study
still needs a matched task-training stage, short causal-edit qualification,
and separate lag/prefix tests. Likewise, the ordinary multilayer model is
outside the controlled scalar theorem even when its first gate is factorized.
Passing all finite contexts does not prove arbitrary-length generalization.

## Run the code

The implementation is separated into:

- [Model variants](../src/fox_experiments/paper_baseline/model.py):
  `build_comparison_fox`, including the balanced factorization.
- [Training runner](../src/fox_experiments/paper_baseline/runner.py): matched
  batches, optimizer states, checkpoints, and evaluation windows.
- [Analysis](../src/fox_experiments/paper_baseline/analysis.py): paired metric
  formulas and plots, operating on saved results without retraining.
- [Configuration](../configs/length_generalization/colab.json): the complete
  model, optimizer, budget, and context-grid settings.

From the repository root:

```bash
pip install -e .
fox-experiment download --data-dir data/longcrawl64_pilot

# First verify software on CPU.
python -m fox_experiments.paper_baseline \
  --config configs/length_generalization/smoke.json \
  --data-dir data/longcrawl64_pilot \
  --out outputs/length_generalization/smoke_v1 --device cpu

# Then train the comparison on a Colab GPU.
python -m fox_experiments.paper_baseline \
  --config configs/length_generalization/colab.json \
  --data-dir data/longcrawl64_pilot \
  --out outputs/length_generalization/colab_v1 --device cuda
```

Or call the API directly:

```python
from fox_experiments.paper_baseline import (
    PaperBaselineConfig, run_paper_comparison, summarize_paper_comparison,
)

config = PaperBaselineConfig.from_json("configs/length_generalization/colab.json")
out = "outputs/length_generalization/colab_v1"
run_paper_comparison(config, "data/longcrawl64_pilot", out, device="cuda")
report = summarize_paper_comparison(out)
print(report["comparison_vs_paper"])
```

The notebook wraps these calls in short, editable cells and exports a report
ZIP. Colab needs the updated repository committed/pushed, `REPO_URL` set, and
a GPU runtime for `PROFILE="colab"`. Start with `PROFILE="smoke"`; use a
fresh run name when switching profiles or changing settings. Optional Drive
storage preserves checkpoints across runtime disconnects.

The default budget is **4,096,000 tokens per branch**, or **40,960,000 tokens
across the ten branches**. The pilot uses the existing document-separated
LongCrawl64 subset, with reproducible random windows that may overlap. It is
not the published full-data/full-scale run. See the reference audit for the
original native-data training configuration.

## Results and interpretation

- `summary.csv`: branch status, optimizer/gate identity, hashes, budget, and fit.
- `context_summary.csv`: NLL and context gain by branch and context length.
- `comparison_vs_paper.csv`: paired differences against original paper FoX.
- `lm_raw.csv`: per-position and matched-target measurements, including initial
  models; these are the source of the summary tables.
- PNG plots: loss versus position, same-target context curves, and context gain.
- Per-branch checkpoints and `failures.json`: retained even for failed runs.

`--analyze-only` regenerates reports from saved measurements. It does not
evaluate new context lengths. Predeclare a larger grid before training, or
conduct a separately identified checkpoint-evaluation study later. Resume
requires matching configuration, source, data, device type, and runtime versions.

One seed and a few test documents are a pilot. For confirmation, repeat
predeclared seeds and use more independent documents. Treat documents, not
individual suffix tokens, as independent evaluation units. Learning-rate or
budget tuning must use short validation; freeze those choices before examining
the long-context test grid. Weak short-context fitting is not evidence of an
SGD length limit.
