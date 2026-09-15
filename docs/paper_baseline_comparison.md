# Compare with the original FoX training recipe

## What was previously run

The saved `downscale_v1_results.zip` includes
`seed0_original_data_paper_adamw`. That branch uses ordinary AdamW ingredients,
but it inherits our modified gate initialization, small architecture, mixed
retrieval objective, and shared-acquisition procedure. It is an **adapted AdamW
control**, not a run of the paper's default training. Its zero retrieval score
does not establish a failure of default FoX.

The new [comparison notebook](../notebooks/03_paper_baseline_comparison.ipynb)
addresses that gap. It trains a downscaled **FoX (LLaMA)** from initialization
on ordinary language modeling and compares optimizers on the same initial
weights, data batches, objective, token budget, and evaluation targets.

## What matches the reference

- Every layer has an ordinary, data-dependent forget gate with bias initialized
  to zero. There is no special strong first-layer gate.
- No positional embeddings; pre-normalized RMSNorm with epsilon `1e-6`, SwiGLU,
  and an untied GPT-2 vocabulary input/output pair.
- Gaussian initialization with standard deviation `0.02` for linear and
  embedding weights, without our earlier residual-projection rescaling.
- Feed-forward dimensions use the upstream rounding to a multiple of 256.
- The paper optimizer uses AdamW `(0.9, 0.95)`, epsilon `1e-8`, weight decay
  `0.1` excluding norms and biases, clipping at `1.0`, linear warmup followed
  by cosine decay. The peak learning rate `0.002` comes from the paper's
  smaller FoX (LLaMA) configuration.
- Natural-text next-token loss gives all target tokens equal weight. Input
  chunks start with the reference EOT/BOS token and shifted target tokens.
  Optimizer moments continue throughout each run; there is no acquisition fork.

This is a portable PyTorch implementation of the plain architecture and its
initialization distribution. It does not promise bitwise agreement with the
upstream fused implementation. FoX **Pro** additionally has KV shifts, QK
normalization, output normalization, and an output gate; it is a separate
architecture and is not silently substituted here.

See the [reference audit](paper_baseline_reference.md) and
[source hashes](paper_baseline_provenance.json).

## Comparison matrix

All four arms use the same ordinary data-dependent FoX architecture:

| Arm | Moments | Epsilon | Other training choices |
|---|---|---|---|
| `paper_adamw` | `(0.9, 0.95)` | Fixed `1e-8` | Reference decay, clipping, warmup/cosine |
| `adam_fixed` | `(0.1, 0.1)` | Fixed `1e-8` | Our polynomial rates and parameter-group multipliers; no decay/clipping |
| `adam_annealed` | `(0.1, 0.1)` | `1e-8 exp(-0.01t)` | Same as fixed Adam apart from epsilon |
| `sgd` | No momentum | — | Our polynomial rates and multipliers; no decay/clipping |

The primary comparison tests the training recipes. It changes multiple settings
between paper AdamW and our Adam; it does not attribute the difference solely
to epsilon. The fixed/annealed pair isolates epsilon. The configured pilot
learning rates are starting settings, **not validated optima** on this objective.
Any learning-rate tuning must use short-context validation with equal tuning
budgets before freezing a long-context evaluation.

The factorized constant-gate experiment remains in the first notebook. It tests
a different architecture assumption. The ordinary-gate comparison here cannot
establish a theorem that assumes `g=softplus(u*v)`.

## Run on Colab

1. Commit and push the updated repository, then open
   `notebooks/03_paper_baseline_comparison.ipynb` in Colab.
2. Set `REPO_URL` and choose a GPU runtime.
3. Run `PROFILE="smoke"` once to verify installation and all four arms.
4. Switch to `PROFILE="colab"` and choose a fresh `RUN_NAME`. Enable Google
   Drive in the storage cell if results should survive a runtime disconnect.
5. Run the remaining cells and export the report ZIP.

The Colab preset has width 128, four layers, two heads, 256-token training
contexts, and 2,000 updates. Batch four with two accumulation steps gives
**4,096,000 training tokens per arm**. It tests context lengths 256, 512, and
1,024. The pilot data are the existing 128 training documents and separate
validation/test documents, repurposed from native heldout rows. Random windows
can repeat; token budget is not a count of unique data. This scale is intended
to measure feasibility and initial trends, and successful language modeling
must be established before making a generalization claim.

From a terminal, the same run is:

```bash
python -m fox_experiments.paper_baseline \
  --config configs/paper_baseline/colab.json \
  --data-dir data/longcrawl64_pilot \
  --out outputs/paper_baseline/colab_v1 \
  --device cuda
```

Use `--resume` only for the same code, configuration, data, and device type.
Changed settings require a fresh output directory. There is no automatic
training or GPU allocation when merely importing the package.

## What to read in the results

**Primary:** held-out per-position language-model NLL, with a marker at the
training context length. Also compare the exact same target tokens with
different amounts of preceding context. Positive context gain means that
additional context reduced their NLL. This avoids mistaking easier targets in
another document region for improved context use.

Reports retain each arm's initial and trained measurements, training token
counts, and run status. A lower final NLL by itself demonstrates better language
modeling; evidence of context use requires the paired context comparison too.
One seed and four test documents provide only a pilot comparison, not reliable
training-seed uncertainty or a universal optimizer ranking.

The optional short synthetic retrieval diagnostic is secondary. A pure language
model has not been explicitly trained on our query grammar, so failing that
probe does not invalidate its language-model evaluation. Do not infer a
retrieval boundary from an unacquired task. To compare retrieval, add the same
post-training task curriculum to every arm and require short causal-edit
success before testing long lags.

## Full published-scale replication

The copied [upstream configuration](../configs/paper_baseline/upstream_llama_125m_2b.yaml)
preserves the authors' smaller published recipe: 12 layers, width 768,
16,384-token contexts, and 2,684,354,560 training tokens. The paper's "125M"
convention excludes input embeddings; the total model is approximately 162M
parameters. Our existing `h200_124m` configuration is a different model.

Use the original upstream trainer and native `train.zarr` for that replication;
the launch example is in the reference audit. The portable pilot's random-window
data order, smaller model/context/budget, and nonfused components are deliberate
scale/implementation changes. This task has not run a full replication or
established default FoX's long-context performance on the user's hardware.
