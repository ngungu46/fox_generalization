# Latest-answer loss versus all-token training

Open **notebook 05: `05_loss_and_constraints.ipynb`**. It defaults to the Colab
training profile; the CPU smoke profile is explicitly software validation only.
The notebook calls readable modules in `src/fox_experiments/loss_study/`.

## Question and experimental unit

Does changing the objective remove the optimizer-dependent retrieval behavior,
and which architectural/training restrictions make that behavior observable?

Each matched pair sees the **same complete sequences, initial model function,
minibatch order, optimizer settings, and update budget**. Only the loss changes.
The core grid is two losses × two binding-gate parameterizations × three
optimizers, for 12 branches per seed. All branches start from initialization;
there is no optimizer-specific acquisition checkpoint or reset during training.

### Experiment A: latest-answer supervision

An example is `BOS k1 v1 ... kL vL QUERY key answer`. The answer is the value of
the most recent record with that key. The answer token is never an input to its
own prediction. Only the final answer position contributes to the loss:

```
L_answer = mean cross_entropy(logits_at_query, latest_value)
```

This preserves the paper's **supervision on the latest matching value**. Its
full-vocabulary softmax is a deliberate modification of the paper's binary
decoder. That same softmax is used in Experiment B, so the main comparison
does not change both the loss mask and output normalization at once.

### Experiment B: ordinary next-token prediction

Use identical examples and predictions, and average cross-entropy over **every
non-padding target token**, with unit token weights:

```
L_all = (sum CE on answer tokens + sum CE on other tokens) / target_token_count
      = answer_fraction * L_answer + (1 - answer_fraction) * L_other
```

This is the standard causal language-model objective, on controlled sequence
data. It is not a claim that the primary data are ordinary web text. Keys and
new values include unpredictable choices, so all-token loss has an entropy
floor; low total loss is not the retrieval success criterion.

The optional `answer_upweight` intervention changes Experiment B's token
weights. Its reports must be read as **weighted NTP**, not the unit-weight
practical control. It diagnoses loss dilution; it does not remove conflicting
non-answer gradients. Baseline rates are identical across the two losses.
Optional short-only LR tuning is a separate, explicitly tuned comparison.

## Data and limits

The default `records` mode has a small vocabulary and generated held-out
histories. It requires no download and avoids spending the pilot budget on
learning a 50,257-token language vocabulary. The default support mixes
single-record calibration, overwrite, and delayed-recall examples, with
randomized keys, values, target lag and older prefix. Training target lag is at
most four records. The `theory_template_support` intervention instead uses the
literal binary calibration/O/C support with mixture .6/.2/.2 and lag at most
two. Template support alone does not make the neural model the theorem model.

The `text_background` extension uses the project's checked LongCrawl64 subset
and GPT-2 tokenizer. Controlled update sentences are interleaved with short real
text spans. This is **semi-synthetic text**, with a much larger vocabulary and
harder acquisition problem. It requires causal routing and is run separately
after the compact experiment. It does not replace a future natural dialogue
state-tracking benchmark. Download through the existing `fox-experiment download`
command; the notebook exposes this only when text mode is selected.

## Architecture restrictions

The core is a two-layer trainable FoX with embeddings, residual paths, RMSNorm,
SwiGLU and a learned full-vocabulary output head. Both constant first-gate
parameterizations begin with matching logits:

* Factorized: `g = softplus(u*v)`, positive balanced factors.
* Direct: `g = softplus(z)`, with `z` equal to the actual initial product `u*v`.

Later retrieval forgetting uses a separate **single-logit constant gate** in
the baseline. Making this gate factorized would change the proposed competition
between matching and forgetting; we do not do so implicitly.

The downscale default initializes retrieval decay at **0.5 per token**. A local
short-only calibration found that the previous 0.1 initialization could fit
zero-prefix examples while failing stale-prefix cases. At 0.5, the factorized
fixed-Adam answer-only branch passed every sampled short edit panel by 1,000
updates. This selected an acquisition setting using short cases only, not a
desired long-range outcome. Other branches still need to qualify. The
`weak_retrieval_init` control restores 0.1. Neither setting changes the first
binding gate's initial decay of approximately 2.944.

Baseline `routing='record'` supplies only the serialization format. In the first
layer, value tokens can bind to preceding key-token positions within distance
three. Other first-layer rows have the same short causal window. Later layers
attend to completed value-token positions. Rows without an eligible token use
self-attention. Masks never identify equal keys, the correct record, or the
answer. This is a restricted FoX experiment, not the paper's default architecture.

With `routing='causal'`, every layer uses ordinary causal attention. That is the
first architecture relaxation to run. Gold target/conflict positions are used
only for diagnostics and never passed as selection masks in the model forward.

## One-condition interventions

Run one named intervention per output directory. Keep all seeds and failures.
These are tests of necessity **within the tested setup and budget**, not proofs
that a condition is mathematically necessary.

| Name | Change | Diagnosis |
|---|---|---|
| `baseline` | Core paired study | Loss × factorization × optimizer |
| `unrestricted_routing` | Replace supplied format masks by causal attention | Is learned parsing/binding the bottleneck? |
| `data_dependent_retrieval` | Learn token-dependent later gates | Does context-dependent forgetting alter the effect? |
| `qk_normalization` | Normalize query/key vectors | Does restricting content-score scale matter? |
| `learned_output_gate` | Gate attention output using the current hidden state | Can the model learn when to use memory? |
| `slow_representations` | Lower representation LR and give it a summable tail | Does representation drift hide the mechanism? |
| `weight_decay` | Add positive optimizer weight decay | Does parameter shrinkage oppose the growth route? |
| `gradient_clipping` | Clip global gradient norm at one | Does clipping materially change updates? |
| `larger_epsilon` | Start both Adam controls at epsilon 1e-4 | Is epsilon actually active? |
| `answer_upweight` | Weight answer tokens by 16 in the all-token loss | Dilution versus conflicting gradients |
| `theory_template_support` | Use literal original short templates | Objective support versus task acquisition |
| `longer_training_lag` | Train up to lag eight | How does empirical range depend on training lag? |
| `dense_stale_prefix` | Make every older prefix record a matching write | Stress the accumulated obsolete mass |
| `weak_retrieval_init` | Restore retrieval decay 0.1 instead of 0.5 | Was stale-prefix acquisition limited by initial recency? |
| `faster_retrieval_updates` | Raise later-gate LR multiplier from .05 to .5 | Is the retrieval gate adapting too slowly? |

One-at-a-time results do not identify all interactions. After identifying a
candidate explanation, cross it with the loss and gate mode in a confirmation
run with fresh seeds. `slow_representations` is a bundled representation-policy
intervention: its LR amplitude and tail exponent should subsequently be varied
separately if that bundle matters. QK normalization, output gating and routing
are architectural changes; they are not function-matched to baseline.

## Independent generalization axes

Default training uses target lags 1–4 and at most four earlier records. The test
grid is lags `[1,2,4,8,16,32]` × older-prefix sizes `[0,4,16]`.

* Increasing target lag adds newer **off-key** records after the latest match.
* Increasing older prefix adds records before the latest match, including stale
  matching values, while retaining the target and newer suffix.
* Latest-value edits must change the answer; stale/unrelated edits must preserve
  it. Alternative-key queries test whether the model actually uses the key.
* Inapplicable edits are omitted, not counted as successful unchanged copies.

The same base history keeps its queried key and answer across both axes. Lag
growth appends a nested suffix; prefix growth preserves the target and later
records. Older matching values are independent draws except for one explicitly
conflicting overwrite anchor. The default stale-key fraction is .5. Changing
the queried key can change its actual geometry; raw metadata records both the
base panel labels and actual lag/prefix, and short tests filter out queries
outside the declared training envelope.

Always read accuracy, true-answer probability, and NLL together. Short
qualification uses held-out histories and joint success on their applicable
edits. A branch that fails this test remains visible but receives **no valid
generalization-radius interpretation**. Its radius is missing, not zero.

Any reported range is measured only on the explicit sampled grid and finite
prefix panel. Passing the largest lag is right-censored: it means **at least the
tested ceiling**, not infinity. Untested lags are not certified. Evaluation at
several training checkpoints distinguishes expanding tested range from a single
final accuracy curve. Finite SGD plateaus do not prove an asymptotic bound.

## Troubleshooting order

1. **Short causal retrieval fails:** inspect all-edit accuracy, key-query changes,
   output collapse and learning curves. Increase/tune short training before
   interpreting any optimizer-dependent length limit.
2. **Answer-only succeeds; all-token fails:** inspect the answer fraction and
   answer/other gradient norms and cosine similarity. A small positive aligned
   gradient suggests dilution; negative alignment suggests interference.
3. **Both fit short, neither extrapolates:** inspect binding, target-versus-other
   content scores, accumulated forgetting, stale-prefix sensitivity and answer
   confidence. Factorization alone is not sufficient.
4. **Adam variants overlap:** check per-group second moments relative to epsilon.
   Annealing an already negligible epsilon need not create a visible effect.
5. **Factorized/direct differ:** compare their matching initialization, short fit
   and effective gate updates. Equal scalar LR does not mean equal effective
   gate speed; the change in optimization geometry is part of the intervention.
6. **A restriction helps:** repeat with fresh seeds and cross the restriction with
   the relevant loss/gate condition. Keep the short-validation selection rule
   fixed before inspecting the long test.

The actual update uses the complete chosen loss once. Gradient diagnostics do
not perform separate Adam steps for answer and non-answer terms.

Annealed epsilon uses ordinary FP32 optimizer arithmetic. A run stops with a
recorded numerical-range failure if scheduled epsilon leaves the normal FP32
range; it never silently floors epsilon or substitutes signed-log arithmetic.
The default 2,000-step budget stays within that range. Longer budgets may need a
separately named, slower epsilon schedule; that is a changed experimental setup.

## Scope relative to the latest theory

Source: `learning_with_cot-3.pdf`, Sections 2–5. The theorem additionally requires
specified scalar matching factors, retained positive key geometry, controlled
amplified binding leakage, a binary oriented decoder, special SGD steps/batches,
and specific Adam/representation schedules. This neural experiment relaxes
several of those requirements. It tests a finite transfer hypothesis and helps
locate its failures; it does not implement a proof or inherit the number four
as an expected universal cutoff.

The existing `mechanism.run_mechanism` remains an optional closer scalar
reference. It has its own documented practical deviations and must be reported
separately from this neural objective comparison.

## Running and saved evidence

```bash
python -m fox_experiments.loss_study --config configs/loss_study/colab.json \
  --out outputs/loss_study/colab_v1 --device cuda
```

The notebook supplies a Drive option, displays the full branch/budget table,
exports reports, and supports resume with saved optimizer moments. Changed
source/configuration/data/runtime requires a fresh run directory. The result
archive excludes weights by default; checkpoints remain in the run directory.
No Colab GPU scientific outcome is claimed by the included local smoke checks.

Optional `tune_learning_rates` uses a separate short-validation namespace and
equal trial budgets; it records every candidate, failure and selected per-branch
rate. Baseline uses the same rates across losses. A tuned study can change rates
across losses and therefore answers a different question: best short-validated
performance within the declared search. Choose a fresh run name for that study.

For confirmation, set `seeds=(0,1,2)` (or another predeclared set), increase the
held-out history count, and retain failed seeds in the report. The default
single-seed 16-history panels are a pilot, not a significance analysis.
