# Why the downscale run reports zeros

## Conclusion

**The text model did not acquire short retrieval. Its zero all-edit scores cannot
locate an SGD or Adam generalization boundary.** In the separate controlled
model, Adam's zero 99% radius is forced by insufficient decoder confidence.
SGD's R=8 zero instead reflects a real failure with older stale records.

The initial acquisition budget was not validated to teach the text task. Software
smoke tests passing did not establish that prerequisite. The fixes below make
this failure explicit; they do not establish a successful replacement recipe.

## Evidence and provenance

This analysis uses the user's actual `downscale_v1_results.zip`, not a new smoke
run. SHA256:

```text
0d600e2eac44bd782e8dc6e00a50c50987c054735f3c3d4682f42cb98cf5599f
```

The archive contains 7,200 retrieval/edit predictions, acquisition scores,
training traces, language-model losses, and controlled-model summaries. It has
no model weights. It records an A100-SXM4-40GB, PyTorch 2.11.0+cu128, and seed 0.
The archive remains untouched. Extracted records are under
`validation/user_downscale_v1/`; that directory is intentionally Git-ignored.
The original probe source hash matches repository commit `ce8c771`.

The supplied picture shows **nonzero language-model negative log likelihood**:
approximately 4–7.5 nats per position and 4.0–4.4 nats on the same-target context
comparison. A nearly flat loss curve means extra context provides little benefit
on those targets. It is a different measurement from retrieval accuracy or the
controlled model's sufficient radius.

## 1. Text retrieval: constant guessing

Both acquired sources failed before continuation:

| Acquired source | Qualified | Worst confirmation accuracy | Mean accuracy | Mean answer NLL |
|---|---|---:|---:|---:|
| Direct constant gate | False | 0% | 6.25% | 2.460449 |
| Original data-dependent gate | False | 0% | 6.25% | 2.460698 |

All 20 branch/checkpoint summaries have `short_qualified=False`. At the final
checkpoint, each branch produces exactly one distinct prediction throughout its
360 test examples:

| Branch family | Number of branches | Constant prediction | Base accuracy | Latest-value-edit accuracy | All-edit accuracy |
|---|---:|---|---:|---:|---:|
| Fixed/annealed Adam, all gate types | 6 | ` 3` (token 513) | 25% | 0% | 0% |
| SGD, all gate types | 3 | ` 8` (token 807) | 0% | 25% | 0% |
| Original FoX / paper AdamW | 1 | ` 8` (token 807) | 0% | 25% | 0% |

**Prediction change rate after editing the latest value is zero in every branch.**
The base answer and edited answer differ, so a constant prediction cannot get
both right. The all-edit conjunction correctly gives zero. There are only four
base histories per panel, reused across lag/prefix conditions; an accidental
constant match can therefore give 25% base accuracy without retrieval.

A reported tested range of zero means the first tested lag (32 tokens here)
fails the conjunction over prefix panels. It does not establish a boundary at
distance zero. Likewise, `failures.json=[]` in this older run records successful
numerical execution, not successful task acquisition.

### CPU reproduction and exposure

The original width-128, full-vocabulary acquisition was rerun on two CPU threads
using the same seed, cached text data, and 100 natural-text plus 400 mixed updates.
Mean accuracy is again 6.25%, all-edit accuracy is zero, and NLL is 2.460436,
within 0.000013 of the A100 constant-gate source. This is reproducible beyond
Colab. Training loss decreases, so the evidence points to failed conditional
task learning rather than a stopped training loop.

The mixed acquisition stream provides only **405 supervised answers** across
204,800 input tokens. Answer weights account for about 18.6% of loss weight,
despite the factor of 128 assigned to an answer. Strong first-layer forgetting
also makes binding nearby name/value tokens harder, but weakening that gate
alone to `g0=0.1` at the original budget still gives 6.25% accuracy and zero
all-edit accuracy. That weak initialization is a diagnostic outside the positive
balanced factor region, not a proposed theorem configuration.

## 2. Controlled model: confidence floor versus stale-prefix failure

For its positive binary decoder, even perfect retrieval cannot exceed

```text
best_possible_answer_probability = sigmoid(w)
99% feasibility requires w >= log(99) = 4.595120...
```

All twelve Adam branches end at `w ≈ 4.13553`. Their best possible probability
is **98.4258%**, below the 99% requirement. Thus the zero sufficient radius is
forced by the decoder floor before examining long-range attention. It is not
evidence that these branches have zero retrieval ability.

The full diagnostic evaluates every integer lag, using the saved scalar state
and exact witness expression including the geometric terms and binding leakage:

| Optimizer, either gate | Training R | Witness radius, no old prefix | Witness radius, infinite stale-prefix limit | Saved arbitrary-prefix sufficient radius |
|---|---:|---:|---:|---:|
| SGD | 2 | 10 | 9 | 9 |
| SGD | 4 | 12 | 9 | 9 |
| SGD | 8 | 16 | 0 | 0 |
| Fixed or annealed Adam | 2, 4, 8 | 0 | 0 | 0 |

The witness is a particular adverse family; passing it alone is not a uniform
certificate. The separate sufficient bound provides the arbitrary-prefix
guarantee within this controlled model.

For SGD at R=8, the decoder is sufficiently confident (`w ≈ 6.02083`), but the
old-prefix witness already has **1.2693% error at lag 1**. Eight stale same-key
records suffice to exceed the 1% threshold in a finite example. This is a real
counterexample to the requested uniform 99% claim for that checkpoint, even
though its no-prefix witness passes through lag 16.

All key-matching gaps are positive. The controlled acquisition flag checks a
positive matching gap and decoder orientation; it never meant 99% acquisition.
The acquired binding gates are already about 10.7–11.1, giving leakage around
2.4–4.7 × 10⁻¹⁰. Consequently, this short continuation barely challenges the
factorized-versus-direct binding distinction. Equal finite scores here neither
establish nor refute the proposed asymptotic difference.

## 3. What the supplied scalar-boundary hint does and does not explain

The older follow-up's saved scalar witness arrays are reproducible. Its
constant-scaling argument is useful for designing a controlled finite-radius
experiment. It does not explain why the text model always answers one digit.
Scalar record lag and text token lag are different quantities.

Its displayed floor formula is an approximation: it drops a finite geometric
numerator and binding leakage. For example, at `h=.001`, `m*delta=5`, `w=10`,
the approximation gives radius 1 while the exact witness gives 54. Use the exact
expression for boundary measurements, especially at small h.

The latest supplied manuscript, `learning_with_cot-3.pdf`, also limits direct
theorem claims about practical runs:

- Page 4, lines 199–201: the usual polynomial SGD scalar schedule is outside
  the theorem's stated guarantee.
- Page 14, Lemma 3, lines 749–750: the sufficient acquisition construction uses
  `gamma <= 1/16`; the older boosted example's `gamma=.2` is outside that construction.

These restrictions do not show that empirical generalization is impossible.
They mean those examples should not be described as satisfying every stated
theorem assumption. Extended-R cutoffs remain hypotheses until separately proved.

## 4. Changes made after this diagnosis

1. **Report the reason for zero.** Saved-result diagnostics separate individual
   correctness, all-edit success, edit sensitivity, constant predictions, and
   failed short acquisition. Range tables distinguish failure at the first
   tested lag from a passing test ceiling and invalid measurements.
2. **Stop unqualified scientific continuations.** The small experiment now
   skips optimizer trials and long tests for a seed if an acquired source fails
   short qualification. It records `short_acquisition_failed`. Smoke runs retain
   untrained branches for software checks; `--allow-unqualified` is an explicit
   diagnostic override. That policy is recorded in the immutable run specification
   and cannot change on resume; unqualified continuation is marked diagnostic.
3. **Remove a separate dataset shortcut.** The v1 stale value was always
   `(answer + 5) % 10`, deterministically revealing the answer. Protocol
   `latest_write_v2_independent_stale` draws an old value among the nine incorrect
   digits. This is a separate latent benchmark defect: the observed constant
   predictions did not learn that shortcut. The new generator is versioned in
   examples and run fingerprints. Use a fresh run directory; do not resume v1
   checkpoints against v2 or relabel the old results.
4. **Expose controlled decoder feasibility.** The notebook shows the maximum
   decoder probability, exact witness radius, sufficient radius, and diagnosis
   separately. Neither the scientific score nor its 99% threshold was changed.

The task grammar still puts the focal latest write in a predictable structural
position. Removing the digit correlation does not eliminate a possible
penultimate-write shortcut. Future key-binding claims need position/order
variation and checks on queries for different keys as well.

### Acquisition-only screens: no validated replacement yet

Two v2 screens used 2,000 updates, 8,000 supervised answers each, length 96,
lags 24/32, task-only data, batch 4, and AdamW LR 0.001. Short confirmation used
32 histories per lag and style with causal edits. No long-test scores selected
these settings.

| First gate | Mean answer accuracy | Worst variant accuracy | Worst all-edit accuracy | NLL |
|---|---:|---:|---:|---:|
| g0=2.9441 | 5.86% | 3.125% | 0% | 2.326259 |
| g0=0.75 | 5.08% | 0% | 0% | 2.326669 |

All-edit accuracy remained zero at every 250-update checkpoint. These negative
checks rule out treating a modest budget increase, task-only mixture, shorter
context, or weaker positive first gate as a validated fix.

Before spending the cluster budget, establish a short-task acquisition recipe
using held-out short confirmation and causal edits. A compact-to-text curriculum
or a pretrained FoX backbone are candidates requiring validation. Then freeze
that acquisition procedure, fork function-matched optimizer/gate branches, and
evaluate lag and prefix independently. For the controlled model, report decoder
feasibility and the sufficient bound throughout a longer training clock; preserve
the prespecified primary threshold. Finite text tests can show increasing tested
range, but cannot certify arbitrarily long generalization.

## Re-analyze the saved run

From the repository root, after extracting the ZIP into the shown directory:

```bash
fox-experiment diagnose --out validation/user_downscale_v1/text
fox-experiment diagnose --out validation/user_downscale_v1/mechanism --mechanism
```

Text diagnostics are written under `text/diagnostics/`; controlled diagnostics
under `mechanism/diagnostics/`. Raw input CSVs are preserved. CPU reproduction
and acquisition-screen summaries are under `validation/zero_diagnosis_agent/`.
Those local diagnostic checkpoints are excluded from Git and report-only exports.

See [validation status](validation.md) and the updated
[downscale notebook](../notebooks/01_downscale_training.ipynb).
