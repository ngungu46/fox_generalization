# Experiment definitions and interpretation

## Shared restricted model and optimization

The unchanged two-head ordinary-answer architecture uses raw Gaussian Q/K
tables, factorized content gain `m=c0*q*p`, binding leakage
`rho=sigmoid(D-2*softplus(u*v))`, retrieval slope `h=softplus(x)`, and a binary
decoder of weight `w`. Parameters and gradients use FP64; gradients and Adam
moments are stored in signed-log form. There is no pointer supervision or
sign-descent replacement. ALiBi fixes h=h0; the binder remains trainable.
The slope per record h corresponds to a raw-token slope h/2.

The default initialization is unchanged: Gaussian scale 1e-6, q=p=0.5,
u=v=1.7, h=1, w=0, zero Adam buffers. Three acquisition updates precede the
practical polynomial continuation. The same deterministic rate constants
are used in both dataset families. SGD's acquisition constants are calibrated
using a zero-table pair-objective probe; **actual random-data parameter updates
use only the random-data loss**. This is an empirical extension, not a proof
that those rates satisfy new distribution-specific acquisition hypotheses.
Matching-gap failure is recorded, and no matching table is supplied or reset.

Adam defaults to beta1=beta2=0.1 and retains all acquisition moment history.
The earlier beta1=0.9,beta2=0.999 control is selectable. Epsilon is
sigma³ exp(-eps_decay*(step-1)), outside the square root; changing the moment
settings changes acquisition as well as continuation. The original theorem's
conservative existence schedule and extra R>2 basin preparation are not
implemented by the practical continuation schedule.

## Finite pair dataset

For each a≠b:

```text
Overwrite: (a,-1),(a,-1),(b,-1),(a,+1),?a
Recall:    (a,+1),(b,-1) repeated R-1 times,?a
```

Include global value complements and two one-record calibration examples per
key. N(N−1) counts pair identities; 4N(N−1)+2N counts the actual serialized
weighted sequences. Objective masses are 0.2 overwrite, 0.2 recall, 0.6
calibration. SGD samples pairs with replacement and evaluates both pair
families, with exact calibration. Adam uses the full population.

## Random structured dataset

The default samples M=N(N−1) complete streams independently once per seed:

1. Query key a is uniform on N keys; target lag L is uniform on 1,…,R.
2. Older-prefix length P is geometric on {0,1,…}, with configurable mean.
3. The P older records have iid uniform keys and iid binary values.
4. Append target key a with an independent uniform binary value y.
5. Append L−1 records with iid uniform keys from the N−1 keys excluding a,
   and independent binary values. The label is y, the latest queried value.

Every legal finite stream has positive probability under this geometric
distribution. A finite sampled dataset does not contain every possible stream.
No truncation or rejection sampling is applied. A memory guard raises on an
oversize padded batch, without silently redrawing it. Optional
`bounded_uniform` draws P uniformly from 0,…,max_prefix instead and has finite
support. Saved manifests record the choice and the exact dataset fingerprint.

Each fixed dataset is reused across the four optimizer/gate arms. Full-batch
Adam computes its empirical objective exactly; SGD's multinomial frequencies
give the same gradient as iid sampling with replacement. Values are not
restricted to the sign pattern of the pair witness. There is no additional
calibration loss. Acquisition and continuation both use this objective.

The optional `online` setting draws new streams each update for both
optimizers. A separate fixed sample is used for diagnostic loss curves and
is labeled accordingly. Online Adam is a different stochastic protocol from
the full-population Adam theorem.

## Error envelopes and the moving lag

E_t(r) is the supremum of 1-P(correct) over every finite stream with latest
target lag ≤r. The plotted lower bound is the infinite stale-prefix witness
limit, maximized over ordered pairs. The conditional upper bound is

```text
A_t(r) = exp(4*m*rho) * [exp(-m*delta_min + (r-1)*h) + exp(-h)] / [1-exp(-h)]
U_t(r) = sigmoid(-w * (1-A_t(r))/(1+A_t(r)))
```

The certificate requires positive gap and forgetting, nonnegative m and w,
valid leakage probability, and raw row norms ≤1. Failed conditions are
reported as unavailable certificates, with trivial upper bound 1 in figures.
Both error and log error are computed directly from logits, avoiding
cancellation in 1-P. The correct-probability interval is [1-U,1-L].

For Adam, save delta_ref=delta_min just after acquisition and predeclare
theta∈(0,1). Let c=theta*delta_ref and r_t=max(1,floor(c*m_t/h_t)). This has a
fixed coefficient rather than one chosen from later test performance. If
delta_ref≤0, the moving probe is unavailable; if the gap later deteriorates,
the current-state upper bound and retention diagnostic expose that change.
The radius has no arbitrary evaluation cap or sequence-allocation cost.

In the prepared restricted learned-SGD regime, E_t(R+1),E_t(R+2)→0 and
E_t(R+3)→1. For frozen retrieval with h0>log2, the answer-supervised extension
predicts expanding recall for both optimizers; its proved setup has R=2.
The sufficient Adam radius is quadratic in its continuation clock S. No
sharp ordinary-answer frozen-SGD time law is assumed here. The random
distribution changes overwrite pressure, so the R+2 cutoff must be tested,
not copied as a theorem for that setting.

## Provenance and finite-run limits

Checkpoints retain parameters, moment buffers, RNG, clock, statuses, and
measurements. Incompatible scientific settings/source changes are rejected.
Validated state is saved atomically; numerical failures restore the previous
commit. Increasing the budget resumes a completed finite run. CPU tests check
bitwise continuation; floating-point CUDA reductions can vary across runs.

The historical kernel under `legacy/` is preserved byte-for-byte so existing
finite-pair checkpoints remain usable. New experiments use separate output
folders. Existing measured CPU pilots remain under `results/` and are not
relabelled as random-data or A100 results. Finite decreasing upper bounds and
increasing radii are evidence to inspect, not an infinite-time proof.

Local theoretical references: `SGD_RECALL_RANGE_MIXTURE_EXTENSION.md` and
`ALIBI_ANSWER_SUPERVISED_EXTENSION.md` in the repository root.
