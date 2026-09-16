# Local checks for notebook 05

These checks support the experiment implementation and its initial settings.
They do not establish the asymptotic theorem or an Adam–SGD separation.

## Software verification

* The complete repository suite passes: **87 tests and 22 subtests**.
* All five notebooks validate and contain at most 30 lines per code cell.
* Notebook 05 has ten code cells. It executed end to end with all twelve smoke
  branches completing, no failures, and four rendered figures.
* Tests cover causal target shifting, absent answer leakage, identical batch
  streams, matched initial gate functions, gradient decomposition, finite-grid
  qualification, censored ranges, short-only tuning, and exact checkpoint resume
  after training/evaluation interruption.

Smoke figures have explicit software-only titles. They contain two updates per
branch and should not be used to interpret optimizer generalization.

## Short-only development calibration

The calibration retained 18 attempted branches across distinct generator-v1 and
generator-v2 ledgers. The earlier v1 results are not pooled with final v2 results.
All runs use one CPU thread and seed zero; all model/settings selection used
short cases only. The long grid was not accessed during calibration.

For the final v2 generator, eight keys, factorized binding, initial retrieval
decay 0.5, and Adam with LR .003 and moments (.1,.1):

| Objective | Updates | Worst short joint-edit panel | Mean joint-edit accuracy |
|---|---:|---:|---:|
| Latest-answer only | 1,000 | 1.000 | 1.000 |
| Latest-answer only | 2,000 | 1.000 | 1.000 |
| Latest-answer only | 4,000 | 1.000 | 1.000 |
| All-token NTP | 2,000 | 0.000 | 0.292 |
| All-token NTP | 4,000 | 0.000 | 0.427 |

Each checkpoint evaluates 96 base histories and 363 applicable edited examples.
A zero worst-panel score does **not** mean every answer is wrong. Joint-edit
success requires every applicable edit of a base history to be correct.

On the same deterministic training batch before update 4,000, the all-token
branch gave answer tokens coefficient 0.1135. Answer/non-answer gradient cosine
was -0.974 for the binding gate and -0.530 for the retrieval gate. The weighted
non-answer retrieval gradient norm was approximately 8.32 times the weighted
answer gradient norm. These are measured signs of objective interference in
this run; they do not prove that all-token training necessarily fails.

The previous retrieval initialization 0.1 could learn zero-prefix cases while
failing older-prefix cases. The notebook therefore starts at 0.5 and exposes
`weak_retrieval_init` to repeat 0.1. This choice used development short cases;
confirm it with fresh seeds and more histories before drawing broad conclusions.

Tested SGD learning rates .03, .1 and .3 remained short-unqualified in the
stronger-retrieval calibration at 4,000 updates. No SGD generalization boundary
can be inferred from those failures. Use the short-only tuning facility and
retain the failure records; a matched-short-fit optimizer comparison remains
an open experimental step.

Full local calibration evidence is preserved in the ignored directory
`validation/loss_study_calibration/`, including the consolidated
`calibration_summary.json`, raw predictions, gradients, configurations, source
hashes, and the calibration script. The notebook's own reports and checkpoints
are saved independently in its chosen output directory.

## Complete default finite pilot

After freezing the implementation and defaults, the complete Colab configuration
was run locally on one CPU thread: **12 branches, 2,000 updates each**, completed
in 243 seconds with no runtime failures. This tests the delivered training path;
it is not a GPU throughput estimate. All 167,076 evaluation rows are retained.

Final outcomes, using the minimum joint-edit accuracy across short panels:

| Binding gate / optimizer | Answer-only short accuracy | Answer-only tested radius | All-token short accuracy | All-token tested radius |
|---|---:|---:|---:|---:|
| Factorized / SGD | .1875 | Unqualified | .0000 | Unqualified |
| Factorized / fixed Adam | 1.0000 | 4; fails at 8 | .0000 | Unqualified |
| Factorized / annealed Adam | .6875 | Unqualified | .0000 | Unqualified |
| Direct / SGD | .2500 | Unqualified | .0000 | Unqualified |
| Direct / fixed Adam | 1.0000 | 8; fails at 16 | .0000 | Unqualified |
| Direct / annealed Adam | .9375 | 4; fails at 8 | .9375 | 2; fails at 4 |

Ranges require success on all sampled older-prefix panels and applicable edits,
on the declared sparse lag grid. Short training covers at most four older
records; long evaluation includes sixteen. Thus a qualified branch can have a
tested radius below the training lag when the larger older prefix causes failure.

These observations **do not establish the predicted optimizer separation**.
SGD has not acquired the short rule. Direct fixed Adam outperformed factorized
fixed Adam on this finite grid. Direct annealed Adam did acquire under all-token
training, so all-token failure is not universal. Factorized annealed Adam passed
short qualification at some intermediate checkpoints and later regressed; final
and intermediate results are both retained.

The next comparisons should use the notebook's condition controls and short-only
tuning, with fresh seeds for confirmation. No default was changed after viewing
this pilot's long-context results. Artifacts are in
`validation/loss_study_cpu_pilot_v1/`, and a report-only ZIP sits alongside it.
