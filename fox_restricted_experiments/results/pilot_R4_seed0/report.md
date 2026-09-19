# Restricted-model empirical report

These results describe the saved finite training runs. They can test consistency with the theory; they do not prove the asymptotic optimizer claims.

Settings: n=8, d=2048, R=4, steps=12000. See `config.json` for initialization, optimizer, and evaluation settings.

- Checkpoint step 0 is the paired Gaussian initialization; step 3 ends optimizer-specific acquisition. Continuation uses each optimizer's own learning-rate sum S.
- The frozen-retrieval comparison freezes only the retrieval forgetting parameter h. The binding gate remains trainable. Any `both_frozen` arm is a separate extra control.
- The SGD continuation schedule is a practical experiment schedule, not the literal conservative theorem schedule.
- Frozen-gate arbitrary-prefix success requires the theorem's conditions, including h₀ > log(2). Positive matching gaps, suitable row bounds, controlled binding contamination, and positive output scale must also hold.
- Learned-gate SGD's asymptotic horizon is 4 in the main R = 2 theorem. The R + 2 reference for R > 2 is conditional on the generalized theorem's hypotheses. At the boundary, output probability must be checked separately from target attention.
- The fully proved frozen-retrieval answer result uses R=2. Frozen R>2 and learned-Adam R>2 runs are empirical extension tests here; the notebook also omits the additional R>2 SGD basin preparation.
- Test prefix lengths: 0, 16, 64. Test lags: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24.

## Run completion and latest observations

Plots use the latest saved evaluation separately for each branch. A stopped run may therefore have an earlier checkpoint than a completed run. Seed envelopes are observed ranges, not confidence intervals.

| Seed | Gate | Optimizer | Completed / requested | Status | Evaluation step | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | 12000 / 12000 | finite_budget_complete | 12000 | — |
| 0 | learned | sgd | 12000 / 12000 | finite_budget_complete | 12000 | — |
| 0 | retrieval_frozen | adam | 12000 / 12000 | finite_budget_complete | 12000 | — |
| 0 | retrieval_frozen | sgd | 12000 / 12000 | finite_budget_complete | 12000 | — |

## Generalization at selected lags

For each seed, the observed minimum is over all tested ordered key pairs and prefix lengths. The table gives the mean and worst of those per-seed minima. It does not cover untested prefixes. An analytic certificate, when present, is a separate conditional arbitrary-prefix lower bound; blank or unavailable certificates are not empirical failures.

| Gate | Optimizer | Lag | Seeds | Mean probability | Mean of observed minima | Worst observed minimum | Analytic lower bound (worst seed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned | adam | 4 | 1 | 0.952 | 0.9329 | 0.9329 | 0.9329 |
| learned | adam | 6 | 1 | 0.8515 | 0.7234 | 0.7234 | 0.7234 |
| learned | adam | 7 | 1 | 0.3155 | 0.1444 | 0.1444 | 0.1444 |
| learned | adam | 12 | 1 | 0.01238 | 0.01237 | 0.01237 | 0.01237 |
| learned | sgd | 4 | 1 | 0.1999 | 0.1999 | 0.1999 | 0.1999 |
| learned | sgd | 6 | 1 | 0.1999 | 0.1999 | 0.1999 | 0.1999 |
| learned | sgd | 7 | 1 | 0.1999 | 0.1999 | 0.1999 | 0.1999 |
| learned | sgd | 12 | 1 | 0.1999 | 0.1999 | 0.1999 | 0.1999 |
| retrieval_frozen | adam | 4 | 1 | 0.8363 | 0.7587 | 0.7587 | 0.7584 |
| retrieval_frozen | adam | 6 | 1 | 0.8184 | 0.7222 | 0.7222 | 0.7219 |
| retrieval_frozen | adam | 7 | 1 | 0.7812 | 0.6488 | 0.6488 | 0.6485 |
| retrieval_frozen | adam | 12 | 1 | 0.01892 | 0.01589 | 0.01589 | 0.01589 |
| retrieval_frozen | sgd | 4 | 1 | 0.8838 | 0.794 | 0.794 | 0.6833 |
| retrieval_frozen | sgd | 6 | 1 | 0.5414 | 0.1952 | 0.1952 | 0.1232 |
| retrieval_frozen | sgd | 7 | 1 | 0.1138 | 0.0229 | 0.0229 | 0.01531 |
| retrieval_frozen | sgd | 12 | 1 | 0.0005743 | 0.0005598 | 0.0005598 | 0.0005566 |

## Final mechanism diagnostics

| Seed | Gate | Optimizer | Step | ln(loss) | δ_min | Max row norm | h | mρ | Content horizon | SGD balance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | 1.2e+04 | -4.289 | 0.00384 | 0.06211 | 1.66 | 3.726e-10 | 6.671 | -6.691 |
| 0 | learned | sgd | 1.2e+04 | -0.6901 | 0.007279 | 0.08463 | 4.588 | 0.002637 | 1.002 | -22.95 |
| 0 | retrieval_frozen | adam | 1.2e+04 | -3.983 | 0.00384 | 0.06211 | 1 | 8.32e-11 | 9.197 | -4.469 |
| 0 | retrieval_frozen | sgd | 1.2e+04 | -6.337 | 0.007279 | 0.08463 | 1 | 0.03717 | 6.578 | -6.064 |

The content horizon is 1 + m δ_min / h; the SGD balance is m δ_min − (R + 1)h − log(m). These are mechanism diagnostics, not substitutes for the actual answer probabilities. `theory_diagnostics.png` displays target attention separately from correct-answer probability, since the two can behave differently.

## Saved figures

- `generalization_by_lag.png` and `.pdf`: probability versus lag at each branch's latest evaluation.
- `probability_by_training.png`: selected-lag probabilities along each saved training trajectory.
- `theory_diagnostics.png`: log loss, content horizon, binding contamination, and attention versus answer probability.

The CSV files contain all measurements and stopped-run statuses. Unobserved checkpoints are never filled in.
