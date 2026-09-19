# Restricted-model empirical report

These results describe the saved finite training runs. They can test consistency with the theory; they do not prove the asymptotic optimizer claims.

Settings: n=8, d=2048, R=2, steps=12000. See `config.json` for initialization, optimizer, and evaluation settings.

- Checkpoint step 0 is the paired Gaussian initialization; step 3 ends optimizer-specific acquisition. Continuation uses each optimizer's own learning-rate sum S.
- The frozen-retrieval comparison freezes only the retrieval forgetting parameter h. The binding gate remains trainable. Any `both_frozen` arm is a separate extra control.
- The SGD continuation schedule is a practical experiment schedule, not the literal conservative theorem schedule.
- Frozen-gate arbitrary-prefix success requires the theorem's conditions, including h₀ > log(2). Positive matching gaps, suitable row bounds, controlled binding contamination, and positive output scale must also hold.
- Learned-gate SGD's asymptotic horizon is 4 in the main R = 2 theorem. The R + 2 reference for R > 2 is conditional on the generalized theorem's hypotheses. At the boundary, output probability must be checked separately from target attention.
- The fully proved frozen-retrieval answer result uses R=2. Frozen R>2 and learned-Adam R>2 runs are empirical extension tests here; the notebook also omits the additional R>2 SGD basin preparation.
- Test prefix lengths: 0, 64. Test lags: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.

## Run completion and latest observations

Plots use the latest saved evaluation separately for each branch. A stopped run may therefore have an earlier checkpoint than a completed run. Seed envelopes are observed ranges, not confidence intervals.

| Seed | Gate | Optimizer | Completed / requested | Status | Evaluation step | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | 12000 / 12000 | finite_budget_complete | 12000 | — |
| 0 | retrieval_frozen | adam | 12000 / 12000 | finite_budget_complete | 12000 | — |

## Generalization at selected lags

For each seed, the observed minimum is over all tested ordered key pairs and prefix lengths. The table gives the mean and worst of those per-seed minima. It does not cover untested prefixes. An analytic certificate, when present, is a separate conditional arbitrary-prefix lower bound; blank or unavailable certificates are not empirical failures.

| Gate | Optimizer | Lag | Seeds | Mean probability | Mean of observed minima | Worst observed minimum | Analytic lower bound (worst seed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| learned | adam | 2 | 1 | 0.993 | 0.9904 | 0.9904 | 0.9904 |
| learned | adam | 4 | 1 | 0.993 | 0.9904 | 0.9904 | 0.9904 |
| learned | adam | 5 | 1 | 0.993 | 0.9904 | 0.9904 | 0.9904 |
| learned | adam | 8 | 1 | 0.9916 | 0.9857 | 0.9857 | 0.9857 |
| retrieval_frozen | adam | 2 | 1 | 0.9011 | 0.8067 | 0.8067 | 0.8067 |
| retrieval_frozen | adam | 4 | 1 | 0.9011 | 0.8067 | 0.8067 | 0.8067 |
| retrieval_frozen | adam | 5 | 1 | 0.9011 | 0.8067 | 0.8067 | 0.8067 |
| retrieval_frozen | adam | 8 | 1 | 0.9011 | 0.8067 | 0.8067 | 0.8067 |

## Final mechanism diagnostics

| Seed | Gate | Optimizer | Step | ln(loss) | δ_min | Max row norm | h | mρ | Content horizon | SGD balance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | 1.2e+04 | -5.398 | 0.007303 | 0.08516 | 2.636 | 0 | 9.208 | 5.733 |
| 0 | retrieval_frozen | adam | 1.2e+04 | -4.783 | 0.007303 | 0.08516 | 1 | 0 | 22.63 | 10.64 |

The content horizon is 1 + m δ_min / h; the SGD balance is m δ_min − (R + 1)h − log(m). These are mechanism diagnostics, not substitutes for the actual answer probabilities. `theory_diagnostics.png` displays target attention separately from correct-answer probability, since the two can behave differently.

## Saved figures

- `generalization_by_lag.png` and `.pdf`: probability versus lag at each branch's latest evaluation.
- `probability_by_training.png`: selected-lag probabilities along each saved training trajectory.
- `theory_diagnostics.png`: log loss, content horizon, binding contamination, and attention versus answer probability.

The CSV files contain all measurements and stopped-run statuses. Unobserved checkpoints are never filled in.
