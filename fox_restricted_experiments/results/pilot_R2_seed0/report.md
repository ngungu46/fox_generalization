# Restricted-model empirical report

These results describe the saved finite training runs. They can test consistency with the theory; they do not prove the asymptotic optimizer claims.

Settings: n=8, d=2048, R=2, steps=12000. See `config.json` for initialization, optimizer, and evaluation settings.

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
| learned | adam | 2 | 1 | 0.9764 | 0.9713 | 0.9713 | 0.9712 |
| learned | adam | 4 | 1 | 0.5876 | 0.3909 | 0.3909 | 0.3906 |
| learned | adam | 5 | 1 | 0.03119 | 0.02403 | 0.02403 | 0.02403 |
| learned | adam | 8 | 1 | 0.01379 | 0.01379 | 0.01379 | 0.01379 |
| learned | sgd | 2 | 1 | 0.9652 | 0.9357 | 0.9357 | 0.8741 |
| learned | sgd | 4 | 1 | 0.04363 | 0.01544 | 0.01544 | 0.01103 |
| learned | sgd | 5 | 1 | 0.001677 | 0.001088 | 0.001088 | 0.000956 |
| learned | sgd | 8 | 1 | 0.0002629 | 0.0002603 | 0.0002603 | 0.0002595 |
| retrieval_frozen | adam | 2 | 1 | 0.8337 | 0.7565 | 0.7565 | 0.7551 |
| retrieval_frozen | adam | 4 | 1 | 0.8209 | 0.7322 | 0.7322 | 0.7308 |
| retrieval_frozen | adam | 5 | 1 | 0.7944 | 0.6833 | 0.6833 | 0.6819 |
| retrieval_frozen | adam | 8 | 1 | 0.1897 | 0.1039 | 0.1039 | 0.1038 |
| retrieval_frozen | sgd | 2 | 1 | 0.8936 | 0.8196 | 0.8196 | 0.6879 |
| retrieval_frozen | sgd | 4 | 1 | 0.2618 | 0.07274 | 0.07274 | 0.04784 |
| retrieval_frozen | sgd | 5 | 1 | 0.01378 | 0.005519 | 0.005519 | 0.004294 |
| retrieval_frozen | sgd | 8 | 1 | 0.0002803 | 0.0002618 | 0.0002618 | 0.0002572 |

## Final mechanism diagnostics

| Seed | Gate | Optimizer | Step | ln(loss) | δ_min | Max row norm | h | mρ | Content horizon | SGD balance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | 1.2e+04 | -4.256 | 0.00384 | 0.06211 | 2.505 | 1.661e-09 | 3.98 | -7.622 |
| 0 | learned | sgd | 1.2e+04 | -7.345 | 0.007279 | 0.08464 | 1.312 | 0.02398 | 3.483 | -6.782 |
| 0 | retrieval_frozen | adam | 1.2e+04 | -3.896 | 0.00384 | 0.06211 | 1 | 1.225e-10 | 7.586 | -3.861 |
| 0 | retrieval_frozen | sgd | 1.2e+04 | -6.932 | 0.007279 | 0.08464 | 1 | 0.0206 | 4.163 | -5.911 |

The content horizon is 1 + m δ_min / h; the SGD balance is m δ_min − (R + 1)h − log(m). These are mechanism diagnostics, not substitutes for the actual answer probabilities. `theory_diagnostics.png` displays target attention separately from correct-answer probability, since the two can behave differently.

## Saved figures

- `generalization_by_lag.png` and `.pdf`: probability versus lag at each branch's latest evaluation.
- `probability_by_training.png`: selected-lag probabilities along each saved training trajectory.
- `theory_diagnostics.png`: log loss, content horizon, binding contamination, and attention versus answer probability.

The CSV files contain all measurements and stopped-run statuses. Unobserved checkpoints are never filled in.
