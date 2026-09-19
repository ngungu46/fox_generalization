# Worst-case generalization error: finite-run bracket report

Define E_t(r) as the supremum of 1 − P_t(correct answer) over every finite record stream whose latest assignment to the queried key is at lag at most r. This E is a model error, not Adam's denominator epsilon.

The unknown E_t(r) is bounded below by an infinite-prefix witness supremum and above by a conditional uniform certificate over arbitrary finite streams. The infinite-prefix witness is a limit of finite streams and therefore supplies a valid lower bound on the supremum. Finite-prefix sample minima and means are reported separately by the ordinary report.

Plots use the saved natural logarithms divided by log(10), so tiny errors remain visible without evaluating 1 − a probability rounded to one. Solid and dotted curves are the upper and lower endpoints for each seed. Their shaded interval brackets the true supremum; it is not a confidence interval or a numerical estimate of E.

Saved error rows: 638. Rows without a valid uniform certificate: 40. Every invalid upper certificate is shown explicitly as the trivial upper bound E ≤ 1 with a cross marker; no uncertified upper curve is silently omitted.

## What the trajectories test

- Fixed-radius probes examine error reduction at an unchanged lag threshold.
- Clock probes choose a radius proportional to S for learned retrieval and to S² for frozen retrieval (with their saved coefficients and integer rounding). These prescribed clocks do not assert that each optimizer supports that growth rate. In particular, frozen-SGD clock probes are stress tests; its ordinary-answer theorem does not give a sharp time law.
- Adaptive probes use the measured content gap and forgetting scale. Their actual integer radii are plotted. A decreasing upper error at a radius that stays bounded does not establish generalization along a radius tending to infinity.
- Certified radii invert the uniform bound at each requested error threshold and are not clipped to the largest finite evaluation lag. Radius zero means no positive radius is certified.
- R = 2 is the proved frozen-gate setting. The theoretical asymptotic conclusions require their sufficient initialization, acquisition, continuation, and basin hypotheses. These practical polynomial schedules are finite experiments, not literal implementations of the existential conservative SGD schedule. Larger R is an extension experiment.
- The frozen intervention holds retrieval h fixed and continues training the binding gate. Uniform frozen-gate success needs h₀ > log(2) and the certificate's other conditions.

## Latest saved brackets

Values are log10(error). A more negative upper endpoint is stronger evidence of a small worst-case error at that saved radius; finite trajectories do not establish a limit.

| Seed | Gate | Optimizer | Probe | Step | S | Radius | log10 lower | log10 upper | Valid uniform bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | adaptive_theta, θ=0.25 | 12000 | 53.93 | 1 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | adaptive_theta, θ=0.5 | 12000 | 53.93 | 3 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | adaptive_theta, θ=0.75 | 12000 | 53.93 | 5 | -2.0162 | -2.0162 | True |
| 0 | learned | adam | clock, c=0.001 | 12000 | 53.93 | 1 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | clock, c=0.003 | 12000 | 53.93 | 1 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | clock, c=0.01 | 12000 | 53.93 | 1 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | clock, c=0.03 | 12000 | 53.93 | 2 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | clock, c=0.1 | 12000 | 53.93 | 6 | -2.0154 | -2.0154 | True |
| 0 | learned | adam | fixed r=1 | 12000 | 53.93 | 1 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | fixed r=128 | 12000 | 53.93 | 128 | -0.0019416 | -0.0019416 | True |
| 0 | learned | adam | fixed r=16 | 12000 | 53.93 | 16 | -0.0019416 | -0.0019416 | True |
| 0 | learned | adam | fixed r=2 | 12000 | 53.93 | 2 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | fixed r=256 | 12000 | 53.93 | 256 | -0.0019416 | -0.0019416 | True |
| 0 | learned | adam | fixed r=3 | 12000 | 53.93 | 3 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | fixed r=32 | 12000 | 53.93 | 32 | -0.0019416 | -0.0019416 | True |
| 0 | learned | adam | fixed r=4 | 12000 | 53.93 | 4 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | fixed r=5 | 12000 | 53.93 | 5 | -2.0162 | -2.0162 | True |
| 0 | learned | adam | fixed r=512 | 12000 | 53.93 | 512 | -0.0019416 | -0.0019416 | True |
| 0 | learned | adam | fixed r=64 | 12000 | 53.93 | 64 | -0.0019416 | -0.0019416 | True |
| 0 | learned | adam | fixed r=8 | 12000 | 53.93 | 8 | -1.8447 | -1.8447 | True |
| 0 | learned | adam | reference_theta, θ=0.25 | 12000 | 53.93 | 1 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | reference_theta, θ=0.5 | 12000 | 53.93 | 3 | -2.0163 | -2.0163 | True |
| 0 | learned | adam | reference_theta, θ=0.75 | 12000 | 53.93 | 5 | -2.0162 | -2.0162 | True |
| 0 | retrieval_frozen | adam | adaptive_theta, θ=0.25 | 12000 | 53.93 | 6 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | adaptive_theta, θ=0.5 | 12000 | 53.93 | 11 | -0.71385 | -0.71385 | True |
| 0 | retrieval_frozen | adam | adaptive_theta, θ=0.75 | 12000 | 53.93 | 17 | -0.70537 | -0.70537 | True |
| 0 | retrieval_frozen | adam | clock, c=0.001 | 12000 | 53.93 | 3 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | clock, c=0.003 | 12000 | 53.93 | 9 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | clock, c=0.01 | 12000 | 53.93 | 30 | -0.00195 | -0.00195 | True |
| 0 | retrieval_frozen | adam | clock, c=0.03 | 12000 | 53.93 | 88 | -0.0019416 | -0.0019416 | True |
| 0 | retrieval_frozen | adam | clock, c=0.1 | 12000 | 53.93 | 291 | -0.0019416 | -0.0019416 | True |
| 0 | retrieval_frozen | adam | fixed r=1 | 12000 | 53.93 | 1 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | fixed r=128 | 12000 | 53.93 | 128 | -0.0019416 | -0.0019416 | True |
| 0 | retrieval_frozen | adam | fixed r=16 | 12000 | 53.93 | 16 | -0.71073 | -0.71073 | True |
| 0 | retrieval_frozen | adam | fixed r=2 | 12000 | 53.93 | 2 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | fixed r=256 | 12000 | 53.93 | 256 | -0.0019416 | -0.0019416 | True |
| 0 | retrieval_frozen | adam | fixed r=3 | 12000 | 53.93 | 3 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | fixed r=32 | 12000 | 53.93 | 32 | -0.0019428 | -0.0019428 | True |
| 0 | retrieval_frozen | adam | fixed r=4 | 12000 | 53.93 | 4 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | fixed r=5 | 12000 | 53.93 | 5 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | fixed r=512 | 12000 | 53.93 | 512 | -0.0019416 | -0.0019416 | True |
| 0 | retrieval_frozen | adam | fixed r=64 | 12000 | 53.93 | 64 | -0.0019416 | -0.0019416 | True |
| 0 | retrieval_frozen | adam | fixed r=8 | 12000 | 53.93 | 8 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | reference_theta, θ=0.25 | 12000 | 53.93 | 6 | -0.71387 | -0.71387 | True |
| 0 | retrieval_frozen | adam | reference_theta, θ=0.5 | 12000 | 53.93 | 11 | -0.71385 | -0.71385 | True |
| 0 | retrieval_frozen | adam | reference_theta, θ=0.75 | 12000 | 53.93 | 17 | -0.70537 | -0.70537 | True |

## Run status

| Seed | Gate | Optimizer | Completed steps | Status | Reason |
| --- | --- | --- | --- | --- | --- |
| 0 | learned | adam | 12000 | finite_budget_complete |  |
| 0 | retrieval_frozen | adam | 12000 | finite_budget_complete |  |

Each branch is plotted through its own last saved checkpoint. Stopped runs can have different budgets. Inspect `runs.csv`, `history.csv`, `asymptotic_errors.csv`, `certified_radii.csv`, and the checkpoint metadata before comparing arms.
