# Worst-case generalization error: finite-run bracket report

Define E_t(r) as the supremum of 1 − P_t(correct answer) over every finite record stream whose latest assignment to the queried key is at lag at most r. This E is a model error, not Adam's denominator epsilon.

The unknown E_t(r) is bounded below by an infinite-prefix witness supremum and above by a conditional uniform certificate over arbitrary finite streams. The infinite-prefix witness is a limit of finite streams and therefore supplies a valid lower bound on the supremum. Finite-prefix sample minima and means are reported separately by the ordinary report.

Plots use the saved natural logarithms divided by log(10), so tiny errors remain visible without evaluating 1 − a probability rounded to one. Solid and dotted curves are the upper and lower endpoints for each seed. Their shaded interval brackets the true supremum; it is not a confidence interval or a numerical estimate of E.

Saved error rows: 1276. Rows without a valid uniform certificate: 80. Every invalid upper certificate is shown explicitly as the trivial upper bound E ≤ 1 with a cross marker; no uncertified upper curve is silently omitted.

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
| 0 | learned | adam | adaptive_theta, θ=0.25 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | adaptive_theta, θ=0.5 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | adaptive_theta, θ=0.75 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | clock, c=0.001 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | clock, c=0.003 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | clock, c=0.01 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | clock, c=0.03 | 12000 | 53.93 | 2 | -1.5424 | -1.5406 | True |
| 0 | learned | adam | clock, c=0.1 | 12000 | 53.93 | 6 | -0.0063329 | -0.0063329 | True |
| 0 | learned | adam | fixed r=1 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | fixed r=128 | 12000 | 53.93 | 128 | -0.0060291 | -0.0060291 | True |
| 0 | learned | adam | fixed r=16 | 12000 | 53.93 | 16 | -0.0060291 | -0.0060291 | True |
| 0 | learned | adam | fixed r=2 | 12000 | 53.93 | 2 | -1.5424 | -1.5406 | True |
| 0 | learned | adam | fixed r=256 | 12000 | 53.93 | 256 | -0.0060291 | -0.0060291 | True |
| 0 | learned | adam | fixed r=3 | 12000 | 53.93 | 3 | -1.3062 | -1.3046 | True |
| 0 | learned | adam | fixed r=32 | 12000 | 53.93 | 32 | -0.0060291 | -0.0060291 | True |
| 0 | learned | adam | fixed r=4 | 12000 | 53.93 | 4 | -0.21528 | -0.2151 | True |
| 0 | learned | adam | fixed r=5 | 12000 | 53.93 | 5 | -0.010563 | -0.010563 | True |
| 0 | learned | adam | fixed r=512 | 12000 | 53.93 | 512 | -0.0060291 | -0.0060291 | True |
| 0 | learned | adam | fixed r=64 | 12000 | 53.93 | 64 | -0.0060291 | -0.0060291 | True |
| 0 | learned | adam | fixed r=8 | 12000 | 53.93 | 8 | -0.0060311 | -0.0060311 | True |
| 0 | learned | adam | reference_theta, θ=0.25 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | reference_theta, θ=0.5 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | adam | reference_theta, θ=0.75 | 12000 | 53.93 | 1 | -1.5636 | -1.5617 | True |
| 0 | learned | sgd | adaptive_theta, θ=0.25 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | learned | sgd | adaptive_theta, θ=0.5 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | learned | sgd | adaptive_theta, θ=0.75 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | learned | sgd | clock, c=0.001 | 12000 | 17977 | 18 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | clock, c=0.003 | 12000 | 17977 | 54 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | clock, c=0.01 | 12000 | 17977 | 180 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | clock, c=0.03 | 12000 | 17977 | 540 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | clock, c=0.1 | 12000 | 17977 | 1798 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=1 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | learned | sgd | fixed r=128 | 12000 | 17977 | 128 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=16 | 12000 | 17977 | 16 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=2 | 12000 | 17977 | 2 | -1.1918 | -0.89993 | True |
| 0 | learned | sgd | fixed r=256 | 12000 | 17977 | 256 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=3 | 12000 | 17977 | 3 | -0.26558 | -0.16526 | True |
| 0 | learned | sgd | fixed r=32 | 12000 | 17977 | 32 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=4 | 12000 | 17977 | 4 | -0.006759 | -0.0048162 | True |
| 0 | learned | sgd | fixed r=5 | 12000 | 17977 | 5 | -0.00047268 | -0.00041539 | True |
| 0 | learned | sgd | fixed r=512 | 12000 | 17977 | 512 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=64 | 12000 | 17977 | 64 | -0.00010947 | -0.00010947 | True |
| 0 | learned | sgd | fixed r=8 | 12000 | 17977 | 8 | -0.00011306 | -0.00011273 | True |
| 0 | learned | sgd | reference_theta, θ=0.25 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | learned | sgd | reference_theta, θ=0.5 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | learned | sgd | reference_theta, θ=0.75 | 12000 | 17977 | 1 | -1.6695 | -1.3389 | True |
| 0 | retrieval_frozen | adam | adaptive_theta, θ=0.25 | 12000 | 53.93 | 2 | -0.61352 | -0.61105 | True |
| 0 | retrieval_frozen | adam | adaptive_theta, θ=0.5 | 12000 | 53.93 | 4 | -0.57222 | -0.56994 | True |
| 0 | retrieval_frozen | adam | adaptive_theta, θ=0.75 | 12000 | 53.93 | 5 | -0.49935 | -0.4974 | True |
| 0 | retrieval_frozen | adam | clock, c=0.001 | 12000 | 53.93 | 3 | -0.60209 | -0.59967 | True |
| 0 | retrieval_frozen | adam | clock, c=0.003 | 12000 | 53.93 | 9 | -0.016266 | -0.016262 | True |
| 0 | retrieval_frozen | adam | clock, c=0.01 | 12000 | 53.93 | 30 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | clock, c=0.03 | 12000 | 53.93 | 88 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | clock, c=0.1 | 12000 | 53.93 | 291 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | fixed r=1 | 12000 | 53.93 | 1 | -0.61779 | -0.6153 | True |
| 0 | retrieval_frozen | adam | fixed r=128 | 12000 | 53.93 | 128 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | fixed r=16 | 12000 | 53.93 | 16 | -0.0056357 | -0.0056357 | True |
| 0 | retrieval_frozen | adam | fixed r=2 | 12000 | 53.93 | 2 | -0.61352 | -0.61105 | True |
| 0 | retrieval_frozen | adam | fixed r=256 | 12000 | 53.93 | 256 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | fixed r=3 | 12000 | 53.93 | 3 | -0.60209 | -0.59967 | True |
| 0 | retrieval_frozen | adam | fixed r=32 | 12000 | 53.93 | 32 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | fixed r=4 | 12000 | 53.93 | 4 | -0.57222 | -0.56994 | True |
| 0 | retrieval_frozen | adam | fixed r=5 | 12000 | 53.93 | 5 | -0.49935 | -0.4974 | True |
| 0 | retrieval_frozen | adam | fixed r=512 | 12000 | 53.93 | 512 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | fixed r=64 | 12000 | 53.93 | 64 | -0.0056289 | -0.0056289 | True |
| 0 | retrieval_frozen | adam | fixed r=8 | 12000 | 53.93 | 8 | -0.047636 | -0.047582 | True |
| 0 | retrieval_frozen | adam | reference_theta, θ=0.25 | 12000 | 53.93 | 2 | -0.61352 | -0.61105 | True |
| 0 | retrieval_frozen | adam | reference_theta, θ=0.5 | 12000 | 53.93 | 4 | -0.57222 | -0.56994 | True |
| 0 | retrieval_frozen | adam | reference_theta, θ=0.75 | 12000 | 53.93 | 5 | -0.49935 | -0.4974 | True |
| 0 | retrieval_frozen | sgd | adaptive_theta, θ=0.25 | 12000 | 17977 | 1 | -1.0164 | -0.72778 | True |
| 0 | retrieval_frozen | sgd | adaptive_theta, θ=0.5 | 12000 | 17977 | 2 | -0.74368 | -0.50568 | True |
| 0 | retrieval_frozen | sgd | adaptive_theta, θ=0.75 | 12000 | 17977 | 3 | -0.29253 | -0.18108 | True |
| 0 | retrieval_frozen | sgd | clock, c=0.001 | 12000 | 17977 | 3.2316e+05 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | clock, c=0.003 | 12000 | 17977 | 9.6947e+05 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | clock, c=0.01 | 12000 | 17977 | 3.2316e+06 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | clock, c=0.03 | 12000 | 17977 | 9.6947e+06 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | clock, c=0.1 | 12000 | 17977 | 3.2316e+07 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=1 | 12000 | 17977 | 1 | -1.0164 | -0.72778 | True |
| 0 | retrieval_frozen | sgd | fixed r=128 | 12000 | 17977 | 128 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=16 | 12000 | 17977 | 16 | -9.0698e-05 | -9.0697e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=2 | 12000 | 17977 | 2 | -0.74368 | -0.50568 | True |
| 0 | retrieval_frozen | sgd | fixed r=256 | 12000 | 17977 | 256 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=3 | 12000 | 17977 | 3 | -0.29253 | -0.18108 | True |
| 0 | retrieval_frozen | sgd | fixed r=32 | 12000 | 17977 | 32 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=4 | 12000 | 17977 | 4 | -0.032798 | -0.021292 | True |
| 0 | retrieval_frozen | sgd | fixed r=5 | 12000 | 17977 | 5 | -0.0024033 | -0.0018687 | True |
| 0 | retrieval_frozen | sgd | fixed r=512 | 12000 | 17977 | 512 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=64 | 12000 | 17977 | 64 | -9.0691e-05 | -9.0691e-05 | True |
| 0 | retrieval_frozen | sgd | fixed r=8 | 12000 | 17977 | 8 | -0.00011373 | -0.00011171 | True |
| 0 | retrieval_frozen | sgd | reference_theta, θ=0.25 | 12000 | 17977 | 1 | -1.0164 | -0.72778 | True |
| 0 | retrieval_frozen | sgd | reference_theta, θ=0.5 | 12000 | 17977 | 2 | -0.74368 | -0.50568 | True |
| 0 | retrieval_frozen | sgd | reference_theta, θ=0.75 | 12000 | 17977 | 3 | -0.29253 | -0.18108 | True |

## Run status

| Seed | Gate | Optimizer | Completed steps | Status | Reason |
| --- | --- | --- | --- | --- | --- |
| 0 | learned | sgd | 12000 | finite_budget_complete |  |
| 0 | learned | adam | 12000 | finite_budget_complete |  |
| 0 | retrieval_frozen | sgd | 12000 | finite_budget_complete |  |
| 0 | retrieval_frozen | adam | 12000 | finite_budget_complete |  |

Each branch is plotted through its own last saved checkpoint. Stopped runs can have different budgets. Inspect `runs.csv`, `history.csv`, `asymptotic_errors.csv`, `certified_radii.csv`, and the checkpoint metadata before comparing arms.
