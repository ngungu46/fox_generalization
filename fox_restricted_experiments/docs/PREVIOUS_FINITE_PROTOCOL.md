> Historical documentation for the earlier four-arm notebooks. Use ../README.md for the current eight-run library notebook.

# Restricted FoX generalization experiments

Train the paper's restricted, two-layer answer-supervised model with SGD and Adam, compare learned versus frozen retrieval forgetting, and plot the probability of the correct answer at different record lags. The standalone notebook contains the implementation and its checks; it requires no repository checkout or uploaded dataset.

For the larger A100 run and the worst-case error `E_t(r_t)` at growing lags,
use [FoX_Restricted_A100_Asymptotics.ipynb](FoX_Restricted_A100_Asymptotics.ipynb)
and read [A100_README.md](A100_README.md). That profile adds uncapped moving
radii, log-error bounds, optimizer diagnostics, and exact checkpoint resumption.
The original notebook below remains the smaller probability-by-lag pilot.

[Measured local pilots and plots](PILOT_RESULTS.md) report the completed `R=2`
and `R=4` experiments, including their limitations and the failed training fit
for learned-gate SGD at `R=4`.

## Run on Colab

1. Upload `FoX_Restricted_Generalization_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime if available. Edit `N`, `R`, `STEPS`, `SEEDS` and `EVAL_LAGS` in the first settings cell.
3. Choose **Runtime → Run all**. The notebook checks the implementation, trains all four arms, displays the report and offers a ZIP containing measurements, plots and source.

The default experiment uses `N=8`, training recall lag `R=2`, width 2,048, three paired seeds and 12,000 total updates per arm, including the three acquisition updates. Set `SMOKE=True` for a short software check. A smoke run is not experimental evidence for the asymptotic claims. Set `USE_DRIVE=True` to persist output in Google Drive; otherwise download the ZIP before the runtime expires. CPU execution is supported, with hardware-dependent runtime.

## Dataset and lag convention

For every ordered distinct-key pair `a != b`, train on:

```text
overwrite O: (a,-1), (a,-1), (b,-1), (a,+1), ?a
recall C_R: (a,+1), (b,-1) repeated R-1 times, ?a
```

Include both global value complements and `(a,+1), ?a` / `(a,-1), ?a` calibration for every key. Thus `K=N(N-1)` counts distinct pair identities. There are `4K+2N` actual serialized sequences, with objective mass 0.2 overwrite, 0.2 recall and 0.6 calibration. The sequences are not weighted uniformly. SGD samples ordered pairs with replacement and evaluates both weighted pair families, with exact calibration. Adam uses the complete finite pair population. Under the output folder's `dataset/`, the runner writes `dataset.jsonl` with the weighted complete population, `recall_at_R.jsonl` with exactly K base recall sequences, and `pairs.jsonl` with pair identities.

The newest record is lag 1. Changing `R` changes the recall template; overwrite remains the original four-record construction. This suite uses only recall range `R`, not a mixture of ranges up to `R`. Its held-out challenge is sequence length, lag and stale-prefix structure on the same vocabulary, rather than unseen vocabulary or unseen training pairs.

## Controlled comparison

| Mode | Binding slope `g` | Retrieval slope `h` | Optimizers |
|---|---|---|---|
| `learned` | learned | learned | SGD, Adam |
| `retrieval_frozen` | learned | fixed at `h0` | SGD, Adam |
| `both_frozen` (optional) | fixed | fixed at `h0` | SGD, Adam |

The main frozen intervention removes the retrieval-logit update while retaining the learned binder. Freezing both gates introduces a different obstruction and is an optional negative control. All modes use matched independent raw Q/K Gaussian draws for each seed and the same initial function at `h0=1`. `h` is a penalty **per record**; the corresponding raw-token retrieval ALiBi slope is `h/2`. The binder's `g` is measured per raw token and determines leakage `rho = sigmoid(D - 2g)`.

The model retains the prescribed parser masks, scalar-factorized content gain, soft binding and scalar binary decoder. It is a restricted architectural experiment, rather than a standard full Transformer. The loss is the ordinary final-answer logistic loss; it does not supervise target attention or target positions. The generalization plots show the correct-answer probability after decoding.

## Initialization and optimization

- Independent Q/K table entries start at Gaussian standard deviation `1e-6`; `q=p=0.5`, `u=v=1.7`, and decoder `w=0`. The learned retrieval logit starts at `softplus^-1(h0)`.
- Three optimizer-specific answer-supervised acquisition updates precede continuation. Adam starts from zero moments and retains the acquisition moment history.
- Adam uses `beta1=0.9`, `beta2=0.999`, bias correction, no weight decay and annealed epsilon `sigma^3 exp(-c t)` outside the square root. There is no gradient clipping or sign-descent substitution.
- Continuation uses an explicit finite polynomial scalar schedule, increasing SGD pair batches, and a strictly positive summable table-rate tail. No long-lag measurement selects a rate or checkpoint.
- Analytic float64 signed-log gradients and signed-log Adam moments retain derivatives and squared gradients that ordinary float64 would underflow. Independent native float64 checks validate the regime where both representations are representable. Signed-log arithmetic still has finite-precision cancellation limits.

The default width meets the frozen theorem's dimension lower bound for `N=8` at failure budget 0.05. This does **not** establish every required rate, initialization, cone, settling or basin constant. The theorem's conservative SGD existence construction is not a practical Colab schedule; these finite experiments approximate its intended dynamics. The actual configuration and diagnostics accompany the results.

## Interpreting the plots

At `R=2`, learned-gate SGD has the paper's asymptotic worst-case cutoff at lag 4, while learned-gate Adam has an expanding range that is linear in its cumulative scalar learning-rate clock under the stated hypotheses. With only retrieval forgetting frozen at `h0 > log(2)`, the answer-supervised extension predicts expanding range for both optimizers; its Adam range is quadratic in that clock. Target attention can retain a nonzero overwrite error floor while correct-answer probability approaches one, so the two metrics must not be conflated.

For `R>2`, the learned-SGD recall-range extension predicts cutoff `R+2` after an additional preparation/basin argument. This suite does not implement that finite preparation construction. The fully proved frozen-retrieval result uses `R=2`. Running `R=4` is an extension experiment, with these qualifications retained in the report.

The final-lag solid curves average the seed-specific minima over evaluated ordered pairs and prefix lengths, with a seed min/max envelope. Dashed curves show mean probabilities over the evaluated examples. A mean can hide the hard pair used by a worst-case theorem. Inspect `lag_probabilities.csv` for each prefix and seed, and use training loss, matching gaps, gate trajectories and curves versus cumulative scalar learning rate to decide whether a run has entered the relevant regime. Equal update counts compare compute; SGD and Adam have different learning-rate clocks. Finite evaluated prefixes are not a certificate over all finite streams, and failure to observe an asymptotic boundary at the chosen budget is inconclusive.

## Local use

Use a Python environment with the packages in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m pytest -q test_core.py test_runner.py
```

The Python API is the same as the notebook:

```python
from core import Config
from runner import run_suite
from report import make_report

cfg = Config(n=8, d=2048, R=2, h0=1.0, steps=12000, device="cpu",
             beta1=0.9, beta2=0.999)
folder = run_suite(
    cfg, "results/my_run", seeds=(0, 1, 2),
    gate_modes=("learned", "retrieval_frozen"),
    optimizers=("sgd", "adam"),
    eval_lags=tuple(range(1, 33)), prefixes=(0, 16, 64), log_every=1000,
)
make_report(folder)
```

`core.py` implements the exact restricted model and optimizer arithmetic; `runner.py` builds the population, runs the suite and records evaluations; `report.py` produces the generalization report. `test_core.py` checks the mathematical implementation, and `test_runner.py` checks population construction and suite behavior. Use a fresh output directory when changing scientific settings.

After editing any implementation module, run `python build_notebook.py` to regenerate the notebook. The generator uses only Python's standard library, embeds the complete source as text, and records its SHA-256 hashes in notebook metadata. The notebook also exports the exact source it executed alongside the results.

## Local theoretical sources

The construction follows the current manuscript and these research notes in the repository root:

- `SGD_RECALL_RANGE_MIXTURE_EXTENSION.md`: exact `C_R` odds, cutoff `R+2`, and the finite preparation caveat.
- `ALIBI_ANSWER_SUPERVISED_EXTENSION.md`: frozen retrieval with a learned binder, answer-supervised convergence, and quadratic Adam range.
- `ALIBI_FROZEN_GATE_THEORY.md`: per-record versus per-token slope conventions, attention floor, `h0 > log(2)` answer threshold, and both-frozen obstruction.

These references document the design; the Colab notebook does not need them at runtime.
