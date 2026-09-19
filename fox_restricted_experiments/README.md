# FoX restricted generalization experiments

Start with **[the readable A100 Colab notebook](notebooks/FoX_Restricted_A100_Colab.ipynb)**.
It imports `fox_restricted`; it contains no embedded model source. The notebook
has eight separate training-and-plotting sections, so each experiment can be
run or resumed independently.

## Run from GitHub in Colab

1. Push this folder’s library, tests, notebook, README, and `pyproject.toml` to your repository.
   Historical `results/` are optional and are not needed to run the experiment.
2. Open or upload `notebooks/FoX_Restricted_A100_Colab.ipynb` in Colab.
3. The first code cell is prefilled for `https://github.com/ngungu46/fox_generalization.git`,
   branch `main`, and `PACKAGE_SUBDIR="fox_restricted_experiments"`, matching the
   local `fox_experiments` Git repository. Push the new folder before running
   the GitHub install. Prefer a commit SHA in `REPO_REF` for reproducibility.
   If this folder is published as a separate repository, set `PACKAGE_SUBDIR="."`.
4. Select an A100 runtime, review the settings, and run all cells or individual
   experiment sections. The setup clones the repository and installs the library.

Defaults are **N=128, d=4096, R=2, 200,000 updates per arm, three seeds**.
There are 24 seed/arm runs. A measured benchmark runs before training; completion
time depends on the actual hardware and data. Drive checkpoints allow multiple
Colab sessions. An A100 run has not been performed in local development.

## The eight runs

| Name | Dataset | Forgetting | Optimizer | Main plot |
|---|---|---|---|---|
| `pairs_learned_sgd` | Paper pair construction at R | learned | SGD | Eₜ(R+1), Eₜ(R+2), Eₜ(R+3) |
| `pairs_learned_adam` | Paper pair construction at R | learned | Adam | Eₜ(rₜ), actual rₜ |
| `pairs_alibi_sgd` | Paper pair construction at R | fixed h₀ | SGD | three fixed-lag errors |
| `pairs_alibi_adam` | Paper pair construction at R | fixed h₀ | Adam | moving-lag error and radius |
| `random_learned_sgd` | Random structured streams, lag ≤ R | learned | SGD | three fixed-lag errors |
| `random_learned_adam` | Random structured streams, lag ≤ R | learned | Adam | moving-lag error and radius |
| `random_alibi_sgd` | Random structured streams, lag ≤ R | fixed h₀ | SGD | three fixed-lag errors |
| `random_alibi_adam` | Random structured streams, lag ≤ R | fixed h₀ | Adam | moving-lag error and radius |

ALiBi here freezes **retrieval forgetting** and retains the learned local
binding gate, matching the frozen-retrieval theory. Initial model draws and
random datasets are paired across arms by seed. The previous initialization,
annealed epsilon, optimizer buffers, and practical rate schedules are retained.
The explicit Adam default is `(beta1,beta2)=(0.1,0.1)`; `(0.9,0.999)` remains a
configurable control.

The random dataset is sampled **once per seed**, with N(N−1) complete streams
by default. SGD draws minibatches from it; Adam uses the full sampled empirical
objective. Prefix lengths are geometric, target lag is uniform in 1…R, and keys
and binary values are random subject to the target being the latest query-key
record. `random_sampling="online"` instead draws fresh minibatches for both
optimizers. The notebook explains the distribution explicitly: no uniform
distribution exists over all finite sequence lengths.

## Reading the plots

`E_t(r)` is the **worst-case answer error over every finite stream whose target
lag is at most r**. We bracket it using a witness lower bound and a conditional
uniform upper bound. It is not replaced by a sampled mean. Each fixed-lag
figure includes error on linear and log scales, plus the corresponding
worst-case correct-probability bracket `1-E`.

For Adam, the notebook fixes `c = theta * acquired_minimum_gap` after the three
acquisition updates, then evaluates

```text
r_t = max(1, floor(c * m_t / h_t))
```

Thus the requested radius is asymptotically linear in `m_t/h_t` if this ratio
diverges. There is no evaluation-lag cap. The actual radius is shown beside its
error; a flat radius is not evidence for generalization along rₜ→∞.

Under the restricted learned-SGD theorem’s hypotheses, the errors at R+1,
R+2, R+3 tend to **0, 0, 1**. Frozen retrieval instead predicts vanishing error
at every fixed lag under its hypotheses. The random-data runs are extension
experiments; these conclusions are not automatically theorems for that new
distribution. See [the protocol and limitations](docs/PROTOCOL.md).

## Read or edit the implementation

```text
notebooks/                           readable experiment notebook
library/fox_restricted/
    __init__.py                      public notebook imports
    study.py                         eight experiment definitions
    runtime.py                       hardware, storage, benchmark
    data/random_streams.py           random structured datasets
    training/experiment.py           train one named experiment
    training/random_training.py      random-data training and checkpoints
    training/random_backend.py       exact arbitrary-stream derivatives
    evaluation/probes.py             fixed and m/h lag probes
    plotting/convergence.py          focused error/probability figures
    legacy/                          preserved, audited finite-pair kernel
tests/                               numerical and resume checks
scripts/build_colab.py               build the readable notebook
results/                             earlier measured CPU pilots
```

The small old filenames at this folder’s root are compatibility shims. The
preserved kernel keeps old checkpoint source hashes valid. Earlier notebooks
that embed source are retained under `notebooks/archive/`; the old root A100
notebook paths now open the new readable notebook. [Full source map](PACKAGE_LAYOUT.md).

## Local use

From the `fox_generalization` repository root (the local `fox_experiments` folder):

```bash
python -m pip install -e './fox_restricted_experiments[test]'
python -m pytest -q fox_restricted_experiments/tests
```

```python
from fox_restricted import a100_config, make_experiments, train, plot_fixed_lags

config = a100_config(R=2)
experiments = make_experiments(config)
run = train(experiments["pairs_learned_sgd"], "results/my_study")
figure = plot_fixed_lags(run, probability=True, save=True)
```

Keep the same settings and source to resume; increase `steps` or `max_seconds`
to extend the budget. Use a new output folder when changing scientific
settings. Saved state includes optimizer buffers and RNG state. Exact
uninterrupted-versus-resumed trajectories are tested on CPU; CUDA reduction
order can affect bitwise reproducibility.
