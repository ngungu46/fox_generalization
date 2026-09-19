"""Build a readable notebook that imports the installed fox_restricted library."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []

    def md(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=source.strip() + "\n"))

    def code(source):
        cells.append(dict(cell_type="code", metadata={}, source=source.strip() + "\n",
                          execution_count=None, outputs=[]))

    md(r"""
# FoX generalization: eight A100 experiments

This notebook contains **settings and library calls only**. Read or edit the Python implementation
under `library/fox_restricted/` in the repository. Run each experiment section independently, or
use **Runtime → Run all**. Select an **A100 GPU** runtime in Colab before a full run.

| Data | Retrieval forgetting | SGD plot | Adam plot |
|---|---|---|---|
| Paper's ordered-pair dataset at lag R | Learned | Eₜ(R+1), Eₜ(R+2), Eₜ(R+3) | Eₜ(rₜ), actual rₜ |
| Paper's ordered-pair dataset at lag R | ALiBi (fixed h₀) | Same three fixed lags | Eₜ(rₜ), actual rₜ |
| Random structured streams, target lag ≤ R | Learned | Same three fixed lags | Eₜ(rₜ), actual rₜ |
| Random structured streams, target lag ≤ R | ALiBi (fixed h₀) | Same three fixed lags | Eₜ(rₜ), actual rₜ |

All arms keep the same restricted two-head architecture and ordinary final-answer loss. ALiBi
means a **frozen retrieval slope**; the local binding gate remains trainable, as in the theory.
Defaults: **N=128, d=4096, R=2, 200,000 updates per arm, three paired seeds**.
The 24 seed/arm runs can span several Colab sessions. Checkpoints persist on Drive.
Actual GPU availability and runtime limits are described in the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
""")
    md("""
## 1. Get the library from GitHub

`REPO_URL` is prefilled from the Git remote of `fox_experiments`.
Push the new `fox_restricted_experiments/` folder before using the GitHub installation.
Set `REPO_REF` to a branch, tag, or preferably an exact commit
SHA for reproducibility. `PACKAGE_SUBDIR` points to this folder inside your repository; use `"."`
if the restricted experiment folder is itself the repository root.

Upload this notebook once, or open its GitHub file directly in Colab after you push it. This cell
clones and installs the library; no model source is copied into the notebook. For local work,
set `LOCAL_PACKAGE` to the package folder and leave `REPO_URL` blank. Private repositories can
be cloned through your usual authenticated Git setup; do not put access tokens in this notebook.
""")
    code(r'''
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/ngungu46/fox_generalization.git"
REPO_REF = "main"             # Prefer a commit SHA when starting a final experiment.
PACKAGE_SUBDIR = "fox_restricted_experiments"
LOCAL_PACKAGE = ""            # Optional local checkout; otherwise use GitHub above.
CHECKOUT = Path("/content/fox-source") if Path("/content").exists() else Path.cwd() / "fox-source"

if LOCAL_PACKAGE:
    PACKAGE = Path(LOCAL_PACKAGE).expanduser().resolve()
    SOURCE_REVISION = "local checkout"
else:
    if not REPO_URL:
        raise ValueError("Fill REPO_URL and REPO_REF, or set LOCAL_PACKAGE for local use.")
    if not CHECKOUT.exists():
        subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(CHECKOUT)], check=True)
    origin = subprocess.check_output(["git", "-C", str(CHECKOUT), "remote", "get-url", "origin"], text=True).strip()
    if origin != REPO_URL:
        raise ValueError("CHECKOUT belongs to another repository. Choose a different CHECKOUT path.")
    changed = subprocess.check_output(["git", "-C", str(CHECKOUT), "diff", "--name-only"], text=True).strip()
    if changed:
        raise RuntimeError("The checkout has local changes; retain them and choose a new CHECKOUT path.")
    subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
    subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "--detach", "FETCH_HEAD"], check=True)
    SOURCE_REVISION = subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip()
    PACKAGE = CHECKOUT / PACKAGE_SUBDIR

if not (PACKAGE / "pyproject.toml").is_file():
    raise FileNotFoundError(f"No installable experiment library at {PACKAGE}; check PACKAGE_SUBDIR.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "-e", str(PACKAGE) + "[test]"], check=True)
# Make a just-installed editable package available without restarting this kernel.
import site
for directory in site.getsitepackages():
    site.addsitedir(directory)
sys.path.insert(0, str(PACKAGE / "library"))
for name in list(sys.modules):
    if name == "fox_restricted" or name.startswith("fox_restricted."):
        del sys.modules[name]
print("Source revision:", SOURCE_REVISION)
print("Library:", PACKAGE)
''')
    md(r"""
## 2. Choose the experiment settings

The finite dataset has **N(N−1) ordered pair identities**; overwrite, recall-at-R, complements,
and calibration expand it to 4N(N−1)+2N weighted sequences. This preserves the previous setup.

The random-data default samples **N(N−1) complete streams once per seed**, shared across its
four arms. The query and target lag 1…R are uniform. An independent geometric older-prefix length
with mean 16 gives full support over finite legal streams. Prefix keys and all values are random;
suffix keys exclude the query so the designated target is its latest occurrence.
There is no uniform probability distribution over all finite sequence lengths.

SGD samples from the fixed dataset; Adam uses its full empirical loss. Set `RANDOM_SAMPLING="online"`
to draw new iid minibatches for **both** optimizers instead. This changes the training protocol.
`prefix_distribution="bounded_uniform"` with `max_prefix=64` is an optional finite-support sampler.
No hidden overwrite/calibration mixture is added to random-data training.

The Adam preset is explicitly `(0.1,0.1)`. Set `(0.9,0.999)` for the previous control and change
`RUN_TAG`. Gaussian initialization, zero Adam moments, annealed epsilon, and the three prescribed
acquisition rate updates are retained. Random-data acquisition uses its own actual sampled loss;
SGD's inherited rate constants are calibrated from the deterministic zero-table pair-model probe.
""")
    code(r'''
from fox_restricted import (
    a100_config, RandomStreamConfig, make_experiments, train, summarize,
    plot_fixed_lags, plot_moving_lag, check_runtime, prepare_output, benchmark,
)

SMOKE = False
N, WIDTH, R, STEPS = 128, 4096, 2, 200_000
SEEDS = (0, 1, 2)
BETA1, BETA2 = 0.1, 0.1
H0 = 1.0
THETA = 0.5
RUN_TAG = "fox_eight_runs_v1"
USE_DRIVE = True
RUN_CHECKS = True
RUN_BENCHMARK = True
REQUIRE_A100 = True
RANDOM_SAMPLING = "fixed"
RANDOM_DATASET_SIZE = None       # None means N(N-1) sampled streams per seed.
LOG_EVERY, CHECKPOINT_EVERY = 1000, 5000

if SMOKE:
    N, WIDTH, STEPS, SEEDS = 4, 64, 20, (0,)
    USE_DRIVE, REQUIRE_A100 = False, False
    RANDOM_DATASET_SIZE = 256
    LOG_EVERY, CHECKPOINT_EVERY = 10, 10

config = a100_config(n=N, d=WIDTH, R=R, steps=STEPS, h0=H0,
                     beta1=BETA1, beta2=BETA2, device="cpu" if SMOKE else "cuda")
random_data = RandomStreamConfig(
    batch_size=32 if SMOKE else 4096,
    prefix_distribution="geometric", prefix_mean=16,
    max_batch_records=8_000_000,  # Raises on oversize draws; never truncates or resamples.
)
experiments = make_experiments(
    config, seeds=SEEDS, random_data=random_data, random_sampling=RANDOM_SAMPLING,
    random_dataset_size=RANDOM_DATASET_SIZE, theta=THETA,
    log_every=LOG_EVERY, checkpoint_every=CHECKPOINT_EVERY,
)
HARDWARE = check_runtime(config.device, require_a100=REQUIRE_A100)
OUTPUT = prepare_output(RUN_TAG + ("_smoke" if SMOKE else ""), use_drive=USE_DRIVE)
print(HARDWARE)
print("Output:", OUTPUT)
print("Experiments:", list(experiments))
''')
    md("""
## 3. Validate and measure this runtime

The tests compare analytical gradients with literal attention and native PyTorch optimizers,
validate the random sampler, and test checkpoint continuation. The benchmark uses disposable
models at the requested size, including the random datasets. It does not spend any updates of
the eight saved experiments. Its time estimates exclude plotting, Drive I/O, and later dynamics.
""")
    code(r'''
if RUN_CHECKS:
    subprocess.run([sys.executable, "-m", "pytest", "-q", str(PACKAGE / "tests")], check=True, cwd=PACKAGE)
if RUN_BENCHMARK:
    throughput = benchmark(experiments, measured_steps=2 if SMOKE else 10, warmup=1 if SMOKE else 3)
''')
    md(r"""
## What the graphs measure

\[
\mathcal E_t(r)=\sup_{X:\,\mathrm{target\ lag}(X)\le r}
\left[1-P_t(y_\star\mid X)\right].
\]

This supremum includes arbitrary finite older prefixes and binary values. We plot a **witness
lower bound and a conditional uniform upper bound**, not an empirical average labeled as the
exact supremum. Linear error axes show 0/1 behavior; log-error axes resolve convergence near zero.
The probability panel shows the corresponding worst-case correct-probability bracket **1−E**.
An unavailable uniform certificate is explicitly marked with the trivial upper bound E≤1.

For all Adam sections choose one coefficient per seed immediately after acquisition:
\[
c=\theta\,\Delta_{\min,\mathrm{acquired}},\qquad
r_t=\max\{1,\lfloor c\,m_t/h_t\rfloor\}.
\]
The coefficient is then fixed, so rₜ is asymptotically proportional to mₜ/hₜ if that ratio diverges.
It is never fitted to test errors. We plot actual rₜ without a lag cap; early clamping at 1 is visible.
A nonpositive acquired gap makes this probe unavailable. The uniform error bound is always
recomputed using the **current** model, rather than assuming the acquired gap persists.

For the restricted learned-SGD setup, the theoretical limits are (0,0,1) at (R+1,R+2,R+3),
subject to its preparation/schedule hypotheses. ALiBi predicts vanishing error at all these
fixed lags under its hypotheses, and an expanding range. The random-data panels are extension
experiments: neither the learned-SGD cutoff nor the full-population Adam theorem automatically
applies to this different distribution. Larger R also retains the previous preparation caveats.
""")
    sections = [
        ("pairs_learned_sgd", "4. Ordered-pair data · learned gate · SGD", "fixed",
         "Inspect Eₜ(R+1), Eₜ(R+2), Eₜ(R+3). The restricted theorem predicts limits 0, 0, 1 under its hypotheses."),
        ("pairs_learned_adam", "5. Ordered-pair data · learned gate · Adam", "moving",
         "Inspect decreasing Eₜ(rₜ) together with growth of rₜ ∝ mₜ/hₜ. A flat rₜ does not establish expanding recall."),
        ("pairs_alibi_sgd", "6. Ordered-pair data · ALiBi · SGD", "fixed",
         "Retrieval h=h₀ stays fixed. All three fixed-lag errors are expected to vanish under the frozen-retrieval theorem's hypotheses."),
        ("pairs_alibi_adam", "7. Ordered-pair data · ALiBi · Adam", "moving",
         "Use the same rₜ ∝ mₜ/h₀ definition. The theory's sufficient Adam range grows quadratically in its scalar learning-rate clock."),
        ("random_learned_sgd", "8. Random streams with lag ≤ R · learned gate · SGD", "fixed",
         "Measure the three fixed-lag errors without assuming the restricted dataset's sharp cutoff survives this broader training distribution."),
        ("random_learned_adam", "9. Random streams with lag ≤ R · learned gate · Adam", "moving",
         "Measure the error envelope and actual moving lag on the random-data objective. Inspect acquisition status if no positive-gap probe is available."),
        ("random_alibi_sgd", "10. Random streams with lag ≤ R · ALiBi · SGD", "fixed",
         "The sampled streams are shared with the learned-gate arm for each seed. Only retrieval forgetting is frozen."),
        ("random_alibi_adam", "11. Random streams with lag ≤ R · ALiBi · Adam", "moving",
         "The fourth random-data arm uses the same acquired-gap coefficient rule and the same error bounds over arbitrary test streams."),
    ]
    for name, title, kind, explanation in sections:
        md(f"## {title}\n\n{explanation}")
        plot = f"plot_fixed_lags({name}, probability=True, save=True)" if kind == "fixed" else f"plot_moving_lag({name}, theta=THETA, save=True)"
        code(f'{name} = train(experiments["{name}"], OUTPUT)\n{plot}')
    md("""
## 12. Review statuses and resume

Re-run a section with the same source revision, run tag, seeds, and scientific settings to resume.
Increasing `STEPS` or the per-seed time budget is allowed; changing the data, moments, learning
rates, or probe coefficient requires a new run tag. A disconnection can lose work since the last
checkpoint (5,000 updates by default). Numerical failures and failed acquisition remain visible.

Every arm saves `config.json`, `history.csv`, `asymptotic_errors.csv`, `certified_radii.csv`,
`runs.csv`, figures, datasets, and complete optimizer/RNG checkpoints. The source hashes accompany
each run. Check both falling upper error bounds and growing radii; finite runs cannot prove an
infinite-time limit. No A100 results are preloaded in this notebook.
""")
    code("runs = [\n    " + ",\n    ".join(row[0] for row in sections) + "\n]\nsummarize(runs)")
    for index, cell in enumerate(cells):
        cell["id"] = f"fox-readable-{index:03d}"
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU",
        "colab": {"name": "FoX_Restricted_A100_Colab.ipynb", "toc_visible": True},
    })
    for target in (ROOT / "notebooks/FoX_Restricted_A100_Colab.ipynb", ROOT / "FoX_Restricted_A100_Asymptotics.ipynb",
                   ROOT / "FoX_Restricted_Generalization_Colab.ipynb"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        print(f"Wrote {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
