"""Build the self-contained, resumable A100 moving-radius experiment notebook."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "library" / "fox_restricted" / "legacy"
TESTS = ROOT / "tests"
OUT = ROOT / "notebooks" / "archive" / "FoX_Restricted_A100_Asymptotics.ipynb"
MODULES = ("core", "runner", "report", "asymptotics", "a100_runner", "asymptotic_report",
           "test_core", "test_runner", "test_asymptotics", "test_a100_runner")


def build():
    cells = []

    def markdown(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=source.strip() + "\n"))

    def code(source, hidden=False):
        cells.append(dict(cell_type="code", execution_count=None, outputs=[],
                          metadata={"jupyter": {"source_hidden": True}} if hidden else {},
                          source=source.strip() + "\n"))

    markdown(r"""
# FoX A100 experiment: worst-case generalization along growing radii

This notebook trains the restricted answer-supervised model on **N=128 keys, width d=4096,
training lag R=2, 200,000 updates, three paired seeds and four optimizer/gate arms**.
It measures both ordinary finite-prefix prediction curves and certified brackets for the
worst-case error **Eₜ(r)** at fixed and growing radii. All executable source and tests are embedded.

In Colab, choose **Runtime → Change runtime type → A100 GPU**, then run all cells.
The notebook checks the hardware you received; it does not provision or guarantee an A100.
GPU availability and runtime lifetimes vary, so the default stores resumable checkpoints on Drive.
See the [official Colab resource and runtime FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)
and its [Drive guidance](https://research.google.com/colaboratory/faq.html#drive-mount).

The execution order is: software checks, a short **seed-0 benchmark at the requested model size**,
then seed 0's four full arms, followed by the remaining seeds. The measured benchmark prints a
rough compute estimate before the full run. Change `SEEDS=(0,)` for an initial single-seed study;
the full default is `(0,1,2)`. `SMOKE=True` runs a tiny CPU software check, not an A100 experiment.

The longer budget does not guarantee that asymptotic behavior has appeared. The report retains
loss, gaps, optimizer clocks, actual growing radii, interval validity, and stopped-run statuses.
""")
    markdown(r"""
## What Eₜ(r) means, and what we can bound

\[
E_t(r)=\sup_{\substack{\text{finite record streams }x\\
\text{latest target lag}(x)\le r}}
\bigl(1-P_t(\text{correct answer}\mid x)\bigr).
\]

This is worst-case **answer error over all finite streams**, not an average over test examples,
and not the epsilon in Adam's denominator. We save a bracket
\(L_t(r)\le E_t(r)\le U_t(r)\):

- The lower endpoint is an analytic infinite-prefix witness limit. Every finite prefix is an
  allowed stream, so the limit lower-bounds the supremum.
- The upper endpoint is a uniform certificate over arbitrary finite streams, conditional on its
  recorded matching-gap, row-norm, scale and sign conditions. When those conditions fail, we mark
  the certificate invalid and show the trivial upper bound **E≤1** explicitly.
- Both endpoints are stored and plotted in log-error space. We do not compute tiny errors by
  subtracting a rounded prediction probability from one.

Fixed probes examine Eₜ(r) at fixed r. Moving probes examine Eₜ(rₜ), and plot the actual integer rₜ
alongside the error bounds. Clock probes use `floor(1+c*S)` for learned retrieval and
`floor(1+c*S**2)` for frozen retrieval. These are prescribed experiments, not growth-rate claims
for every optimizer. Adaptive and acquired-reference probes use the observed matching gaps.
The largest radius certified at each error threshold is computed from the uniform bound with
**no fixed evaluation-lag cap**.

For frozen SGD, the polynomial clock probes are stress tests: the ordinary-answer theorem does
not provide a sharp time law for its expanding range. Pointer-supervised time laws are not used here.

To support a growing-range conclusion, inspect both a decreasing uniform upper error and an
increasing actual radius. A small error at a radius that stays bounded does not establish
generalization along rₜ→∞, and a finite trajectory cannot prove either limiting claim.
""")
    markdown(r"""
## Dataset, interventions, and theoretical scope

Every ordered pair a≠b is included: **K=N(N−1)=16,256 pair identities** at N=128.
For each pair we train the paper's overwrite sequence
`(a,-1),(a,-1),(b,-1),(a,+1),?a` and recall sequence
`(a,+1),(b,-1)×(R−1),?a`, together with both global value complements.
Two one-record calibration sequences are included per key. Thus the complete weighted population
has **4K+2N serialized sequences**. Lags count records, newest record at lag 1.
The overwrite/recall/calibration weights are 0.2/0.2/0.6.

The four arms are learned retrieval / SGD, learned retrieval / Adam, frozen retrieval / SGD,
and frozen retrieval / Adam. Frozen retrieval holds **h=h₀** while the local binding gate remains
trainable. Raw Q/K table draws and scalar initialization are paired within seed. Three acquisition
updates precede continuation; Adam keeps its zero-initialized moment history without resets.

At R=2, the paper predicts a learned-SGD worst-case cutoff at lag 4, an expanding learned-Adam
range, and expanding frozen-retrieval ranges under the sufficient protocol, with h₀>log(2).
The frozen-Adam range has a quadratic sufficient clock scaling. The practical polynomial schedule
here is an empirical protocol, not the theorem's very conservative existence construction.
R>2 is available as an extension experiment, with additional hypotheses and preparation caveats.

N=128,d=4096 satisfies the stated Gaussian dimension lower bound at failure budget 0.05.
That verifies this one dimension condition only; it does not certify every initialization, rate,
settling, cone or continuation hypothesis. Increasing compute alone is not proof of asymptotics.
""")
    code(r"""
# USER SETTINGS — keep RUN_TAG and settings unchanged to resume interrupted work.
from pathlib import Path
import importlib.util
import os
import sys
import tempfile

SMOKE = False
N, WIDTH, R, STEPS = 128, 4096, 2, 200000
SEEDS = (0, 1, 2)                 # Use (0,) for a first full seed.
H0 = 1.0
MOMENT_PRESET = "paper_short_memory"  # Or "standard_memory"; uses a separate run folder.
RUN_TAG = "fox_a100_asymptotics_v1"
ALLOW_OTHER_GPU = False
USE_DRIVE = True
RUN_QA = True
RUN_BENCHMARK = True
RUN_TRAINING = True
DOWNLOAD_AT_END = False           # Results already persist on Drive by default.
COMPILE = False                   # Experimental compilation is deliberately opt-in.
RESUME = True
LOG_EVERY = 1000
CHECKPOINT_EVERY = 5000
BENCHMARK_STEPS, BENCHMARK_WARMUP = 30, 5
MAX_BRANCH_SECONDS = 24 * 3600
EVAL_LAGS = (1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024)
PREFIXES = (0, 64)
FIXED_LAGS = (1, 2, 3, 4, 5, 8, 16, 32, 64)
THETA_VALUES = (0.25, 0.5, 0.75)
CLOCK_COEFFICIENTS = (0.001, 0.003, 0.01, 0.03, 0.1)
ERROR_TARGETS = (0.1, 0.01, 0.001)

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fox_a100_mpl"))
import numpy as np
import torch
from IPython.display import display, Image, FileLink, Markdown

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except (ModuleNotFoundError, ValueError):
    IN_COLAB = False

if MOMENT_PRESET not in {"paper_short_memory", "standard_memory"}:
    raise ValueError("Choose paper_short_memory or standard_memory.")
BETA1, BETA2 = {"paper_short_memory": (0.1, 0.1), "standard_memory": (0.9, 0.999)}[MOMENT_PRESET]
assert BETA1**2 < BETA2
if SMOKE:
    N, WIDTH, STEPS, SEEDS = 4, 64, 20, (0,)
    EVAL_LAGS, PREFIXES = (1, 2, 3, 4, 5, 8), (0, 4)
    FIXED_LAGS, THETA_VALUES, CLOCK_COEFFICIENTS = (1, 2, 4, 8), (0.5,), (0.01,)
    LOG_EVERY, CHECKPOINT_EVERY = 10, 10
    BENCHMARK_STEPS, BENCHMARK_WARMUP = 3, 1
    USE_DRIVE, DOWNLOAD_AT_END = False, False
    DEVICE = "cpu"
else:
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires a CUDA GPU. Select an A100 runtime in Colab, or set SMOKE=True for a CPU check.")
    DEVICE = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    if "A100" not in GPU_NAME.upper() and not ALLOW_OTHER_GPU:
        raise RuntimeError(f"Received {GPU_NAME}, not an A100. Select an A100 or explicitly set ALLOW_OTHER_GPU=True.")
    if "A100" not in GPU_NAME.upper():
        print("Explicit override: running the A100 profile on", GPU_NAME)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

if USE_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("Drive mounting is supported here in Colab. Set USE_DRIVE=False for another CUDA environment.")
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_PARENT = Path("/content/drive/MyDrive/fox_a100_runs")
else:
    OUTPUT_PARENT = Path(os.environ.get("FOX_A100_OUTPUT_PARENT", str(Path.cwd() / "fox_a100_runs")))
EFFECTIVE_RUN_TAG = RUN_TAG + "_" + MOMENT_PRESET + ("_smoke" if SMOKE else "")
OUT_DIR = OUTPUT_PARENT / EFFECTIVE_RUN_TAG
CODE_DIR = Path(tempfile.mkdtemp(prefix="fox_a100_source_"))
SOURCES = {}
HARDWARE = dict(device=DEVICE, gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
                torch_version=torch.__version__, parameter_dtype="float64", tf32=False, mixed_precision=False)
print("Hardware:", HARDWARE)
print("Persistent output:", OUT_DIR)
print(f"Plan: seed {SEEDS[0]} first, followed by {SEEDS[1:]}; {4*len(SEEDS)} arms × {STEPS:,} total updates.")
print(f"K={N*(N-1):,} pair identities; {4*N*(N-1)+2*N:,} complete serialized sequences.")
print("Moment preset:", MOMENT_PRESET, "| betas:", (BETA1, BETA2))
""")
    markdown(r"""
## Exact FP64 implementation and preflight checks

Parameters, signed-log gradients, and signed-log Adam moments remain float64. There is no BF16,
FP16, autocast, TF32, gradient clipping, moment reset or epsilon floor. Adam's prescribed epsilon
is outside the square root and anneals as sigma³ exp(−c t). Signed-log arithmetic retains very
small derivatives but still has finite precision. Independent native-autodiff checks, witness
bounds and resume tests run before training.

The default `paper_short_memory` betas `(0.1,0.1)` follow the existing mechanism experiment's
short-memory setting. They are selected to study moment tracking of changing gradients, not by
long-lag validation performance. The earlier baseline used `(0.9,0.999)`; retain that comparison
with `standard_memory`. Both satisfy beta1²<beta2, and both initialize all moments at zero.
Their separate run folders prevent accidental mixing or overwriting.
Changing the betas affects both acquisition and later moment dynamics, so this comparison does
not isolate the continuation-memory effect while holding the acquired key geometry fixed.
""")
    hashes = {}
    for name in MODULES:
        path = (TESTS if name.startswith("test_") else HERE) / (name + ".py")
        source = path.read_text(encoding="utf-8")
        hashes[path.name] = hashlib.sha256(source.encode()).hexdigest()
        code(f"SOURCES[{name!r}] = {source!r}", hidden=True)
    code(r"""
import hashlib
import importlib
import inspect
import json
import subprocess
from dataclasses import asdict

for name, source in SOURCES.items():
    (CODE_DIR / (name + ".py")).write_text(source, encoding="utf-8")
    sys.modules.pop(name, None)
sys.path.insert(0, str(CODE_DIR))
importlib.invalidate_caches()
from core import Config
from a100_runner import run_large_suite, benchmark_config
from report import make_report
from asymptotic_report import make_asymptotic_report

SOURCE_HASHES = {name + ".py": hashlib.sha256(source.encode()).hexdigest()
                 for name, source in SOURCES.items()}
print("Embedded source SHA-256:", json.dumps(SOURCE_HASHES, indent=2))
if RUN_QA:
    if importlib.util.find_spec("pytest") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest>=7"])
    tests = [str(CODE_DIR / (name + ".py")) for name in SOURCES if name.startswith("test_")]
    subprocess.check_call([sys.executable, "-m", "pytest", "-q", *tests], cwd=CODE_DIR,
                          env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"))

cfg = Config(n=N, d=WIDTH, R=R, steps=STEPS, device=DEVICE, h0=H0,
             sigma=1e-6, q0=0.5, u0=1.7, beta1=BETA1, beta2=BETA2,
             lr_sgd=5.0, lr_adam=0.015, offset=1000.0, power=0.75,
             table_lr=1e-9, table_power=2.0, pair_batch=4096, batch_growth=0.5,
             eps_decay=0.1, max_seconds=MAX_BRANCH_SECONDS)
GATE_MODES, OPTIMIZERS = ("learned", "retrieval_frozen"), ("sgd", "adam")
dimension_bound = int(np.ceil(144*np.log(16*N*(N-1)/0.05)))
print(json.dumps(asdict(cfg), indent=2))
print("Width:", WIDTH, "| stated Gaussian dimension bound at failure budget 0.05:", dimension_bound,
      "| satisfied:", WIDTH >= dimension_bound)
print("Frozen h0 > log(2):", H0 > np.log(2))
""")
    markdown(r"""
## Measured seed-0 throughput before the long run

The benchmark uses independent seed-0 models and their acquisition protocol. Its updates are
discarded; the actual experiment starts from the original draws or an exact saved checkpoint.
It measures the requested model size on the detected device and estimates all requested arms.
Later optimizer states, growing SGD batches, certificate evaluations, checkpoint writes and
Drive latency can change runtime. The estimate is not a promised completion time.
""")
    code(r"""
BENCHMARK = None
if COMPILE:
    print("Experimental compilation requested. The runner must pass its eager/compiled signed-log gradient audit; failures raise an error.")
if RUN_BENCHMARK and RUN_TRAINING:
    BENCHMARK = benchmark_config(cfg, steps=BENCHMARK_STEPS, warmup=BENCHMARK_WARMUP,
                                 num_seeds=len(SEEDS), compile_gradients=COMPILE)
    print("Measured preflight:", json.dumps(BENCHMARK, indent=2))
else:
    print("Throughput benchmark skipped by settings.")
""")
    markdown(r"""
## Run, checkpoint, and resume

The stable default tag is `fox_a100_asymptotics_v1`, followed by the moment-preset suffix.
Leave the tag and training/evaluation settings unchanged to resume after a disconnect.
Checkpoints include model parameters, signed-log Adam state, RNG state, continuation clock and
progress. Saved rows and run statuses are retained. Completed arms are skipped on resume.
Use a **new tag** when changing protocol settings; incompatible checkpoints are rejected.

By default, diagnostics are collected every 1,000 updates and committed with resumable
checkpoints every 5,000 updates. With Drive enabled,
the files live outside the temporary runtime. A disconnect can lose work since the latest
checkpoint (up to 5,000 updates by default); it does not prove a branch converged or failed
theoretically. Inspect `runs.csv`.
For a first full seed set `SEEDS=(0,)`; run the additional seeds as a separate tag if the saved
manifest requires the original seed list to remain unchanged.
""")
    code(r"""
if RUN_TRAINING:
    OUT_DIR = Path(run_large_suite(
        cfg, OUT_DIR, seeds=SEEDS, gate_modes=GATE_MODES, optimizers=OPTIMIZERS,
        eval_lags=EVAL_LAGS, prefixes=PREFIXES, log_every=LOG_EVERY,
        checkpoint_every=CHECKPOINT_EVERY, resume=RESUME,
        compile_gradients=COMPILE,
        fixed_lags=FIXED_LAGS, theta_values=THETA_VALUES,
        clock_coefficients=CLOCK_COEFFICIENTS, error_targets=ERROR_TARGETS,
    ))
else:
    print("Training skipped; using saved output folder:", OUT_DIR)

if not OUT_DIR.exists():
    raise FileNotFoundError(f"No output at {OUT_DIR}. Run training or set the saved run tag.")
embedded = OUT_DIR / "embedded_source"
embedded.mkdir(exist_ok=True)
for name, source in SOURCES.items():
    (embedded / (name + ".py")).write_text(source, encoding="utf-8")
(embedded / "sha256.json").write_text(json.dumps(SOURCE_HASHES, indent=2), encoding="utf-8")
(OUT_DIR / "notebook_hardware.json").write_text(json.dumps(HARDWARE, indent=2), encoding="utf-8")
if BENCHMARK is not None:
    (OUT_DIR / "notebook_benchmark.json").write_text(json.dumps(BENCHMARK, indent=2), encoding="utf-8")
""")
    markdown(r"""
## Eₜ(r), Eₜ(rₜ), and the actual radius trajectories

The main plots show fixed and moving radius brackets against both update count and each run's
own continuation learning-rate sum S. The next plot shows the actual rₜ values and the certified
radius at each error threshold. Read the interval endpoints as bounds on the unknown supremum.
Crosses at log10(error)=0 explicitly mark unavailable uniform certificates.

The optimizer panel compares q/S, u/S and w/S with their nominal Adam multipliers and shows
the current signed normalized growth directions separately. This helps distinguish slow moment
tracking from failure of the predicted asymptotic geometry. Its ratios include acquisition offsets,
and the displayed references remain conditional asymptotic values.

The ordinary lag plot remains useful for finite empirical behavior, including cases where Adam
already generalized beyond some tested lag at an earlier checkpoint. A final fixed-lag curve
alone does not distinguish bounded from expanding asymptotic generalization.
""")
    code(r"""
ASYMPTOTIC_REPORT = make_asymptotic_report(OUT_DIR)
ORDINARY_REPORT = make_report(OUT_DIR)
for name in ("asymptotic_fixed_error_by_step", "asymptotic_fixed_error_by_S",
             "asymptotic_moving_error_by_step", "asymptotic_moving_error_by_S",
             "radius_trajectories", "optimizer_diagnostics"):
    display(Image(filename=ASYMPTOTIC_REPORT[name], width=1200))
display(Image(filename=ORDINARY_REPORT["generalization_by_lag"], width=1200))
display(Image(filename=ORDINARY_REPORT["theory_diagnostics"], width=1400))
display(Markdown(Path(ASYMPTOTIC_REPORT["report"]).read_text(encoding="utf-8")))
print("All reports:", {"asymptotic": ASYMPTOTIC_REPORT, "finite_prefix": ORDINARY_REPORT})
""")
    markdown(r"""
## Retain results and checkpoints

Drive outputs are already persistent. The optional ZIP contains CSV measurements, certificates,
reports, exact source and resumable checkpoints; it can be large for the full profile.
No local A100 result is preloaded or fabricated by this notebook. Its displayed evidence comes
only from the runs you execute or explicitly resume.
""")
    code(r"""
if DOWNLOAD_AT_END:
    import shutil
    archive_parent = Path("/content") if IN_COLAB else Path(tempfile.gettempdir())
    zip_path = Path(shutil.make_archive(str(archive_parent / EFFECTIVE_RUN_TAG), "zip",
                                       root_dir=OUT_DIR.parent, base_dir=OUT_DIR.name))
    if IN_COLAB:
        from google.colab import files
        files.download(str(zip_path))
    else:
        display(FileLink(str(zip_path)))
else:
    print("Measurements, reports and checkpoints retained at:", OUT_DIR)
""")
    notebook = dict(cells=cells, metadata={
        "colab": {"name": OUT.name, "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU", "fox_restricted_source_sha256": hashes,
    }, nbformat=4, nbformat_minor=5)
    for index, cell in enumerate(cells):
        cell["id"] = f"fox-a100-{index:03d}"
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells; {OUT.stat().st_size:,} bytes)")
    return OUT


if __name__ == "__main__":
    build()
