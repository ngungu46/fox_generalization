"""Build a Colab notebook containing all experiment source and its checks.

Run this script whenever implementation modules or embedded tests change.
Only Python's standard library is required to build the notebook.
"""

from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE / "FoX_Restricted_Generalization_Colab.ipynb"


def build():
    cells = []

    def markdown(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"})

    def code(source, hidden=False):
        cells.append({
            "cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"jupyter": {"source_hidden": True}} if hidden else {},
            "source": source.strip() + "\n",
        })

    markdown(r"""
# FoX restricted experiment: learned versus frozen retrieval forgetting

Run **Runtime → Run all** to train SGD and Adam on the paper's exact finite pair population,
then plot the model's **correct-answer probability at different test lags**.
The first settings cell controls vocabulary size **N**, training recall lag **R**, seeds and compute.
All source and checks are included below: no repository, data upload, API key or prior result is needed.

The default is a modest experiment: N=8, R=2, width 2,048, three paired seeds,
12,000 total updates (including acquisition), and four arms. A GPU runtime is useful; CPU also works.
Set `SMOKE=True` for a short software check, whose results are not evidence for asymptotic theory.
Outputs, configurations, source and plots are saved and offered as one downloadable ZIP.

This is the paper's **restricted two-layer answer-supervised architecture** with its parser masks,
local binder, trainable raw query/key tables and scalar decoder. It is an empirical sanity check
of those dynamics, rather than a language-model benchmark. A finite run can support or challenge
the predicted trend; it cannot certify asymptotic or uniform generalization.
""")
    markdown(r"""
## The finite dataset and the meaning of lag R

Let K=N(N−1). We use **every ordered pair of distinct keys** a≠b. For each pair,

\[
O^{a,b}=(a,-1),(a,-1),(b,-1),(a,+1),?a,
\]
\[
C_R^{a,b}=(a,+1),\underbrace{(b,-1),\ldots,(b,-1)}_{R-1},?a.
\]

Both templates include their global value complements. Every key additionally has the two
one-record calibration examples. Therefore K is the number of **pair identities**;
the full supervised population contains **4K+2N serialized sequences**, not K sequences total.
The objective weights the overwrite, recall and calibration families by 0.2, 0.2 and 0.6.
SGD samples pair identities with replacement and evaluates both weighted families; calibration
is evaluated exactly. Adam uses the full pair population.

**Lags count records, with the newest record at lag 1.** Only recall training changes when R
changes; the four-record overwrite template stays fixed. Training uses one recall range R,
not a mixture of every shorter range. All losses supervise the final binary answer only.

Evaluation keeps the same vocabulary and tests a target at each requested lag with newer,
opposite-valued distractors. Additional old, opposite-valued same-key prefixes challenge overwrite
robustness. The plots report the probability assigned by the scalar decoder to the correct answer,
not merely target attention or thresholded classification accuracy.
""")
    markdown(r"""
## The four arms and predictions to examine

| Arm | Local binding penalty g | Retrieval penalty h | Optimizer |
|---|---|---|---|
| Learned / SGD | learned | learned | SGD |
| Learned / Adam | learned | learned | full-batch annealed-epsilon Adam |
| Frozen retrieval / SGD | learned | fixed h₀ | SGD |
| Frozen retrieval / Adam | learned | fixed h₀ | full-batch annealed-epsilon Adam |

The frozen intervention fixes **only retrieval forgetting**. The binder must still learn.
An optional `both_frozen` control is a separate, obstructed case. The learned and frozen arms
start with identical h₀ and matched raw table draws for each seed.

At R=2, the learned-gate SGD theorem gives a worst-case cutoff at lag 4; learned-gate Adam
has an expanding range, linear in its cumulative scalar learning-rate clock under the theorem's
conditions. With frozen retrieval h₀>log 2, the answer-supervised extension predicts an expanding
range for both optimizers; its Adam range is quadratic in that clock. Frozen retrieval can
produce confident correct answers even though target attention retains an overwrite error floor.

For R>2, the learned-SGD extension predicts cutoff R+2 under its basin/preparation and
continuation conditions. This notebook does not implement the extension's extra finite
preparation construction. The frozen-retrieval convergence proof in the accompanying theory
uses R=2. Setting R=4 is a useful extension experiment, but must not be described as a direct
empirical verification of every proved hypothesis.

The theorem's very conservative SGD existence schedule is not a practical Colab schedule.
We use an explicit finite polynomial schedule with increasing pair batches and a positive,
summable table-rate tail. The plots retain diagnostics needed to assess this approximation.
A curve that has not reached its predicted regime is inconclusive, and a mean curve can conceal
a hard key pair. Inspect minimum probabilities, individual seeds, training loss and gap diagnostics.
""")
    code(r"""
# USER SETTINGS — edit this cell, then run all.
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import os
import sys
import tempfile

SMOKE = False
N = 8
R = 2                         # Try R=4 separately; read the extension caveat above.
WIDTH = 2048
STEPS = 12000                 # Total updates, including three acquisition updates.
SEEDS = (0, 1, 2)              # Paired across both optimizers and gate modes.
H0 = 1.0                      # Penalty PER RECORD. Default exceeds log(2).
EVAL_LAGS = tuple(range(1, 33))
PREFIXES = (0, 16, 64)
INCLUDE_BOTH_FROZEN = False
LOG_EVERY = 1000
RUN_QA = True
RUN_TRAINING = True
DOWNLOAD_AT_END = True
USE_DRIVE = False
RUN_TAG = "fox_restricted_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Colab normally includes these numerical packages. Elsewhere install requirements.txt.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fox_restricted_mpl"))
import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Image, FileLink, Markdown

def show_table(columns, rows):
    def clean(value):
        return str(value).replace("|", "&#124;").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, columns)) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
    display(Markdown("\n".join(lines)))

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except (ModuleNotFoundError, ValueError):
    IN_COLAB = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)       # Tiny matrix operations are faster without thread-pool overhead.
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
if SMOKE:
    N, WIDTH, STEPS, SEEDS = 4, 64, 20, (0,)
    EVAL_LAGS, PREFIXES, LOG_EVERY = tuple(range(1, 9)), (0, 4), 10

if USE_DRIVE:
    if not IN_COLAB:
        raise RuntimeError("USE_DRIVE applies to Colab; set OUTPUT_PARENT to a local path instead.")
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_PARENT = Path("/content/drive/MyDrive/fox_restricted_runs")
else:
    OUTPUT_PARENT = Path("/content/fox_restricted_runs") if IN_COLAB else Path.cwd() / "fox_restricted_runs"
OUT_DIR = OUTPUT_PARENT / (RUN_TAG + ("_smoke" if SMOKE else ""))
CODE_DIR = Path(tempfile.mkdtemp(prefix="fox_restricted_source_"))
SOURCES = {}
print("Device:", DEVICE, "| PyTorch:", torch.__version__, "| Outputs:", OUT_DIR)
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name())
print(f"N={N}, R={R}; {N*(N-1)} pair identities; {4*N*(N-1)+2*N} serialized sequences")
print(f"{len(SEEDS) * (3 if INCLUDE_BOTH_FROZEN else 2) * 2} runs × {STEPS:,} total updates")
""")
    markdown(r"""
## Embedded implementation

These folded cells contain the complete Python modules. Their SHA-256 hashes are saved with the
outputs, and the tests run before training. The implementation uses float64 signed-log gradients
and signed-log Adam moments to retain small derivatives without squaring them to zero.
This is the ordinary answer objective and actual Adam update, with an independent native
float64 cross-check in the preflight suite. Signed-log arithmetic still has finite precision.
""")
    names = ["core", "runner", "report"]
    for test_name in ("test_core", "test_runner"):
        if (HERE / (test_name + ".py")).exists():
            names.append(test_name)
    hashes = {}
    for name in names:
        source = (HERE / (name + ".py")).read_text(encoding="utf-8")
        hashes[name + ".py"] = hashlib.sha256(source.encode()).hexdigest()
        code(f"SOURCES[{name!r}] = {source!r}", hidden=True)
    code(r"""
import hashlib
import importlib
import json
import subprocess

for name, source in SOURCES.items():
    (CODE_DIR / (name + ".py")).write_text(source, encoding="utf-8")
    sys.modules.pop(name, None)
sys.path.insert(0, str(CODE_DIR))
importlib.invalidate_caches()
from core import Config
from runner import run_suite
from report import make_report

SOURCE_HASHES = {name + ".py": hashlib.sha256(src.encode()).hexdigest()
                 for name, src in SOURCES.items()}
print("Embedded sources loaded from", CODE_DIR)
show_table(("file", "sha256"), SOURCE_HASHES.items())
if RUN_QA:
    if "test_core" not in SOURCES:
        raise RuntimeError("No embedded test module; regenerate the notebook with test_core.py present.")
    if importlib.util.find_spec("pytest") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest>=7"])
    tests = [str(CODE_DIR / (name + ".py")) for name in SOURCES if name.startswith("test_")]
    test_env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    subprocess.check_call([sys.executable, "-m", "pytest", "-q", *tests],
                          cwd=CODE_DIR, env=test_env)
""")
    markdown(r"""
## Initialization and optimizer settings

Every seed initializes independent Q/K Gaussian tables with standard deviation 10⁻⁶,
q=p=0.5, u=v=1.7 and decoder w=0. The learned retrieval logit is set to
softplus⁻¹(h₀). Three optimizer-specific, answer-supervised acquisition updates acquire
matching; Adam starts with zero moments and retains its full acquisition history.
There is no weight decay, target-position loss, gradient clipping or optimizer-state reset.

Adam uses β₁=0.9, β₂=0.999 (so β₁²<β₂) and epsilon σ³ exp(−c t) outside the square root.
The notebook records the complete configuration. Learning rates differ by optimizer because
the paper prescribes different acquisition and continuation protocols. Equal update counts
are a compute comparison, not equality of optimizer clocks; also inspect the plots versus
cumulative scalar learning rate. No long-lag test result is used to select these rates.

The width is configurable. The default N=8, d=2,048 meets the stated Gaussian dimension
lower bound at failure budget 0.05, but this alone does not verify the theorem's rate,
initialization, cone, settling or basin constants. The smoke width deliberately does not.
""")
    code(r"""
cfg = Config(
    n=N, d=WIDTH, R=R, h0=H0, steps=STEPS, device=DEVICE,
    sigma=1e-6, q0=0.5, u0=1.7,
    beta1=0.9, beta2=0.999,
    lr_sgd=5.0, lr_adam=0.015,
    offset=1000.0, power=0.75,
    table_lr=1e-9, table_power=2.0,
    pair_batch=4096, batch_growth=0.5,
    eps_decay=0.1,
)
GATE_MODES = ("learned", "retrieval_frozen") + (("both_frozen",) if INCLUDE_BOTH_FROZEN else ())
OPTIMIZERS = ("sgd", "adam")
from dataclasses import asdict
show_table(("setting", "value"), asdict(cfg).items())
print("Frozen retention per record:", np.exp(-H0), "| h0 > log(2):", H0 > np.log(2))
print("At R=2: learned-SGD reference boundary is lag 4. At R>2, R+2 is conditional here.")
""")
    markdown(r"""
## Train all requested arms

The runner saves the full weighted population (`dataset/dataset.jsonl`), exactly K base recall
sequences (`dataset/recall_at_R.jsonl`), pair identities (`dataset/pairs.jsonl`), configurations
and diagnostics.
Use a fresh `RUN_TAG` when changing settings; retain the downloaded ZIP or enable Drive for persistence.
Inspect any failed acquisition or numerical status before interpreting a curve.
""")
    code(r"""
if RUN_TRAINING:
    saved_root = run_suite(
        cfg, OUT_DIR, seeds=SEEDS, gate_modes=GATE_MODES,
        optimizers=OPTIMIZERS, eval_lags=EVAL_LAGS,
        prefixes=PREFIXES, log_every=LOG_EVERY,
    )
    OUT_DIR = Path(saved_root)
else:
    print("Training skipped. Set OUT_DIR to a previously generated output folder to plot it.")

# Keep the exact executable code next to the numeric results.
if OUT_DIR.exists():
    audit = OUT_DIR / "embedded_source"
    audit.mkdir(exist_ok=True)
    for name, source in SOURCES.items():
        (audit / (name + ".py")).write_text(source, encoding="utf-8")
    (audit / "sha256.json").write_text(json.dumps(SOURCE_HASHES, indent=2), encoding="utf-8")
""")
    markdown(r"""
## Generalization plots and run diagnostics

The final-lag panels answer the main question: how does correct-answer confidence change as the
target moves farther back? The time/clock panels show whether that range is still expanding.
The solid curves average each seed's minimum over the evaluated key pairs and prefix lengths;
their envelopes span seeds. Dashed curves show mean probabilities over the evaluated examples.
Compare all four arms and inspect the saved prefix-specific measurements.
The theoretical cutoff is a worst-case asymptotic statement; these finite prefix tests are not
a proof over arbitrary streams.
""")
    code(r"""
if not OUT_DIR.exists():
    raise FileNotFoundError(f"No results at {OUT_DIR}; run training or set the correct output path.")
REPORT_PATHS = make_report(OUT_DIR)
print("Report outputs:", REPORT_PATHS)
for path in sorted(OUT_DIR.rglob("*.png")):
    display(Image(filename=str(path), width=1100))
summary_candidates = [p for p in sorted(OUT_DIR.rglob("*.csv"))
                      if "summary" in p.name or p.name == "runs.csv"]
for path in summary_candidates:
    display(Markdown(f"**{path.relative_to(OUT_DIR)}**"))
    import csv
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or ())
    if path.name == "runs.csv":
        columns = [name for name in ("seed", "gate_mode", "optimizer", "status", "completed_steps",
                                      "requested_steps", "acquired_positive_gaps", "reason") if name in columns]
    show_table(columns, ([row.get(name, "") for name in columns] for row in rows))
for path in sorted(OUT_DIR.rglob("*.md")):
    if path.name.lower() in {"report.md", "summary.md"}:
        display(Markdown(path.read_text(encoding="utf-8")))
""")
    markdown(r"""
## Export everything

The ZIP contains saved measurements, reports, plots and the exact embedded code. Retain it before
the Colab runtime expires. This notebook itself remains a clean, rerunnable experiment artifact;
save a copy in Colab if you also want its displayed outputs.
""")
    code(r"""
import shutil

zip_path = Path(shutil.make_archive(str(OUT_DIR), "zip", root_dir=OUT_DIR.parent, base_dir=OUT_DIR.name))
print("Saved", zip_path, f"({zip_path.stat().st_size / 2**20:.2f} MiB)")
if IN_COLAB and DOWNLOAD_AT_END:
    from google.colab import files
    files.download(str(zip_path))
else:
    display(FileLink(str(zip_path)))
""")
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": OUT.name, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "fox_restricted_source_sha256": hashes,
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    for i, cell in enumerate(cells):
        cell["id"] = f"fox-restricted-{i:03d}"
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cells)} cells; {OUT.stat().st_size:,} bytes)")
    return OUT


if __name__ == "__main__":
    build()
