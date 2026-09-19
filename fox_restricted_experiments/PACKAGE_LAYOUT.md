# Source layout

The notebook is a client of an installable Python library. It holds experiment
settings and calls; dataset construction, optimization, error estimates, and
plotting live in the package.

```text
fox_restricted_experiments/
├── pyproject.toml                    # package dependencies and installation
├── notebooks/
│   ├── FoX_Restricted_A100_Colab.ipynb  # readable eight-experiment notebook
│   └── archive/                      # earlier embedded-source notebooks
├── library/fox_restricted/
│   ├── __init__.py                   # notebook-facing public imports
│   ├── study.py                      # eight experiment definitions
│   ├── runtime.py                    # Colab storage, hardware, benchmark
│   ├── models/                       # architecture and literal-stream audits
│   ├── data/                         # sampled structured-token streams
│   ├── training/                     # optimization and experiment orchestration
│   ├── evaluation/                   # error bounds and lag probes
│   ├── plotting/                     # focused fixed/moving error figures
│   └── legacy/                       # unchanged, validated restricted kernel
├── tests/                            # model, sampler, optimizer, resume checks
├── scripts/                          # notebook building and pilot reanalysis
│   └── archive/                      # old embedded-notebook builders
└── results/                          # existing measured pilots (not installed)
```

Start at the public API and the notebook to understand what is run. The existing
scientific implementation remains in `legacy/` byte-for-byte: earlier resumable
checkpoints require the exact source hashes of `core.py`, `runner.py`,
`a100_runner.py`, and `asymptotics.py`. The small same-named files at the experiment
root are compatibility entry points, not separate implementations. Existing
result folders and their original source snapshots are retained.

## Kernel map

| File under `library/fox_restricted/legacy/` | Responsibility |
|---|---|
| `core.py` | `Config`, two-layer `Model`, literal stream audit, signed-log Adam, finite pair dataset |
| `runner.py` | Small finite-dataset pilot and shared CSV/audit utilities |
| `a100_runner.py` | Resumable finite-dataset training, health checks, hardware benchmark |
| `asymptotics.py` | Worst-case error envelopes, moving lag definitions, certified radii |
| `report.py` | Original probability-by-lag pilot plots |
| `asymptotic_report.py` | Original asymptotic diagnostic report |

The public `models`, `training`, and `evaluation` namespaces expose these stable
operations alongside new experiment code. New source should import from the
package, rather than relying on a working directory containing `core.py`.

## Install a checkout locally or in Colab

From the `fox_generalization` repository root (the local `fox_experiments` folder):

```python
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-e",
    "./fox_restricted_experiments",
])
from fox_restricted import Config
```

The notebook contains GitHub clone and package-install cells prefilled from the
local `fox_experiments` Git remote: `ngungu46/fox_generalization`, branch `main`.
The new restricted experiment folder must be pushed before the GitHub install
can use it. Colab does not need the historical `results/` files. A branch, tag,
or exact commit SHA can be selected with `REPO_REF`.

For package checks:

```bash
python -m pip install -e './fox_restricted_experiments[test]'
python -m pytest fox_restricted_experiments/tests
```

For reproducibility, the training outputs record the implementation hashes as
well as the complete configuration. Cloning a pinned commit provides an
additional record of the exact library used by a Colab run.
