# Validation status

## Verified locally

- **33 CPU tests passed**: 12 model checks, 5 data checks, 4 small-training
  integration checks, 6 controlled-mechanism checks, and 6 full-runner checks.
- The small Adam trainer resumes an interrupted run to bitwise-identical model
  parameters and optimizer moments. Its saved specification rejects changed
  configuration, data or implementation files.
- The full trainer completed a reduced CPU source/continuation experiment using
  synthetic native Zarr stores. Interrupted/resumed training matched model,
  optimizer and scheduler states with dropout enabled, exercising RNG recovery.
- Full-path tests cover weighted loss accumulation, native split separation,
  missing-chunk rejection, epsilon limits, source/optimizer pairing, command
  matrix coverage, checkpoint evaluation, and report generation.
- **Both notebooks executed without error outputs in fresh local Jupyter kernels.**
  The downscale notebook ran the complete `smoke` experiment on the actual cached
  LongCrawl64 subset. The full notebook ran its planning and empty-result analysis
  cells with download, hardware checks and training disabled.
- Notebooks have **10 and 11 code cells**, respectively; the longest code cell
  has **22 lines**. Neither notebook embeds, writes or executes source-module
  strings. The two delivered notebook files have no execution outputs.
- The real-data smoke produced two acquisition rows, 20 branch/checkpoint
  summaries, 660 retrieval/edit predictions and 1,320 language-model rows, with
  no failed branches. The separate controlled run produced six branches.
- The pilot's verified 18 MiB payload, manifest and GPT-2 tokenizer cache are
  present locally. Token IDs, complete row boundaries and the reference SHA256
  were checked. Dataset binaries and validation outputs are ignored by Git.
- The vendored attention source is byte-identical to the supplied source folder.
  Its SHA256 is `6a71b9ebef4dbfaf489ca32fb4ad32ed51a3dc474b6d681c21ddf026eef1b5d2`.
  Upstream license notices accompany it in the repository and installable package.
- Editable installation, Python package building, notebook schema checks and the
  command-line entry points were checked locally.

## Not yet verified

No Colab GPU pilot, CUDA/Triton kernel execution, multi-process GPU training,
H200 throughput/memory measurement, full corpus download, or full scientific
training run was performed here. The full notebook's required hardware preflight
compares weak/strong forgetting and nondivisible sequence lengths against the
dense reference before a GPU launch. It must pass on the chosen environment.

Full native-Zarr random access can incur substantial decompression and storage
overhead. Measure loading time and GPU utilization on cluster scratch before
choosing the final run budget. Dependency ranges are install constraints, not a
claim that every compatible CUDA/Triton version has been tested.

These results establish software behavior only. They do **not** support an
SGD/Adam generalization separation or infinite-distance retrieval. Keep failed
short acquisition, numerical stops and censored test ceilings in the research
report. A finite model-specific analytic bound and a finite language-model test
have different interpretations.

## Reproduce checks

```bash
pip install -e '.[dev,full]'
pytest -q
python scripts/check_notebooks.py
```

To exercise actual data, run the first notebook with `PROFILE="smoke"` or use
the CLI commands in the main README. The reduced full-runner tests do not
download the full dataset or start a GPU job. Development execution copies and
raw validation artifacts are retained under the Git-ignored `validation/` folder.
