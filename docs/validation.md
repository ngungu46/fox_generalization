# Validation status

## Original versus factorized/direct FoX comparison — 2026-09-15

- **58 CPU tests and 22 subtests passed**. The added checks cover shared
  non-gate initialization, exactly matching direct/factorized initial logits,
  positive balanced factors, checkpoint reconstruction, the 10-branch matrix,
  signed paired differences, and explicit missing-control/missing-target results.
- `04_length_generalization_comparison.ipynb` executed all **7 code cells** in
  a fresh local kernel using the actual cached corpus. All **10 smoke branches**
  completed with zero failed arms and successful report/ZIP export. The run took
  4.09 seconds; each branch saw only 64 training tokens, so it validates software.
- The saved smoke comparison has one common non-gate initial hash, 40 paired
  branch/context rows, and all pairing statuses `ok`. Context-gain differences
  at the training length are exactly zero, as their definition requires.
- Model-specific plots and the plot of differences against paper FoX were
  visually checked. All four notebooks pass schema/Python/short-cell checks.
- The Colab preset has **not** been trained here. Its ten branches require
  40,960,000 training tokens in total. No length-generalization advantage or
  arbitrary-length guarantee is claimed from the smoke run.

See [the comparison design/code](length_generalization_comparison.md). Local
execution records are in `validation/length_generalization_notebook/`. Analysis
is separated from training in `paper_baseline/analysis.py`; earlier original-only
saved reports remain analyzable. Resuming older code still requires its original
source, as enforced by the run specification.

## Original-FoX baseline comparison — 2026-09-15

- **52 CPU tests and 22 subtests passed** after adding the separate baseline.
  New checks cover the reference architecture/initialization, retention-sign
  output and gradient equivalence, paper warmup/cosine, paired initial weights
  and data, identical evaluation targets, failed-arm reporting, and exact
  interrupted Adam resume. The four runner tests also passed after the final
  plot-axis correction.
- `03_paper_baseline_comparison.ipynb` executed all **7 short code cells** on
  the actual cached LongCrawl64 pilot in a fresh local Jupyter kernel. All four
  smoke arms completed, reports and ZIP export succeeded, and there were no
  error outputs. The final execution took 2.79 seconds and is explicitly a
  software check: each arm trained on only 64 tokens.
- The plot was visually checked. Per-position loss uses 32-token bin centers
  on the full evaluation-length axis; the training-length marker therefore
  represents the correct location. Raw per-token measurements are retained.
- The copied upstream Hydra configuration is byte-identical to the supplied
  source, and all 17 recorded source hashes were verified.
- **The 4,096,000-token-per-arm Colab preset has not been trained here.** No
  published-scale FoX replication or new optimizer generalization result is
  claimed. The previous `paper_adamw` result was an adapted control.

See [the baseline guide](paper_baseline_comparison.md) and
[reference audit](paper_baseline_reference.md). Local execution copies and
reports are under `validation/paper_baseline_notebook/`; checkpoints and raw
validation outputs remain ignored by Git.

## Saved-pilot troubleshooting — 2026-09-15

- **42 CPU tests passed** after the diagnostic and acquisition-guard changes.
  New coverage checks constant-prediction diagnosis, all-edit scoring, failed and
  nonfinite ranges, preservation of raw report bytes, stopping before optimizer
  trials, acquisition-policy resume rejection, controlled decoder feasibility,
  exact witness diagnostics, and the removal of the deterministic stale digit.
- The updated downscale notebook executed all **11 code cells** without errors
  in a fresh local Jupyter kernel, using protocol v2 and the real cached corpus
  with `PROFILE="smoke"`. Execution took 8.16 seconds. This checks software only.
  The delivered notebook has no execution outputs. Both notebooks pass schema
  and Python checks; both now have 11 code cells, each at most 30 lines.
- The actual user-supplied A100 report archive was analyzed independently of
  smoke outputs. Both text sources failed short acquisition; every final branch
  predicts a constant digit. The original acquisition was reproduced on CPU
  with NLL within 0.000013 of the archived result.
- Two additional acquisition-only v2 screens each trained on 8,000 answers.
  Both retained zero all-edit accuracy. **No successful acquisition recipe or
  optimizer length-generalization separation was validated.**
- The controlled saved states were checked with exact witness expressions.
  Adam's 99% radius is blocked by decoder confidence. SGD R=8 has a real finite
  stale-prefix failure; SGD R=2 and R=4 have sufficient radius 9.
- The acquisition override is now part of the immutable run specification;
  resumed runs cannot silently change it. Intentional unqualified continuation
  is marked in the final status as diagnostic.

See [the full diagnosis](downscale_v1_diagnosis.md). Raw records and execution
copies remain in the Git-ignored `validation/` directory. Original v1 results
were preserved; protocol v2 requires a fresh run.

## Original package validation

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

The user's saved Colab/A100 pilot has now been inspected as described above.
No CUDA/Triton kernel execution, multi-process GPU training, H200 throughput/memory
measurement, full corpus download, or full scientific training run was performed
locally. The full notebook's required hardware preflight
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
