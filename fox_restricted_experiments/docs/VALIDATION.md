# Validation of the library notebook

Validated locally on 2026-09-18 with Python/PyTorch on CPU.

- **142 tests and 76 subtests passed.** These include literal two-head gradient
  checks, native SGD/Adam agreement, random-stream sampling, signed-log
  underflow cases, error-bound checks, pairing, packed reductions, compiled
  updates, profiling, and checkpoint continuation.
- All **12 executable cells** of the final readable notebook ran in CPU smoke
  mode, including actual package installation, the test suite, benchmark,
  original/packed profiles, and local/results checkpoint-write probes.
- All **eight experiment arms** completed 20 updates at N=4, d=64, R=2, seed 0;
  the smoke random dataset had 256 streams. All final statuses were
  `finite_budget_complete`. Eight PNG and eight PDF figures were generated.
- The clone/install setup was exercised against a temporary local Git remote
  using a branch, tag, and pinned commit. Imports resolved to the chosen
  checkout; tracked local edits were preserved and rejected for checkout.
  This does not imply the new folder has already been pushed to GitHub.
- The notebook validates as nbformat 4, contains eight training calls and
  eight plot calls, and contains no embedded model source.
- Full-update graph capture and pause/resume tests cover all eight arms.
  Actual CPU Inductor compilation and its numerical audit also passed for
  pair, padded-random, and packed-random objectives. The notebook smoke run
  uses eager packed execution; CUDA compilation remains to be tested on A100.
- The finite-pair model and optimizer mathematics retain their source hashes.
  Runner orchestration changed to accept the performance engine. A narrowly
  scoped, journaled source-metadata upgrade recognizes the exact prior release
  and preserves its complete training state; unknown source changes are rejected.
- Continuation was checked using the actual previous release at Git commit
  `92b72c55f868869684b8239dcc06250630cc7452` for pairs/random × SGD/Adam.
  A no-update upgrade preserved all parameter, optimizer, RNG, clock and
  measurement values exactly. Five further updates preserved RNG and clocks;
  pair trajectories were bitwise equal. Random SGD parameters differed by at
  most 1.11e-16; random Adam parameters were bitwise equal and moments differed
  by at most 1.07e-14. Archived original source hashes were verified.

Notebook SHA-256:

```text
7555f15c14c20c35a531ed0e6f0131562e17fdff236536ee6540e0cce4f9fd29
```

The smoke run validates execution, not asymptotic convergence. No A100 was
available for local execution. GPU throughput is measured by the notebook
on the actual selected Colab runtime.
