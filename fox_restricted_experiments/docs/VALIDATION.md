# Validation of the library notebook

Validated locally on 2026-09-18 with Python/PyTorch on CPU.

- **76 tests and 76 subtests passed.** These include literal two-head gradient
  checks, native SGD/Adam agreement, random-stream sampling, signed-log
  underflow cases, error-bound checks, pairing, and checkpoint continuation.
- All **12 executable cells** of the final readable notebook ran in CPU smoke
  mode, including actual package installation, the test suite, and benchmark.
- All **eight experiment arms** completed 20 updates at N=4, d=64, R=2, seed 0;
  the smoke random dataset had 256 streams. All final statuses were
  `finite_budget_complete`. Eight PNG and eight PDF figures were generated.
- The clone/install setup was exercised against a temporary local Git remote
  using a branch, tag, and pinned commit. Imports resolved to the chosen
  checkout; tracked local edits were preserved and rejected for checkout.
  This does not imply the new folder has already been pushed to GitHub.
- The notebook validates as nbformat 4, contains eight training calls and
  eight plot calls, and contains no embedded model source.
- Existing finite-pair kernel source hashes remain unchanged, preserving the
  old checkpoint guard. CPU resumed trajectories match uninterrupted ones.

Notebook SHA-256:

```text
c1883d11f628df0e0d7f5fea370609701a8d09b6a83c94919c1ae1cfe26c4883
```

The smoke run validates execution, not asymptotic convergence. No A100 was
available for local execution. GPU throughput is measured by the notebook
on the actual selected Colab runtime.
