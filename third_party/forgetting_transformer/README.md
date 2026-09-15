# Vendored upstream attention kernel

Source supplied by the user: `forgetting-transformer-main/src/forgetting_transformer/ops/forgetting_attention.py`.

Public project: <https://github.com/zhixuan-lin/forgetting-transformer>.
The supplied archive did not include a Git commit identifier, so the exact file
SHA256 is recorded instead:

`6a71b9ebef4dbfaf489ca32fb4ad32ed51a3dc474b6d681c21ddf026eef1b5d2`

The file is copied without modifications to
`src/fox_experiments/full_training/_vendor/forgetting_attention.py`.
The repository MIT license is retained in `LICENSE-MIT`. The kernel derives from
FlagAttention and retains its Apache 2.0 copyright header; the Apache license text
is included in `LICENSE-APACHE-2.0`.

This project uses the exact, causal kernel with `adaptive_threshold=None`.
Adaptive computation pruning from the supplied newer code is deliberately
disabled in the primary generalization experiments.
