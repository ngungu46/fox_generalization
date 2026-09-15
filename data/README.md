# Corpus storage

`longcrawl64_pilot/` contains the previously downloaded and verified 18 MiB
native token subset, its manifest, and the GPT-2 tokenizer cache. Data and caches
are ignored by Git. They are available locally now; cloning the repository into
Colab will download/reconstruct them again unless you reuse a Drive copy.

```bash
fox-experiment download --data-dir data/longcrawl64_pilot
```

The downloader pins a public mirror revision, checks native Zarr layout, and
reconstructs complete 65,536-token source rows. It downloads about 248 MB the
first time to create the 18 MiB subset. Source rows are storage blocks; they are
not guaranteed to correspond to single original web documents.

The pilot assigns 128 source-heldout rows to training, 8 to validation, and 8
to testing. These are disjoint **pilot partitions**, not the paper's original
benchmark splits. The full runner requires the separate native `train.zarr`
and `heldout.zarr` arrays; see `docs/full_training.md`.

Default subset SHA256:
`4f12e470db96593c281e194f25658d755c1603c3c39df9debabf869e2fc0b2a2`

Dataset source, tokenization and provenance:

- <https://manifestai.com/articles/longcrawl64/index.html>
- <https://huggingface.co/datasets/clankur/longcrawl64/tree/5e2f38e601ca7c16fa304f30536d9e2945bb2a32>

The corpus is web-derived research data. This repository does not apply its
source-code license to the underlying third-party text.
