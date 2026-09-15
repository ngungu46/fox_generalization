# Original FoX baseline: source audit

The primary reference is **FoX (LLaMA)**, the paper's plain Forgetting Transformer. **FoX (Pro)** adds QK normalization, key/value shifts, an output gate and output normalization; it is a separate architecture. The supplied repository's example command happens to select Pro, which does not make Pro the plain-FoX default.

Sources: [paper v2](https://arxiv.org/html/2503.02130v2), [official repository](https://github.com/zhixuan-lin/forgetting-transformer), and the user-supplied `forgetting-transformer-main` directory. [The provenance manifest](paper_baseline_provenance.json) records SHA256 hashes for every audited local source. A Git commit identity was not inferred from the downloaded directory.

## Exact smaller published reference

[`upstream_llama_125m_2b.yaml`](../configs/paper_baseline/upstream_llama_125m_2b.yaml) is a **byte-for-byte copy** of the supplied `configs/experiment/longcrawl64/forgetting_transformer/llama_125m_2b.yaml`. It is an upstream Hydra configuration, not an input to this repository's pilot runner.

| Setting | Upstream original FoX reference |
|---|---|
| Model | 12 blocks, width 768, 12 heads, head width 64, SwiGLU intermediate width 2048 |
| Vocabulary | 50,257; input embedding and output head untied |
| Attention | Data-dependent forget gate in every block; no positional embeddings or Pro components |
| Gate initialization | Gate projection weights have standard deviation 0.02; biases are zero |
| Other initialization | All linear/embedding weights have standard deviation 0.02; no residual-projection rescaling |
| Normalization | RMSNorm epsilon `1e-6` |
| Training | Pure causal language modeling on native LongCrawl64 training data |
| Context and global batch | 16,384 tokens; 32 sequences = 524,288 target tokens per update |
| Token budget | 2,684,354,560 target tokens = 5,120 optimizer updates |
| Optimizer | AdamW, peak LR `0.002`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1` |
| Exceptions and clipping | No decay for biases/RMSNorm; global gradient norm clipped at 1.0 |
| Schedule | Warmup from zero over 268,435,456 tokens (512 updates), then cosine decay to zero at the total token budget |
| Precision | BF16 mixed precision; distributed FSDP training in the supplied recipe |

These settings follow the copied YAML and its inherited model/optimizer defaults. The optimizer configuration does not override epsilon, so it uses PyTorch AdamW's `1e-8` default. The paper describes this setup in §4.1 and Appendix B.1. [Paper training details](https://arxiv.org/html/2503.02130v2)

**Size convention:** the paper excludes the input embedding when naming model sizes, while retaining the untied output head. Counting the supplied plain-FoX tensor shapes gives **123,661,968 non-embedding parameters and 162,259,344 total parameters**—approximately 124M and 162M, respectively. The earlier informal “about 163M total” referred to this larger total, not an additional model. This is an analytic count, with its formula recorded in the manifest. Our older `h200_124m` configuration uses width 640 and a different parameter budget; it is not the paper's “125M” architecture.

The main published setting is larger: 760M non-embedding parameters and roughly 48B training tokens. Its plain-FoX peak LR is `0.001`; Pro uses `0.002`. Do not substitute those main-setting rates or Pro results for the smaller reference above. [Paper, Appendix B.1](https://arxiv.org/html/2503.02130v2)

## Why the existing `paper_adamw` control is different

The earlier `original_data + paper_adamw` branch shares the forgetting-attention equation and AdamW moments, but changes gate initialization, RMSNorm epsilon, residual initialization, feed-forward rounding, data budget and objective. It uses a strongly forgetting first gate, weak later gates, weighted synthetic retrieval examples, a common acquisition checkpoint and reset continuation moments. Its name identifies an **optimizer control**, not a completed original-FoX replication.

In the actual `downscale_v1` results, that branch finished 400 continuation updates with short-task qualification false. Those measurements remain useful as diagnostics; they cannot serve as published original-FoX performance numbers or establish optimizer generalization.

## The portable baseline in this repository

The dedicated plain-FoX model factory restores the upstream architecture choices and initialization: data-dependent gates with zero bias in every block, RMSNorm epsilon `1e-6`, all linear weights at standard deviation `0.02`, no residual rescaling, no Pro components, and feed-forward width rounded as in the source. In our positive-decay notation, a zero gate bias has reference decay `softplus(0)=log(2)`; input-dependent weights still make each token's actual decay vary.

Its primary baseline trains **pure language modeling** with equal target-token weights and the original AdamW recipe. The pilot configuration scales the width, depth, context length, batch and number of tokens to an affordable run. It remains a downscaled comparison, not a reproduction of the published parameter/token budget. Porting the architecture also does not mean the complete upstream dependency stack or training implementation is being executed.

The data protocol differs as well:

- Upstream `LongCrawlDataset` visits fixed, non-overlapping chunks in a deterministic column/row order from `train.zarr`. It shifts each chunk right and inserts EOT token `50256` at the start, predicting every original chunk token.
- The portable pilot uses reproducible **random windows**, with the same EOT-shift convention. Windows can overlap and revisit tokens; an 8M-token training budget does not imply 8M unique tokens.
- The bundled pilot corpus uses disjoint documents reconstructed from native `heldout.zarr` and repurposes them into pilot train/validation/test partitions. These are not the paper's original training and validation splits.

Compare optimizer arms at the same model scale, starting weights, data-window seeds, objective and token budget. Report the full optimizer recipe when schedules or decay differ. Compare factorized gates as a separately labeled architectural intervention. Synthetic retrieval post-training, if added, must be applied consistently across comparison arms and reported separately from this pure-language-model baseline.

## Full upstream replication command — reference only

This command has **not been launched**. Run it from a checkout of the original repository after following its [dependency and data instructions](https://github.com/zhixuan-lin/forgetting-transformer/blob/main/REPRODUCE.md). The data root must contain `longcrawl64/train.zarr` and `longcrawl64/heldout.zarr` with the native dimensions; the small pilot subset is not sufficient. Select a fresh output directory for this experiment.

```bash
cd /path/to/forgetting-transformer
FOX_REFERENCE_OUT=/path/to/fresh/fox_llama_125m_2b_seed0
FOX_REFERENCE_DATA=/path/to/data
mkdir -p "$FOX_REFERENCE_OUT" "$FOX_REFERENCE_OUT/wandb"

fabric run train.py \
    --devices 4 \
    --num-nodes 1 \
    --node-rank 0 \
    --main-address localhost \
    --main-port 1234 \
    +experiment/longcrawl64/forgetting_transformer=llama_125m_2b \
    seed=0 \
    exp=replication \
    tag=fox_llama_125m_2b_seed0 \
    "output_dir=$FOX_REFERENCE_OUT" \
    "data_dir=$FOX_REFERENCE_DATA" \
    "wandb.log_dir=$FOX_REFERENCE_OUT/wandb" \
    wandb.mode=disabled \
    resume=true
```

The supplied reference sets `skip_eval: true`; use the original repository's post-training model export and evaluation procedures to reproduce its plots. Training, evaluation and a comparison against the new optimizer must be completed before claiming a replicated result.
