# Scientific design

The experiment is preserved from the original Colab prototype.
The current entry points are the two small notebooks in `notebooks/`;
implementation lives under `src/fox_experiments/`.

## 1. The question the experiment can answer

Keep the two size axes separate:

- **Target lag (r):** newer irrelevant material between the latest relevant write and the query.
- **Older prefix (N):** obsolete writes prepended before that same target. Increasing (N) must leave (r) unchanged.

Your main theorem gives SGD a **record-lag-four** boundary under its specified schedule. Annealed-epsilon Adam has a certified range (R_t) that increases with cumulative learning rate (S_t). It does **not** say one finite checkpoint retrieves at every distance. A finite test that passes at the largest lag is **right-censored**: its boundary was not found.

For a controlled checkpoint, the sufficient bound can remove the older-prefix length analytically. For a language model, a flat finite prefix curve remains finite evidence. Never label an empirical curve “infinite generalization.” Also distinguish correct greedy answers from probabilities tending to one.

### A useful extension of the training lag (R)

**Controlled model:** retain the overwrite template (O=(a,-),(a,-),(b,-),(a,+),?a). Replace recall by (C_R=(a,+),(b,-),\ldots,(b,-),?a), with (R-1) newer off-key records. Run (R=2,4,8) separately, keeping loss weights, source checkpoint, and budgets paired. Test densely near (R+1,R+2,R+3), as well as (2R,4R), and sweep old prefixes independently.

A **candidate balance**, not an established extension of the theorem, is obtained by comparing recall odds (e^{-\delta m+(R-1)h}) with the overwrite pressure (e^{-2h}): (\delta m\approx(R+1)h+\log m+O(1)). Conditional on negligible leakage and the other asymptotic requirements, this suggests a boundary near **(r=R+2)**, recovering four when (R=2). The experiment tests this hypothesis and reports measured curves; failure of this scaling is informative. The auxiliary supplied-score model's (R+1) theorem has different geometry and is not substituted for this one.

**Text model:** (r) is measured in actual GPT-2 **tokens**, not records. Keep training length fixed at 256, compare maximum training lags 64 and 128, and vary distance within each range. Do not transfer the numerical record-lag boundary to token distances. Both runs keep the same long test grid and old-prefix grid.

## 2. Which factorization is being tested?

In your draft's Eq. (2), the *local binding head* has
\[
g=\operatorname{softplus}(uv),\qquad \rho=\sigma(D-2g).
\]
The retrieval head separately has (m=c_0qp) and (h=\operatorname{softplus}(x)). Changing the factorization of (m) or replacing (h) is a different ablation.

| Arm | Local gate | Other quantities | Interpretation |
|---|---|---|---|
| Controlled factorized | (g=\mathrm{softplus}(uv)) | Keep (m=c_0qp,h=\mathrm{softplus}(x)) | Direct test of the local-gate assumption |
| Controlled direct | (g=\mathrm{softplus}(z)) | Identical at the fork, (z=uv) | Remove that assumption alone |
| Text factorized constant | First block: (g_h=\mathrm{softplus}(u_hv_h)) | Other blocks retain token-dependent gates | Transfer of the local-gate idea, without a supplied binder |
| Text direct constant | First block: (g_h=\mathrm{softplus}(b_h)) | Function-matched to factorized branch | Primary paired architecture comparison |
| Original data-dependent FoX | Every block: (a_{th}=\mathrm{softplus}(W_hx_t+b_h)) | Ordinary learned projections | Closest architecture reference |

An optional `factorized_data` mode uses (\mathrm{softplus}(u_h(W_hx_t+v_h))) in the first block. This is a distinct extension, **not exactly** the scalar (\mathrm{softplus}(uv)) theorem.

The latest draft's **Proposition 8** gives a scalar-first-gate obstruction to simultaneous linear factor growth in its specified finite full-batch model. It does not prove that every nonfactorized Transformer fails at long recall. Measure this instead of building the expected answer into the experiment.

The first-block bias starts at \(g_0=\mathrm{softplus}(1.7^2)\approx2.944\); later blocks initialize near decay .1 with learned token dependence. At a checkpoint, positive constant gate logits convert to balanced \(u=v=\sqrt{b}\). A nonpositive logit uses the function-matched \(u=1,v=b\) fallback, outside the prescribed positive symmetric region. The ordinary model still lacks the theorem's parser, masks, scalar content factors, and table-rate tail.

For matched continuation, convert an acquired direct gate to factorized coordinates without changing its function, and start both branch optimizers with empty buffers. Thereafter moments persist. The acquisition optimizer, checkpoint identity, and function-equivalence checks are recorded. This intervention differs from the paper's separate acquisition trajectories and retained acquisition moments.

## 3. Optimizers: source settings and practical adaptations

| Setting | SGD | Adam |
|---|---|---|
| Latest draft, Appendix S | (5(1+j/1000)^{-3/4}); growing sampled pair batch | (0.015(1+j/1000)^{-3/4}); full pair batch |
| Source scalar multipliers | \((1,1,1.5,1.5,.05,.1)\) for \((q,p,u,v,x,w)\) | Same |
| Source table tail | (10^{-9}(1+j/1000)^{-2}) | Same |
| Source moments / epsilon | No momentum | \((.1,.1)\), (10^{-18}e^{-2k}), signed-log arithmetic |
| Main theorem | More conservative tiny-step and growing-batch conditions | \(\beta_1^2<\beta_2\), \(\epsilon_t=\sigma_0^3e^{-ct}\), prescribed polynomial scalar and summable table rates |

**Practical primary comparison:** SGD without momentum, fixed-epsilon Adam, and annealed-epsilon Adam. Native Adam arms use the same moments **(.1,.1)** to isolate epsilon. The native epsilon schedule is (10^{-8}e^{-.01j}); the rate schedule retains the (3/4) power, with amplitudes tuned only on short validation. This slower epsilon annealing is an explicit practical modification. Native arithmetic stops before its scheduled epsilon leaves the supported normal floating-point range; it never silently replaces it by a floor or resets moments.

For the ordinary text model all representation weights continue training. The source multipliers motivate first-gate ×1.5, later-gate ×.05, output ×.1, and other weights ×1. These groups are an **analogy**, not the theorem's scalar variables. No weight decay, clipping, or momentum is added to SGD; no clipping or weight decay is added to the two native Adam arms.

Also run a **paper-style AdamW control** with moments (.9,.95), weight decay .1 excluding norms and biases, clipping 1, and warmup/cosine. The FoX study uses these broad training ingredients and real-text next-token prediction. Our tiny model, data volume, gate initialization, and mixed retrieval continuation make this a downscaled adaptation, not a reproduction of its published scores.

**Sensitivity runs:** change both native Adam arms together to (.9,.999); test epsilon-decay 0,.002,.01,.02; compare strong first-gate initialization with `g0=.1`; and include `mixture_probability=0` for a pure language-model ablation. These are separate named runs. Do not select their winner using long-test scores.

## 4. Real data and the training objective

The downloader uses a **version-pinned public mirror of LongCrawl64**, the corpus used by FoX, and saves source URLs, ranges, dtype, file hashes, and provenance. The original publisher bucket returned HTTP 404 during preparation. The [dataset release article](https://manifestai.com/articles/longcrawl64/index.html) describes its GPT-2 tokenization; the [public mirror](https://huggingface.co/datasets/clankur/longcrawl64) provides the native Zarr arrays. We create **pilot-specific train/validation/test partitions** from nonoverlapping complete 65,536-token rows reconstructed from the mirrored held-out Zarr array. These are not the original benchmark splits. Validation is divided again into tuning and confirmation blocks. For H200 confirmation, use different source ranges and re-establish a held-out benchmark split.

A convenient mirrored binary failed the independent row-alignment check and is **not used** in the delivered dataset. The native Zarr metadata and chunks determine the actual row boundaries. The source is already tokenized with GPT-2. Windows never cross a stored source-block boundary. A block is not assumed to be a verified single original web document; source preprocessing and possible near-duplicate content remain limitations. This bounded leading subset is not a representative random sample of the whole corpus. Increase disjoint source ranges and data diversity for the cluster study.

**Training phases:** (1) ordinary next-token prediction on real text; (2) short acquisition with a 50/50 mixture of real-text windows and text containing plain-language code updates; (3) paired optimizer continuation on the same mixture. Every token has next-token supervision; the final answer token in task examples has weight 128. These examples are **semi-synthetic retrieval probes**, which make the target dependence controllable with little compute. They are not naturally occurring QA labels. Random digits and keys prevent memorizing a fixed answer.

A task example interleaves natural text with `Update: Ada code = 7`, a conflicting older update, and unrelated-key updates. The query asks for the latest code. Target distance and stale-copy count are explicit experimental interventions. Target indices are never passed to the model or used as pointer supervision. Evaluation checks the original, a latest-value edit, edits of old values, and an unrelated-value edit. Greedy accuracy is over the **full 50,257-token vocabulary**, not a restricted candidate list.

## 7. Controlled mechanism: exact local-gate ablation

The mechanism model supplies the parser and signed value channel, learns Q/K and the decoder, and trains ordinary answer loss with weights .6 calibration, .2 overwrite, .2 recall. Its small practical acquisition replaces the asymptotic theorem's dimension and three-step construction. The `sgd` mechanism arm uses full-batch gradients through `torch.optim.SGD`, so it is GD rather than the theorem's growing-batch stochastic procedure. The local gate arms fork from one acquired function; their optimizer states start empty at the intervention. Positive summable Q/K rates then retain the acquired geometry approximately, with actual displacement logged.

For current nonnegative raw matching margin \(\delta\), row-norm product \(B=\max_a\|Q_a\|\max_b\|K_b\|\), and positive \(m,h,w\), use
\[
A_r=e^{4m\rho B}\frac{e^{-m\delta+(r-1)h}+e^{-h}}{1-e^{-h}},\qquad
E_r\le\sigma\!\left[-w\frac{1-A_r}{1+A_r}\right].
\]
The bound accounts for arbitrarily many older records. When assumptions fail, record an invalid bound rather than silently substituting a positive margin. Compute in log space. The largest contiguous lag with sufficient answer-error bound \(\le .01\) is a model-specific sufficient radius, with an explicit evaluation ceiling; it is not an empirical confidence interval or an interval-arithmetic proof.

Record \(m,g,h,w,\rho,m\rho,\delta,m\delta/h\), table displacement, the SGD balance residual, epsilon/moment diagnostics, and the radius at several checkpoints. A zero sufficient radius means **no positive lag certified by this bound**, not that every answer is wrong. Runs with failed acquisition and numerical stops remain visible.

## 9. Evaluate and interpret

**Retrieval:** measure full-vocabulary greedy accuracy, true-answer probability, NLL, and success on all applicable edits of a base history. Report lag-by-prefix panels and short-rule retention. The same held-out histories and edits are paired across optimizer branches. Last-query attention/content/forgetting diagnostics describe selected layers; they do not certify the mechanism theorem in an ordinary multilayer network.

**Language modeling:** report per-position next-token NLL and the same final target predicted with increasing context windows. The latter controls for variation in target difficulty. Whole-sequence perplexity alone is not a measure of retrieved distance. These tiny data panels are exploratory.

**Uncertainty:** the report resamples training seeds and base histories for paired Adam–SGD contrasts; edits stay together. One-seed pilots cannot estimate variability across training seeds. Intervals are pointwise, not simultaneous across the lag grid. For H200 confirmation use fresh seeds, held-out source ranges, at least 256 histories per panel, and predeclare primary contrasts before running them.

**Positive evidence:** at matched short accuracy and training budget, the factorized/annealed arm expands its useful lag grid over checkpoints while rejecting stale prefixes; the controlled bound and matching/forgetting ratio expand consistently. **Negative evidence:** loss of the short rule, epsilon domination, gate leakage, both optimizers reaching the test ceiling, or no separation. Every outcome should remain in the report.
