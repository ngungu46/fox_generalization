# A100 performance settings

The original implementation was CUDA-capable, not profiled or fully optimized
on an A100. The current changes address measured wasted work and eager kernel
launch overhead; actual A100 throughput still needs to be measured on the
user's runtime.

## Use the faster path

The main notebook now defines:

```python
PERFORMANCE = PerformanceConfig(pack_random=True, compile_updates=True)
run = train(experiments["random_learned_adam"], OUTPUT, performance=PERFORMANCE)
```

`pack_random=True` removes padded slots from computation. It retains every
real record and all frequency weights; full-population Adam remains
full-population Adam. Static grouping indices are constructed once and reused.

`compile_updates=True` compiles the gradient calculation and full SGD/Adam
update together. Rates, epsilon and bias corrections enter as tensor controls,
so each new training step does not require a new graph. This targets many
small eager kernels, including Adam's full-size moment arrays. Compilation and
its first numerical audit can take several minutes per new shape/configuration;
the subsequent throughput estimate excludes this startup cost.

The first accelerated update checks signed-log gradients, loss, parameters,
and Adam buffers against eager execution. A failed audit or compiler error
does not silently switch implementations. Resume with `compile_updates=False`
to keep packing and use eager updates. Use compilation only for fixed datasets;
online random batches have changing shapes and should set it to False.

The experiment keeps FP64, signed-log gradients/moments, all data, optimizer
settings, and schedules. Switching to BF16 would change the numerical
experiment and is not used as a performance shortcut.

## What was measured locally

Default random-data seed 0 at N=128, d=4096, R=2 has 16,256 streams:

| Quantity | Count |
|---|---:|
| Valid records | 284,230 |
| Slots in the previous padded representation | 3,186,176 |
| Padding fraction | 91.1% |

The original path processes about 11.2 times as many record slots as exist.
The packed path computes only the 284,230 real records.

These are **CPU measurements**, not A100 forecasts:

| Measurement | Original | Packed | Observed ratio |
|---|---:|---:|---:|
| Full-population gradient, 4-thread median of four timed calls | 253.0 ms | 31.3 ms | 8.09× |
| Random-Adam full update, 1-thread short two-update profile | 582.4 ms | 118.9 ms | 4.90× |

The second measurement includes the unchanged eager optimizer. After packing,
its optimizer accounted for roughly half of measured CPU update time, which
motivated compiling the entire update. Actual CPU Inductor compilation and
numerical audits passed for pair, padded-random, and packed-random objectives.
CUDA Inductor and A100 speed have not been validated locally.

## Measure the actual Colab runtime

Set `RUN_PROFILE=True` and choose `PROFILE_EXPERIMENT` in the notebook. Or run:

```python
from fox_restricted import profile_experiment, format_profile

report = profile_experiment(
    experiments["random_learned_adam"],
    out_dir=OUTPUT / "profiles",
    performance=PERFORMANCE,
    warmup=5,
    steps=20,
    io_directories={"local": "/content", "drive": OUTPUT},
)
print(format_profile(report))
```

Omit `performance` to measure the original eager/padded implementation.
The report includes hardware, actual record counts, end-to-end update
throughput, synchronized phase times, diagnostics, and disposable checkpoint
writes. A compiled update is reported as one fused phase, not split into
invented gradient/optimizer times. Optional `trace=True` exports a profiler
trace. Benchmark models are disposable; saved experiment checkpoints are not
used or modified by profiling.

The default whole study is eight arms × three seeds × 200,000 updates =
**4.8 million updates**. Even a hypothetical 10 ms/update would mean about
13.3 hours of compute before diagnostics and I/O. Use the measured per-arm
times, rather than assuming that fitting on an A100 makes the whole study short.

## Continue an existing run

Use the same run tag, scientific settings and seed list. The known prior
runner release is recognized by exact source hashes. Its original manifest
and source are archived under `source_before_performance_upgrade/`, and a
journal makes the metadata transition resumable. Parameters, optimizer
buffers, RNG state, clocks and prior measurements are not reset. Unknown
scientific source changes remain rejected.

Performance options can change on resume and are logged in each checkpoint's
`performance_sessions`. Their source is retained under `performance_source/`.
CPU tests cover both switching from reference to packed execution and the
actual previous release's continuation. Floating-point reduction ordering
can produce tiny differences; GPU bitwise identity is not promised.
