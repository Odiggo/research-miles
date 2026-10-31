# Opt-in SkyRL KDA for GLM-5.3-Flash

This package integrates SkyRL's head-sharded KDA with Miles' Megatron GLM-5.3-Flash
model. It replaces only KDA and adds the separate Q/K/V convolution checkpoint-name
mappings. Optimizer placement, score-centering loss recomputation, DSA and the rest
of the training loop are unchanged.

## Enable

Replace the existing GLM model's `--spec` pair with:

```text
--spec miles_plugins.models.glm5_next.skyrl_kda.spec get_spec
```

The entrypoint installs the adapter before building Miles' GLM block spec and
retains the tested uncached-TE setting (`disable_parameter_transpose_cache=True`).
For a custom worker bootstrap that needs the checkpoint mappings earlier, call
`miles_plugins.models.glm5_next.skyrl_kda.adapter.install()` before constructing the
model/loading weights. Do not install the previous custom TP-KDA patch afterward.
Importing this package alone does not change the default model.

Requirements and behavior:

- Megatron/CUDA, Transformer Engine and FLA with KDA support.
- CP1 and packed THD inputs; TP > 1 requires sequence parallelism. The Miles and
  Megatron sequence-parallel settings must agree.
- SkyRL's native projections own the sequence-parallel gather/reduce-scatter;
  the adapter does not wrap them in Miles' replicated HF attention collectives.
- Megatron's normal replicated-gradient finalization must run. The low-rank
  `f_a_proj`/`g_a_proj` and shared `o_norm` weights require the SP gradient sum.
- KDA projections remain BF16 even inside a hybrid-FP8 training region: the
  adapter overrides FP8 only for its copied KDA config and disables TE FP8
  autocast around KDA construction/forward. Decay parameters stay FP32.
- Cached HF and SkyRL convolution weights both use `[channels, 1, kernel]`.
  Only their names change in the import/export mappings; no reshape is applied.
- Training only; inference caches are rejected.

Retain the existing RAM-backed optimizer and score-centering recomputation
configuration when selecting this spec. This package does not configure or
reimplement either feature.

## Existing component evidence

The integration was tested on two Modal B200s with actual TP2/SP, real layer-0
weights, a deterministic 256-token fixture, and a fixed upstream gradient.
No full model was loaded. The independent reference used a direct FP64 recurrence
and plain autograd. All output/input-gradient elements and all 15 canonical
parameter-gradient tensors were compared after native TP gradient finalization.

| Measurement | Result |
|---|---:|
| Output relative L2 error vs FP64 | 0.588442% |
| Input-gradient relative L2 error | 0.753011% |
| Aggregate parameter-gradient relative L2 error | 0.616915% |
| Worst parameter-gradient error (`dt_bias`) | 2.263822% |
| Parameter storage per GPU | 132.36 MiB |
| Incremental peak allocation per GPU | 141.36 MiB |
| Slower-rank forward median | 3.85446 ms |
| Slower-rank backward + finalization median | 5.48755 ms |

Finalized replicated gradients matched exactly between the two ranks; no compared
values were nonfinite. Incremental peak includes new gradients, activations and
temporaries above resident baseline allocations, not optimizer state or total
device occupancy. Timing uses three warmed measurements.

This establishes component execution and measured errors, not a numerical
correctness pass: no tolerance was agreed. It does not validate optimizer updates,
full-model RL, TP8, long-sequence throughput, maximum batch size, checkpoint export
or weight synchronization. At 256 tokens this TP2 measurement was slower than the
earlier TP1 measurement. The package relocation was checked on CPU; the GPU run
predates this packaging-only move.

## Source and license

`kda.py` is vendored from
[NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL), path
`skyrl/backends/skyrl_train/patches/megatron/mcore_ext/kda.py`, under the
[Apache-2.0 license](LICENSE.skyrl). The integration does not require the SkyRL
Python package. Local changes are attribution, formatting, narrow lint annotations
preserving upstream style, and an optional-config-access comment. The computational implementation is
retained. `adapter.py` and `spec.py` provide the Miles integration.
