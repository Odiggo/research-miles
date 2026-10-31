# Native Megatron KDA for GLM-5.3-Flash

This opt-in adapter uses MCore's native `KimiDeltaAttention` inside Miles' existing
GLM-5.3-Flash model. It replaces only KDA, not DSA, mHC, optimizer placement or the
score-centering loss implementation. Keep the existing RAM-backed optimizer and
score-centering recomputation settings.

Replace the existing model-spec pair with:

```text
--spec miles_plugins.models.glm5_next.native_kda.spec get_spec
```

The entrypoint installs the native KDA class and its HF mappings before building
the original Miles GLM spec. It retains uncached TE weights. Custom bootstraps
that need mappings before spec construction can call
`miles_plugins.models.glm5_next.native_kda.adapter.install()` explicitly. Do not
install the SkyRL or previous custom TP-KDA adapter afterward.

## Runtime requirements

Requires a Megatron-Core runtime exposing:

- `megatron.core.ssm.gated_delta_net.kda.KimiDeltaAttention` and its submodule spec;
- `megatron.core.ssm.kda_layer_config.KDALayerConfig`, including `from_config`;
- GLM two-stage gates, bounded safe gating and `kda_disable_fp8`.

These are the native KDA capabilities introduced by
[Megatron-LM #7054](https://github.com/NVIDIA/Megatron-LM/pull/7054).
The adapter does not vendor or silently upgrade Megatron. An older runtime without
these APIs fails on import. Native KDA requires the corresponding TE/FLA runtime.

The bounded integration supports CP1, packed THD inputs, and sequence parallelism
when TP > 1. Native projections own SP communication. Megatron's standard
replicated-gradient finalization must run for the two low-rank down-projections
and output norm. Inference caches are rejected.

GLM's checkpoint config supplies head counts/dimensions, convolution width, gate
bound and RMS epsilon. The local KDA config selects two-stage gates, safe gating,
SiLU convolution and BF16 projections through `kda_disable_fp8=True`; FP32 decay
parameters are preserved. The surrounding model config is not overwritten.

## Checkpoint layout

Native MCore fuses Q/K/V projections and convolutions. Each TP rank must receive
`[q_rank, k_rank, v_rank]`. A global `[q_all, k_all, v_all]` tensor cannot simply be
split into contiguous rank slices. The bridge conversion therefore packs a
rank-major representation before Miles' normal sharding and reverses it after
gathering for HF export. Beta, output projection and output norm also receive
their native-to-HF name mappings. Decay and low-rank gate names are retained.

This entrypoint is separate from the preserved SkyRL comparison implementation.
It does not change the default GLM model spec or automatically select a different
optimizer, loss, context-parallel mode or recomputation policy.


## Component evidence and limits

Tested on two B200s with actual TP2/SP, a saved 256-token fixture, real layer-0
weights and an independent FP64 recurrence reference. Full-element comparisons:

| Measurement | Native TP2 |
|---|---:|
| Output relative L2 error | 0.588442% |
| Input-gradient relative L2 error | 0.723670% |
| Aggregate parameter-gradient relative L2 error | 0.616917% |
| Worst parameter-gradient error (`dt_bias`) | 2.263822% |
| Parameter storage per GPU | 132.36 MiB |
| Incremental peak allocation per GPU | 145.36 MiB |
| Slower-rank forward median | 3.91245 ms |
| Slower-rank backward + finalization median | 4.69104 ms |

All 15 canonical parameter-gradient tensors were compared after native replicated
TP gradient finalization; replicated gradients matched between ranks and all
compared values were finite. Fused QKV mapping roundtrips passed. No tolerance was
agreed, so these results are not a numerical correctness pass. Incremental peak
includes new gradients/activations/temporaries, not total device occupancy or
optimizer states. Timings are three warmed measurements, not full-training steps.

Compared with the earlier SkyRL TP2 component, native KDA had lower input-gradient
error and a lower measured backward time, but about 4 MiB more incremental peak
allocation. Different allocations and MCore versions prevent attributing changes
solely to KDA. No long-sequence speed, maximum batch, TP8, full model, optimizer
update, checkpoint/export or weight-sync validation was performed.

The test used an already cached native MCore source tree through a process-local
Python path because the installed training-image MCore predates this API. That
newer tree emitted `te_checkpoint` import warnings for optional Miles bridge,
Nemotron and MIMO plugins. The selected component ran, but this does not establish
full-stack dependency compatibility. No production runtime was upgraded. The two
GPU allocation was stopped after results were saved. The prior SkyRL entrypoint
remains available; the default Miles GLM path is unchanged.
