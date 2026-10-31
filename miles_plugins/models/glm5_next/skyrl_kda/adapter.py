"""Opt-in SkyRL KDA replacement for Miles; optimizer and loss code are untouched."""

import copy

import transformer_engine.pytorch as te
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.transformer.module import MegatronModule

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.glm5_next.kda import _get_text_config
from miles_plugins.models.glm5_next.skyrl_kda.kda import KimiDeltaAttention, KimiDeltaAttentionSubmodules


class SkyRLKDAAttention(MegatronModule):
    """SkyRL owns sequence-parallel communication; do not inherit HF gather/scatter."""

    def __init__(self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None, name=None):
        super().__init__(config=config)
        if config.context_parallel_size != 1:
            raise ValueError("This integration is scoped to CP1")
        if config.sequence_parallel != args.backend.sequence_parallel:
            raise ValueError("Miles and Megatron sequence_parallel must agree")
        if config.tensor_model_parallel_size > 1 and not config.sequence_parallel:
            raise ValueError("TP integration requires SP for replicated-gradient finalization")
        text = _get_text_config(load_hf_config(args.hf_checkpoint))
        cfg = copy.copy(config)
        cfg.linear_num_key_heads = cfg.linear_num_value_heads = text.linear_num_heads
        cfg.linear_key_head_dim = cfg.linear_value_head_dim = text.linear_head_dim
        cfg.linear_conv_kernel_dim = text.linear_conv_kernel_dim
        cfg.kda_gate_lower_bound = text.linear_lower_bound
        cfg.layernorm_epsilon = text.rms_norm_eps
        # Preserve original Miles' BF16 KDA projections even inside hybrid FP8 training.
        cfg.fp8 = None
        cfg.fp8_param = False
        backend = TESpecProvider()
        column = backend.column_parallel_linear()
        sub = KimiDeltaAttentionSubmodules(
            q_proj=column,
            k_proj=column,
            v_proj=column,
            f_a_proj=backend.linear(),
            f_b_proj=column,
            g_a_proj=backend.linear(),
            g_b_proj=column,
            b_proj=column,
            o_proj=backend.row_parallel_linear(),
        )
        with te.fp8_autocast(enabled=False):
            self.kda = KimiDeltaAttention(
                cfg, sub, layer_number=layer_number, pg_collection=pg_collection, cp_comm_type=cp_comm_type, name=name
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        packed_seq_params=None,
        inference_context=None,
        inference_params=None,
        **kwargs,
    ):
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("This integration requires packed THD inputs")
        if inference_context is not None or inference_params is not None:
            raise ValueError("Training component only; no inference caches")
        with te.fp8_autocast(enabled=False):
            return self.kda(hidden_states, attention_mask=attention_mask, packed_seq_params=packed_seq_params)


_installed = False


def install():
    """Replace only Miles' KDA class and its three convolution checkpoint mappings."""
    global _installed
    import miles_plugins.models.glm5_next.glm5_next as spec
    from miles_plugins.mbridge.glm5_next import Glm5NextBridge

    if _installed:
        if spec.Glm5NextKDAAttention is not SkyRLKDAAttention:
            raise RuntimeError("A later patch replaced the installed SkyRL adapter")
        return
    spec.Glm5NextKDAAttention = SkyRLKDAAttention
    conv_names = tuple(f"self_attention.kda.{n}_conv1d.weight" for n in ("q", "k", "v"))
    Glm5NextBridge._ATTENTION_MAPPING = {
        **Glm5NextBridge._ATTENTION_MAPPING,
        **{
            name: ["model.layers.{layer_number}.self_attn." + name.split("self_attention.kda.", 1)[1]]
            for name in conv_names
        },
    }
    # The cached HF checkpoint and SkyRL both store [channels,1,kernel].
    # Only names change; keep Miles' existing import/export tensor conversion.
    from miles.backends.megatron_utils.megatron_to_hf import glm5_next as export

    export._KDA_SUFFIX_MAPPING.update(
        {name: "self_attn." + name.split("self_attention.kda.", 1)[1] for name in conv_names}
    )
    _installed = True
