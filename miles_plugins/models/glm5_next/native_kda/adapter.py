"""Native Megatron KDA inside Miles; no changes to DSA, mHC, optimizer or loss."""

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention, KimiDeltaAttentionSubmodules
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.transformer.module import MegatronModule

from miles.utils.hf_utils.config import load_hf_config
from miles_plugins.models.glm5_next.kda import _get_text_config

FUSED = {
    "in_proj.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
    "conv1d.weight": ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"),
}
RENAMED = {"beta_proj.weight": "b_proj.weight", "out_proj.weight": "o_proj.weight", "out_norm.weight": "o_norm.weight"}


def pack_qkv_for_tp(values, tp_size):
    """Global loader layout: [q_rank0,k_rank0,v_rank0,q_rank1,...]."""
    if len(values) != 3 or any(t.shape != values[0].shape for t in values):
        raise ValueError("GLM KDA requires three equal-shape Q/K/V tensors")
    if tp_size < 1 or values[0].shape[0] % tp_size:
        raise ValueError("Q/K/V channels must divide TP size")
    shards = [t.chunk(tp_size, dim=0) for t in values]
    return torch.cat([shards[section][rank] for rank in range(tp_size) for section in range(3)], dim=0).contiguous()


def unpack_qkv_from_tp(value, tp_size):
    """Inverse of pack_qkv_for_tp after gathering rank-local native QKV blocks."""
    if tp_size < 1 or value.shape[0] % (3 * tp_size):
        raise ValueError("Fused QKV rows must divide 3*TP")
    rank_parts = [rank.chunk(3, dim=0) for rank in value.chunk(tp_size, dim=0)]
    return [torch.cat([parts[section] for parts in rank_parts], dim=0).contiguous() for section in range(3)]


class NativeKDAAttention(MegatronModule):
    def __init__(self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None, name=None):
        super().__init__(config=config)
        if config.context_parallel_size != 1:
            raise ValueError("This integration is scoped to CP1")
        if config.sequence_parallel != args.backend.sequence_parallel:
            raise ValueError("Miles and Megatron sequence-parallel settings must agree")
        if config.tensor_model_parallel_size > 1 and not config.sequence_parallel:
            raise ValueError("TP requires SP and native replicated-gradient finalization")
        text = _get_text_config(load_hf_config(args.hf_checkpoint))
        cfg = KDALayerConfig.from_config(config)
        cfg.linear_num_key_heads = cfg.linear_num_value_heads = text.linear_num_heads
        cfg.linear_key_head_dim = cfg.linear_value_head_dim = text.linear_head_dim
        cfg.linear_conv_kernel_dim = text.linear_conv_kernel_dim
        cfg.kda_two_stage_gates = True
        cfg.kda_safe_gate = True
        cfg.kda_lower_bound = text.linear_lower_bound
        cfg.kda_disable_fp8 = True
        cfg.layernorm_epsilon = text.rms_norm_eps
        # GLM KDA convolution uses SiLU independently of the surrounding FFN.
        cfg.activation_func = F.silu
        cfg.validate_kda()
        backend = TESpecProvider()
        column = backend.column_parallel_linear()
        sub = KimiDeltaAttentionSubmodules(
            in_proj=column,
            beta_proj=column,
            f_a_proj=backend.linear(),
            f_b_proj=column,
            g_a_proj=backend.linear(),
            g_b_proj=column,
            out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        )
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
        return self.kda(hidden_states, attention_mask=attention_mask, packed_seq_params=packed_seq_params)


_installed = False


def install():
    """Install native KDA and section-aware HF mappings before model construction."""
    global _installed
    import miles_plugins.models.glm5_next.glm5_next as spec
    from miles.backends.megatron_utils import megatron_to_hf as converters
    from miles.backends.megatron_utils.megatron_to_hf import glm5_next as export
    from miles_plugins.mbridge.glm5_next import Glm5NextBridge

    if _installed:
        if spec.Glm5NextKDAAttention is not NativeKDAAttention:
            raise RuntimeError("Another adapter replaced native KDA after installation")
        return
    prefix = "self_attention.kda."
    mappings = {**FUSED, **{name: (target,) for name, target in RENAMED.items()}}
    Glm5NextBridge._ATTENTION_MAPPING = {
        **Glm5NextBridge._ATTENTION_MAPPING,
        **{
            prefix + name: ["model.layers.{layer_number}.self_attn." + n for n in targets]
            for name, targets in mappings.items()
        },
    }
    old_import = Glm5NextBridge._weight_to_mcore_format
    old_export = Glm5NextBridge._weight_to_hf_format

    def import_weights(self, name, values):
        if any(name.endswith(prefix + key) for key in FUSED):
            return pack_qkv_for_tp(values, self.mpu.tp_size)
        return old_import(self, name, values)

    def export_weights(self, name, value):
        if any(name.endswith(prefix + key) for key in FUSED):
            return self._weight_name_mapping_mcore_to_hf(name), unpack_qkv_from_tp(value, self.mpu.tp_size)
        return old_export(self, name, value)

    Glm5NextBridge._weight_to_mcore_format = import_weights
    Glm5NextBridge._weight_to_hf_format = export_weights
    export._KDA_SUFFIX_MAPPING.update({prefix + name: "self_attn." + target for name, target in RENAMED.items()})
    old_convert = export.convert_glm5_next_to_hf

    def convert(args, name, value):
        before, separator, tail = name.partition("." + prefix)
        if separator and tail in FUSED:
            layer = before.rsplit(".", 1)[-1]
            if not layer.isdigit():
                raise ValueError("Expected numeric global layer in KDA tensor name")
            values = unpack_qkv_from_tp(value, args.tensor_model_parallel_size)
            return [
                (f"model.layers.{layer}.self_attn.{target}", t) for target, t in zip(FUSED[tail], values, strict=True)
            ]
        return old_convert(args, name, value)

    export.convert_glm5_next_to_hf = convert
    converters.convert_glm5_next_to_hf = convert
    spec.Glm5NextKDAAttention = NativeKDAAttention
    _installed = True
