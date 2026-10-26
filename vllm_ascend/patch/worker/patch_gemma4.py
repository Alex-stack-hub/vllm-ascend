# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch upstream Gemma 4 full-attention K=V projection on Ascend A5.

The upstream model expands one checkpoint K projection into separate K and V
shards and executes both matrix products. Gemma 4 declares that these raw
projections are equal on global-attention layers, so Ascend can project Q+K
once and still apply the model's distinct K/V normalization and K-only RoPE.

This worker patch is intentionally limited to Gemma 4 full-attention layers
that declare ``attention_k_eq_v``. Remove it after the upstream Gemma 4 model
exposes a projection hook or implements the same reduced projection.
"""

from typing import Any

import torch
from typing_extensions import TypeIs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.gemma4 import (
    Gemma4Attention,
    Gemma4ForCausalLM,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.utils.math_utils import divide

from vllm_ascend.ops.linear import (
    AscendColumnParallelLinear,
    AscendQKVParallelLinear,
)

_ORIGINAL_GEMMA4_FOR_CAUSAL_LM_INIT = Gemma4ForCausalLM.__init__


class Gemma4QKParallelLinear(AscendQKVParallelLinear):
    """Tensor-parallel Q/K projection for Gemma 4 K=V layers.

    Gemma 4 full-attention checkpoints contain one K projection and declare
    ``attention_k_eq_v``.  The generic QKV layer loads that K weight into both
    K and V shards and executes both matrix products.  This layer retains the
    QKV loader's replicated-KV-head behavior but allocates and computes only
    the Q and K shards.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size >= total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        self.output_sizes = [
            self.num_heads * head_size * tp_size,
            self.num_kv_heads * head_size * tp_size,
        ]
        AscendColumnParallelLinear.__init__(
            self,
            input_size=hidden_size,
            output_size=sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, shard_id: Any) -> TypeIs[str | None]:
        if shard_id in {"q", "k"}:
            return True
        raise ValueError(
            f"Gemma4QKParallelLinear requires separate q_proj/k_proj checkpoint shards, got shard id {shard_id}."
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size,
        }.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str) -> int | None:
        return {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
        }.get(loaded_shard_id)


class AscendGemma4Attention(Gemma4Attention):
    """Gemma 4 attention that derives normalized K and V from one projection."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qk, _ = self.qkv_proj(hidden_states)
        q, raw_kv = qk.split([self.q_size, self.kv_size], dim=-1)

        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim)))
        q = q.flatten(-2, -1)
        k = self.k_norm(raw_kv.unflatten(-1, (self.num_kv_heads, self.head_dim))).flatten(-2, -1)
        v = self.v_norm(raw_kv.unflatten(-1, (self.num_kv_heads, self.head_dim))).flatten(-2, -1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


def _get_text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _gemma4_ascend_weights_mapper(config: Any) -> WeightsMapper:
    """Build the standard Gemma 4 mapper without a duplicate V shard."""
    num_layers = int(config.num_hidden_layers)
    first_kv_shared = num_layers - int(getattr(config, "num_kv_shared_layers", 0))
    use_k_eq_v = bool(getattr(config, "attention_k_eq_v", False))
    stacked: dict[str, tuple[str, Any]] = {
        ".mlp.gate_proj.": (".mlp.gate_up_proj.", 0),
        ".mlp.up_proj.": (".mlp.gate_up_proj.", 1),
    }
    for layer_idx, layer_type in enumerate(config.layer_types[:first_kv_shared]):
        shards = (("q", "q"), ("k", "k"))
        if not (use_k_eq_v and layer_type == "full_attention"):
            shards += (("v", "v"),)
        for source_shard, target_shard in shards:
            stacked[f"layers.{layer_idx}.self_attn.{source_shard}_proj."] = (
                f"layers.{layer_idx}.self_attn.qkv_proj.",
                target_shard,
            )
    return WeightsMapper(
        orig_to_new_substr={
            f"layers.{layer_idx}.self_attn.{name}.": None
            for layer_idx in range(first_kv_shared, num_layers)
            for name in ("k_proj", "v_proj", "k_norm")
        },
        orig_to_new_stacked=stacked,
    )


def _patched_gemma4_for_causal_lm_init(
    self: Gemma4ForCausalLM,
    *,
    vllm_config: VllmConfig,
    prefix: str = "",
) -> None:
    """Install the reduced projection before the model is compiled."""
    _ORIGINAL_GEMMA4_FOR_CAUSAL_LM_INIT(
        self,
        vllm_config=vllm_config,
        prefix=prefix,
    )
    config = _get_text_config(vllm_config.model_config.hf_config)
    if not bool(getattr(config, "attention_k_eq_v", False)):
        return

    model_prefix = maybe_prefix(prefix, "model")
    for layer_idx, layer in enumerate(self.model.layers):
        if isinstance(layer, PPMissingLayer):
            continue
        if config.layer_types[layer_idx] != "full_attention":
            continue
        attn = layer.self_attn
        if attn.is_kv_shared_layer:
            continue
        layer_config = config.per_layer_config[layer_idx]
        attn.qkv_proj = Gemma4QKParallelLinear(
            hidden_size=config.hidden_size,
            head_size=layer_config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=layer_config.num_key_value_heads,
            bias=config.attention_bias,
            quant_config=vllm_config.quant_config,
            prefix=f"{model_prefix}.layers.{layer_idx}.self_attn.qkv_proj",
        )
        # Both classes are Python nn.Module types with the same instance
        # layout. Retain all initialized attention/normalization modules and
        # switch only the forward implementation before compilation starts.
        attn.__class__ = AscendGemma4Attention

    self.model.hf_to_vllm_mapper = _gemma4_ascend_weights_mapper(config)


Gemma4ForCausalLM.__init__ = _patched_gemma4_for_causal_lm_init


__all__ = [
    "AscendGemma4Attention",
    "Gemma4QKParallelLinear",
]
