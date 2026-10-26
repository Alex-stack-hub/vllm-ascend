# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4 A5 FIA prolog with an in-kernel paged KV-cache write."""

from importlib import import_module
from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import LayerNameType, _resolve_layer_name, direct_register_custom_op


def gemma4_fia_prolog_available() -> bool:
    """Return whether the installed ops package exports the fused prolog."""
    try:
        ops = import_module("cann_ops_transformer.ops")
    except ModuleNotFoundError as exc:
        if exc.name in ("cann_ops_transformer", "cann_ops_transformer.ops"):
            return False
        raise
    return callable(getattr(ops, "und_gen_qkv_rms_norm_rope_cache", None))


def get_gemma4_fia_prolog_context(
    layer_name: LayerNameType,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Resolve explicit cache arguments or select the existing mixed path."""
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)
    attn_state = getattr(attn_metadata, "attn_state", None)
    is_mixed_chunked_prefill = (
        getattr(attn_state, "name", None) == "ChunkedPrefill"
        and getattr(attn_metadata, "num_decodes", 0) > 0
        and getattr(attn_metadata, "num_prefills", 0) > 0
    )
    if is_mixed_chunked_prefill:
        return None
    if slot_mapping is None:
        raise RuntimeError("Gemma4 fused FIA prolog requires a slot mapping")
    if getattr(attn_layer.impl, "use_bnsd_kv_cache", False):
        raise RuntimeError("Gemma4 fused FIA prolog requires contiguous BBND KV cache")

    key_cache, value_cache = _split_bbnd_cache(kv_cache)
    slot_mapping = _prepare_slot_mapping(slot_mapping, num_tokens)
    return key_cache, value_cache, slot_mapping


def _split_bbnd_cache(kv_cache: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.dim() == 0 or kv_cache.shape[0] < 2:
            raise RuntimeError("Gemma4 fused FIA prolog requires a two-plane KV cache")
        return kv_cache[0], kv_cache[1]
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        return kv_cache[0], kv_cache[1]
    raise RuntimeError(f"Unsupported Gemma4 KV cache container: {type(kv_cache)!r}")


def _prepare_slot_mapping(slot_mapping: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Cast once per cache group/forward and pad graph-only tokens with -1."""
    forward_context = get_forward_context()
    prepared_mappings = forward_context.additional_kwargs.setdefault("gemma4_fia_slot_mappings", {})
    # Avoid Tensor.data_ptr() in the compiled model path. The forward context
    # owns the exact slot-mapping objects for this invocation, so object
    # identity is sufficient to share one cast/pad result across its layers.
    cache_key = (id(slot_mapping), num_tokens)
    prepared = prepared_mappings.get(cache_key)
    if prepared is not None:
        return prepared

    prepared = slot_mapping.to(dtype=torch.int64)
    if prepared.numel() < num_tokens:
        padding = prepared.new_full((num_tokens - prepared.numel(),), -1)
        prepared = torch.cat((prepared, padding))
    elif prepared.numel() > num_tokens:
        prepared = prepared[:num_tokens]
    prepared_mappings[cache_key] = prepared
    return prepared


def _gemma4_fia_prolog_impl(
    qk: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    norm_eps: float,
) -> torch.Tensor:
    """Normalize Q/K/V, apply RoPE, and write K/V directly into BBND cache."""
    if qk.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise RuntimeError("Gemma4 fused FIA prolog currently requires BF16 activations and KV cache")
    if slot_mapping.dtype != torch.int64 or positions.dtype != torch.int64:
        raise RuntimeError("Gemma4 fused FIA prolog requires INT64 slot mapping and positions")

    ops = import_module("cann_ops_transformer.ops")
    head_dim = qk.shape[-1] // (num_heads_q + num_heads_kv)
    query = ops.und_gen_qkv_rms_norm_rope_cache(
        qk.unflatten(-1, (num_heads_q + num_heads_kv, head_dim)),
        q_weight,
        k_weight,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        positions.flatten(),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_kv,
        num_heads_v=num_heads_kv,
        norm_eps=norm_eps,
    )
    return query.flatten(-2, -1)


def gemma4_fia_prolog_fake(
    qk: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    norm_eps: float,
) -> torch.Tensor:
    del positions, q_weight, k_weight, cos_sin_cache, key_cache, value_cache, slot_mapping, norm_eps
    head_dim = qk.shape[-1] // (num_heads_q + num_heads_kv)
    return qk.new_empty((qk.shape[0], num_heads_q * head_dim))


direct_register_custom_op(
    op_name="gemma4_fia_prolog",
    op_func=_gemma4_fia_prolog_impl,
    fake_impl=gemma4_fia_prolog_fake,
    mutates_args=["key_cache", "value_cache"],
)


def gemma4_fia_prolog(
    qk: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    norm_eps: float,
) -> torch.Tensor:
    """Invoke the registered opaque op so mutation and fake schemas apply."""
    return torch.ops.vllm.gemma4_fia_prolog(
        qk,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        num_heads_q,
        num_heads_kv,
        norm_eps,
    )


__all__ = [
    "gemma4_fia_prolog",
    "gemma4_fia_prolog_available",
    "get_gemma4_fia_prolog_context",
]
