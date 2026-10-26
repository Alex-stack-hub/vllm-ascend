# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm_ascend.ops.gemma4_fia_prolog as prolog


def test_prepare_slot_mapping_casts_once_and_pads_graph_tokens(monkeypatch):
    forward_context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(prolog, "get_forward_context", lambda: forward_context)
    slots = torch.tensor([3, 8], dtype=torch.int32)

    prepared = prolog._prepare_slot_mapping(slots, num_tokens=4)
    reused = prolog._prepare_slot_mapping(slots, num_tokens=4)

    assert prepared.dtype == torch.int64
    assert prepared.tolist() == [3, 8, -1, -1]
    assert reused is prepared


def test_fused_prolog_falls_back_for_mixed_chunked_prefill(monkeypatch):
    forward_context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(prolog, "get_forward_context", lambda: forward_context)
    metadata = SimpleNamespace(
        attn_state=SimpleNamespace(name="ChunkedPrefill"),
        num_decodes=1,
        num_prefills=1,
    )
    monkeypatch.setattr(
        prolog,
        "get_attention_context",
        lambda _: (metadata, None, None, None),
    )

    assert prolog.get_gemma4_fia_prolog_context("layer", 1) is None

    metadata.num_prefills = 0
    attn_layer = SimpleNamespace(impl=SimpleNamespace(use_bnsd_kv_cache=False))
    kv_cache = torch.empty(2, 1, 1, 1, 1)
    slot_mapping = torch.zeros(1, dtype=torch.int64)
    monkeypatch.setattr(
        prolog,
        "get_attention_context",
        lambda _: (metadata, attn_layer, kv_cache, slot_mapping),
    )
    key_cache, value_cache, prepared_slots = prolog.get_gemma4_fia_prolog_context("layer", 1)
    assert key_cache.data_ptr() == kv_cache[0].data_ptr()
    assert value_cache.data_ptr() == kv_cache[1].data_ptr()
    assert prepared_slots is slot_mapping
