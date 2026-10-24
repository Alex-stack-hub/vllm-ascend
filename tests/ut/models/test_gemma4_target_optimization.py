# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm_ascend.models.gemma4 import (
    AscendGemma4Attention,
    _gemma4_ascend_weights_mapper,
)
from vllm_ascend.models.gemma4_dspark import AscendGemma4DSparkModel


class _Linear(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output
        self.calls = 0

    def forward(self, _hidden_states):
        self.calls += 1
        return self.output, None


class _Scale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, value):
        return value * self.scale


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = None
        self.value = None

    def forward(self, query, key, value):
        self.key = key
        self.value = value
        return query


def test_k_eq_v_target_projects_once_but_normalizes_k_and_v_separately():
    module = AscendGemma4Attention.__new__(AscendGemma4Attention)
    nn.Module.__init__(module)
    raw_qk = torch.tensor([[1.0, 2.0, 3.0]])
    module.qkv_proj = _Linear(raw_qk)
    module.q_size = 2
    module.kv_size = 1
    module.num_heads = 2
    module.num_kv_heads = 1
    module.head_dim = 1
    module.q_norm = _Scale(1.0)
    module.k_norm = _Scale(2.0)
    module.v_norm = _Scale(3.0)
    module.rotary_emb = lambda _positions, q, k: (q, k + 5.0)
    module.attn = _Attention()
    module.o_proj = _Linear(torch.tensor([[7.0, 8.0]]))

    output = module(torch.tensor([0]), torch.zeros(1, 4))

    assert module.qkv_proj.calls == 1
    torch.testing.assert_close(module.attn.key, torch.tensor([[11.0]]))
    torch.testing.assert_close(module.attn.value, torch.tensor([[9.0]]))
    torch.testing.assert_close(output, torch.tensor([[7.0, 8.0]]))


def test_k_eq_v_mapper_drops_only_the_duplicate_full_attention_v_shard():
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_kv_shared_layers=0,
        attention_k_eq_v=True,
        layer_types=["sliding_attention", "full_attention"],
    )
    mapper = _gemma4_ascend_weights_mapper(config)

    assert mapper._map_name_with_shard("layers.0.self_attn.v_proj.weight") == (
        "layers.0.self_attn.qkv_proj.weight",
        "v",
    )
    assert mapper._map_name_with_shard("layers.1.self_attn.k_proj.weight") == (
        "layers.1.self_attn.qkv_proj.weight",
        "k",
    )
    assert not any("layers.1.self_attn.v_proj." in source for source in mapper.orig_to_new_stacked)


def test_dspark_fuses_processed_mxfp8_context_weights_along_output_axis():
    model = AscendGemma4DSparkModel.__new__(AscendGemma4DSparkModel)
    nn.Module.__init__(model)
    wrappers = []
    for value in (1, 2, 3):
        linear = SimpleNamespace(
            _mxfp8_transformed=True,
            weight=torch.full((4, 2), value, dtype=torch.float32),
            weight_scale=torch.full((2, 2, 2), value, dtype=torch.uint8),
            bias=torch.full((2,), value, dtype=torch.float32),
        )
        wrappers.append(SimpleNamespace(linear=linear, weight=linear.weight, bias=linear.bias))
    model._context_k_proj_wrappers = wrappers

    model.prepare_fused_context_k_projection()

    assert model._fused_context_k_weight.shape == (4, 6)
    assert model._fused_context_k_scale.shape == (2, 6, 2)
    assert model._fused_context_k_bias.shape == (6,)
    torch.testing.assert_close(
        model._fused_context_k_weight[:, 2:4],
        torch.full((4, 2), 2.0),
    )


def test_dspark_keeps_unfused_fallback_until_mxfp8_weights_are_processed():
    model = AscendGemma4DSparkModel.__new__(AscendGemma4DSparkModel)
    nn.Module.__init__(model)
    model._context_k_proj_wrappers = [SimpleNamespace(linear=SimpleNamespace(_mxfp8_transformed=False))]
    model._fused_context_k_weight = None

    model.prepare_fused_context_k_projection()

    assert model._fused_context_k_weight is None
