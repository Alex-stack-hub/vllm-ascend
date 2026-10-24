# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend adaptation for the Gemma 4 DSpark draft model."""

import torch
import torch.nn as nn
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gemma4_dspark import Gemma4DSparkForCausalLM, Gemma4DSparkModel
from vllm.model_executor.models.utils import maybe_prefix

from vllm_ascend.models.qwen3_dspark import align_draft_weights
from vllm_ascend.ops.cv_linear import CVLinearWrapper


class AscendGemma4DSparkModel(Gemma4DSparkModel):
    """Gemma 4 draft backbone with a quantization-aware context projection.

    Upstream fuses the draft layers' K projections by concatenating their
    floating-point weights and calling ``F.linear``.  MXFP8 weights have been
    packed together with E8M0 scales, so that path both loses the scales and
    bypasses the NPU quantized matmul.  For MXFP8 checkpoints, quantize the
    normalized context once and reuse it across every draft layer's K matmul.
    """

    def _build_context_kv_buffers(self, layers_attn: list[nn.Module], has_bias: bool) -> None:
        wrappers = [CVLinearWrapper(attn.k_proj) for attn in layers_attn]
        use_mxfp8 = all(wrapper.supports_mxfp8_split for wrapper in wrappers)
        if not use_mxfp8:
            self._context_k_proj_wrappers = None
            super()._build_context_kv_buffers(layers_attn, has_bias)
            return

        self._context_k_proj_wrappers = wrappers
        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._k_norm_weights = torch.stack([attn.k_norm.weight.data for attn in layers_attn], dim=0).contiguous()
        self._v_norm_weights = torch.ones(
            len(layers_attn),
            layers_attn[0].head_dim,
            dtype=self._k_norm_weights.dtype,
            device=self._k_norm_weights.device,
        )

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wrappers = getattr(self, "_context_k_proj_wrappers", None)
        if wrappers is None:
            return super()._project_context_kv(context_states, num_ctx, num_layers, num_kv_heads, head_dim)

        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, self._hidden_norm_weight, self._rms_norm_eps)
        quantized, scale = wrappers[0].quantize(normed)
        all_k = torch.stack(
            [wrapper.matmul(quantized, scale, wrapper.bias) for wrapper in wrappers],
            dim=0,
        ).view(num_layers, num_ctx, num_kv_heads, head_dim)
        all_v = torch.empty_like(all_k)
        ops.rms_norm(all_v, all_k, self._v_norm_weights, self._rms_norm_eps)
        return all_k, all_v


class AscendGemma4DSparkForCausalLM(Gemma4DSparkForCausalLM):
    """Keep the self-contained draft in its original basis on Ascend.

    Gemma 4 DSpark checkpoints carry their own embedding and LM head. When the
    target is QuaRot-rotated, only the target auxiliary hidden states are in
    the rotated basis. The draft ``fc`` projection therefore absorbs that
    rotation while its token embedding and output head remain unchanged.
    """

    has_own_embed_tokens = True
    has_own_lm_head = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.model = AscendGemma4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
        )
        self.draft_id_to_target_id = None

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)
