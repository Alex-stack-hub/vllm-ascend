# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend adaptation for the Gemma 4 DSpark draft model."""

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_dspark import Gemma4DSparkForCausalLM

from vllm_ascend.models.qwen3_dspark import align_draft_weights


class AscendGemma4DSparkForCausalLM(Gemma4DSparkForCausalLM):
    """Keep the self-contained draft in its original basis on Ascend.

    Gemma 4 DSpark checkpoints carry their own embedding and LM head. When the
    target is QuaRot-rotated, only the target auxiliary hidden states are in
    the rotated basis. The draft ``fc`` projection therefore absorbs that
    rotation while its token embedding and output head remain unchanged.
    """

    has_own_embed_tokens = True
    has_own_lm_head = True

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)
