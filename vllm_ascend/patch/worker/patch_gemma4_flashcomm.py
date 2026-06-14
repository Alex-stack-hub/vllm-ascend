"""Patch Gemma4DecoderLayer: chunk residual at manual add points for FC1."""
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

def _sp(): return bool(_EXTRA_CTX.flash_comm_v1_enabled)
def _chunk(x, r):
    if not _sp() or x.size(0) == r.size(0): return r
    tp, rk = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    ps = _EXTRA_CTX.pad_size
    if ps > 0: r = F.pad(r, (0, 0, 0, ps))
    return r.chunk(tp, 0)[rk]

class AscendGemma4DecoderLayer(Gemma4DecoderLayer):
    def forward(self, positions, hidden_states, residual=None,
                per_layer_input=None, **kwargs):
        if residual is not None:
            residual = _chunk(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + _chunk(hidden_states, residual)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.enable_moe_block:
            h1 = self.post_feedforward_layernorm_1(hidden_states)
            rlogits = self.router(residual)
            h2 = self.pre_feedforward_layernorm_2(residual)
            h2 = self.moe(h2, rlogits)
            h2 = self.post_feedforward_layernorm_2(h2)
            hidden_states = h1 + _chunk(h1, h2)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + _chunk(hidden_states, residual)
        if per_layer_input is not None and self.per_layer_input_gate is not None:
            import torch
            g = self.per_layer_input_gate(hidden_states)
            g = torch.nn.functional.gelu(g, approximate="tanh")
            c = self.per_layer_projection(g * per_layer_input)
            c = self.post_per_layer_input_norm(c)
            hidden_states = hidden_states + c
        return hidden_states * self.layer_scalar, None

Gemma4DecoderLayer.forward = AscendGemma4DecoderLayer.forward
