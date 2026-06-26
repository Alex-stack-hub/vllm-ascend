#!/bin/bash
# Install vllm site-packages stubs for PD compatibility with v0.20.2
# Run ONCE after pip install vllm==0.20.2

VLLM_LAYERS=$(python3 -c "import vllm.model_executor.layers as m; import os; print(os.path.dirname(m.__file__))")
VLLM_DIST=$(python3 -c "import vllm.distributed.parallel_state as p; import os; print(os.path.dirname(p.__file__))")
VLLM_DIST_INIT="$VLLM_DIST/__init__.py"
VLLM_PS="$VLLM_DIST/parallel_state.py"

echo "VLLM_LAYERS=$VLLM_LAYERS"
echo "VLLM_DIST=$VLLM_DIST"

# ---- Stub module files ----
declare -A STUBS
STUBS=(
  ["$VLLM_LAYERS/deepseek_compressor.py"]="class CompressorStateCache:
    @classmethod
    def update_cache_size(cls,*a,**kw):pass
    @classmethod
    def increment_cache_size(cls):pass
    @classmethod
    def decrement_cache_size(cls):return 0"

  ["$VLLM_LAYERS/deepseek_v4_attention.py"]="class DeepseekV4IndexerCache:
    def __init__(self,*a,**kw):pass
    def get_kv_cache_spec(self,*a,**kw):pass
    def forward(self):pass
    def get_attn_backend(self):pass"

  ["$VLLM_LAYERS/mamba/linear_attn.py"]="from torch import nn
class CustomOp(nn.Module):pass
class MiniMaxText01RMSNormTP:
    def __init__(self,*a,**kw):pass
class MiniMaxM2RMSNormTP(CustomOp):
    def __init__(self,*a,**kw):pass"

  ["$VLLM_LAYERS/mamba/gdn_linear_attn.py"]="class GatedDeltaNetAttention:
    def __init__(self,*a,**kw):pass"
)

for filepath in "${!STUBS[@]}"; do
    mkdir -p "$(dirname "$filepath")"
    if [ ! -f "$filepath" ]; then
        echo "${STUBS[$filepath]}" > "$filepath"
        echo "Created: $filepath"
    else
        echo "Already exists: $filepath"
    fi
done

# ---- parallel_state.py stubs ----
if ! grep -q "get_decode_context_model_parallel_rank" "$VLLM_PS" 2>/dev/null; then
    cat >> "$VLLM_PS" << 'PYEOF'

# vllm-ascend PD stubs
def patch_tensor_parallel_group(*a,**k):pass
def get_decode_context_model_parallel_rank():
    try:
        from vllm.distributed.parallel_state import get_dcp_group
        return get_dcp_group().rank_in_group
    except: return 0
def get_decode_context_model_parallel_world_size():
    try:
        from vllm.distributed.parallel_state import get_dcp_group
        return get_dcp_group().world_size
    except: return 1
PYEOF
    echo "Added stubs to $VLLM_PS"
else
    echo "parallel_state stubs already present"
fi

# ---- distributed/__init__.py re-exports ----
if ! grep -q "get_decode_context_model_parallel_rank" "$VLLM_DIST_INIT" 2>/dev/null; then
    cat >> "$VLLM_DIST_INIT" << 'PYEOF'

from vllm.distributed.parallel_state import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    patch_tensor_parallel_group,
)
PYEOF
    echo "Added re-exports to $VLLM_DIST_INIT"
else
    echo "distributed re-exports already present"
fi

echo ""
echo "All PD stubs installed. Verify with:"
echo "  python3 -c 'from vllm.distributed import get_decode_context_model_parallel_rank; from vllm.model_executor.layers.deepseek_compressor import CompressorStateCache; print(\"OK\")'"
