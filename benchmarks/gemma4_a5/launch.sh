#!/usr/bin/env bash
set -euo pipefail

# Required:
#   TARGET_KIND=dense|a4b MODE=... ./launch.sh
# Set TARGET_BF16_MODEL and TARGET_MXFP_MODEL to switch ablations by MODE only.
# TARGET_MODEL remains a convenient single-checkpoint fallback.
TARGET_KIND=${TARGET_KIND:?set TARGET_KIND to dense or a4b}
MODE=${MODE:-dspark-adaptive}
TARGET_MODEL=${TARGET_MODEL:-}
TARGET_BF16_MODEL=${TARGET_BF16_MODEL:-${TARGET_MODEL}}
TARGET_MXFP_MODEL=${TARGET_MXFP_MODEL:-${TARGET_MODEL}}
DRAFT_MODEL=${DRAFT_MODEL:-}
SERVED_MODEL=${SERVED_MODEL:-gemma4}
TP=${TP:-4}
DP=${DP:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
PORT=${PORT:-8000}
PROFILER_DIR=${PROFILER_DIR:-}

export VLLM_USE_V2_MODEL_RUNNER=1

if [[ "${MODE}" == "target-bf16" ]]; then
  selected_target=${TARGET_BF16_MODEL:?set TARGET_BF16_MODEL for target-bf16}
else
  selected_target=${TARGET_MXFP_MODEL:?set TARGET_MXFP_MODEL for MXFP and DSpark modes}
fi

args=(
  vllm serve "${selected_target}"
  --served-model-name "${SERVED_MODEL}"
  --tensor-parallel-size "${TP}"
  --data-parallel-size "${DP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
  --port "${PORT}"
)

if [[ -n "${PROFILER_DIR}" ]]; then
  profiler_config=$(printf \
    '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false}' \
    "${PROFILER_DIR}")
  args+=(--profiler-config "${profiler_config}")
fi

case "${MODE}" in
  target-bf16)
    args+=(--enforce-eager)
    ;;
  target-mxfp)
    args+=(--quantization ascend --enforce-eager)
    ;;
  dspark-eager|dspark-graph|dspark-adaptive)
    : "${DRAFT_MODEL:?set DRAFT_MODEL for DSpark modes}"
    args+=(--quantization ascend)
    speculative_config=$(printf \
      '{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_sample_method":"greedy"%s}' \
      "${DRAFT_MODEL}" \
      "$([[ "${MODE}" == "dspark-adaptive" ]] && printf ',"enable_adaptive_verification":true')")
    args+=(--speculative-config "${speculative_config}")
    if [[ "${MODE}" == "dspark-eager" ]]; then
      args+=(--enforce-eager)
    else
      # Query length is 1 bonus + 7 drafts. These token buckets cover BS 1..16.
      args+=(--compilation-config
        '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,24,32,48,64,96,128]}')
    fi
    ;;
  *)
    echo "unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

if [[ "${TARGET_KIND}" == "a4b" ]]; then
  args+=(--enable-expert-parallel --additional-config '{"enable_fused_mc2":2}')
elif [[ "${TARGET_KIND}" != "dense" ]]; then
  echo "unsupported TARGET_KIND=${TARGET_KIND}" >&2
  exit 2
fi

printf 'Launching:'
printf ' %q' "${args[@]}"
printf '\n'
exec "${args[@]}"
