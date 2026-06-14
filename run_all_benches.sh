#!/bin/bash
# Feature benchmark: weight_nz, async, cpu_bind, chunked_prefill
# Usage: bash run_all_benches.sh

MODEL=/home/xty/gemma4/31B
MODEL_NAME=gemma-4-31B
PORT=18005
BENCH_SCRIPT=/home/gemma4_quant_xty/vllm-ascend/bench_features.py
LOG_DIR=/home/gemma4_quant_xty/vllm-ascend
RESULT_FILE=$LOG_DIR/bench_features_results.json

echo "{}" > $RESULT_FILE

run_bench() {
    local FEATURE=$1
    local ON_OFF=$2
    local EXTRA_ENV=$3
    local EXTRA_ARGS=$4
    local LOGFILE=$LOG_DIR/bench_${FEATURE}_${ON_OFF}.log

    echo "=== $FEATURE $ON_OFF ==="
    rm -f /tmp/bench_result.json

    # Start service
    eval "$EXTRA_ENV ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=256 \
      vllm serve $MODEL --served-model-name $MODEL_NAME --tensor-parallel-size 2 \
      --enforce-eager --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
      --enable-prefix-caching $EXTRA_ARGS --port $PORT \
      > $LOGFILE 2>&1 &"
    local VPID=$!

    # Wait for startup (max 5 min)
    for i in $(seq 1 60); do
        sleep 5
        if grep -q "Application startup complete" $LOGFILE 2>/dev/null; then
            echo "  Service ready after $((i*5))s"
            break
        fi
        if ! kill -0 $VPID 2>/dev/null; then
            echo "  Service died! Check $LOGFILE"
            grep "Error\|ERROR\|RuntimeError" $LOGFILE | tail -5
            return 1
        fi
    done

    # Run benchmark
    python3 $BENCH_SCRIPT $PORT $MODEL_NAME /tmp/bench_result.json 2>&1 | tail -1
    kill $VPID 2>/dev/null
    wait $VPID 2>/dev/null
    sleep 3

    # Collect results
    if [ -f /tmp/bench_result.json ]; then
        python3 -c "
import json
with open('/tmp/bench_result.json') as f:
    r = json.load(f)
with open('$RESULT_FILE') as f:
    all_r = json.load(f)
all_r['${FEATURE}_${ON_OFF}'] = r
with open('$RESULT_FILE', 'w') as f:
    json.dump(all_r, f, indent=2)
print(f\"  Result: {r.get('throughput_tok_s', 'N/A')} tok/s, TTFT p50: {r.get('ttft_p50_ms', 'N/A')}ms\")
"
    fi
    rm -rf /root/.cache/vllm/torch_compile_cache/* /root/.cache/vllm/torch_aot_compile/* 2>/dev/null
}

# ===== weight_nz =====
run_bench "weight_nz" "ON"  "VLLM_ASCEND_ENABLE_NZ=2"  ""
run_bench "weight_nz" "OFF" "VLLM_ASCEND_ENABLE_NZ=0"  ""

# ===== Async Scheduling =====
run_bench "async_sched" "ON"  "" "--async-scheduling"
run_bench "async_sched" "OFF" "" "--no-async-scheduling"

# ===== CPU Core Binding =====
run_bench "cpu_bind" "ON"  "" ""
run_bench "cpu_bind" "OFF" "" "--additional-config {\"enable_cpu_binding\": false}"

# ===== Chunked Prefill =====
run_bench "chunked_prefill" "ON"  "" "--enable-chunked-prefill"
run_bench "chunked_prefill" "OFF" "" "--no-enable-chunked-prefill"

echo "=== ALL DONE ==="
python3 -c "
import json
with open('$RESULT_FILE') as f:
    r = json.load(f)
print(json.dumps(r, indent=2))

# Comparison table
for feat in ['weight_nz', 'async_sched', 'cpu_bind', 'chunked_prefill']:
    on = r.get(f'{feat}_ON', {})
    off = r.get(f'{feat}_OFF', {})
    t_on = on.get('throughput_tok_s', 0)
    t_off = off.get('throughput_tok_s', 0)
    ttft_on = on.get('ttft_p50_ms', 0)
    ttft_off = off.get('ttft_p50_ms', 0)
    ratio = t_on / t_off if t_off > 0 else 0
    print(f'{feat:20s}: ON={t_on:.1f} tok/s, OFF={t_off:.1f} tok/s, ratio={ratio:.2f}x | TTFT: ON={ttft_on:.0f}ms OFF={ttft_off:.0f}ms')
"
