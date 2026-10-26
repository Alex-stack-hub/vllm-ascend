# Gemma 4 A5 performance validation

This directory validates the text-only A5 execution stacks for
`google/gemma-4-31B-it` and `google/gemma-4-26B-A4B-it`. Run every comparison
with the same target checkpoint, prompt set, server limits, and visible NPUs.

## Implemented paths

The dense and A4B paths share these optimizations:

1. The target checkpoints use MXFP8 attention and MXFP4 dense MLP weights.
2. Gemma 4 full-attention layers project Q and K once. The raw K projection is
   normalized independently into K and V, preserving K-RoPE and V semantics.
3. FIA keeps the five sliding layers bounded to 1024 tokens and runs every
   sixth global layer over the full context.
4. The DSpark draft uses global-layer target features, MXFP8 attention and
   MXFP4 MLP weights. Context K for all draft layers is one wide MXFP8 QMM,
   with one shared activation quantization.
5. `dspark_draft_topk=64` limits sequential Markov corrections to the 64
   strongest backbone candidates. Target verification remains exact.
6. FULL graph captures target verification and the complete parallel draft
   step. Adaptive verification uses the trained confidence head to avoid
   verifying low-confidence tail positions. For Gemma4 on A5, exact adaptive
   query/KV lengths remain in persistent device buffers, removing both
   per-target-step device-to-host length synchronizations.
7. Full-attention layers use a fused FIA prolog. One AIV kernel performs Q/K
   weighted RMSNorm, standard RoPE, V's weightless RMSNorm, and the paged
   BBND K/V write. FIA then consumes Q plus the already updated cache, so the
   generic reshape-and-cache path is skipped.

The A4B path additionally enables EP and MegaMoe. MegaMoe fuses expert
dispatch, MXFP GMM, gated tanh-GELU, the down projection, and expert combine.
Its wave pipeline overlaps token movement with expert computation inside the
operator; Gemma4 has no shared-expert branch requiring a separate overlap
stream.

## Checkpoint preparation

Train each draft with the corresponding config in the private DeepSpec fork:

```bash
torchrun --nproc-per-node 8 train.py --config \
  config/dspark/dspark_gemma4_31b.py

torchrun --nproc-per-node 8 train.py --config \
  config/dspark/dspark_gemma4_26b_a4b.py
```

Quantize the exported drafts with the matching ModelSlim practice:

```bash
msmodelslim quant --model_path /models/dspark-gemma4-31b \
  --model_type gemma4_dspark \
  --config_path lab_practice/gemma4_dspark/gemma4_31b_dspark_mxfp8_mxfp4.yaml \
  --save_path /models/dspark-gemma4-31b-mxfp

msmodelslim quant --model_path /models/dspark-gemma4-26b-a4b \
  --model_type gemma4_dspark \
  --config_path lab_practice/gemma4_dspark/gemma4_26b_a4b_dspark_mxfp8_mxfp4.yaml \
  --save_path /models/dspark-gemma4-26b-a4b-mxfp
```

## Server ablation matrix

Set both target paths once, then use `launch.sh` and change only `MODE` between
runs:

| Mode | Purpose |
| --- | --- |
| `target-bf16` | eager BF16 functional and performance baseline |
| `target-mxfp` | isolates target MXFP8/MXFP4 benefit |
| `dspark-eager` | isolates DSpark acceptance without graph effects |
| `dspark-graph` | adds target/draft ACLGraph |
| `dspark-adaptive` | final path; adds confidence-based verify length |

Dense example:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TARGET_KIND=dense MODE=dspark-adaptive \
TARGET_BF16_MODEL=/models/gemma-4-31b-it \
TARGET_MXFP_MODEL=/models/gemma-4-31b-mxfp \
DRAFT_MODEL=/models/dspark-gemma4-31b-mxfp \
bash benchmarks/gemma4_a5/launch.sh
```

A4B example uses `TARGET_KIND=a4b`; the script adds EP and MegaMoe.

## Workload matrix

Run at least three steady-state repetitions after warmup. Save server logs,
`npu-smi info` samples, profiler output, and every JSON report.

| Scenario | Input/output | Concurrency | What it proves |
| --- | --- | --- | --- |
| decode latency | 128/512 | 1, 8, 16 | DSpark, graph, Markov top-k, TPOT |
| balanced serving | 2048/256 | 1, 8, 16, 32 | end-to-end throughput and queueing |
| long context | 8192, 32768/128 | 1, 4, 8 | SWA/global FIA and KV bandwidth |
| prefix reuse | 8192/128 | 8, 16 | shared physical KV pools and cache isolation |
| global retrieval | 8192/32, marker at start | 8, 16 | global-layer correctness beyond window 1024 |
| A4B saturation | 2048/256 | 16, 32, 64 | MegaMoe dispatch/compute/combine and HCCL overlap |

Example balanced run:

```bash
python benchmarks/gemma4_a5/benchmark.py \
  --tokenizer /models/gemma-4-31B-it \
  --input-tokens 2048 --output-tokens 256 \
  --requests 64 --concurrency 16 \
  --output /workspace/results/31b-dspark-adaptive-2k-256-c16.json
```

Add `--shared-prefix` for prefix-cache testing and `--marker-at-start` for
cross-window retrieval. A run passes correctness when HTTP errors are zero and
`marker_success == http_success`, and `marker_isolation_failures` is zero. Run
these two flags separately because a request-unique marker at the start cannot
also be part of one shared prefix. For ordinary performance prompts,
`marker_success` is diagnostic; use the marker-at-start case as the strict
memory test.

The report includes Prometheus counter deltas. Confirm that draft and accepted
token counters increase, calculate accepted tokens divided by draft tokens,
and compare output tokens/s with `target-mxfp`. For prefix reuse, confirm the
prefix hit counters increase while every request returns only its own marker.

## Component checks

Use these checks before accepting an end-to-end speedup:

- Target QK projection: profiler shows a smaller full-layer projection output
  (`Q+K`, not `Q+K+V`) and outputs match the unoptimized model at temperature 0.
  Measure this component against the same checkpoint on an unpatched checkout;
  both `target-bf16` and `target-mxfp` use the optimized projection in this PR.
  The implementation is a worker patch of the upstream Gemma 4 initializer;
  it activates only when `attention_k_eq_v` is true, the layer is full
  attention, and the layer does not consume shared KV.
- DSpark context projection: profiler shows one context-K QMM per request
  initialization, with N equal to `draft_layers * local_kv_width`.
- Markov top-k: compare otherwise identical drafts with the config field
  removed. Markov gathered GEMMs use K=64; acceptance loss must be measured.
- ACLGraph: logs contain target and draft graph replay after capture. No new
  graph capture occurs during the steady-state repetition. Use `FULL`, not
  `FULL_DECODE_ONLY`, and verify both prefill and adaptive verification replay.
- Adaptive verification: accepted output is identical to fixed-width exact
  rejection sampling; target query-token count decreases on mixed prompts.
  In the target-step host trace there must be no D2H copy for
  `query_start_loc` or `seq_lens` after adaptive draft redistribution.
- Fused FIA prolog: only the ten dense-model/five A4B global-attention layers
  select `und_gen_qkv_rms_norm_rope_cache`; sliding-window layers retain the
  existing path. The profiler must show one fused prolog launch per global
  layer and no separate Q/K/V normalization, RoPE, or KV-write launch for that
  layer. One shared INT32-to-INT64 slot preparation per full-attention cache
  group/forward is expected and is reused by all global layers in that group.
  Compare Q, K-cache, V-cache, and final logits against `target-mxfp` eager at
  temperature 0.
- MegaMoe: A4B logs select MegaMoe and the profiler shows fused dispatch/GMM/
  activation/combine instead of separate expert kernels.

Set `PROFILER_DIR=/workspace/profile/<mode>` before `launch.sh` to enable the
service profiler. Capture only steady-state requests:

```bash
curl -X POST http://127.0.0.1:8000/start_profile
python benchmarks/gemma4_a5/benchmark.py \
  --tokenizer /models/gemma-4-31B-it \
  --input-tokens 2048 --output-tokens 256 \
  --requests 16 --concurrency 8 \
  --output /workspace/results/profile-window.json
curl -X POST http://127.0.0.1:8000/stop_profile
```

Inspect Cube time, vector time, HCCL time, QMM launch counts, graph replay,
and host gaps in the generated trace. Keep profiler runs out of throughput
comparisons because trace collection changes latency.

## Theory report

Generate architecture-derived upper bounds before testing:

```bash
python benchmarks/gemma4_a5/theory.py 31b --context 32768 \
  --acceptance 0.70 --draft-cost-ratio 0.12
python benchmarks/gemma4_a5/theory.py 26b-a4b --context 32768 \
  --acceptance 0.70 --draft-cost-ratio 0.10 --comm-fraction 0.25
```

Generate the assumption-driven serving curves for all standard workloads and
concurrency points:

```bash
python benchmarks/gemma4_a5/theory.py 31b --matrix --format markdown
python benchmarks/gemma4_a5/theory.py 26b-a4b --draft-cost-ratio 0.10 \
  --matrix --format markdown
```

The curve model names four baselines explicitly: B0 is BF16 eager
autoregressive serving, B1 is MXFP target eager, B2 is adaptive DSpark before
the P1/P2 device-metadata and FIA-prolog changes, and B3 is the current final
path. The JSON form emits every central Amdahl assumption so A5 profiler shares
can replace them without changing the formulas. See
[`THEORETICAL_BASELINE.md`](THEORETICAL_BASELINE.md) for the complete workload
interpretation, uncertainty bands, and expected curves.

The report intentionally keeps independent bounds separate. Quantization,
speculative decoding, graph replay, and communication overlap share bottlenecks,
so their factors must not be multiplied into an end-to-end claim.

With the example assumptions above, the report gives these independent upper
bounds:

| Component | 31B dense | 26B-A4B | Interpretation |
| --- | ---: | ---: | --- |
| BF16 to MXFP8 weight bandwidth | 1.94x | 1.94x | Attention and draft projection weights only |
| BF16 to MXFP4 weight bandwidth | 3.76x | 3.76x | Quantized MLP or expert weights only |
| SWA/global KV reads at 32k | 8.38x less | 8.38x less | Versus reading full context in every layer, with each layer's real KV width |
| K=V target projection | 1.02x | 1.02x | QKV input-projection FLOP ceiling across all layers; full layers alone remove 10% |
| Draft context K launches | 5 to 1 | 4 to 1 | Launch reduction; QMM FLOPs are unchanged |
| Markov correction candidates | 4096x less | 4096x less | 64 of 262,144 vocabulary rows; backbone LM head remains full-vocabulary |
| Ideal speculative decode | 2.80x | 2.86x | Acceptance 0.70, seven drafts, draft/target cost 0.12 or 0.10 |
| Perfect communication overlap | n/a | 1.33x | Only when communication is 25% of serial MegaMoe time |
| Adaptive length D2H synchronizations | 2 to 0 | 2 to 0 | Per adaptive target step under FULL graph; removes the host blocking term |
| Global FIA prolog logical launches | 5 to 1 | 5 to 1 | Q/K/V norm, RoPE and paged KV write collapse into one launch |

Graph replay has no architecture-only multiplier: report its measured launch
time reduction and TPOT. Quantization speedups are bandwidth ceilings, while
the actual kernels can remain compute-, launch-, or synchronization-bound.

## P1/P2 acceptance matrix

Run every row on both checkpoints and retain eager output as the oracle:

| Check | Command/config delta | Pass condition |
| --- | --- | --- |
| P1 graph correctness | `MODE=dspark-adaptive`, graph mode `FULL` | temperature-0 tokens equal eager for short, 8k and 32k prompts; zero graph recapture after warmup |
| P1 adaptive boundaries | mixed acceptance requests at concurrency 1/8/16/32 | FIA receives monotonically increasing device q-boundaries ending at padded token count; no error 561002 |
| P1 host synchronization | profile one warmed target step | zero NPU-to-CPU copies for query boundaries and sequence lengths after draft compaction |
| P2 operator accuracy | operator whitebox `w_gemma4_global_d512` | Q/K/V match FP32 golden within BF16 tolerance for prefill and decode-shaped token counts |
| P2 cache isolation | concurrent unique markers plus cancellations | each response contains only its own marker; reused slots match eager; zero illegal-memory/OOM errors |
| P2 activation scope | inspect global and sliding layer traces | fused prolog only on global D512 BF16-BBND layers; sliding D256 layers unchanged |
| Combined throughput | 2k/256 and 32k/128 matrices | higher output tokens/s and no TPOT regression versus `dspark-adaptive` before P1/P2 |

The two launch/synchronization reductions above are exact structural benefits,
not end-to-end speedup claims. Derive measured contribution from the trace as
`2 * D2H_sync_latency` per adaptive target step plus four removed launch gaps
per global layer. Report this separately from Cube/Vector execution time so the
fusion and graph benefits are explainable without multiplying unrelated upper
bounds.
