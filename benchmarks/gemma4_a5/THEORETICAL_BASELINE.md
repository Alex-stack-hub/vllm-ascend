# Gemma 4 A5 theoretical performance baseline

This document separates exact structural ratios from assumption-driven serving
estimates. It is a planning baseline, not A5 benchmark evidence.

## Baselines

- **B0**: BF16 eager autoregressive target with generic operator flow.
- **B1**: MXFP target eager before DSpark and P1/P2.
- **B2**: adaptive DSpark before device-resident FIA lengths and fused prolog.
- **B3**: current final path.

All B3/B0 values include target weight quantization, DSpark, graph/control-flow
work, the target projection change, and the A4B MegaMoe path where applicable.
B3/B1 isolates the speculative and graph/fusion stack from target
quantization. B3/B2 isolates P1/P2.

## Implemented optimization ledger

| Area | Implemented change | Exact structural effect | Primary metric |
| --- | --- | --- | --- |
| Target projection | Full-attention K=V layers compute Q+K instead of Q+K+V | 10% less QKV projection work in full layers; 2% across all QKV projections | Cube time |
| Target/draft quantization | MXFP8 attention/FC and MXFP4 MLP/expert weights | 1.94x/3.76x weight-byte ceilings; 2.98x dense and 3.17x A4B mixed-weight ceilings under the documented shares | TPOT, throughput, HBM |
| DSpark | Seven drafts, adaptive verification, confidence head | At acceptance 0.70, 2.14 accepted draft tokens and 3.14 output tokens per target step in the independence model | TPOT, output tokens/s |
| Draft context | One wide context-K QMM and one activation quantization | 5 to 1 launches for 31B; 4 to 1 for A4B; unchanged QMM FLOPs | TTFT, host gaps |
| Markov correction | Top-64 gathered candidates | 64/262144 rows, or 4096x fewer candidate rows; full vocabulary LM head unchanged | Draft latency |
| Graph/control | FULL graph plus device-resident adaptive FIA lengths | Two blocking D2H length synchronizations to zero per adaptive target step | Host gap, TPOT |
| FIA prolog | Q/K RMSNorm, RoPE, V RMSNorm and paged KV write in one AIV op | Five logical stages to one and no generic second KV write | Vector/launch time |
| A4B expert path | MegaMoe gated tanh-GELU and wave overlap | 8/128 active experts; perfect overlap ceiling 1.33x when communication is 25% | MoE/HCCL time |

The SWA/global 5:1 pattern is part of Gemma 4 rather than an incremental PR
gain. It is nevertheless essential to the engine: at 32k it reads 8.38x less
KV data than treating all layers as global. Do not add 8.38x to B3/B0 because
B0 already executes the same model architecture.

## Independent bounds

At acceptance `a=0.70`, draft count `n=7`, and total draft/target cost ratio
`r`, the independent speculative ceiling is:

```text
accepted_drafts = sum(a^i, i=1..n) = 2.1412
tokens_per_target_step = 1 + accepted_drafts = 3.1412
speedup_ceiling = tokens_per_target_step / (1 + r)
```

That gives 2.80x for 31B at `r=0.12` and 2.86x for A4B at `r=0.10`.
This ceiling assumes the multi-token verifier costs one ordinary target step;
the expected curves below realize only part of the incremental gain as
concurrency and context increase.

| Independent component | 31B | 26B-A4B | Why it is not an end-to-end multiplier |
| --- | ---: | ---: | --- |
| Mixed MXFP weight bytes | 2.98x | 3.17x | Only the weight-transfer fraction benefits |
| Ideal speculative decode | 2.80x | 2.86x | Verification utilization and draft overhead vary |
| P1 adaptive length copies | 2 to 0 | 2 to 0 | Saves host latency, not Cube work |
| P2 prolog logical stages | 5 to 1 | 5 to 1 | Applies only to 1/6 global layers |
| Context-K launches | 5 to 1 | 4 to 1 | QMM FLOPs are unchanged and this is request initialization |
| Markov candidate rows | 4096x fewer | 4096x fewer | The full-vocabulary backbone head remains |
| Perfect MoE communication overlap | n/a | 1.33x | Only the communication fraction can be hidden |

## Central expected curves

The central model uses Amdahl composition. At concurrency 1/4/8/16/32/64,
the assumed decode weight-transfer shares are 82/72/62/48/32/20%; the realized
fractions of the ideal speculative *increment* are 72/66/57/45/32/20%; and
host/control shares are 12/10/8.5/6.5/4.5/3%. Long context reduces weight and
host shares because KV attention occupies more of the step. A4B assumes
45-80% of a 25% communication term is overlapped as concurrency rises.

### Decode-heavy 128/512

Expected B3/B0 output-throughput (equivalently inverse-TPOT) curve:

| Concurrency | 1 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31B central | 5.64x | 4.61x | 3.74x | 2.84x | 2.11x | 1.63x |
| 31B planning band | 4.4-6.3x | 3.7-5.4x | 3.0-4.5x | 2.3-3.5x | 1.7-2.7x | 1.3-2.0x |
| A4B central | 6.80x | 5.67x | 4.70x | 3.62x | 2.71x | 2.10x |
| A4B planning band | 5.0-7.8x | 4.3-6.7x | 3.6-5.7x | 2.8-4.5x | 2.1-3.4x | 1.6-2.6x |

The curve falls with concurrency because BF16 weight reads and launch gaps are
already amortized by batching. A4B retains more high-concurrency benefit from
expert communication overlap. B3/B1 is the fairer comparison when production
already uses MXFP: for decode-heavy requests, the request-level central curve
is about 2.24/2.12/1.98/1.78/1.56/1.36x for C1/4/8/16/32/64.

### End-to-end request curve

These central B3/B0 values combine TTFT and decode through the stated baseline
time shares; they are not simple token-count averages.

| Workload | Decode share | C1 | C8 | C16 | C32 | C64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 31B decode 128/512 | 90% | 4.58x | 3.29x | 2.62x | 2.03x | 1.62x |
| 31B balanced 2k/256 | 65% | 2.85x | 2.37x | 2.08x | 1.77x | 1.52x |
| 31B long 8k/128 | 35% | 1.72x | 1.59x | 1.51x | 1.41x | 1.31x |
| 31B long 32k/128 | 15% | 1.24x | 1.20x | 1.18x | 1.15x | 1.12x |
| A4B decode 128/512 | 90% | 5.47x | 4.12x | 3.33x | 2.61x | 2.08x |
| A4B balanced 2k/256 | 65% | 3.35x | 2.93x | 2.62x | 2.27x | 1.96x |
| A4B long 8k/128 | 35% | 2.00x | 1.95x | 1.89x | 1.79x | 1.68x |
| A4B long 32k/128 | 15% | 1.43x | 1.47x | 1.47x | 1.46x | 1.43x |

Long-prefill workloads have modest end-to-end gains because DSpark accelerates
decode, not the original prompt pass, and global attention increasingly
dominates TTFT. For a serving trace with different phase shares, rerun
`theory.py --matrix` after replacing the workload shares.

### P1/P2-only curve

B3/B2 isolates the two newest changes. The central request-level curve ranges
from 1.108x at 31B C1 decode-heavy to 1.011x at 31B C64/32k, and from 1.102x
to 1.010x for A4B. It is largest at low concurrency and short context, where
two host synchronizations and four removed logical launch gaps per global
layer occupy the largest fraction. The gain approaches 1-3% at 32k or high
concurrency because attention/Cube work dominates. This remains valuable
because it removes serial host tail latency and should reduce variance.

## Context and acceptance curves

The architecture-only SWA/global KV-read reduction versus an all-global
counterfactual is:

| Context | 1k | 2k | 4k | 8k | 16k | 32k | 64k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KV-read reduction | 1.00x | 1.83x | 3.14x | 4.89x | 6.77x | 8.38x | 9.51x |

At seven draft tokens and `r=0.12`, the independent 31B speculative ceiling
versus acceptance is:

| Acceptance | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ideal speedup | 1.49x | 1.78x | 2.19x | 2.80x | 3.72x | 5.09x |

Acceptance is therefore the highest-leverage unknown. Report the full
distribution of accepted positions, not only its mean, because adaptive
verification and batch compaction depend on the tail.

## Prefix reuse and capacity

Prefix caching is orthogonal to the new kernel path. A 94.3% token hit rate has
an ideal prefill-work ceiling of `1/(1-0.943)=17.54x`; fixed scheduling,
suffix, block-alignment and decode costs make TTFT improvement smaller. Compare
cached and uncached TTFT directly and do not multiply 17.54x into the tables.

The shared physical KV-pool correction restores six times the allocatable
tokens for the 50 sliding + 10 global layer grouping. That is a capacity and
admission-control gain, not a per-request kernel speedup: it raises the
concurrency at which memory becomes the limiter but does not bypass
`max_num_batched_tokens`, Cube, vector, or HCCL saturation.

## Reading the curves

Use the central curve for capacity planning and the band for experiment design.
Reject rather than rationalize a measurement outside the band after confirming
identical checkpoints and scheduler limits. Replace the emitted weight, host,
communication, phase-share and speculative-realization assumptions with
profiler measurements; the formulas then become a fitted performance model
instead of a forecast.
