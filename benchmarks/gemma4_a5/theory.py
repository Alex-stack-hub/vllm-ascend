#!/usr/bin/env python3
"""Model architecture bounds and expected Gemma 4 A5 serving curves.

The architecture report contains exact ratios derived from model geometry.
The scenario matrix is an assumption-driven Amdahl model, not a benchmark.
Every assumption used by the matrix is emitted with the result so measured
profiles can replace it without changing the formulas.
"""

import argparse
import json
import math
from typing import Any

MODELS = {
    "31b": {
        "layers": 60,
        "full_layers": 10,
        "q_heads": 32,
        "sliding_kv_heads": 16,
        "sliding_head_dim": 256,
        "full_kv_heads": 4,
        "full_head_dim": 512,
        "draft_layers": 5,
        "mxfp4_weight_share": 0.72,
        "moe": False,
    },
    "26b-a4b": {
        "layers": 30,
        "full_layers": 5,
        "q_heads": 16,
        "sliding_kv_heads": 8,
        "sliding_head_dim": 256,
        "full_kv_heads": 2,
        "full_head_dim": 512,
        "draft_layers": 4,
        "mxfp4_weight_share": 0.80,
        "moe": True,
    },
}

CONCURRENCY_POINTS = (1, 4, 8, 16, 32, 64)
WORKLOADS = {
    "decode-128-512": {"input": 128, "output": 512, "decode_time_share": 0.90},
    "balanced-2k-256": {"input": 2048, "output": 256, "decode_time_share": 0.65},
    "long-8k-128": {"input": 8192, "output": 128, "decode_time_share": 0.35},
    "long-32k-128": {"input": 32768, "output": 128, "decode_time_share": 0.15},
}

# Central utilization assumptions for the expected curve. These values are
# deliberately conservative relative to the independent architecture ceilings.
# Replace them with measured profiler shares when A5 data is available.
DECODE_WEIGHT_FRACTION = {1: 0.82, 4: 0.72, 8: 0.62, 16: 0.48, 32: 0.32, 64: 0.20}
SPECULATIVE_GAIN_REALIZATION = {1: 0.72, 4: 0.66, 8: 0.57, 16: 0.45, 32: 0.32, 64: 0.20}
CONTROL_OVERHEAD_FRACTION = {1: 0.12, 4: 0.10, 8: 0.085, 16: 0.065, 32: 0.045, 64: 0.03}
MOE_COMM_OVERLAP = {1: 0.45, 4: 0.55, 8: 0.65, 16: 0.72, 32: 0.78, 64: 0.80}


def _validate_ratio(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        parser.error(f"{name} must be in [0, 1]")


def _interpolate(points: dict[int, float], concurrency: int) -> float:
    """Log2-interpolate a concurrency-dependent assumption."""
    keys = sorted(points)
    if concurrency <= keys[0]:
        return points[keys[0]]
    if concurrency >= keys[-1]:
        return points[keys[-1]]
    x = math.log2(concurrency)
    for left, right in zip(keys, keys[1:]):
        if left <= concurrency <= right:
            alpha = (x - math.log2(left)) / (math.log2(right) - math.log2(left))
            return points[left] + alpha * (points[right] - points[left])
    raise AssertionError("unreachable interpolation interval")


def _amdahl(fraction: float, component_speedup: float) -> float:
    return 1.0 / ((1.0 - fraction) + fraction / component_speedup)


def _harmonic_phase_speedup(prefill: float, decode: float, decode_time_share: float) -> float:
    return 1.0 / ((1.0 - decode_time_share) / prefill + decode_time_share / decode)


def architecture_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = MODELS[args.model]
    mxfp8_bits = 8.0 + 8.0 / 32.0
    mxfp4_bits = 4.0 + 8.0 / 32.0
    sliding_layers = cfg["layers"] - cfg["full_layers"]
    sliding_kv_width = cfg["sliding_kv_heads"] * cfg["sliding_head_dim"]
    full_kv_width = cfg["full_kv_heads"] * cfg["full_head_dim"]
    actual_kv_reads = (
        sliding_layers * sliding_kv_width * min(args.context, args.window)
        + cfg["full_layers"] * full_kv_width * args.context
    )
    all_global_kv_reads = (sliding_layers * sliding_kv_width + cfg["full_layers"] * full_kv_width) * args.context
    swa_read_ratio = actual_kv_reads / all_global_kv_reads
    full_q_width = cfg["q_heads"] * cfg["full_head_dim"]
    full_projection_before = full_q_width + 2 * full_kv_width
    full_projection_after = full_q_width + full_kv_width
    full_projection_ratio = full_projection_after / full_projection_before
    sliding_projection = (cfg["q_heads"] + 2 * cfg["sliding_kv_heads"]) * cfg["sliding_head_dim"]
    all_projection_before = cfg["full_layers"] * full_projection_before + sliding_layers * sliding_projection
    all_projection_after = cfg["full_layers"] * full_projection_after + sliding_layers * sliding_projection
    all_layer_projection_ratio = all_projection_after / all_projection_before
    expected_accepted = sum(args.acceptance**position for position in range(1, args.draft_tokens + 1))
    ideal_spec_speedup = (1.0 + expected_accepted) / (1.0 + args.draft_cost_ratio)
    comm_overlap_speedup = 1.0 / max(1.0 - args.comm_fraction, args.comm_fraction)
    mixed_weight_ratio = (
        cfg["mxfp4_weight_share"] * mxfp4_bits / 16.0 + (1.0 - cfg["mxfp4_weight_share"]) * mxfp8_bits / 16.0
    )

    return {
        "model": args.model,
        "weight_bandwidth_upper_bound": {
            "bf16_to_mxfp8": 16.0 / mxfp8_bits,
            "bf16_to_mxfp4": 16.0 / mxfp4_bits,
            "model_mix": 1.0 / mixed_weight_ratio,
            "assumed_mxfp4_weight_share": cfg["mxfp4_weight_share"],
        },
        "attention": {
            "swa_plus_global_kv_read_ratio_vs_all_global": swa_read_ratio,
            "swa_plus_global_kv_read_reduction": 1.0 / swa_read_ratio,
            "k_eq_v_full_layer_projection_flop_ratio": full_projection_ratio,
            "k_eq_v_all_layer_projection_flop_ratio": all_layer_projection_ratio,
            "k_eq_v_qkv_projection_upper_bound": 1.0 / all_layer_projection_ratio,
            "global_fia_prolog_logical_launches_before": 5,
            "global_fia_prolog_logical_launches_after": 1,
            "global_fia_prolog_launches_removed_per_layer": 4,
            "global_fia_prolog_layers_per_target_forward": cfg["full_layers"],
        },
        "dspark": {
            "expected_accepted_drafts_per_step": expected_accepted,
            "ideal_decode_speedup": ideal_spec_speedup,
            "context_k_qmm_launches_before": cfg["draft_layers"],
            "context_k_qmm_launches_after": 1,
            "markov_candidate_ratio": args.markov_topk / 262144.0,
            "adaptive_length_d2h_syncs_before_per_target_step": 2,
            "adaptive_length_d2h_syncs_after_per_target_step": 0,
        },
        "moe": (
            {
                "active_expert_fraction": 8.0 / 128.0,
                "communication_overlap_upper_bound": comm_overlap_speedup,
            }
            if cfg["moe"]
            else None
        ),
    }


def expected_curve(args: argparse.Namespace, bounds: dict[str, Any]) -> list[dict[str, Any]]:
    """Return central expected curves for three explicit baselines.

    B0: BF16 eager autoregressive target, generic operator flow.
    B1: MXFP target eager, before DSpark and P1/P2.
    B2: adaptive DSpark before device-length and FIA-prolog fusion.
    B3: current final path.
    """
    cfg = MODELS[args.model]
    compression = bounds["weight_bandwidth_upper_bound"]["model_mix"]
    ideal_spec = bounds["dspark"]["ideal_decode_speedup"]
    rows: list[dict[str, Any]] = []

    for workload_name, workload in WORKLOADS.items():
        input_tokens = workload["input"]
        output_tokens = workload["output"]
        decode_share = workload["decode_time_share"]
        average_decode_context = input_tokens + output_tokens / 2.0
        context_pressure = 1.0 / (1.0 + max(average_decode_context - 1024.0, 0.0) / 32768.0 * 0.8)

        for concurrency in CONCURRENCY_POINTS:
            decode_weight_fraction = _interpolate(DECODE_WEIGHT_FRACTION, concurrency) * context_pressure
            prefill_weight_fraction = max(
                0.08,
                0.58 / (1.0 + input_tokens / 8192.0) / (1.0 + 0.04 * math.log2(concurrency)),
            )
            decode_quant = _amdahl(decode_weight_fraction, compression)
            prefill_quant = _amdahl(prefill_weight_fraction, compression)

            spec_realization = _interpolate(SPECULATIVE_GAIN_REALIZATION, concurrency) * context_pressure**0.25
            spec = 1.0 + (ideal_spec - 1.0) * spec_realization

            control_fraction = _interpolate(CONTROL_OVERHEAD_FRACTION, concurrency)
            device_graph = _amdahl(control_fraction, 1.0 / (1.0 - 0.65))
            prolog_fraction = (
                (0.055 if args.model == "31b" else 0.045)
                / (1.0 + average_decode_context / 8192.0)
                / (1.0 + 0.08 * math.log2(concurrency))
            )
            fia_prolog = _amdahl(prolog_fraction, 2.0)
            p1_p2_decode = device_graph * fia_prolog
            p1_p2_prefill = _amdahl(prolog_fraction * 1.35, 2.0) * (1.0 + (device_graph - 1.0) * 0.20)

            # The 0.3% target-wide central assumption keeps the exact 2% QKV
            # subsystem ceiling from being misreported as an end-to-end gain.
            k_eq_v_target = 1.003
            moe = 1.0
            if cfg["moe"]:
                overlap = _interpolate(MOE_COMM_OVERLAP, concurrency)
                moe = 1.0 / (1.0 - args.comm_fraction * overlap)
                # Small extra allowance for fused gated activation/cache flow.
                moe *= 1.02

            b3_vs_b2_decode = p1_p2_decode
            b3_vs_b2_prefill = p1_p2_prefill
            b3_vs_b2_request = _harmonic_phase_speedup(
                b3_vs_b2_prefill,
                b3_vs_b2_decode,
                decode_share,
            )

            b3_vs_b1_decode = spec * p1_p2_decode
            b3_vs_b1_prefill = p1_p2_prefill
            b3_vs_b1_request = _harmonic_phase_speedup(
                b3_vs_b1_prefill,
                b3_vs_b1_decode,
                decode_share,
            )

            b3_vs_b0_decode = decode_quant * spec * p1_p2_decode * k_eq_v_target * moe
            b3_vs_b0_prefill = prefill_quant * p1_p2_prefill * k_eq_v_target * moe
            b3_vs_b0_request = _harmonic_phase_speedup(
                b3_vs_b0_prefill,
                b3_vs_b0_decode,
                decode_share,
            )

            rows.append(
                {
                    "workload": workload_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "concurrency": concurrency,
                    "expected_speedup": {
                        "b3_vs_b0_request": b3_vs_b0_request,
                        "b3_vs_b0_ttft": b3_vs_b0_prefill,
                        "b3_vs_b0_tpot_or_decode_throughput": b3_vs_b0_decode,
                        "b3_vs_b1_request": b3_vs_b1_request,
                        "b3_vs_b2_request": b3_vs_b2_request,
                    },
                    "central_factors": {
                        "target_quant_decode": decode_quant,
                        "target_quant_prefill": prefill_quant,
                        "dspark_decode": spec,
                        "p1_p2_decode": p1_p2_decode,
                        "p1_p2_prefill": p1_p2_prefill,
                        "a4b_megamoe": moe,
                    },
                }
            )
    return rows


def _markdown_matrix(model: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {model} central expected speedup matrix",
        "",
        "B0 = BF16 eager AR; B1 = MXFP eager AR; B2 = pre-P1/P2 adaptive DSpark; B3 = current final path.",
        "All values are assumption-driven central estimates, not measurements.",
        "",
        "| Workload | C | B3/B0 request | B3/B0 TTFT | B3/B0 decode | B3/B1 request | B3/B2 request |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = row["expected_speedup"]
        lines.append(
            f"| {row['workload']} | {row['concurrency']} | "
            f"{speedup['b3_vs_b0_request']:.2f}x | "
            f"{speedup['b3_vs_b0_ttft']:.2f}x | "
            f"{speedup['b3_vs_b0_tpot_or_decode_throughput']:.2f}x | "
            f"{speedup['b3_vs_b1_request']:.2f}x | "
            f"{speedup['b3_vs_b2_request']:.3f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--acceptance", type=float, default=0.7)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--draft-cost-ratio", type=float, default=0.12)
    parser.add_argument("--comm-fraction", type=float, default=0.25)
    parser.add_argument("--markov-topk", type=int, default=64)
    parser.add_argument("--matrix", action="store_true", help="emit the expected serving curve matrix")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()

    _validate_ratio(parser, "--acceptance", args.acceptance)
    _validate_ratio(parser, "--comm-fraction", args.comm_fraction)
    if args.context <= 0 or args.window <= 0:
        parser.error("--context and --window must be positive")
    if args.draft_tokens <= 0 or args.draft_cost_ratio < 0.0:
        parser.error("--draft-tokens must be positive and --draft-cost-ratio must be non-negative")
    if not 1 <= args.markov_topk <= 262144:
        parser.error("--markov-topk must be in [1, 262144]")

    bounds = architecture_report(args)
    if not args.matrix:
        report = {
            **bounds,
            "notes": [
                "Architecture ratios are exact for the supplied geometry and assumptions.",
                "Bounds exclude alignment, non-quantized layers, scheduler cost, and memory stalls.",
                "Speculative speedup uses independent per-position acceptance "
                "and the supplied draft/target cost ratio.",
                "Do not multiply independent ceilings into an end-to-end claim.",
                "SWA/global KV reduction is a model-architecture dividend, not an incremental PR speedup.",
                "FIA prolog launch counts are logical stages; confirm physical launches in the A5 profiler.",
            ],
        }
        print(json.dumps(report, indent=2))
        return

    curve = expected_curve(args, bounds)
    if args.format == "markdown":
        print(_markdown_matrix(args.model, curve))
        return
    print(
        json.dumps(
            {
                "model": args.model,
                "baselines": {
                    "b0": "BF16 eager autoregressive target with generic operator flow",
                    "b1": "MXFP target eager before DSpark and P1/P2",
                    "b2": "adaptive DSpark before device-length and FIA-prolog fusion",
                    "b3": "current final path",
                },
                "assumptions": {
                    "decode_weight_fraction_by_concurrency": DECODE_WEIGHT_FRACTION,
                    "speculative_gain_realization_by_concurrency": SPECULATIVE_GAIN_REALIZATION,
                    "control_overhead_fraction_by_concurrency": CONTROL_OVERHEAD_FRACTION,
                    "moe_comm_overlap_by_concurrency": MOE_COMM_OVERLAP,
                    "workload_decode_time_share": {
                        name: workload["decode_time_share"] for name, workload in WORKLOADS.items()
                    },
                    "p1_device_graph_removes_control_overhead_fraction": 0.65,
                    "p2_fia_prolog_component_speedup": 2.0,
                    "k_eq_v_target_wide_central_speedup": 1.003,
                },
                "curve": curve,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
