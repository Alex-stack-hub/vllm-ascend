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
        "total_params": 31e9,
        "active_params": 31e9,
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
        "total_params": 26e9,
        "active_params": 4e9,
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

HARDWARE = {
    "950dt": {
        "hbm_capacity_bytes": 144e9,
        "hbm_bandwidth_bytes_per_second": 4e12,
        "interconnect_bytes_per_second": 2e12,
        "mxfp8_flops": 1e15,
        "mxfp4_flops": 2e15,
        # Huawei publishes FP8/MXFP8 and MXFP4 peaks but not a BF16 peak in
        # the cited 950DT announcement. Keep the inferred 2:1 ratio explicit.
        "assumed_bf16_flops": 0.5e15,
    }
}

DENSE_HBM_EFFICIENCY = {1: 0.62, 4: 0.68, 8: 0.72, 16: 0.75, 32: 0.77, 64: 0.78}
DENSE_TP_EFFICIENCY = {1: 0.82, 4: 0.84, 8: 0.85, 16: 0.85, 32: 0.84, 64: 0.82}
A4B_HBM_EFFICIENCY = {1: 0.50, 4: 0.58, 8: 0.62, 16: 0.66, 32: 0.70, 64: 0.72}
A4B_EP_EFFICIENCY = {1: 0.75, 4: 0.72, 8: 0.68, 16: 0.64, 32: 0.60, 64: 0.58}
DENSE_PREFILL_CUBE_EFFICIENCY = {1: 0.42, 4: 0.48, 8: 0.52, 16: 0.55, 32: 0.57, 64: 0.58}
A4B_PREFILL_CUBE_EFFICIENCY = {1: 0.32, 4: 0.38, 8: 0.42, 16: 0.45, 32: 0.47, 64: 0.48}

# Closed-batch planning model. The fixed term covers scheduler and launch work
# that does not scale with prompt tokens. It is deliberately visible so a
# measured intercept can replace it without changing the roofline equations.
PREFILL_FIXED_OVERHEAD_SECONDS = 0.006
EFFECTIVE_TPOT_FLOOR_SECONDS = {
    "31b": {
        "generic_bf16": 0.00250,
        "mxfp_ar": 0.00120,
        "dspark_prefusion": 0.00075,
        "gemma4_a5_final": 0.00055,
    },
    "26b-a4b": {
        "generic_bf16": 0.00120,
        "mxfp_ar": 0.00075,
        "dspark_prefusion": 0.00048,
        "gemma4_a5_final": 0.00035,
    },
}


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
    """Return central expected curves for four explicit configurations.

    Generic BF16: eager autoregressive target with generic operator flow.
    MXFP autoregressive: quantized target before DSpark and P1/P2.
    DSpark before fusion: adaptive DSpark before device-length and prolog fusion.
    Gemma4-A5 final: current complete path.
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


def _prefill_flops(cfg: dict[str, Any], prompt_tokens: int, window: int) -> float:
    """Approximate one-request prefill FLOPs from active weights and attention."""
    sliding_layers = cfg["layers"] - cfg["full_layers"]
    sliding_q_width = cfg["q_heads"] * cfg["sliding_head_dim"]
    full_q_width = cfg["q_heads"] * cfg["full_head_dim"]
    linear_flops = 2.0 * cfg["active_params"] * prompt_tokens
    attention_flops = (
        2.0
        * prompt_tokens
        * (
            sliding_layers * sliding_q_width * min(prompt_tokens, window)
            + cfg["full_layers"] * full_q_width * prompt_tokens
        )
    )
    return linear_flops + attention_flops


def _batch_weight_bytes(cfg: dict[str, Any], concurrency: int, bytes_per_parameter: float) -> float:
    """Model the union of weights touched by one decode batch.

    Dense decoding visits all weights once per batch. For A4B, independent
    requests gradually cover more experts, so the active-weight union grows
    with concurrency and is capped by the total checkpoint size.
    """
    total = cfg["total_params"] * bytes_per_parameter
    if not cfg["moe"]:
        return total
    active_union = cfg["active_params"] * bytes_per_parameter * concurrency
    return min(total, active_union)


def absolute_serving_report(
    args: argparse.Namespace,
    bounds: dict[str, Any],
    curve: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert relative curves into a capped 950DT closed-batch forecast."""
    cfg = MODELS[args.model]
    hardware = HARDWARE[args.hardware]
    aggregate_bandwidth = args.chips * hardware["hbm_bandwidth_bytes_per_second"]
    aggregate_bf16_flops = args.chips * hardware["assumed_bf16_flops"]
    mixed_weight_ratio = 1.0 / bounds["weight_bandwidth_upper_bound"]["model_mix"]
    mixed_bytes_per_parameter = 2.0 * mixed_weight_ratio
    mixed_peak_per_chip = 1.0 / (
        cfg["mxfp4_weight_share"] / hardware["mxfp4_flops"]
        + (1.0 - cfg["mxfp4_weight_share"]) / hardware["mxfp8_flops"]
    )
    aggregate_mixed_flops = args.chips * mixed_peak_per_chip
    ideal_spec = bounds["dspark"]["ideal_decode_speedup"]
    rows: list[dict[str, Any]] = []

    def latency_ceiling(baseline: str, concurrency: int) -> float:
        floor = EFFECTIVE_TPOT_FLOOR_SECONDS[args.model][baseline]
        batch_floor = floor * (1.0 + 0.06 * math.log2(concurrency))
        return concurrency / batch_floor

    def result_values(
        prefill_seconds: float,
        decode_tokens_per_second: float,
        concurrency: int,
        output_tokens: int,
    ) -> dict[str, float]:
        tpot_seconds = concurrency / decode_tokens_per_second
        batch_latency = prefill_seconds + output_tokens * tpot_seconds
        return {
            "ttft_ms": 1000.0 * prefill_seconds,
            "tpot_ms": 1000.0 * tpot_seconds,
            "output_tokens_per_second": decode_tokens_per_second,
            "requests_per_second": concurrency / batch_latency,
            "closed_batch_latency_seconds": batch_latency,
        }

    for row in curve:
        concurrency = row["concurrency"]
        prompt_tokens = row["input_tokens"]
        output_tokens = row["output_tokens"]
        if cfg["moe"]:
            hbm_efficiency = _interpolate(A4B_HBM_EFFICIENCY, concurrency)
            topology_efficiency = _interpolate(A4B_EP_EFFICIENCY, concurrency)
            prefill_efficiency = _interpolate(A4B_PREFILL_CUBE_EFFICIENCY, concurrency)
        else:
            hbm_efficiency = _interpolate(DENSE_HBM_EFFICIENCY, concurrency)
            topology_efficiency = _interpolate(DENSE_TP_EFFICIENCY, concurrency)
            prefill_efficiency = _interpolate(DENSE_PREFILL_CUBE_EFFICIENCY, concurrency)

        bf16_batch_weight_bytes = _batch_weight_bytes(cfg, concurrency, 2.0)
        mixed_batch_weight_bytes = _batch_weight_bytes(cfg, concurrency, mixed_bytes_per_parameter)
        bf16_memory_ceiling = (
            concurrency * aggregate_bandwidth * hbm_efficiency * topology_efficiency / bf16_batch_weight_bytes
        )
        bf16_compute_ceiling = (
            aggregate_bf16_flops * prefill_efficiency * topology_efficiency / (2.0 * cfg["active_params"])
        )
        generic_decode = min(
            bf16_memory_ceiling,
            bf16_compute_ceiling,
            latency_ceiling("generic_bf16", concurrency),
        )

        quant_memory_ceiling = (
            concurrency * aggregate_bandwidth * hbm_efficiency * topology_efficiency / mixed_batch_weight_bytes
        )
        quant_compute_ceiling = (
            aggregate_mixed_flops * prefill_efficiency * topology_efficiency / (2.0 * cfg["active_params"])
        )
        factors = row["central_factors"]
        k_eq_v_target = 1.003
        quant_decode = min(
            generic_decode * factors["target_quant_decode"] * k_eq_v_target * factors["a4b_megamoe"],
            quant_memory_ceiling,
            quant_compute_ceiling,
            latency_ceiling("mxfp_ar", concurrency),
        )
        dspark_decode = min(
            quant_decode * factors["dspark_decode"],
            quant_memory_ceiling * ideal_spec,
            quant_compute_ceiling * ideal_spec,
            latency_ceiling("dspark_prefusion", concurrency),
        )
        final_decode = min(
            generic_decode * row["expected_speedup"]["b3_vs_b0_tpot_or_decode_throughput"],
            quant_memory_ceiling * ideal_spec,
            quant_compute_ceiling * ideal_spec,
            latency_ceiling("gemma4_a5_final", concurrency),
        )

        prompt_flops = _prefill_flops(cfg, prompt_tokens, args.window)
        generic_prefill = (
            concurrency * prompt_flops / (aggregate_bf16_flops * prefill_efficiency * topology_efficiency)
            + PREFILL_FIXED_OVERHEAD_SECONDS
        )
        quant_prefill_speedup = factors["target_quant_prefill"] * k_eq_v_target * factors["a4b_megamoe"]
        quant_prefill = generic_prefill / quant_prefill_speedup
        dspark_prefill = quant_prefill
        final_prefill = dspark_prefill / factors["p1_p2_prefill"]

        generic = result_values(generic_prefill, generic_decode, concurrency, output_tokens)
        quant_ar = result_values(quant_prefill, quant_decode, concurrency, output_tokens)
        dspark_prefusion = result_values(dspark_prefill, dspark_decode, concurrency, output_tokens)
        final = result_values(final_prefill, final_decode, concurrency, output_tokens)
        rows.append(
            {
                **{key: row[key] for key in ("workload", "input_tokens", "output_tokens", "concurrency")},
                "generic_bf16": generic,
                "mxfp_ar": quant_ar,
                "dspark_prefusion": dspark_prefusion,
                "gemma4_a5_final": final,
                "realized_speedup": {
                    "mxfp_ar_vs_generic_requests_per_second": (
                        quant_ar["requests_per_second"] / generic["requests_per_second"]
                    ),
                    "dspark_state_vs_generic_requests_per_second": (
                        dspark_prefusion["requests_per_second"] / generic["requests_per_second"]
                    ),
                    "final_vs_generic_ttft": generic["ttft_ms"] / final["ttft_ms"],
                    "final_vs_generic_tpot": generic["tpot_ms"] / final["tpot_ms"],
                    "final_vs_generic_requests_per_second": (
                        final["requests_per_second"] / generic["requests_per_second"]
                    ),
                },
                "active_ceiling": {
                    "generic_bf16": min(
                        (
                            (bf16_memory_ceiling, "memory"),
                            (bf16_compute_ceiling, "compute"),
                            (latency_ceiling("generic_bf16", concurrency), "effective-tpot"),
                        )
                    )[1],
                    "gemma4_a5_final": min(
                        (
                            (
                                generic_decode * row["expected_speedup"]["b3_vs_b0_tpot_or_decode_throughput"],
                                "relative-curve",
                            ),
                            (quant_memory_ceiling * ideal_spec, "memory"),
                            (quant_compute_ceiling * ideal_spec, "compute"),
                            (latency_ceiling("gemma4_a5_final", concurrency), "effective-tpot"),
                        )
                    )[1],
                },
            }
        )

    one_chip_bf16_dense = hardware["hbm_bandwidth_bytes_per_second"] / (2.0 * cfg["total_params"])
    one_chip_mixed_active = hardware["hbm_bandwidth_bytes_per_second"] / (
        mixed_bytes_per_parameter * cfg["active_params"]
    )
    return {
        "model": args.model,
        "hardware": args.hardware,
        "chips": args.chips,
        "official_per_chip_spec": {key: value for key, value in hardware.items() if key != "assumed_bf16_flops"},
        "explicit_inference": {
            "assumed_bf16_flops_per_chip": hardware["assumed_bf16_flops"],
            "basis": "one-half of the published MXFP8 peak; replace with a measured BF16 roofline",
        },
        "topology": "TP=8" if not cfg["moe"] else "TP/EP=8",
        "baselines": {
            "generic_bf16": "BF16 eager autoregressive target with generic operator flow",
            "mxfp_ar": "MXFP eager autoregressive target before speculative decoding",
            "dspark_prefusion": "adaptive DSpark before device-length and FIA-prolog fusion",
            "gemma4_a5_final": "final Gemma4 A5 path with graph and operator fusion",
        },
        "single_chip_raw_weight_roofline_tokens_per_second": {
            "bf16_total_checkpoint": one_chip_bf16_dense,
            "mixed_precision_active_weights": one_chip_mixed_active,
        },
        "assumptions": {
            "closed_batch_without_external_queueing": True,
            "prefix_cache": False,
            "scheduler_admits_all_concurrent_requests": True,
            "fixed_prefill_overhead_ms": 1000.0 * PREFILL_FIXED_OVERHEAD_SECONDS,
            "effective_tpot_floor_ms": {
                model: {baseline: 1000.0 * floor for baseline, floor in floors.items()}
                for model, floors in EFFECTIVE_TPOT_FLOOR_SECONDS.items()
            },
            "dense_hbm_efficiency": DENSE_HBM_EFFICIENCY,
            "dense_tp_efficiency": DENSE_TP_EFFICIENCY,
            "a4b_hbm_efficiency": A4B_HBM_EFFICIENCY,
            "a4b_ep_efficiency": A4B_EP_EFFICIENCY,
            "dense_prefill_cube_efficiency": DENSE_PREFILL_CUBE_EFFICIENCY,
            "a4b_prefill_cube_efficiency": A4B_PREFILL_CUBE_EFFICIENCY,
            "a4b_expert_union": "min(total weights, active weights times concurrency)",
        },
        "curve": rows,
    }


def _markdown_matrix(model: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {model} central expected speedup matrix",
        "",
        "Generic = BF16 eager AR; MXFP AR = quantized target; DSpark pre-fusion = before graph/prolog fusion; "
        "Final = current Gemma4 A5 path.",
        "All values are assumption-driven central estimates, not measurements.",
        "",
        "| Workload | C | Final/Generic request | Final/Generic TTFT | Final/Generic decode |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        speedup = row["expected_speedup"]
        lines.append(
            f"| {row['workload']} | {row['concurrency']} | "
            f"{speedup['b3_vs_b0_request']:.2f}x | "
            f"{speedup['b3_vs_b0_ttft']:.2f}x | "
            f"{speedup['b3_vs_b0_tpot_or_decode_throughput']:.2f}x |"
        )
    return "\n".join(lines)


def _markdown_absolute(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['model']} theoretical serving forecast on {report['chips']}x950DT",
        "",
        "Generic is BF16 eager AR; Final is the Gemma4 A5 MXFP + DSpark + graph/fusion path.",
        "Values are closed-batch planning estimates, not measurements.",
        "",
        "| Workload | C | Generic TTFT ms | Final TTFT ms | Generic TPOT ms | Final TPOT ms | "
        "Generic tok/s | Final tok/s | Generic req/s | Final req/s | Final/Generic req/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["curve"]:
        generic = row["generic_bf16"]
        final = row["gemma4_a5_final"]
        lines.append(
            f"| {row['workload']} | {row['concurrency']} | "
            f"{generic['ttft_ms']:.1f} | {final['ttft_ms']:.1f} | "
            f"{generic['tpot_ms']:.2f} | {final['tpot_ms']:.2f} | "
            f"{generic['output_tokens_per_second']:.0f} | {final['output_tokens_per_second']:.0f} | "
            f"{generic['requests_per_second']:.2f} | {final['requests_per_second']:.2f} | "
            f"{row['realized_speedup']['final_vs_generic_requests_per_second']:.2f}x |"
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
    parser.add_argument("--absolute", action="store_true", help="emit the capped absolute serving forecast")
    parser.add_argument("--hardware", choices=HARDWARE, default="950dt")
    parser.add_argument("--chips", type=int, default=8)
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
    if args.chips <= 0:
        parser.error("--chips must be positive")

    bounds = architecture_report(args)
    if not args.matrix and not args.absolute:
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
    if args.absolute:
        report = absolute_serving_report(args, bounds, curve)
        if args.format == "markdown":
            print(_markdown_absolute(report))
            return
        print(json.dumps(report, indent=2))
        return
    if args.format == "markdown":
        print(_markdown_matrix(args.model, curve))
        return
    print(
        json.dumps(
            {
                "model": args.model,
                "baselines": {
                    "generic_bf16": "BF16 eager autoregressive target with generic operator flow",
                    "mxfp_ar": "MXFP target eager before DSpark and P1/P2",
                    "dspark_prefusion": "adaptive DSpark before device-length and FIA-prolog fusion",
                    "gemma4_a5_final": "current final path",
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
