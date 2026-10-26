#!/usr/bin/env python3
"""Print architecture-derived upper bounds for the Gemma 4 A5 paths."""

import argparse
import json

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
        "moe": True,
    },
}


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
    args = parser.parse_args()
    cfg = MODELS[args.model]

    if not 0.0 <= args.acceptance <= 1.0:
        parser.error("--acceptance must be in [0, 1]")
    if not 0.0 <= args.comm_fraction <= 1.0:
        parser.error("--comm-fraction must be in [0, 1]")
    if not 1 <= args.markov_topk <= 262144:
        parser.error("--markov-topk must be in [1, 262144]")

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

    report = {
        "model": args.model,
        "weight_bandwidth_upper_bound": {
            "bf16_to_mxfp8": 16.0 / mxfp8_bits,
            "bf16_to_mxfp4": 16.0 / mxfp4_bits,
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
        "notes": [
            "Bounds exclude scale metadata alignment, non-quantized layers, scheduler cost, and memory stalls.",
            "Speculative speedup uses independent per-position acceptance and the supplied draft/target cost ratio.",
            "Measure end-to-end throughput and TPOT; do not add these factors multiplicatively.",
            "FIA prolog launch counts are logical stages; confirm physical launch counts in the A5 profiler.",
            "The adaptive graph path removes two blocking D2H length copies; "
            "its latency gain is twice the measured sync latency.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
