import importlib.util
import math
from argparse import Namespace
from pathlib import Path

THEORY_PATH = Path(__file__).parents[3] / "benchmarks" / "gemma4_a5" / "theory.py"
SPEC = importlib.util.spec_from_file_location("gemma4_a5_theory", THEORY_PATH)
assert SPEC is not None and SPEC.loader is not None
THEORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THEORY)


def _args(model: str) -> Namespace:
    return Namespace(
        model=model,
        context=32768,
        window=1024,
        acceptance=0.7,
        draft_tokens=7,
        draft_cost_ratio=0.10 if model == "26b-a4b" else 0.12,
        comm_fraction=0.25,
        markov_topk=64,
        hardware="950dt",
        chips=8,
    )


def test_absolute_forecast_covers_standard_workload_matrix() -> None:
    args = _args("31b")
    bounds = THEORY.architecture_report(args)
    curve = THEORY.expected_curve(args, bounds)
    report = THEORY.absolute_serving_report(args, bounds, curve)

    assert len(report["curve"]) == len(THEORY.WORKLOADS) * len(THEORY.CONCURRENCY_POINTS)
    assert report["chips"] == 8
    assert report["topology"] == "TP=8"
    assert all(row["b0"]["ttft_ms"] > 0 for row in report["curve"])
    assert all(row["b3"]["output_tokens_per_second"] > 0 for row in report["curve"])


def test_a4b_single_stream_forecast_obeys_step_latency_floor() -> None:
    args = _args("26b-a4b")
    bounds = THEORY.architecture_report(args)
    curve = THEORY.expected_curve(args, bounds)
    report = THEORY.absolute_serving_report(args, bounds, curve)
    single_stream_rows = [row for row in report["curve"] if row["concurrency"] == 1]
    expected_ceiling = 1.0 / THEORY.B3_SINGLE_STREAM_STEP_FLOOR_SECONDS[args.model]

    assert report["topology"] == "TP/EP=8"
    assert single_stream_rows
    for row in single_stream_rows:
        assert row["b3"]["output_tokens_per_second"] <= expected_ceiling
        assert math.isclose(
            row["b3"]["tpot_ms"],
            1000.0 / row["b3"]["output_tokens_per_second"],
        )


def test_closed_batch_request_rate_is_derived_from_prefill_and_decode() -> None:
    args = _args("31b")
    bounds = THEORY.architecture_report(args)
    curve = THEORY.expected_curve(args, bounds)
    report = THEORY.absolute_serving_report(args, bounds, curve)

    for row in report["curve"]:
        for baseline in ("b0", "b3"):
            values = row[baseline]
            expected_latency = (
                values["ttft_ms"] / 1000.0
                + row["output_tokens"] * values["tpot_ms"] / 1000.0
            )
            assert math.isclose(values["closed_batch_latency_seconds"], expected_latency)
            assert math.isclose(
                values["requests_per_second"],
                row["concurrency"] / expected_latency,
            )
