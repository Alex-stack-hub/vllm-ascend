#!/usr/bin/env python3
"""Reproducible OpenAI-API benchmark for the Gemma 4 A5 optimization stack."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


@dataclass
class RequestResult:
    request_id: int
    status: int
    latency_s: float
    ttft_s: float
    tpot_s: float
    prompt_tokens: int
    completion_tokens: int
    marker: str
    marker_found: bool
    foreign_marker_found: bool
    generated_text: str
    error: str | None = None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _get_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _metric_totals(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            key, value = line.rsplit(None, 1)
            name = key.split("{", 1)[0]
            if any(token in name for token in ("spec_decode", "prefix_cache", "request")):
                totals[name] = totals.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return totals


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: value - before.get(name, 0.0) for name, value in after.items() if value != before.get(name, 0.0)}


def _fit_prompt(tokenizer, target_tokens: int, marker: str, marker_at_start: bool) -> tuple[str, int]:
    instruction = (
        f"The exact verification code is {marker}. Remember it.\n"
        if marker_at_start
        else "Read the following context carefully.\n"
    )
    suffix = (
        "\nReturn only the exact verification code, without punctuation."
        if marker_at_start
        else f"\nThe exact verification code is {marker}. Return only that code."
    )
    filler_ids = tokenizer.encode(
        "Gemma four uses local sliding attention and periodic global attention. ",
        add_special_tokens=False,
    )
    fixed_ids = tokenizer.encode(instruction + suffix, add_special_tokens=False)
    wanted = max(1, target_tokens - len(fixed_ids))
    repeated = (filler_ids * math.ceil(wanted / len(filler_ids)))[:wanted]
    prompt = instruction + tokenizer.decode(repeated, skip_special_tokens=True) + suffix
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual


def _fit_shared_context(tokenizer, target_tokens: int) -> tuple[str, int]:
    prefix = "Read the following shared context carefully.\n"
    filler_ids = tokenizer.encode(
        "Gemma four uses local sliding attention and periodic global attention. ",
        add_special_tokens=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    wanted = max(1, target_tokens - len(prefix_ids))
    repeated = (filler_ids * math.ceil(wanted / len(filler_ids)))[:wanted]
    prompt = prefix + tokenizer.decode(repeated, skip_special_tokens=True)
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, actual


def _run_one(
    request_id: int,
    endpoint: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    marker: str,
    output_tokens: int,
    timeout: float,
) -> RequestResult:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        text_parts: list[str] = []
        usage: dict[str, Any] = {}
        first_token_at: float | None = None
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                event_usage = event.get("usage")
                if event_usage:
                    usage = event_usage
                choices = event.get("choices") or ()
                piece = choices[0].get("text", "") if choices else ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(piece)
        latency = time.perf_counter() - started
        ttft = (first_token_at - started) if first_token_at is not None else latency
        completion_tokens = int(usage.get("completion_tokens", 0))
        tpot = (latency - ttft) / (completion_tokens - 1) if completion_tokens > 1 else 0.0
        text = "".join(text_parts)
        return RequestResult(
            request_id=request_id,
            status=status,
            latency_s=latency,
            ttft_s=ttft,
            tpot_s=tpot,
            prompt_tokens=int(usage.get("prompt_tokens", prompt_tokens)),
            completion_tokens=completion_tokens,
            marker=marker,
            marker_found=marker in text,
            foreign_marker_found=any(found != marker for found in re.findall(r"G4A5_\d{6}", text)),
            generated_text=text,
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return RequestResult(
            request_id=request_id,
            status=0,
            latency_s=time.perf_counter() - started,
            ttft_s=math.nan,
            tpot_s=math.nan,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            marker=marker,
            marker_found=False,
            foreign_marker_found=False,
            generated_text="",
            error=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--marker-at-start", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.shared_prefix and args.marker_at_start:
        parser.error("--shared-prefix and --marker-at-start test different paths; run them separately")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    endpoint = f"{args.base_url.rstrip('/')}/v1/completions"
    metrics_url = f"{args.base_url.rstrip('/')}/metrics"

    shared_prompt, _ = _fit_shared_context(
        tokenizer, max(1, args.input_tokens - 64)
    )

    def case(index: int) -> tuple[str, int, str]:
        marker = f"G4A5_{index:06d}"
        if args.shared_prefix:
            tail = f"\nRequest {index}. Return only {marker}."
            prompt = shared_prompt + tail
            return prompt, len(tokenizer.encode(prompt, add_special_tokens=False)), marker
        prompt, count = _fit_prompt(
            tokenizer,
            args.input_tokens,
            marker,
            args.marker_at_start,
        )
        return prompt, count, marker

    for warmup_id in range(args.warmup):
        prompt, count, marker = case(-warmup_id - 1)
        result = _run_one(
            warmup_id,
            endpoint,
            args.model,
            prompt,
            count,
            marker,
            min(args.output_tokens, 16),
            args.timeout,
        )
        if result.status != 200:
            raise RuntimeError(f"warmup failed: {result}")

    before = _metric_totals(_get_text(metrics_url))
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for request_id in range(args.requests):
            prompt, count, marker = case(request_id)
            futures.append(
                pool.submit(
                    _run_one,
                    request_id,
                    endpoint,
                    args.model,
                    prompt,
                    count,
                    marker,
                    args.output_tokens,
                    args.timeout,
                )
            )
        results = [future.result() for future in futures]
    wall_time = time.perf_counter() - started
    after = _metric_totals(_get_text(metrics_url))

    successful = [result for result in results if result.status == 200]
    for result in successful:
        if result.completion_tokens == 0 and result.generated_text:
            result.completion_tokens = len(
                tokenizer.encode(result.generated_text, add_special_tokens=False)
            )
            result.tpot_s = (
                (result.latency_s - result.ttft_s) / (result.completion_tokens - 1)
                if result.completion_tokens > 1
                else 0.0
            )
    latencies = [result.latency_s for result in successful]
    ttfts = [result.ttft_s for result in successful]
    tpots = [result.tpot_s for result in successful if result.completion_tokens > 1]
    completion_tokens = sum(result.completion_tokens for result in successful)
    report = {
        "config": vars(args) | {"output": str(args.output)},
        "summary": {
            "http_success": len(successful),
            "http_errors": len(results) - len(successful),
            "marker_success": sum(result.marker_found for result in successful),
            "marker_isolation_failures": sum(result.foreign_marker_found for result in successful),
            "wall_time_s": wall_time,
            "request_throughput_rps": len(successful) / wall_time,
            "output_throughput_tps": completion_tokens / wall_time,
            "latency_mean_s": statistics.fmean(latencies) if latencies else math.nan,
            "latency_p50_s": _percentile(latencies, 0.50),
            "latency_p90_s": _percentile(latencies, 0.90),
            "latency_p99_s": _percentile(latencies, 0.99),
            "ttft_mean_s": statistics.fmean(ttfts) if ttfts else math.nan,
            "ttft_p50_s": _percentile(ttfts, 0.50),
            "ttft_p99_s": _percentile(ttfts, 0.99),
            "tpot_mean_s": statistics.fmean(tpots) if tpots else math.nan,
            "tpot_p50_s": _percentile(tpots, 0.50),
            "tpot_p99_s": _percentile(tpots, 0.99),
            "prompt_tokens_min": min((r.prompt_tokens for r in successful), default=0),
            "prompt_tokens_max": max((r.prompt_tokens for r in successful), default=0),
            "completion_tokens": completion_tokens,
        },
        "metric_delta": _metric_delta(before, after),
        "requests": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["metric_delta"], indent=2))


if __name__ == "__main__":
    main()
