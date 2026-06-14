"""Benchmark FC1 ON vs OFF for Gemma4-31B.

FC1 (FlashComm1) reduces AllReduce to ReduceScatter+AllGather,
benefiting prefill throughput when num_tokens > 1000 (dense threshold).

We use concurrent long-prompt requests to ensure the prefill batch
exceeds 1000 tokens, triggering FC1.
"""
import time, json, asyncio, sys, statistics
import aiohttp

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18005
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma-4-31B"

# ~350 tokens per request, 3 concurrent = ~1050 tokens > 1000 threshold
PROMPT = (
    "Provide a comprehensive technical overview of transformer neural network "
    "architectures. Cover the following topics in depth: attention mechanisms "
    "including scaled dot-product attention, multi-head attention, and the "
    "quadratic complexity challenge; positional encodings including absolute, "
    "relative, rotary (RoPE), and ALiBi; the encoder-decoder architecture and "
    "its variants; normalization techniques including layer normalization, "
    "RMS normalization, and pre-norm vs post-norm placement; feed-forward "
    "networks and activation functions including GELU, SiLU, and SwiGLU; "
    "training techniques including learning rate schedules, gradient clipping, "
    "and mixed precision training; and recent innovations including mixture of "
    "experts, grouped query attention, multi-query attention, sliding window "
    "attention, and flash attention. Conclude with a discussion of scaling laws."
) * 2  # ~350 tokens

NUM_REQUESTS = 12
CONCURRENCY = 4
MAX_TOKENS = 256


async def run(label, port, model):
    sem = asyncio.Semaphore(CONCURRENCY)
    ttfts = []
    latencies = []
    total_out = 0
    errors = 0
    t_start = time.time()

    async def req(idx):
        nonlocal total_out, errors
        async with sem:
            t0 = time.time()
            payload = {
                "model": model, "prompt": PROMPT,
                "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            }
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.post(
                        f"http://localhost:{port}/v1/completions", json=payload,
                        timeout=aiohttp.ClientTimeout(total=600),
                    ) as r:
                        first = True
                        local_tokens = 0
                        async for line in r.content:
                            line = line.decode().strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    chunk = json.loads(line[6:])
                                    text = chunk["choices"][0].get("text", "")
                                    if first and text:
                                        ttfts.append(time.time() - t0)
                                        first = False
                                    local_tokens += 1
                                except Exception:
                                    pass
                        total_out += local_tokens
                        latencies.append(time.time() - t0)
            except Exception as e:
                errors += 1
                print(f"  [{idx}] ERROR: {e}")

    tasks = [req(i) for i in range(NUM_REQUESTS)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t_start

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Requests:      {NUM_REQUESTS}  |  Concurrency: {CONCURRENCY}")
    print(f"  Errors:        {errors}")
    print(f"  Total time:    {elapsed:.1f}s")
    print(f"  Throughput:    {total_out/elapsed:.1f} tok/s" if elapsed > 0 else "  Throughput:    N/A")
    if ttfts:
        print(f"  TTFT mean: {statistics.mean(ttfts)*1000:.0f}ms  "
              f"p50: {statistics.median(ttfts)*1000:.0f}ms  "
              f"p95: {sorted(ttfts)[int(len(ttfts)*.95)]*1000:.0f}ms")
    if latencies:
        print(f"  Latency mean: {statistics.mean(latencies):.1f}s  "
              f"p50: {statistics.median(latencies):.1f}s")
    print()

    return {
        "throughput": total_out / elapsed if elapsed > 0 else 0,
        "ttft_mean": statistics.mean(ttfts) * 1000 if ttfts else 0,
        "ttft_p50": statistics.median(ttfts) * 1000 if ttfts else 0,
        "errors": errors,
    }


async def main():
    results = {}
    results["on"] = await run("FC1 ON  (VLLM_ASCEND_ENABLE_FLASHCOMM1=1)", 18005, MODEL)
    results["off"] = await run("FC1 OFF (baseline)", 18006, MODEL)

    print("\n" + "=" * 55)
    print("  COMPARISON")
    print("=" * 55)
    for k, label in [("on", "FC1 ON"), ("off", "FC1 OFF")]:
        r = results[k]
        print(f"  {label}: {r['throughput']:.1f} tok/s  |  "
              f"TTFT p50: {r['ttft_p50']:.0f}ms  |  "
              f"errors: {r['errors']}")

    if results["on"]["throughput"] > 0 and results["off"]["throughput"] > 0:
        ratio = results["on"]["throughput"] / results["off"]["throughput"]
        print(f"\n  Throughput ratio (ON/OFF): {ratio:.2f}x")
        if results["on"]["ttft_p50"] > 0 and results["off"]["ttft_p50"] > 0:
            ttft_ratio = results["on"]["ttft_p50"] / results["off"]["ttft_p50"]
            print(f"  TTFT ratio     (ON/OFF): {ttft_ratio:.2f}x")

if __name__ == "__main__":
    asyncio.run(main())
