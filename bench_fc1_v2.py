"""Benchmark FC1 ON vs OFF. Run twice: once with FC1 ON, once OFF.

Usage:
  python3 bench_fc1_v2.py 18005 gemma-4-31B > /tmp/bench_result.txt

Uses ~1400 token prompts with concurrency=3 to trigger FC1 (>1000).
"""
import time, json, asyncio, sys, statistics
import aiohttp

PORT = int(sys.argv[1])
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma-4-31B"

QM_TEXT = (
    "Quantum mechanics is a fundamental theory in physics that provides a description of "
    "the physical properties of nature at the scale of atoms and subatomic particles. "
    "It is the foundation of all quantum physics including quantum chemistry, quantum field theory, "
    "quantum technology, and quantum information science. Classical physics, the collection of theories "
    "that existed before the advent of quantum mechanics, describes many aspects of nature at an ordinary "
    "macroscopic and microscopic scale, but is not sufficient for describing them at very small submicroscopic "
    "atomic and subatomic scales. Most theories in classical physics can be derived from quantum mechanics "
    "as an approximation valid at large macroscopic scale. Quantum mechanics differs from classical physics "
    "in that energy, momentum, angular momentum, and other quantities of a bound system are restricted to "
    "discrete values quantization; objects have characteristics of both particles and waves wave-particle "
    "duality; and there are limits to how accurately the value of a physical quantity can be predicted prior "
    "to its measurement, given a complete set of initial conditions the uncertainty principle. "
)
PROMPT = QM_TEXT * 5  # ~1400 tokens, triggers FC1 on dense models (>1000)
NUM_REQUESTS = 6
CONCURRENT = 3
MAX_TOKENS = 128


async def run(label, port, model):
    sem = asyncio.Semaphore(CONCURRENT)
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
                print(f"  [{idx}] ERROR: {e}", flush=True)

    tasks = [req(i) for i in range(NUM_REQUESTS)]
    await asyncio.gather(*tasks)
    elapsed = time.time() - t_start

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Requests:     {NUM_REQUESTS}  |  Concurrency: {CONCURRENT}")
    print(f"  Errors:       {errors}")
    print(f"  Total time:   {elapsed:.1f}s")
    print(f"  Output tokens:{total_out}")
    if elapsed > 0 and total_out > 0:
        print(f"  Throughput:   {total_out/elapsed:.1f} tok/s")
        print(f"  Tokens/req/s: {total_out/elapsed/CONCURRENT:.1f}")
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
        "ttft_p50": statistics.median(ttfts) * 1000 if ttfts else 0,
        "ttft_mean": statistics.mean(ttfts) * 1000 if ttfts else 0,
        "errors": errors,
    }


if __name__ == "__main__":
    r = asyncio.run(run("Benchmark", PORT, MODEL))
    print(f"RESULT: throughput={r['throughput']:.1f} ttft_p50={r['ttft_p50']:.0f}ms errors={r['errors']}")
