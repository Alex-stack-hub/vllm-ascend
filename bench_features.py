"""Benchmark script for feature ON/OFF comparison.

Usage: python3 bench_features.py <port> <model_name> <output_file>
"""
import time, json, sys, statistics
import urllib.request

PORT = int(sys.argv[1])
MODEL = sys.argv[2]
OUTPUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/bench_result.json"

# ~1400 token prompt to trigger features that have token thresholds
PROMPT = (
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
) * 5

NUM_REQUESTS = 8
MAX_TOKENS = 128

def run_sequential():
    """Send requests sequentially, measure TTFT and total time."""
    ttfts = []
    latencies = []
    errors = 0
    total_out = 0

    for i in range(NUM_REQUESTS):
        payload = json.dumps({
            "model": MODEL, "prompt": PROMPT,
            "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
        }).encode()

        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"http://localhost:{PORT}/v1/completions",
                payload, {"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=600)
            first_token = True
            local_tokens = 0
            buffer = b""
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk_data = json.loads(line[6:])
                            text = chunk_data["choices"][0].get("text", "")
                            if first_token and text:
                                ttfts.append(time.time() - t0)
                                first_token = False
                            local_tokens += 1
                        except Exception:
                            pass
            total_out += local_tokens
            latencies.append(time.time() - t0)
        except Exception as e:
            errors += 1
            print(f"  [{i}] ERROR: {e}", flush=True)

    elapsed = time.time()
    return ttfts, latencies, total_out, errors

print(f"Benchmarking {MODEL} on port {PORT}...", flush=True)
print(f"Requests: {NUM_REQUESTS}, max_tokens: {MAX_TOKENS}", flush=True)

t_start = time.time()
ttfts, latencies, total_out, errors = run_sequential()
total_time = time.time() - t_start

result = {
    "model": MODEL, "port": PORT, "requests": NUM_REQUESTS,
    "total_time_s": round(total_time, 2), "output_tokens": total_out,
    "errors": errors,
}
if total_out > 0 and total_time > 0:
    result["throughput_tok_s"] = round(total_out / total_time, 2)
if ttfts:
    result["ttft_mean_ms"] = round(statistics.mean(ttfts) * 1000, 1)
    result["ttft_p50_ms"] = round(statistics.median(ttfts) * 1000, 1)
    result["ttft_min_ms"] = round(min(ttfts) * 1000, 1)
    result["ttft_max_ms"] = round(max(ttfts) * 1000, 1)

print(json.dumps(result, indent=2))
with open(OUTPUT, "w") as f:
    json.dump(result, f, indent=2)
