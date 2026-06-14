"""Quick concurrent benchmark for feature comparison.

Sends N concurrent long-prompt requests and measures TTFT + throughput.
"""
import time, json, sys, statistics
from concurrent.futures import ThreadPoolExecutor
import urllib.request

PORT = int(sys.argv[1])
MODEL = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 6

PROMPT = ("Quantum mechanics is a fundamental theory in physics. " * 50)  # ~400 tokens

def do_one(idx):
    t0 = time.time()
    payload = json.dumps({
        "model": MODEL, "prompt": PROMPT,
        "max_tokens": 64, "temperature": 0, "stream": True,
    }).encode()
    try:
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/completions",
            payload, {"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=300)
        first_token = None
        tokens = 0
        buffer = b""
        while True:
            chunk = resp.read(65536)
            if not chunk: break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        text = json.loads(line[6:])["choices"][0].get("text","")
                        if first_token is None and text:
                            first_token = time.time() - t0
                        tokens += 1
                    except: pass
        return first_token or (time.time()-t0), time.time()-t0, tokens, None
    except Exception as e:
        return 0, time.time()-t0, 0, str(e)

print(f"Bench {MODEL}:{PORT} x{N} concurrent...", flush=True)
t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=N) as ex:
    results = list(ex.map(do_one, range(N)))
total_t = time.time() - t0

ttfts = [r[0] for r in results if r[0] > 0]
lats = [r[1] for r in results]
tokens = sum(r[2] for r in results)
errors = sum(1 for r in results if r[3])

print(f"  Time: {total_t:.1f}s  Tokens: {tokens}  Errors: {errors}")
if tokens > 0:
    print(f"  Throughput: {tokens/total_t:.1f} tok/s")
if ttfts:
    print(f"  TTFT mean/p50/min/max: {statistics.mean(ttfts)*1000:.0f} / {statistics.median(ttfts)*1000:.0f} / {min(ttfts)*1000:.0f} / {max(ttfts)*1000:.0f} ms")
print(f"RESULT: tp={tokens/total_t:.1f} ttft_p50={statistics.median(ttfts)*1000:.0f}ms err={errors}" if tokens > 0 else f"RESULT: FAILED err={errors}")
