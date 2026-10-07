import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)

LABEL = "fp16"

with open(RESULTS / f"metrics-{LABEL}.json") as f:
    samples = json.load(f)

# Trim idle time before and after the load test
tok_all = [s.get("vllm:generation_tokens_total", 0.0) for s in samples]
active = [i for i in range(1, len(tok_all)) if tok_all[i] > tok_all[i - 1]]
start = max(active[0] - 2, 0)
end = min(active[-1] + 2, len(samples) - 1)
samples = samples[start:end + 1]

t0 = samples[0]["t"]
t = [s["t"] - t0 for s in samples]
running = [s.get("vllm:num_requests_running", 0) for s in samples]
waiting = [s.get("vllm:num_requests_waiting", 0) for s in samples]
kv = [s.get("vllm:kv_cache_usage_perc", 0) * 100 for s in samples]
tok = [s.get("vllm:generation_tokens_total", 0.0) for s in samples]

tput = [0.0] + [(tok[i] - tok[i - 1]) / (t[i] - t[i - 1]) for i in range(1, len(tok))]


def rolling(xs, w=5):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
axes[0].plot(t, running, label="running")
axes[0].plot(t, waiting, label="waiting")
axes[0].set_ylabel("Requests")
axes[0].set_title(f"vLLM server metrics during load test ({LABEL})")
axes[0].legend()
axes[1].plot(t, kv, color="tab:green")
axes[1].set_ylabel("KV cache usage (%)")
axes[2].plot(t, rolling(tput), color="tab:red")
axes[2].set_ylabel("Generation tok/s")
axes[2].set_xlabel("Seconds since load start")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / f"server_metrics_{LABEL}.png", dpi=150)
plt.close()

last = samples[-1]


def mean(prefix):
    c = last.get(f"vllm:{prefix}_count", 0)
    return last.get(f"vllm:{prefix}_sum", 0) / c if c else float("nan")


print(f"Peak running requests : {max(running):.0f}")
print(f"Peak waiting requests : {max(waiting):.0f}")
print(f"Peak KV cache usage   : {max(kv):.1f}%")
print(f"Total preemptions     : {last.get('vllm:num_preemptions_total', 0):.0f}")
print(f"Mean queue time       : {mean('request_queue_time_seconds') * 1000:.0f} ms")
print(f"Mean prefill time     : {mean('request_prefill_time_seconds') * 1000:.0f} ms")
print(f"Mean decode time      : {mean('request_decode_time_seconds') * 1000:.0f} ms")

# Repeatability: baseline run vs run 2
with open(RESULTS / "fp16-baseline.json") as f:
    r1 = json.load(f)["results"]
with open(RESULTS / "fp16-run2.json") as f:
    r2 = json.load(f)["results"]
print("\nRepeatability (run 2 vs run 1)")
print("users  tok/s run1  tok/s run2  diff    p95 run1  p95 run2  diff")
for a, b in zip(r1, r2):
    print(
        f"{a['concurrency']:>5}  {a['throughput_tok_s']:>10.1f}  {b['throughput_tok_s']:>10.1f}  "
        f"{(b['throughput_tok_s'] / a['throughput_tok_s'] - 1) * 100:>+5.1f}%  "
        f"{a['latency_p95']:>8.2f}  {b['latency_p95']:>8.2f}  "
        f"{(b['latency_p95'] / a['latency_p95'] - 1) * 100:>+5.1f}%"
    )