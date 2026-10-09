import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "dashboards" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Fixed start time so all runs overlay on the same time axis
BASE = datetime(2026, 10, 9, tzinfo=timezone.utc).timestamp()

RUNS = {
    "fp16-cache-on": "metrics-fp16.json",
    "fp16-cache-off": "metrics-fp16-nocache.json",
    "awq-cache-off": "metrics-awq-nocache.json",
}

KEEP = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_queue_time_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
]

MONOTONIC = [
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:e2e_request_latency_seconds_count",
]

data = {}
for run, fname in RUNS.items():
    with open(RESULTS / fname) as f:
        raw = sorted(json.load(f), key=lambda s: s["t"])
    kept, last_t = [], -1e9
    last_vals = {k: -1.0 for k in MONOTONIC}
    for s in raw:
        if s["t"] - last_t < 0.9:          # at most one sample per second
            continue
        vals = {k: s.get(k, 0.0) for k in MONOTONIC}
        if any(vals[k] < last_vals[k] for k in MONOTONIC):
            continue                        # counter went backwards: duplicate recorder thread
        kept.append(s)
        last_t, last_vals = s["t"], vals
        gen = [s.get("vllm:generation_tokens_total", 0.0) for s in kept]
    active = [i for i in range(1, len(gen)) if gen[i] > gen[i - 1]]
    a, b = max(active[0] - 2, 0), min(active[-1] + 2, len(kept) - 1)
    window = kept[a:b + 1]
    t0 = window[0]["t"]
    data[run] = [{**s, "t": s["t"] - t0} for s in window]

lines = []
for name in KEEP:
    lines.append(f"# TYPE {name} gauge")
    for run, samples in data.items():
        for s in samples:
            if name in s:
                lines.append(f'{name}{{run="{run}"}} {s[name]} {BASE + s["t"]:.3f}')
lines.append("# EOF")

path = OUT / "runs.om"
with open(path, "w", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines)} lines to {path}")
for run, samples in data.items():
    print(f"  {run}: {len(samples)} samples, {samples[-1]['t'] / 60:.1f} minutes")