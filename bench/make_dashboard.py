import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "dashboards" / "grafana" / "provisioning" / "dashboards"
OUT.mkdir(parents=True, exist_ok=True)

DS = {"type": "prometheus", "uid": "prometheus"}
W = "30s"  # smoothing window for rate() queries

PANELS = [
    ("Requests running", "vllm:num_requests_running", "short"),
    ("Requests waiting", "vllm:num_requests_waiting", "short"),
    ("KV cache usage", "vllm:kv_cache_usage_perc * 100", "percent"),
    ("Generation throughput (tok/s)", f"rate(vllm:generation_tokens_total[{W}])", "short"),
    ("Prompt tokens processed (tok/s)", f"rate(vllm:prompt_tokens_total[{W}])", "short"),
    ("Mean TTFT (s)",
     f"rate(vllm:time_to_first_token_seconds_sum[{W}]) / rate(vllm:time_to_first_token_seconds_count[{W}])", "s"),
    ("Mean queue time (s)",
     f"rate(vllm:request_queue_time_seconds_sum[{W}]) / rate(vllm:request_queue_time_seconds_count[{W}])", "s"),
    ("Mean inter-token latency (s)",
     f"rate(vllm:inter_token_latency_seconds_sum[{W}]) / rate(vllm:inter_token_latency_seconds_count[{W}])", "s"),
]

panels = []
for i, (title, expr, unit) in enumerate(PANELS):
    panels.append({
        "id": i + 1,
        "type": "timeseries",
        "title": title,
        "datasource": DS,
        "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [{"datasource": DS, "expr": expr, "legendFormat": "{{run}}", "refId": "A"}],
    })

dashboard = {
    "uid": "vllm-runs",
    "title": "vLLM load test runs",
    "schemaVersion": 39,
    "timezone": "utc",
    "time": {"from": "2026-10-09T00:00:00.000Z", "to": "2026-10-09T00:09:00.000Z"},
    "panels": panels,
}

with open(OUT / "vllm.json", "w", newline="\n") as f:
    json.dump(dashboard, f, indent=2)

provider = [
    "apiVersion: 1",
    "providers:",
    "  - name: vllm",
    "    type: file",
    "    options:",
    "      path: /etc/grafana/provisioning/dashboards",
]
with open(OUT / "provider.yml", "w", newline="\n") as f:
    f.write("\n".join(provider) + "\n")

print("wrote dashboard files to", OUT)