import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
samples = json.load(open(ROOT / "results" / "metrics-fp16.json"))
last = samples[-1]

queries = last.get("vllm:prefix_cache_queries_total", 0)
hits = last.get("vllm:prefix_cache_hits_total", 0)
cached = last.get("vllm:prompt_tokens_cached_total", 0)
prompt = last.get("vllm:prompt_tokens_total", 0)

print(f"Prefix cache queries (tokens): {queries:.0f}")
print(f"Prefix cache hits (tokens)   : {hits:.0f}")
print(f"Hit rate                     : {100 * hits / queries:.1f}%" if queries else "Hit rate: n/a (no queries)")
print(f"Prompt tokens total          : {prompt:.0f}")
print(f"Prompt tokens served from cache: {cached:.0f}")
