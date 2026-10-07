import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)

VARIANTS = ["fp16-baseline", "awq-4bit"]
# Weight memory from the vLLM "Model loading took" log lines
WEIGHT_GIB = {"fp16-baseline": 2.98, "awq-4bit": 1.1}
LEVELS = [1, 2, 4, 8, 16, 32]


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


bench = {v: load(f"{v}.json") for v in VARIANTS}
evals = {v: load(f"eval-{v}.json") for v in VARIANTS}


def line_chart(metric, ylabel, title, fname):
    plt.figure(figsize=(7, 4.5))
    for v in VARIANTS:
        rows = bench[v]["results"]
        plt.plot(
            [r["concurrency"] for r in rows],
            [r[metric] for r in rows],
            marker="o",
            label=v,
        )
    plt.xscale("log", base=2)
    plt.xticks(LEVELS, LEVELS)
    plt.xlabel("Concurrent users")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=150)
    plt.close()


line_chart("throughput_tok_s", "Tokens / second", "Throughput vs concurrency", "throughput.png")
line_chart("latency_p95", "p95 latency (s)", "p95 latency vs concurrency", "latency_p95.png")
line_chart("ttft_p95", "p95 TTFT (s)", "p95 time to first token vs concurrency", "ttft_p95.png")

# Accuracy with 95% CI, labelled with weight memory
plt.figure(figsize=(6, 4.5))
labels = [f"{v}\n{WEIGHT_GIB[v]:.2f} GiB weights" for v in VARIANTS]
accs = [evals[v]["accuracy"] * 100 for v in VARIANTS]
cis = [evals[v]["ci95"] * 100 for v in VARIANTS]
plt.bar(labels, accs, yerr=cis, capsize=6, color=["#4c78a8", "#f58518"])
plt.ylabel("MMLU accuracy (%)")
plt.title("Accuracy vs model size (1,000 questions, 95% CI)")
plt.ylim(0, 100)
for i, a in enumerate(accs):
    plt.text(i, a + cis[i] + 1.5, f"{a:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(OUT / "accuracy_vs_memory.png", dpi=150)
plt.close()

print("saved charts to", OUT)