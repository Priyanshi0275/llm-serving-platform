import json, math, sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

MAX_ACC_DROP = 0.03          # candidate may lose at most 3 accuracy points
MAX_P95_REGRESSION = 0.10    # candidate p95 latency may be at most 10% worse
GATE_CONCURRENCY = 16        # load level used for the latency check


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def main(baseline, candidate, eval_baseline=None, eval_candidate=None):
    eval_baseline = eval_baseline or baseline
    eval_candidate = eval_candidate or candidate
    bb, cb = load(f"{baseline}.json"), load(f"{candidate}.json")
    be, ce = load(f"eval-{eval_baseline}.json"), load(f"eval-{eval_candidate}.json")

    # Quality: paired comparison on the same questions
    a, b = be["correct"], ce["correct"]
    n = len(a)
    only_base = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    only_cand = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    drop = (only_base - only_cand) / n
    se = math.sqrt(only_base + only_cand - (only_base - only_cand) ** 2 / n) / n
    lo, hi = drop - 1.96 * se, drop + 1.96 * se

    # Latency at the gate concurrency level
    br = next(r for r in bb["results"] if r["concurrency"] == GATE_CONCURRENCY)
    cr = next(r for r in cb["results"] if r["concurrency"] == GATE_CONCURRENCY)
    p95_change = cr["latency_p95"] / br["latency_p95"] - 1

    checks = [
        (
            f"accuracy drop {drop*100:.1f} pts (95% CI {lo*100:.1f} to {hi*100:.1f}), limit {MAX_ACC_DROP*100:.0f} pts",
            drop <= MAX_ACC_DROP,
        ),
        (
            f"p95 latency @ {GATE_CONCURRENCY} users {p95_change*100:+.1f}%, limit +{MAX_P95_REGRESSION*100:.0f}%",
            p95_change <= MAX_P95_REGRESSION,
        ),
    ]

    print(f"Gate: {candidate} vs {baseline}")
    ok = True
    for text, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}")
        ok = ok and passed
    print("RESULT:", "APPROVED" if ok else "REJECTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
        main(*sys.argv[1:5])