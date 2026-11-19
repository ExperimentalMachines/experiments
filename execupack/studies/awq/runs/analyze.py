"""Tables for the AWQ / GPTQ comparison: per model and arm, and paired differences against GPTQ."""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODELS = sys.argv[1:] or ["lfm25-1.2b", "qwen3-1.7b", "llama32-1b"]
ARMS = ["rtn", "published", "gptq", "awq", "awq_gptq"]


def load(model, arm):
    d = ROOT / model / arm
    res = d / f"result-{arm}.json"
    rows = d / f"rows-{arm}.json"
    if not res.exists():
        return None, None
    return json.loads(res.read_text()), json.loads(rows.read_text())


def boot(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(diffs) / k, means[int(0.025 * n)], means[int(0.975 * n)]


for model in MODELS:
    ref = json.loads((ROOT / model / "ref" / "reference.json").read_text())
    print(f"\n## {model}  (calibration {ref['timings_s'].get('calibration', float('nan')):.0f}s, "
          f"held-out {ref['heldout_rows']}, reply positions {ref['reply_positions']}, "
          f"gate confident rows {ref['gate_confident_rows']})")
    print(f"{'arm':10} {'solve_s':>8} {'MB':>7} {'smoke':>6} {'gate':>10} {'gate mean':>10} "
          f"{'HO agree':>9} {'HO KL':>7} {'RP agree':>9} {'RP KL':>7}")
    data = {}
    for arm in ARMS:
        r, rows = load(model, arm)
        if r is None:
            print(f"{arm:10} (missing)")
            continue
        data[arm] = rows
        g = r["gate"]
        smoke = (r.get("smoke") or {}).get("passed")
        solve = r["timings_s"].get("solve")
        print(
            f"{arm:10} {solve if solve is not None else '-':>8} {r['bytes'] / 1e6:7.1f} {str(smoke):>6} "
            f"{('PASS ' if g.get('passed') else 'FAIL ') + str(g.get('agreed')) + '/' + str(g.get('graded')):>10} "
            f"{g.get('export_mean', float('nan')):10.3f} {r['heldout']['top1_agree']:9.3f} {r['heldout']['mean_kl']:7.3f} "
            f"{r['replies']['top1_agree']:9.3f} {r['replies']['mean_kl']:7.3f}"
        )
    if "gptq" not in data:
        continue
    for arm in ("awq", "awq_gptq", "published", "rtn"):
        if arm not in data:
            continue
        a, b = data[arm], data["gptq"]
        ho = [x["kl"] - y["kl"] for x, y in zip(a["per_row"], b["per_row"])]
        rp = [
            p["kl"] - q["kl"]
            for ra, rb in zip(a["per_reply"], b["per_reply"])
            for p, q in zip(ra["positions"], rb["positions"])
        ]
        m1, l1, h1 = boot(ho)
        m2, l2, h2 = boot(rp)
        flips = sum(
            (x["top"] == x["fp32_top"]) != (y["top"] == y["fp32_top"]) for x, y in zip(a["per_row"], b["per_row"])
        )
        print(
            f"  {arm} - gptq: held-out KL {m1:+.4f} [{l1:+.4f}, {h1:+.4f}], "
            f"reply KL {m2:+.4f} [{l2:+.4f}, {h2:+.4f}], rows where only one agrees with fp32: {flips}"
        )
