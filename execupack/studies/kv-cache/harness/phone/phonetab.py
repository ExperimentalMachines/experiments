"""Tabulate KvBench results pulled from the phone: out/<window>-<model>-<kv>-<rep>.json.

Per window and model: the two runs of each cache side by side (prefill and decode tok/s from the runner's own
stats, resident memory after load and the process peak), and whether the two caches wrote the same text.
"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
runs: dict = {}
for f in sorted(out.glob("*.json")):
    w, m, kv, rep = f.stem.split("-")
    runs.setdefault((w, m), {}).setdefault(kv, []).append(json.loads(f.read_text()))


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


for (w, m), by_kv in sorted(runs.items()):
    phases = [k for k in next(iter(by_kv.values()))[0] if k not in ("load", "warm")]
    print(f"\n## {w} {m}")
    print("| phase | ctx after | " + " | ".join(f"{kv} prefill | {kv} decode" for kv in ("f32kv", "q8kv")) + " | same text (chars) |")
    print("|---|---|" + "---|---|" * 2 + "---|")
    for p in phases:
        cells = []
        ctx = ""
        for kv in ("f32kv", "q8kv"):
            rs = by_kv.get(kv, [])
            pf = "/".join(f"{r[p]['stats']['prefill_token_per_sec']:.0f}" for r in rs if r[p].get("stats"))
            dc = "/".join(f"{r[p]['stats']['decode_token_per_sec']:.1f}" for r in rs if r[p].get("stats"))
            errs = [r[p]["error"] for r in rs if r[p].get("error")]
            cells += [pf + (" ERR" if errs else ""), dc]
        a, b = by_kv.get("f32kv", [{}])[0].get(p, {}), by_kv.get("q8kv", [{}])[0].get(p, {})
        same = f"{common_prefix(a.get('text', ''), b.get('text', ''))}/{len(a.get('text', ''))}"
        print(f"| {p} | | " + " | ".join(cells) + f" | {same} |")
    for kv in ("f32kv", "q8kv"):
        for r in by_kv.get(kv, []):
            last = r[phases[-1]]
            print(f"- {kv}: load {r['load']['ms']} ms, RSS after load {r['load']['rss_kb'] / 1024:.0f} MB, "
                  f"RSS at end {last['rss_kb'] / 1024:.0f} MB, peak {last['hwm_kb'] / 1024:.0f} MB")
