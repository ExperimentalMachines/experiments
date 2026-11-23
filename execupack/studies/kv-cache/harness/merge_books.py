"""Merge the GPU pass's per-book stream results (stream-<w>-<m>-<kv>.b<book>.json, one harness process per book from a
cool GPU) into the stream-<w>-<m>-<kv>.json layout report.py reads, and attach to every checkpoint the GPU's thermal
cap while its decode ran, estimated: checkpoint["gpu"] = {"level": [min, max], "max_clk_mhz": highest clock sampled, "src": ...}.
The decode window is reconstructed (step times x1.3 for scoring, 0.5 s start-up), so the level is an estimate.

Sources, best first:
- phase.log (phasemon.sh, every 2 s: epoch, thermal_pwrlevel, gpuclk MHz, newest stdout, its line count, last line):
  the decode of checkpoint d ends when "book <b> checkpoint <d> done" first appears; it spans the sum of its step times
  (x1.3 for the harness's per-step scoring) before that.
- <run>.therm (gpu.sh, every 5 s: epoch, thermal_pwrlevel, gpuclk Hz, busy %): a run that never left level 0 puts every
  checkpoint at level 0; otherwise the decode window is reconstructed from load time, prefill chunks and step times.
The instantaneous clock also falls whenever the GPU idles between steps, so the thermal level (0 = uncapped, 1,450
MHz; 12 = 636 MHz on this SM8975) is the number that says whether a measurement was throttled.

usage: python merge_books.py <gpu out dir>
"""

import json
import re
import sys
from pathlib import Path

BOOKS = {"2k": 3, "8k": 3, "32k": 1}


def read_phase(out: Path) -> list[tuple]:
    p = out / "phase.log"
    rows = []
    for line in p.read_text().splitlines() if p.exists() else []:
        f = line.split(" ", 5)
        if len(f) >= 5 and f[0].isdigit() and f[1].isdigit():
            rows.append((int(f[0]), int(f[1]), int(f[2]), f[3], f[5] if len(f) > 5 else ""))
    return rows


def read_therm(p: Path) -> list[tuple]:
    rows = []
    for line in p.read_text().splitlines() if p.exists() else []:
        f = line.split()
        if len(f) >= 3 and f[1].isdigit():
            rows.append((int(f[0]), int(f[1]), int(f[2]) // 10**6))
    return rows


def window(samples: list[tuple], lo: float, hi: float, src: str) -> dict | None:
    s = [r for r in samples if lo <= r[0] <= hi]
    if not s:
        return None
    return {"level": [min(r[1] for r in s), max(r[1] for r in s)], "max_clk_mhz": max(r[2] for r in s), "src": src}


def attach(run: dict, stdout_name: str, therm: list[tuple], phase: list[tuple]) -> None:
    mine = [r for r in phase if r[3] == stdout_name]
    allp = [(r[0], r[1], r[2]) for r in phase]
    for book in run["books"]:
        for c in book["checkpoints"]:
            dec = sum(c["step_us"]) / 1e6 * 1.3
            done = next((r[0] for r in mine if r[4] == f"book_{book['book']}_checkpoint_{c['d']}_done"), None)
            g = window(allp, done - dec - 2, done, "phase.log") if done else None
            if g is None and therm and max(r[1] for r in therm) == 0:
                g = {"level": [0, 0], "max_clk_mhz": max(r[2] for r in therm), "src": "therm (run never capped)"}
            if g is None and therm and len(run["books"]) == 1:
                t = therm[0][0] + 0.5 + book["load_ms"] / 1000
                for start, n, us in sorted(book["prefill_chunks"]):
                    if start < c["d"] - 128:
                        t += us / 1e6
                for c2 in book["checkpoints"]:
                    if c2["d"] < c["d"]:
                        t += sum(c2["step_us"]) / 1e6 * 1.3
                g = window(therm, t - 2.5, t + dec + 2.5, "therm (reconstructed)")
            c["gpu"] = g


def main(out: Path) -> None:
    phase = read_phase(out)
    groups: dict[str, list[Path]] = {}
    for p in sorted(out.glob("stream-*.b*.json")):
        if not re.search(r"\.b\d+\.json$", p.name):  # repeats (gpu.sh REP=...: *.b<book>.<rep>.json) stay apart
            continue
        groups.setdefault(re.sub(r"\.b\d+\.json$", "", p.name), []).append(p)
    for stem, parts in sorted(groups.items()):
        w = stem.split("-")[1]
        if len(parts) < BOOKS[w]:
            print(f"{stem}: {len(parts)}/{BOOKS[w]} books, not merged yet")
            continue
        merged = None
        for p in parts:
            d = json.loads(p.read_text())
            attach(d, p.with_suffix(".stdout").name, read_therm(p.with_suffix(".therm")), phase)
            merged = d if merged is None else {**merged, "books": merged["books"] + d["books"]}
        merged["books"].sort(key=lambda b: b["book"])
        (out / f"{stem}.json").write_text(json.dumps(merged))
        print(f"{stem}: " + "; ".join(f"{b['book']}: " + ", ".join(
            f"@{c['d']} L{c['gpu']['level'][0]}-{c['gpu']['level'][1]}" if c.get("gpu") else f"@{c['d']} ?"
            for c in b["checkpoints"]) for b in merged["books"]))
    for p in sorted(out.glob("stream-*.json")):  # single-process runs (Llama 2k): annotate in place
        if ".b" in p.name or p.stem in groups or not p.with_suffix(".therm").exists():
            continue
        d = json.loads(p.read_text())
        attach(d, p.with_suffix(".stdout").name, read_therm(p.with_suffix(".therm")), phase)
        p.write_text(json.dumps(d))
        print(f"{p.stem}: " + ", ".join(f"{b['book']}@{c['d']} L{c['gpu']['level']}" if c.get("gpu") else "?"
                                         for b in d["books"] for c in b["checkpoints"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
