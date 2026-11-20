#!/usr/bin/env python3
"""Draw the charts and the banner the READMEs show, from the run files themselves.

Every value plotted here is read from a file under a project's `studies/` (or computed by that
study's own analysis script) when this runs, never typed in, so a chart cannot drift from its
data. Each chart is written twice, `<name>.svg` for GitHub's light theme and `<name>-dark.svg`
for its dark theme, and a README shows the right one with `<picture>`.

    python3 tools/figures.py            # rewrite figures/
    python3 tools/figures.py --check    # fail if figures/ is not what the data draws

Colours are the first three slots of a palette checked for colour-blind separation against
both GitHub surfaces (#ffffff and #0d1117); text never takes a series colour.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"

THEMES = {
    "light": {"surface": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "track": "#eff2f5",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#0d1117", "text": "#f0f6fc", "muted": "#9198a1", "grid": "#3d444d", "track": "#262c36",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}
FONT = ('-apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, &quot;Noto Sans&quot;, '
        'Helvetica, Arial, sans-serif')
W = 760


def text_width(s: str, size: float) -> float:
    return len(s) * size * 0.56


def fmt(v: float) -> str:
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if v >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.2g}"


class Svg:
    def __init__(self, theme: dict, height: int):
        self.t, self.h, self.parts = theme, height, []
        self.surface = theme["surface"]

    def text(self, x, y, s, size=13, fill="text", weight=400, anchor="start", extra=""):
        colour = self.t.get(fill, fill)
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{colour}" text-anchor="{anchor}"{extra}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke="grid", width=1, extra=""):
        colour = self.t.get(stroke, stroke)
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{colour}" stroke-width="{width}"{extra}/>')

    def dot(self, x, y, colour, r=5):
        # The ring in the page colour keeps a dot legible where it sits on a line or another dot.
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 2}" fill="{self.surface}"/>'
                          f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{colour}"/>')

    def bar(self, x, y, w, h, colour):
        # Square at the baseline, rounded at the data end.
        r = min(4, w / 2, h / 2)
        self.parts.append(f'<path d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
                          f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z" fill="{colour}"/>')

    def raw(self, s):
        self.parts.append(s)

    def render(self, title: str, desc: str) -> str:
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" '
                f'viewBox="0 0 {W} {self.h}" role="img" font-family="{FONT}">'
                f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>'
                + "".join(self.parts) + "</svg>\n")


def header(s: Svg, title: str, subtitle: str) -> None:
    s.text(24, 34, title, size=17, weight=600)
    s.text(24, 56, subtitle, size=13, fill="muted")


def legend(s: Svg, x: float, y: float, items: list[tuple[str, str]]) -> None:
    for colour, label in items:
        s.dot(x + 5, y - 4, colour, r=5)
        s.text(x + 16, y, label, size=13, fill="muted")
        x += 16 + text_width(label, 13) + 22


def footer(s: Svg, source: str) -> None:
    s.text(24, s.h - 14, source, size=11, fill="muted")


def log_axis(lo: float, hi: float, x0: float, x1: float):
    a, b = math.log10(lo), math.log10(hi)
    return lambda v: x0 + (math.log10(v) - a) / (b - a) * (x1 - x0)


def log_ticks(lo: float, hi: float) -> list[float]:
    ticks, p = [], math.floor(math.log10(lo))
    while 10 ** p <= hi:
        for m in (1, 2, 5):
            v = m * 10 ** p
            if lo <= v <= hi:
                ticks.append(v)
        p += 1
    return ticks


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- prefix reuse

COLDWARM = "openweights/studies/prefix-reuse/runs"
COLDWARM_FILES = [
    ("POCO X8 Pro Max", "own phone, MediaTek MT6991", "2026-11-15-coldwarm-poco/greedy-LFM2.5-1.2B-Instruct-8da4w-gptq-4k.coldwarm.jsonl"),
    ("Galaxy S25 Ultra", "Test Lab, Snapdragon 8 Elite", "2026-11-16-coldwarm-matrix/pa3q/greedy-lfm25.coldwarm.jsonl"),
    ("Galaxy Tab S10+", "Test Lab, Dimensity 9300+", "2026-11-16-coldwarm-matrix/gts10pwifi/greedy-lfm25.coldwarm.jsonl"),
    ("Galaxy S24", "Test Lab, Snapdragon 8 Gen 3", "2026-11-16-coldwarm-matrix/SC-51E/greedy-lfm25.coldwarm.jsonl"),
    ("Pixel 9", "Test Lab, Tensor G4", "2026-11-16-coldwarm-matrix/tokay/greedy-lfm25.coldwarm.jsonl"),
]


def coldwarm_rows() -> list[dict]:
    rows = []
    for name, where, rel in COLDWARM_FILES:
        pairs: dict[int, dict] = {}
        for r in jsonl(REPO / COLDWARM / rel):
            pairs.setdefault(r["pair"], {})[r["arm"]] = r
        whole = [p for p in pairs.values() if "warm" in p and "cold" in p and p["warm"]["cached_tokens"] > 0]
        rows.append({
            "name": name, "where": where,
            "warm": statistics.median(p["warm"]["prefill_ms"] for p in whole),
            "cold": statistics.median(p["cold"]["prefill_ms"] for p in whole),
            "ratio": statistics.median(p["cold"]["prefill_ms"] / p["warm"]["prefill_ms"] for p in whole),
            "pairs": len(whole),
        })
    return rows


def prefix_reuse(theme: str) -> str:
    t = THEMES[theme]
    rows = coldwarm_rows()
    top, row_h = 112, 46
    s = Svg(t, top + row_h * len(rows) + 64)
    header(s, "Keeping the cache makes the next turn 30 to 41 times faster",
           "Second turn of a ~1,000-token chat, LFM2.5 1.2B on ExecuTorch, greedy. Median of 8 to 12 pairs.")
    legend(s, 24, 86, [(t["series"][0], "Cache kept: reads the 20 new tokens"),
                       (t["series"][1], "Cache reset: reads all ~1,000 tokens")])
    x0, x1 = 236, W - 128
    xs = log_axis(50, 10000, x0, x1)
    y_axis = top + row_h * len(rows) + 2
    for v in log_ticks(50, 10000):
        s.line(xs(v), top - 8, xs(v), y_axis, stroke="track")
        s.text(xs(v), y_axis + 16, f"{v:,.0f}" if v < 1000 else f"{v / 1000:g}k",
               size=11, fill="muted", anchor="middle", extra=' style="font-variant-numeric:tabular-nums"')
    s.text(x1, y_axis + 32, "time to read the prompt, ms (log scale)", size=11, fill="muted", anchor="end")
    s.text(W - 24, top - 14, "speed-up", size=11, fill="muted", anchor="end")
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        s.text(24, y - 2, r["name"], size=13, weight=600)
        s.text(24, y + 14, r["where"], size=11, fill="muted")
        s.line(xs(r["warm"]), y, xs(r["cold"]), y, stroke="grid", width=2,
               extra=' stroke-linecap="round"')
        s.dot(xs(r["warm"]), y, t["series"][0])
        s.dot(xs(r["cold"]), y, t["series"][1])
        s.text(xs(r["warm"]) - 12, y + 4, f"{r['warm']:,.0f}", size=11, fill="muted", anchor="end")
        s.text(xs(r["cold"]) + 12, y + 4, f"{r['cold']:,.0f}", size=11, fill="muted")
        s.text(W - 24, y + 5, f"{r['ratio']:.1f}x", size=14, weight=600, anchor="end")
    footer(s, f"Data: {COLDWARM}/ (five *.coldwarm.jsonl files). Ratio: median over pairs.")
    return s.render("Second-turn prompt read with the cache kept and reset, five phones",
                    "; ".join(f"{r['name']}: {r['warm']:.0f} ms kept, {r['cold']:.0f} ms reset, "
                              f"{r['ratio']:.1f}x" for r in rows))


# ---------------------------------------------------------------- threads under throttle

THREADS_DIR = "openweights/studies/prefill-decode/runs/2026-10-24-sm8850"


def thread_rows() -> list[dict]:
    script = REPO / "openweights/harness/prefill-decode/snapdragon/analyze_threads49.py"
    run = subprocess.run([sys.executable, "-I", "-B", str(script), "table", "chain49.log", "logcat-r49.txt", "."],
                         cwd=REPO / THREADS_DIR, capture_output=True, text=True, check=True)
    return json.loads(run.stdout.strip().splitlines()[-1])


def threads(theme: str) -> str:
    t = THEMES[theme]
    rows = thread_rows()
    s = Svg(t, 420)
    header(s, "With most cores paused, more threads make decode slower, not faster",
           "llama.cpp decode on a Snapdragon 8 Elite Gen 5, cores paused the way its hard throttle does. Median of 3.")
    legend(s, 24, 86, [(t["series"][0], "Normal"), (t["series"][1], "6 of 8 cores paused")])
    x0, x1, y0, y1 = 84, W - 150, 126, 350
    xs = lambda n: x0 + (n - 1) / 7 * (x1 - x0)
    lo, hi = 0.05, 50
    a, b = math.log10(lo), math.log10(hi)
    ys = lambda v: y1 - (math.log10(v) - a) / (b - a) * (y1 - y0)
    for v in log_ticks(lo, hi):
        s.line(x0, ys(v), x1, ys(v), stroke="track")
        s.text(x0 - 10, ys(v) + 4, fmt(v), size=11, fill="muted", anchor="end",
               extra=' style="font-variant-numeric:tabular-nums"')
    s.text(24, y0 - 12, "tokens/s (log scale)", size=11, fill="muted")
    for n in range(1, 9):
        s.text(xs(n), y1 + 20, str(n), size=11, fill="muted", anchor="middle")
    s.text((x0 + x1) / 2, y1 + 38, "CPU threads", size=11, fill="muted", anchor="middle")
    s.line(xs(4), y0, xs(4), y1, stroke="grid", extra=' stroke-dasharray="0"')
    s.text(xs(4) + 6, y1 - 6, "app default", size=11, fill="muted")
    for key, colour in (("decodeNormal", t["series"][0]), ("decodePaused", t["series"][1])):
        pts = [(xs(r["threads"]), ys(r[key])) for r in rows]
        s.raw(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
              f'stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            s.dot(x, y, colour, r=4)
    by = {r["threads"]: r for r in rows}
    for n, key, dx, dy, anchor in ((2, "decodePaused", 18, 4, "start"), (4, "decodePaused", 0, 22, "middle"),
                                   (8, "decodePaused", -8, -12, "end"), (4, "decodeNormal", 0, 22, "middle")):
        s.text(xs(n) + dx, ys(by[n][key]) + dy, f"{fmt(by[n][key])} tok/s", size=12, weight=600, anchor=anchor)
    s.text(x1 + 14, ys(by[8]["decodeNormal"]) + 4, "normal", size=12, fill="muted")
    s.text(x1 + 14, ys(by[8]["decodePaused"]) + 4, "6 paused", size=12, fill="muted")
    footer(s, f"Data: {THREADS_DIR}/chain49.log and logcat-r49.txt, through analyze_threads49.py.")
    return s.render("Decode speed by CPU thread count, normal and throttled",
                    "; ".join(f"{r['threads']} threads: {r['decodeNormal']:.2f} normal, "
                              f"{r['decodePaused']:.3f} throttled tok/s" for r in rows))


# ---------------------------------------------------------------- NPU serving

NPU_DIR = "execuserve/studies/2026-10-06-npu"


def npu_rows() -> dict:
    out = {}
    for proc, name in (("cpu", "sm8850-execuserve-cpu-gpu-runs.csv"), ("npu", "sm8850-execuserve-qnn-runs.csv")):
        for r in csv.DictReader((REPO / NPU_DIR / name).open()):
            unit = "npu" if proc == "npu" else ("gpu" if "vulkan" in r["model"] else "cpu")
            long = int(r["prompt_tokens"]) > 600
            out.setdefault(unit, {"ttft": [], "decode": []})
            if long:
                out[unit]["ttft"].append(float(r["first_token_ms"]))
            else:
                out[unit]["decode"].append(float(r["decode_tokens_per_second"]))
    return {u: {k: statistics.median(v) for k, v in d.items()} for u, d in out.items()}


def npu(theme: str) -> str:
    t = THEMES[theme]
    rows = npu_rows()
    order = [("cpu", "CPU", t["series"][0]), ("gpu", "GPU (Vulkan)", t["series"][1]), ("npu", "NPU (QNN)", t["series"][2])]
    s = Svg(t, 300)
    header(s, "The NPU answers first and then writes slowest",
           "Qwen3 1.7B served by ExecuServe on a Snapdragon 8 Elite Gen 5. Median of 3 requests each.")
    panels = [("ttft", "First token after a ~700-token prompt", "ms, lower is better", 2500),
              ("decode", "Writing speed after a 21-token prompt", "tokens/s, higher is better", 60)]
    pw = (W - 48 - 40) / 2
    for p, (key, title, unit, hi) in enumerate(panels):
        px = 24 + p * (pw + 40)
        s.text(px, 96, title, size=13, weight=600)
        s.text(px, 113, unit, size=11, fill="muted")
        lx, rx = px + 96, px + pw - 64
        for i, (u, label, colour) in enumerate(order):
            y = 132 + i * 44
            v = rows[u][key]
            s.text(px, y + 16, label, size=12, fill="muted")
            s.bar(lx, y + 4, max(2, (rx - lx) * v / hi), 18, colour)
            s.text(lx + (rx - lx) * v / hi + 8, y + 18,
                   f"{v:,.0f} ms" if key == "ttft" else f"{v:.1f}", size=12, weight=600)
        s.line(lx, 128, lx, 132 + 3 * 44, stroke="grid")
    footer(s, f"Data: {NPU_DIR}/sm8850-execuserve-cpu-gpu-runs.csv and -qnn-runs.csv.")
    return s.render("First token and decode speed on CPU, GPU and NPU",
                    "; ".join(f"{label}: first token {rows[u]['ttft']:.0f} ms, decode {rows[u]['decode']:.1f} tok/s"
                              for u, label, _ in order))


# ---------------------------------------------------------------- MediaTek A16W4

A16W4_DIR = "execupack/studies/mtk-a16w4/runs"
A16W4_MODELS = [("Llama 3.2 1B", "llama"), ("Qwen3 1.7B", "qwen"), ("LFM2.5 1.2B", "lfm")]


def a16w4_rows() -> list[dict]:
    rows = []
    for name, sub in A16W4_MODELS:
        v = json.loads((REPO / A16W4_DIR / sub / "eval-2048.json").read_text())["variants"]
        rows.append({"name": name, **{k: v[k]["kl"] for k in ("a16w8", "a16w4", "a16w4-gptq")}})
    return rows


def a16w4(theme: str) -> str:
    t = THEMES[theme]
    rows = a16w4_rows()
    top, row_h = 112, 52
    s = Svg(t, top + row_h * len(rows) + 64)
    header(s, "With MediaTek's NPU quantization, 4-bit weights drift far from the model",
           "KL from fp32 over a book passage (lower is closer), simulated on a GPU host. GPTQ helps two of three.")
    keys = [("a16w8", "8-bit weights (published)", t["series"][0]), ("a16w4", "4-bit weights", t["series"][1]),
            ("a16w4-gptq", "4-bit weights + GPTQ", t["series"][2])]
    legend(s, 24, 86, [(c, label) for _, label, c in keys])
    x0, x1 = 160, W - 40
    xs = log_axis(0.002, 2, x0, x1)
    y_axis = top + row_h * len(rows) + 2
    for v in log_ticks(0.002, 2):
        s.line(xs(v), top - 8, xs(v), y_axis, stroke="track")
        s.text(xs(v), y_axis + 16, fmt(v), size=11, fill="muted", anchor="middle",
               extra=' style="font-variant-numeric:tabular-nums"')
    s.text(x1, y_axis + 32, "KL divergence (log scale)", size=11, fill="muted", anchor="end")
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        s.text(24, y + 5, r["name"], size=13, weight=600)
        lo, hi = min(r[k] for k, _, _ in keys), max(r[k] for k, _, _ in keys)
        s.line(xs(lo), y, xs(hi), y, stroke="grid", width=2)
        for j, (k, _, colour) in enumerate(keys):
            s.dot(xs(r[k]), y, colour)
            above = (j == 2) if r["a16w4-gptq"] < r["a16w4"] else (j == 1)
            s.text(xs(r[k]), y + (-12 if above or j == 0 else 22), fmt(r[k]), size=11, fill="muted",
                   anchor="middle")
    footer(s, f"Data: {A16W4_DIR}/<model>/eval-2048.json, field variants.<recipe>.kl.")
    return s.render("KL from fp32 for A16W8, A16W4 and A16W4 with GPTQ on MediaTek's NPU",
                    "; ".join(f"{r['name']}: A16W8 {r['a16w8']:.4f}, A16W4 {r['a16w4']:.3f}, "
                              f"A16W4+GPTQ {r['a16w4-gptq']:.3f}" for r in rows))


# ---------------------------------------------------------------- banner

def counts() -> dict:
    studies = sum(1 for p in ("openweights", "execupack") for d in (REPO / p / "studies").iterdir() if d.is_dir())
    studies += sum(1 for f in (REPO / "execuserve/studies").glob("*.md"))
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.splitlines()
    data = sum(1 for f in tracked if "/studies/" in f or "/research/evidence/" in f)
    # The rows of the first table in docs/devices.md, minus its header.
    table = (REPO / "docs/devices.md").read_text().split("## Every device", 1)[1].split("\n## ", 1)[0]
    devices = sum(1 for line in table.splitlines() if line.startswith("| ")) - 1
    return {"studies": studies, "files": data, "devices": devices}


def banner(theme: str) -> str:
    t = THEMES[theme]
    c = counts()
    s = Svg(t, 210)
    s.raw(f'<rect x="24" y="30" width="4" height="96" rx="2" fill="{t["series"][0]}"/>')
    s.text(44, 72, "Experiments", size=40, weight=700)
    s.text(44, 104, "What it really takes to run open-weight language models on a phone,", size=15, fill="muted")
    s.text(44, 124, "measured on real devices, with the scripts and every run file kept.", size=15, fill="muted")
    stats = [(c["studies"], "studies"), (c["devices"], "phones, tablets and emulators"), (c["files"], "run and evidence files")]
    x = 44
    for value, label in stats:
        s.text(x, 176, f"{value:,}", size=24, weight=700)
        s.text(x, 196, label, size=12, fill="muted")
        x += max(text_width(label, 12), text_width(f"{value:,}", 24)) + 48
    return s.render("Experiments", f"{c['studies']} studies, {c['devices']} devices, {c['files']} files")


FIGURES = {"banner": banner, "prefix-reuse": prefix_reuse, "threads-under-throttle": threads,
           "npu-serving": npu, "mtk-a16w4": a16w4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with figures/ instead of writing")
    check = ap.parse_args().check
    stale = 0
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        for theme in THEMES:
            path = OUT / (f"{name}.svg" if theme == "light" else f"{name}-dark.svg")
            svg = draw(theme)
            if check:
                if not path.exists() or path.read_text() != svg:
                    print(f"stale: {path.relative_to(REPO)}")
                    stale += 1
            else:
                path.write_text(svg)
    print(f"figures: {len(FIGURES) * len(THEMES)} files" + (f", {stale} stale" if check else " written"))
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
