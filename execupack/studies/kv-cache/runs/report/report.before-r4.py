"""The KV study's report page: every table and chart computed here from the phone's results and the reference data.

    python scripts/kvstudy/report.py work/kvstudy/phone-out work/kvstudy/data work/kvstudy/report/narrative.html \
        > work/kvstudy/report/index.html

narrative.html holds the hand-written verdict and notes (HTML fragments separated by <!-- section: NAME --> lines);
everything numeric comes from analyze.py's loaders, so the page and the markdown tables never disagree.
"""

from __future__ import annotations

import html
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]]
NARRATIVE = Path(sys.argv[3])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

DEPTHS = (1900, 7900, 32500)
WIN_ORDER = ("2k", "8k", "32k")
REVISION = A.REVISION


def direct_kl(lp_a: list[float], lp_b: list[float]) -> float:
    """KL(a || b) between two files' distributions on the reference's top-64 tokens plus one bucket for the rest."""
    pa = [math.exp(v) for v in lp_a]
    pb = [math.exp(v) for v in lp_b]
    ta, tb = max(1 - sum(pa), 1e-12), max(1 - sum(pb), 1e-12)
    return sum(x * (la - lb) for x, la, lb in zip(pa, lp_a, lp_b)) + ta * (math.log(ta) - math.log(tb))


def collect(out_dir: Path, pair: tuple[str, str], models: tuple[str, ...]) -> dict:
    from transformers import AutoTokenizer

    def ld(kind: str, w: str, m: str, kv: str):
        f = out_dir / f"{kind}-{w}-{m}-{kv}.json"
        return json.loads(f.read_text()) if f.exists() else None

    ka_, kb_ = pair
    out: dict = {"pair": pair, "quality": [], "needles": [], "app": [], "speed": [], "memory": []}
    for m in models:
        tok = None
        for w in WIN_ORDER:
            a, b = ld("stream", w, m, ka_), ld("stream", w, m, kb_)
            if not a or not b:
                continue
            pa, pb = A.per_depth(a), A.per_depth(b)
            hf: dict = {}
            for book in a["books"]:
                meta = json.loads((A.DATA / m / f"book{book['book']}.json").read_text())
                for d, v in meta["hf_nll"].items():
                    hf.setdefault(int(d), []).append(v)
            for d in sorted(pa):
                if d not in pb:
                    continue
                diff = [y - x for x, y in zip(pa[d]["kl"], pb[d]["kl"])]
                lo, hi = A.boot_blocks(diff)
                ppl = lambda v: math.exp(sum(v) / len(v))  # noqa: E731
                row = {
                    "model": m, "window": w, "books": len(a["books"]), "depth": d,
                    "kl_f": st.mean(pa[d]["kl"]), "kl_q": st.mean(pb[d]["kl"]), "d": sum(diff) / len(diff), "lo": lo, "hi": hi,
                    "ppl_f": ppl(pa[d]["nll"]), "ppl_q": ppl(pb[d]["nll"]), "ppl_hf": math.exp(st.mean(hf[d])),
                    "t1_f": st.mean(pa[d]["top1"]), "t1_q": st.mean(pb[d]["top1"]), "direct": None,
                }
                if pa[d]["ref_lp"] and pb[d]["ref_lp"]:
                    dk = [direct_kl(x, y) for x, y in zip(pa[d]["ref_lp"], pb[d]["ref_lp"])]
                    row["direct"] = (st.mean(dk), max(dk))
                out["quality"].append(row)
            tok = tok or AutoTokenizer.from_pretrained(REVISION[m][0], revision=REVISION[m][1])
            win = {"2k": "2048", "8k": "8192", "32k": "32768"}[w]
            res: dict = {}
            hfc = [0, 0]
            for kv, run in ((ka_, a), (kb_, b)):
                for book in run["books"]:
                    meta = json.loads((A.DATA / m / f"book{book['book']}.json").read_text())
                    qs = {q["key"]: q for q in meta["questions"][win]}
                    for nd in book["needles"]:
                        ok = qs[nd["key"]]["number"] in tok.decode(nd["ids"], skip_special_tokens=True)
                        res.setdefault((book["book"], nd["key"]), {})[kv] = ok
                        if kv == ka_:
                            hfc[0] += qs[nd["key"]]["hf_correct"]
                            hfc[1] += 1
            pairs = [v for v in res.values() if ka_ in v and kb_ in v]
            out["needles"].append({
                "model": m, "window": w, "books": len(a["books"]), "hf": hfc, "n": len(pairs),
                "f": sum(v[ka_] for v in pairs), "q": sum(v[kb_] for v in pairs),
                "both": sum(v[ka_] and v[kb_] for v in pairs), "only_f": sum(v[ka_] and not v[kb_] for v in pairs),
                "only_q": sum(v[kb_] and not v[ka_] for v in pairs), "neither": sum(not v[ka_] and not v[kb_] for v in pairs),
            })
            bands = []
            for lo_b, hi_b in A.BANDS:
                vals = []
                for run in (a, b):
                    toks = us = 0
                    for book in run["books"]:
                        for start, n, t in book["prefill_chunks"]:
                            if lo_b <= start < hi_b:
                                toks, us = toks + n, us + t
                    vals.append(toks / us * 1e6 if us else None)
                bands.append(vals)
            decode = {d: (1e6 / st.median(pa[d]["step_us"]), 1e6 / st.median(pb[d]["step_us"])) for d in pa if d in pb}
            def levels(run: dict, d: int) -> str | None:  # GPU thermal cap during this depth's decode (merge_books.py)
                g = [c["gpu"]["level"] for bk in run["books"] for c in bk["checkpoints"] if c["d"] == d and c.get("gpu")]
                if not g:
                    return None
                lo, hi = min(x[0] for x in g), max(x[1] for x in g)
                return f"L{lo}" if lo == hi else f"L{lo}–{hi}"
            clk = {d: (levels(a, d), levels(b, d)) for d in decode}
            out["speed"].append({"model": m, "window": w, "books": len(a["books"]), "prefill": bands, "decode": decode,
                                 "clk": clk if any(v[0] or v[1] for v in clk.values()) else None})
            gpu = [max((bk["end"].get("kgsl") or {}).get("kernel_max", 0) for bk in r["books"]) / 2**20 for r in (a, b)]
            out["memory"].append({
                "model": m, "window": w,
                "after": [st.mean(bk["after_load"]["rss_kb"] for bk in r["books"]) / 1024 for r in (a, b)],
                "peak": [max(bk["end"]["hwm_kb"] for bk in r["books"]) / 1024 for r in (a, b)],
                "gpu": gpu if any(gpu) else None,
            })
        fa, fb = ld("app", "2k", m, ka_), ld("app", "2k", m, kb_)
        if fa and fb:
            ka, kb = [r["kl"] for r in fa["heldout"]], [r["kl"] for r in fb["heldout"]]
            groups = A.reply_groups(fa, fb)
            ra = [k for r in fa["replies"] for k in r["kl"]]
            rb = [k for r in fb["replies"] for k in r["kl"]]
            hd = [y - x for x, y in zip(ka, kb)]
            worst = max(range(len(hd)), key=lambda i: hd[i])
            out["app"].append({
                "model": m, "ho_f": st.mean(ka), "ho_q": st.mean(kb), "hd": st.mean(hd), "hci": A.boot(hd),
                "t1_f": sum(r["top"] == r["fp32_top"] for r in fa["heldout"]), "t1_q": sum(r["top"] == r["fp32_top"] for r in fb["heldout"]),
                "rows": len(ka), "worse": sum(x > 0.1 for x in hd), "better": sum(x < -0.1 for x in hd),
                "worst": (ka[worst], kb[worst]),
                "rk_f": st.mean(ra), "rk_q": st.mean(rb), "rd": st.mean([v for g in groups for v in g]), "rci": A.boot_groups(groups),
            })
    return out


# ---------- rendering ----------

L = A.LABEL
esc = html.escape


def fmt(v: float, p: int = 1) -> str:
    return f"{v:,.{p}f}"


def sig(lo: float, hi: float) -> bool:
    return not (lo <= 0 <= hi)


def decode_chart(rows: list[dict]) -> str:
    """Small multiples: decode tok/s against depth in each model's 32k file, fp32 vs int8 cache, log scale."""
    # Each model's largest measured window (the 32k file where it exists).
    best: dict = {}
    for r in rows:
        if r["model"] not in best or WIN_ORDER.index(r["window"]) > WIN_ORDER.index(best[r["model"]]["window"]):
            best[r["model"]] = r
    panels = list(best.values())
    W, H, pl, pr, pt, pb = 250, 210, 44, 14, 26, 34
    vals = [v for r in panels for pair in r["decode"].values() for v in pair]
    ymin = 10 ** math.floor(math.log10(min(vals))) if vals else 1
    ymax = 10 ** math.ceil(math.log10(max(vals))) if vals else 100
    y = lambda v: pt + (H - pt - pb) * (1 - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)))  # noqa: E731
    x = lambda i: pl + (W - pl - pr) * (i / 2)  # noqa: E731
    out = []
    for r in panels:
        g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Decode speed by depth, {esc(L[r["model"]])}">',
             f'<text x="{pl}" y="16" class="ct">{esc(L[r["model"]])} · {r["window"]} file</text>']
        for t in [x for x in (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000) if ymin <= x <= ymax]:
            g.append(f'<line x1="{pl}" x2="{W - pr}" y1="{y(t):.1f}" y2="{y(t):.1f}" class="grid"/>'
                     f'<text x="{pl - 6}" y="{y(t) + 3.5:.1f}" class="ax" text-anchor="end">{t:g}</text>')
        for i, d in enumerate(DEPTHS):
            g.append(f'<text x="{x(i):.1f}" y="{H - 12}" class="ax" text-anchor="middle">{d / 1000:.1f}k</text>')
        for k, cls in ((0, "s-f"), (1, "s-q")):
            pts = [(x(i), y(r["decode"][d][k])) for i, d in enumerate(DEPTHS) if d in r["decode"]]
            g.append('<polyline class="' + cls + '" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
            for a, b in pts:
                g.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3.2" class="{cls}"/>')
        last = max(r["decode"])
        if last in r["decode"]:
            f, q = r["decode"][last]
            g.append(f'<text x="{x(DEPTHS.index(last)) - 4:.1f}" y="{y(q) - 8:.1f}" class="lab" text-anchor="end">{q / f:.1f}×</text>')
        g.append("</svg>")
        out.append("".join(g))
    return '<div class="multiples">' + "".join(f"<figure>{p}</figure>" for p in out) + "</div>"


def window_chart(rows: list[dict]) -> str:
    """Decode tok/s at 1.9k depth, fp32 cache, in the 2k, 8k and 32k files of each model."""
    W, H, pl, pr, pt, pb = 560, 220, 96, 60, 12, 26
    models = [m for m in A.MODELS if any(r["model"] == m and 1900 in r["decode"] for r in rows)]
    vmax = max(r["decode"][1900][0] for r in rows if 1900 in r["decode"])
    vmax = math.ceil(vmax / 10) * 10
    band = (H - pt - pb) / max(len(models), 1)
    bh = band / 4.2
    xs = lambda v: pl + (W - pl - pr) * v / vmax  # noqa: E731
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Decode speed at 1.9k tokens by window size">']
    for t in range(0, vmax + 1, 20):
        g.append(f'<line x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{pt}" y2="{H - pb}" class="grid"/>'
                 f'<text x="{xs(t):.1f}" y="{H - 8}" class="ax" text-anchor="middle">{t}</text>')
    for i, m in enumerate(models):
        top = pt + i * band + band * 0.12
        g.append(f'<text x="{pl - 8}" y="{top + bh * 1.6:.1f}" class="ax" text-anchor="end">{esc(L[m])}</text>')
        for j, w in enumerate(WIN_ORDER):
            r = next((r for r in rows if r["model"] == m and r["window"] == w and 1900 in r["decode"]), None)
            if not r:
                continue
            v = r["decode"][1900][0]
            yy = top + j * bh
            g.append(f'<rect x="{pl}" y="{yy:.1f}" width="{xs(v) - pl:.1f}" height="{bh - 2:.1f}" class="w{j}"/>'
                     f'<text x="{xs(v) + 5:.1f}" y="{yy + bh - 4:.1f}" class="lab">{w} · {v:.0f}</text>')
    g.append("</svg>")
    return "".join(g)


def forest(rows: list[dict], nb: str = "int8") -> str:
    """KL difference (int8 minus fp32 cache) with its 95% interval, one row per model, window and depth."""
    rows = sorted(rows, key=lambda r: (A.MODELS.index(r["model"]), WIN_ORDER.index(r["window"]), r["depth"]))
    W, rh, pl, pr, pt = 560, 20, 190, 20, 26
    H = pt + rh * len(rows) + 30
    lim = max(0.05, max(max(abs(r["lo"]), abs(r["hi"])) for r in rows))
    lim = math.ceil(lim * 50) / 50
    xs = lambda v: pl + (W - pl - pr) * (v + lim) / (2 * lim)  # noqa: E731
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="KL difference between int8 and fp32 cache with 95% intervals">']
    ticks = [round(-lim + i * lim / 2, 3) for i in range(5)]
    for t in ticks:
        g.append(f'<line x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{pt - 6}" y2="{H - 26}" class="{"zero" if t == 0 else "grid"}"/>'
                 f'<text x="{xs(t):.1f}" y="{H - 10}" class="ax" text-anchor="middle">{t:+.3f}</text>')
    g.append(f'<text x="{xs(-lim):.1f}" y="14" class="ax">{nb} closer to fp32 model</text>'
             f'<text x="{xs(lim):.1f}" y="14" class="ax" text-anchor="end">{nb} further from fp32 model</text>')
    for i, r in enumerate(rows):
        yy = pt + i * rh + rh / 2
        cls = "pt-sig" if sig(r["lo"], r["hi"]) else "pt"
        books = "" if r["books"] == 3 else f" ({r['books']} book)"
        g.append(f'<text x="{pl - 8}" y="{yy + 3.5:.1f}" class="ax" text-anchor="end">{esc(L[r["model"]])} · {r["window"]} @ {r["depth"] / 1000:.1f}k{books}</text>'
                 f'<line x1="{xs(r["lo"]):.1f}" x2="{xs(r["hi"]):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" class="{cls}"/>'
                 f'<circle cx="{xs(r["d"]):.1f}" cy="{yy:.1f}" r="3.4" class="{cls}"/>')
    g.append("</svg>")
    return "".join(g)


def table(head: list[str], body: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def ratio(f: float | None, q: float | None, p: int = 0) -> str:
    if f is None or q is None:
        return "–"
    cls = "up" if q / f >= 1.1 else "down" if q / f <= 0.9 else ""
    return f'{fmt(f, p)} → {fmt(q, p)} <span class="r {cls}">{q / f:.2f}×</span>'


CACHE = {"f32kv": "fp32", "q8kv": "int8", "f16kv": "fp16"}


def pass_block(d: dict, title: str, intro: str, notes: dict) -> str:
    ka, kb = d["pair"]
    na, nb = CACHE[ka], CACHE[kb]
    q = sorted(d["quality"], key=lambda r: (A.MODELS.index(r["model"]), WIN_ORDER.index(r["window"]), r["depth"]))
    direct = any(r["direct"] for r in q)
    qhead = ["model", "window", "books", "depth", f"KL to HF, {na} cache", f"KL to HF, {nb} cache",
             f"change {nb} − {na} (95% CI)"] + ([f"KL({na} ‖ {nb}) mean / max"] if direct else []) + \
            [f"PPL {na} / {nb} / HF", f"top-1 agreement with HF {na} / {nb}"]
    qrows = [[esc(L[r["model"]]), r["window"], str(r["books"]), f'{r["depth"]:,}', f'{r["kl_f"]:.3f}', f'{r["kl_q"]:.3f}',
              f'<span class="{"sigv" if sig(r["lo"], r["hi"]) else ""}">{r["d"]:+.4f}</span> <span class="ci">({r["lo"]:+.4f}, {r["hi"]:+.4f})</span>']
             + ([f'{r["direct"][0]:.5f} / {r["direct"][1]:.4f}' if r["direct"] else "–"] if direct else [])
             + [f'{r["ppl_f"]:.1f} / {r["ppl_q"]:.1f} / {r["ppl_hf"]:.1f}', f'{r["t1_f"]:.0%} / {r["t1_q"]:.0%}'] for r in q]
    nrows = [[esc(L[r["model"]]), r["window"], str(r["books"]), f'{r["hf"][0]}/{r["hf"][1]}', f'{r["f"]}/{r["n"]}', f'{r["q"]}/{r["n"]}',
              f'{r["both"]} / {r["only_f"]} / {r["only_q"]} / {r["neither"]}'] for r in d["needles"]]
    arows = [[esc(L[r["model"]]), f'{r["ho_f"]:.3f} → {r["ho_q"]:.3f}',
              f'<span class="{"sigv" if sig(*r["hci"]) else ""}">{r["hd"]:+.3f}</span> <span class="ci">({r["hci"][0]:+.3f}, {r["hci"][1]:+.3f})</span>',
              f'{r["t1_f"]}/{r["rows"]} → {r["t1_q"]}/{r["rows"]}', f'{r["worse"]} / {r["better"]}', f'{r["worst"][0]:.2f} → {r["worst"][1]:.2f}',
              f'{r["rk_f"]:.3f} → {r["rk_q"]:.3f}',
              f'<span class="{"sigv" if sig(*r["rci"]) else ""}">{r["rd"]:+.3f}</span> <span class="ci">({r["rci"][0]:+.3f}, {r["rci"][1]:+.3f})</span>']
             for r in d["app"]]
    srows = [[esc(L[r["model"]]), r["window"], str(r["books"])] + [ratio(*b) for b in r["prefill"]]
             + [ratio(*r["decode"][dd], 1) if dd in r["decode"] else "–" for dd in DEPTHS] for r in d["speed"]]
    has_clk = any(r.get("clk") for r in d["speed"])
    if has_clk:  # GPU: the thermal cap each decode ran under (0 = uncapped, 1,450 MHz; 12 = 636 MHz)
        for row, r in zip(srows, d["speed"]):
            c = r.get("clk") or {}
            row.append(" · ".join(f"{dd // 100 / 10:g}k: {c[dd][0] or '?'} | {c[dd][1] or '?'}" for dd in DEPTHS if dd in c) or "–")
    gpu = any(r["gpu"] for r in d["memory"])
    mhead = ["model", "window", "resident after load (MiB)", "resident peak (MiB)"] + (["GPU driver memory peak (MiB)"] if gpu else []) + [("GPU driver memory saved (MiB)" if gpu else "resident peak saved (MiB)")]
    mrows = [[esc(L[r["model"]]), r["window"], ratio(*r["after"]), ratio(*r["peak"])]
             + ([ratio(*r["gpu"]) if r["gpu"] else "–"] if gpu else [])
             + [f'{(r["gpu"][0] - r["gpu"][1]) if r["gpu"] else (r["peak"][0] - r["peak"][1]):,.0f}']
             for r in d["memory"]]
    legend = (f'<div class="legend"><span><i style="background:var(--fp32)"></i>{na} cache</span>'
              f'<span><i style="background:var(--int8)"></i>{nb} cache</span><span>tok/s, log scale; each model\'s largest measured window; '
              f'label = {nb} ÷ {na} at the deepest point</span></div>')
    parts = [f'<section class="pass"><h2>{title}</h2>{intro}</section>',
             f'<section><h3>Decode speed against depth</h3>{notes.get("speed", "")}{legend}{decode_chart(d["speed"])}',
             table(["model", "window", "books", "prefill, chunks starting 0–2k", "2–8k", "8–16k", "16–32k", "decode @1.9k", "@7.9k", "@32.5k"]
                   + ([f"GPU thermal level during decode, estimated ({na} | {nb})"] if has_clk else []), srows, "num"),
             "</section>"]
    if notes.get("window"):
        parts += [f'<section><h3>The window itself costs speed</h3>{notes["window"]}<figure>{window_chart(d["speed"])}</figure></section>']
    parts += [f'<section><h3>Memory</h3>{notes.get("memory", "")}', table(mhead, mrows, "num"), "</section>",
              f'<section><h3>Quality</h3>{notes.get("quality", "")}<figure>{forest(q, nb)}</figure>'
              '<div class="legend"><span><i style="background:var(--sig)"></i>95% interval excludes 0</span>'
              '<span><i style="background:var(--fp32)"></i>no detectable difference</span><span>intervals: block bootstrap, 16-position blocks</span></div>',
              "<h4>PG-19 books, against the fp32 Hugging Face model</h4>", table(qhead, qrows, "num"),
              f"<h4>Needles: correct answers, and paired outcomes (both / {na} only / {nb} only / neither). LFM rows on the CPU are not valid: a rewind does not restore its convolution state.</h4>",
              table(["model", "window", "books", "HF fp32", f"{na} cache", f"{nb} cache", "both / only / only / neither"], nrows, "num")]
    if arows:
        parts += [f"<h4>141 held-out app questions (2k file)</h4>",
                  table(["model", "held-out KL", "change (95% CI)", "top-1 agreement", f"rows worse / better by > 0.1", "worst row", "reply KL", "change (95% CI, by reply)"], arows, "num")]
    parts += ["</section>"]
    return "".join(parts)


def rerun_block(d: dict, first: dict, intro: str) -> str:
    """The CPU re-run on the second phone: direct divergence between the two caches' outputs, needles (LFM2's from a
    fresh model) and decode speed set against the first phone's."""
    q = sorted(d["quality"], key=lambda r: (A.MODELS.index(r["model"]), WIN_ORDER.index(r["window"]), r["depth"]))
    qrows = [[esc(L[r["model"]]), r["window"], f'{r["depth"]:,}',
              (f'{r["direct"][0]:.4f}' if r["direct"] else "–"), (f'{r["direct"][1]:.3f}' if r["direct"] else "–"),
              f'{r["d"]:+.4f}'] for r in q]
    nrows = [[esc(L[r["model"]]), r["window"], f'{r["hf"][0]}/{r["hf"][1]}', f'{r["f"]}/{r["n"]}', f'{r["q"]}/{r["n"]}',
              f'{r["both"]} / {r["only_f"]} / {r["only_q"]} / {r["neither"]}'] for r in d["needles"]]
    firsts = {(r["model"], r["window"]): r for r in first["speed"]}
    srows = []
    for r in d["speed"]:
        f0 = firsts.get((r["model"], r["window"]))
        for dd in sorted(r["decode"]):
            a1 = f0["decode"].get(dd) if f0 else None
            srows.append([esc(L[r["model"]]), r["window"], f"{dd:,}", ratio(*r["decode"][dd], 1),
                          ratio(*a1, 1) if a1 else "–"])
    return "".join([
        f'<section class="pass"><h2>CPU re-run on a second phone: direct comparison of the two caches</h2>{intro}</section>',
        "<section><h4>Divergence between the fp32-cache and int8-cache files' outputs (KL(fp32 ‖ int8), coarsened to the "
        "reference's top 64 tokens plus one bucket), next to the change in divergence from the fp32 HF model</h4>",
        table(["model", "window", "depth", "direct KL mean", "direct KL worst position", "change in KL to HF"], qrows, "num"),
        "<h4>Needles on the second phone (LFM2 answered from a fresh model and the full prefix)</h4>",
        table(["model", "window", "HF fp32", "fp32 cache", "int8 cache", "both / fp32 only / int8 only / neither"], nrows, "num"),
        "<h4>Decode tok/s, fp32 → int8 cache: second phone against the first</h4>",
        table(["model", "window", "depth", "second phone", "first phone"], srows, "num"),
        "</section>",
    ])


def render(cpu: dict, gpu: dict | None, rerun: dict | None = None) -> str:
    sections = {}
    if NARRATIVE.exists():
        bits = re.split(r"<!-- section: ([a-z_]+) -->", NARRATIVE.read_text())
        for name, body in zip(bits[1::2], bits[2::2]):
            sections[name] = body.strip()
    s = sections.get
    cpu_block = pass_block(cpu, "CPU (XNNPACK): fp32 against int8 cache", s("cpu_intro", ""),
                           {"speed": s("cpu_speed", ""), "window": s("cpu_window", ""), "memory": s("cpu_memory", ""), "quality": s("cpu_quality", "")})
    gpu_block = pass_block(gpu, "GPU (Vulkan): fp32 against fp16 cache", s("gpu_intro", ""),
                           {"speed": s("gpu_speed", ""), "memory": s("gpu_memory", ""), "quality": s("gpu_quality", "")}) if gpu and gpu["quality"] else ""
    rerun_html = rerun_block(rerun, cpu, s("rerun_intro", "")) if rerun and rerun["quality"] else ""
    return TEMPLATE.format(verdict=s("verdict", ""), setup=s("setup", ""), cpu=cpu_block + rerun_html, gpu=gpu_block,
                           method=s("method", ""), notes=s("notes", ""), codex=s("codex", ""))




TEMPLATE = """<title>KV Cache Study</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: one reading column of prose; charts and tables may run wider; the phone's numbers set in mono. */
:root {{
  --bg: #f6f7f8; --surface: #ffffff; --fg: #17202a; --muted: #5a6672; --rule: #d9dee3;
  --fp32: #6b7785; --int8: #0f7c74; --sig: #b4441f; --w0: #9fb7c8; --w1: #5d86a3; --w2: #2d4f69;
  --display: "Space Grotesk", "IBM Plex Sans", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12171c; --surface: #182029; --fg: #e4e9ee; --muted: #9aa6b2; --rule: #2a343e;
  --fp32: #8c98a5; --int8: #3fb8ac; --sig: #ec7b52; --w0: #3b5568; --w1: #5f8aa8; --w2: #9cc3de; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #12171c; --surface: #182029; --fg: #e4e9ee; --muted: #9aa6b2; --rule: #2a343e;
  --fp32: #8c98a5; --int8: #3fb8ac; --sig: #ec7b52; --w0: #3b5568; --w1: #5f8aa8; --w2: #9cc3de; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }}
main {{ max-width: 1000px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 40px 72px; display: grid; gap: 34px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 12px/1.4 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 600 clamp(28px, 4.6vw, 40px)/1.12 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 24px/1.25 var(--display); margin: 0; text-wrap: balance; }}
h3 {{ font: 600 18px/1.3 var(--display); margin: 0; }}
h4 {{ font: 600 14px/1.3 var(--body); margin: 8px 0 0; }}
section {{ display: grid; gap: 12px; min-width: 0; }}
section.pass {{ border-top: 2px solid var(--fg); padding-top: 18px; }}
p, li {{ max-width: 70ch; margin: 0; }}
ul {{ margin: 0; padding-left: 1.2em; display: grid; gap: 6px; }}
.verdict {{ background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--int8); padding: 18px 20px; display: grid; gap: 10px; }}
.key {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }}
.key div {{ background: var(--surface); border: 1px solid var(--rule); padding: 12px 14px; display: grid; gap: 2px; }}
.key b {{ font: 600 22px/1.2 var(--mono); }}
.key span {{ color: var(--muted); font-size: 13px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font: 13px var(--mono); color: var(--muted); }}
.legend i {{ display: inline-block; width: 14px; height: 3px; vertical-align: middle; margin-right: 6px; }}
.multiples {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
figure {{ margin: 0; background: var(--surface); border: 1px solid var(--rule); padding: 8px; min-width: 0; }}
svg {{ width: 100%; height: auto; display: block; }}
.ct {{ font: 600 13px var(--body); fill: var(--fg); }}
.ax {{ font: 11px var(--mono); fill: var(--muted); }}
.lab {{ font: 500 11px var(--mono); fill: var(--fg); }}
.grid {{ stroke: var(--rule); stroke-width: 1; }}
.zero {{ stroke: var(--muted); stroke-width: 1.2; }}
polyline {{ fill: none; stroke-width: 2.2; }}
polyline.s-f, circle.s-f {{ stroke: var(--fp32); }} circle.s-f {{ fill: var(--fp32); }}
polyline.s-q, circle.s-q {{ stroke: var(--int8); }} circle.s-q {{ fill: var(--int8); }}
rect.w0 {{ fill: var(--w0); }} rect.w1 {{ fill: var(--w1); }} rect.w2 {{ fill: var(--w2); }}
line.pt {{ stroke: var(--fp32); stroke-width: 2; }} circle.pt {{ fill: var(--fp32); }}
line.pt-sig {{ stroke: var(--sig); stroke-width: 2; }} circle.pt-sig {{ fill: var(--sig); }}
.tw {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th {{ font: 500 11px var(--mono); letter-spacing: .03em; text-transform: uppercase; color: var(--muted); }}
table.num td {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
table.num td:first-child {{ font-family: var(--body); }}
tbody tr:last-child td {{ border-bottom: 0; }}
.r {{ color: var(--muted); }} .r.up {{ color: var(--int8); font-weight: 500; }} .r.down {{ color: var(--sig); font-weight: 500; }}
.sigv {{ color: var(--sig); font-weight: 500; }} .ci {{ color: var(--muted); }}
.notes {{ display: grid; gap: 8px; color: var(--muted); }}
code {{ font: 13px var(--mono); }}
</style>
<main>
<header>
  <div class="eyebrow">ExecuPack · ExecuTorch 1.5.1 Android runtime · Snapdragon SM8975 reference phone · 2026-11-21</div>
  <h1>KV cache precision on a phone: CPU and GPU</h1>
  <p>Four small models, 2k / 8k / 32k windows, the same 4-bit GPTQ weights in every file of a pair. On the CPU each pair differs only in how the KV cache is stored; on the GPU the fp16 file also computes in fp16.</p>
</header>
<section class="verdict">{verdict}</section>
<section><h2>What was measured</h2>{setup}</section>
{cpu}
{gpu}
<section class="pass"><h2>Independent review</h2>{codex}</section>
<section><h2>Method</h2>{method}</section>
<section><h2>Notes and limits</h2><div class="notes">{notes}</div></section>
</main>
"""

if __name__ == "__main__":
    cpu = collect(Path(sys.argv[1]), ("f32kv", "q8kv"), A.MODELS)
    gpu_dir = Path(sys.argv[1]).parent / "gpu-out"
    gpu = collect(gpu_dir, ("f32kv", "f16kv"), ("llama", "qwen")) if gpu_dir.exists() else None
    rerun_dir = Path(sys.argv[1]).parent / "phone-out2"
    rerun = collect(rerun_dir, ("f32kv", "q8kv"), A.MODELS) if rerun_dir.exists() else None
    print(render(cpu, gpu, rerun))
