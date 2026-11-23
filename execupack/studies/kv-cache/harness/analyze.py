"""KV study tables from the phone's results (StudyBench) and the reference data.

    python scripts/kvstudy/analyze.py work/kvstudy/phone-out work/kvstudy/data > work/kvstudy/report.md

Quality is against the fp32 Hugging Face model; every int8-cache number is also given as int8 minus fp32 cache,
paired by position (same file weights, same tokens), with a 95% bootstrap interval over the scored positions.
Speed is the runtime's own forward() time on the phone: prefill tok/s by depth band (each chunk of up to 2,047
tokens is timed whole), decode tok/s as the median of the 127 single-token steps at each checkpoint.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from pathlib import Path

OUT, DATA = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = ("llama", "lfm", "qwen", "gemma")
WINDOWS = ("2k", "8k", "32k")
import os
# The cache pair compared: the CPU pass is fp32 against int8; the GPU pass fp32 against fp16 (KV_PAIR=f32kv,f16kv).
KVS = tuple(os.environ.get("KV_PAIR", "f32kv,q8kv").split(","))
LABEL = {"llama": "Llama-3.2-1B", "lfm": "LFM2.5-1.2B", "qwen": "Qwen3-1.7B", "gemma": "Gemma-3-1B"}
REVISION = {
    "llama": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
    "lfm": ("LiquidAI/LFM2.5-1.2B-Instruct", "0f604ada3f766f9f257460c4c9f0b5d6f69d431b"),
    "qwen": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
    "gemma": ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752"),
}
BANDS = ((0, 2048), (2048, 8192), (8192, 16384), (16384, 32768))


def load(kind: str, w: str, m: str, kv: str):
    p = OUT / f"{kind}-{w}-{m}-{kv}.json"
    return json.loads(p.read_text()) if p.exists() else None


def boot(diffs: list[float], n: int = 2000) -> tuple[float, float]:
    """Percentile interval of the mean, resampling single values (for independent rows such as held-out questions)."""
    rng = random.Random(0)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def boot_groups(groups: list[list[float]], n: int = 2000) -> tuple[float, float]:
    """Percentile interval of the mean, resampling whole groups (clusters) with replacement."""
    rng = random.Random(0)
    means = []
    for _ in range(n):
        pick = rng.choices(groups, k=len(groups))
        flat = [v for g in pick for v in g]
        means.append(sum(flat) / len(flat))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


BLOCK = 16


def boot_blocks(diffs: list[float], n: int = 2000) -> tuple[float, float]:
    """Block bootstrap: neighbouring teacher-forced positions share context, so resample runs of BLOCK consecutive
    positions. Each book contributes 128 positions per checkpoint, so blocks never span two books."""
    return boot_groups([diffs[i : i + BLOCK] for i in range(0, len(diffs), BLOCK)], n)


def fmt_ci(d: list[float], blocks: bool = True) -> str:
    lo, hi = boot_blocks(d) if blocks else boot(d)
    star = "" if lo <= 0 <= hi else " *"
    return f"{sum(d) / len(d):+.4f} ({lo:+.4f}, {hi:+.4f}){star}"


def reply_groups(a, b) -> list[list[float]]:
    """Per-reply lists of (b - a) KL: positions within one reply share its prompt and trajectory, so a reply is one
    cluster (47 replies, every third held-out question)."""
    return [[y - x for x, y in zip(ra["kl"], rb["kl"])] for ra, rb in zip(a["replies"], b["replies"])]


def fmt_reply(a, b) -> str:
    g = reply_groups(a, b)
    flat = [v for x in g for v in x]
    lo, hi = boot_groups(g)
    star = "" if lo <= 0 <= hi else " *"
    return f"{sum(flat) / len(flat):+.4f} ({lo:+.4f}, {hi:+.4f}){star}"


def per_depth(run) -> dict[int, dict[str, list[float]]]:
    """Checkpoint d -> concatenated per-position metrics over the books."""
    out: dict[int, dict[str, list[float]]] = {}
    for book in run["books"]:
        for cp in book["checkpoints"]:
            o = out.setdefault(cp["d"], {"kl": [], "nll": [], "top1": [], "step_us": [], "ref_lp": []})
            o["kl"] += cp["kl"]
            o["nll"] += cp["nll"]
            o["top1"] += cp["top1"]
            o["step_us"] += cp["step_us"][1:]  # step 0 is the prefill's output, not a decode step
            o["ref_lp"] += cp.get("ref_lp", [])
    return out


def quality(lines: list[str]) -> None:
    lines.append("## Quality on PG-19 (3 books), against the fp32 Hugging Face model\n")
    lines.append("KL is over fp32's top-64 tokens plus a bucket for the rest; PPL from the file's NLL of the true "
                 "token; * marks a difference whose 95% interval excludes 0.\n")
    lines.append("| model | window | books | depth | KL fp32 cache | KL int8 cache | KL Δ int8 − fp32 (95% CI) | PPL fp32 / int8 / HF | top-1 fp32 / int8 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for w in WINDOWS:
            a, b = load("stream", w, m, "f32kv"), load("stream", w, m, "q8kv")
            if not a or not b:
                continue
            pa, pb = per_depth(a), per_depth(b)
            hf = {}
            for book in a["books"]:
                meta = json.loads((DATA / m / f"book{book['book']}.json").read_text())
                for d, v in meta["hf_nll"].items():
                    hf.setdefault(int(d), []).append(v)
            for d in sorted(pa):
                if d not in pb:
                    continue
                ka, kb = pa[d]["kl"], pb[d]["kl"]
                diff = [y - x for x, y in zip(ka, kb)]
                ppl = lambda v: math.exp(sum(v) / len(v))  # noqa: E731
                lines.append(
                    f"| {LABEL[m]} | {w} | {len(a['books'])} | {d:,} | {st.mean(ka):.4f} | {st.mean(kb):.4f} | {fmt_ci(diff)} | "
                    f"{ppl(pa[d]['nll']):.2f} / {ppl(pb[d]['nll']):.2f} / {math.exp(st.mean(hf[d])):.2f} | "
                    f"{st.mean(pa[d]['top1']):.1%} / {st.mean(pb[d]['top1']):.1%} |"
                )
    lines.append("")


def needles(lines: list[str]) -> None:
    from transformers import AutoTokenizer

    lines.append("## Needle retrieval (RULER-style single needles, 10/50/90% of each window, 3 books)\n")
    lines.append("| model | window | books | HF fp32 | fp32 cache | int8 cache |")
    lines.append("|---|---|---|---|---|---|")
    for m in MODELS:
        tok = None
        for w in WINDOWS:
            runs = {kv: load("stream", w, m, kv) for kv in KVS}
            if not all(runs.values()):
                continue
            tok = tok or AutoTokenizer.from_pretrained(REVISION[m][0], revision=REVISION[m][1])
            win = {"2k": "2048", "8k": "8192", "32k": "32768"}[w]
            score = {kv: [0, 0] for kv in KVS}
            hf = [0, 0]
            for kv, run in runs.items():
                for book in run["books"]:
                    meta = json.loads((DATA / m / f"book{book['book']}.json").read_text())
                    qs = {q["key"]: q for q in meta["questions"][win]}
                    for nd in book["needles"]:
                        answer = tok.decode(nd["ids"], skip_special_tokens=True)
                        score[kv][0] += qs[nd["key"]]["number"] in answer
                        score[kv][1] += 1
                        if kv == "f32kv":
                            hf[0] += qs[nd["key"]]["hf_correct"]
                            hf[1] += 1
            lines.append(f"| {LABEL[m]} | {w} | {len(runs['f32kv']['books'])} | {hf[0]}/{hf[1]} | " + " | ".join(f"{score[kv][0]}/{score[kv][1]}" for kv in KVS) + " |")
    lines.append("")


def app(lines: list[str]) -> None:
    lines.append("## The 141 held-out app questions (2k file), against the fp32 Hugging Face model\n")
    lines.append("| model | held-out KL fp32 / int8 | Δ (95% CI) | top-1 fp32 / int8 | reply KL fp32 / int8 | Δ (95% CI) |")
    lines.append("|---|---|---|---|---|---|")
    for m in MODELS:
        a, b = load("app", "2k", m, "f32kv"), load("app", "2k", m, "q8kv")
        if not a or not b:
            continue
        ka = [r["kl"] for r in a["heldout"]]
        kb = [r["kl"] for r in b["heldout"]]
        ta = st.mean(r["top"] == r["fp32_top"] for r in a["heldout"])
        tb = st.mean(r["top"] == r["fp32_top"] for r in b["heldout"])
        ra = [k for r in a["replies"] for k in r["kl"]]
        rb = [k for r in b["replies"] for k in r["kl"]]
        lines.append(
            f"| {LABEL[m]} | {st.mean(ka):.4f} / {st.mean(kb):.4f} | {fmt_ci([y - x for x, y in zip(ka, kb)], blocks=False)} | "
            f"{ta:.1%} / {tb:.1%} | {st.mean(ra):.4f} / {st.mean(rb):.4f} | {fmt_reply(a, b)} |"
        )
    lines.append("")


def speed(lines: list[str]) -> None:
    lines.append("## Speed on the phone (tok/s), fp32 cache → int8 cache\n")
    lines.append("Prefill: tokens / time over the chunks that start in each band, all books. Decode: 1e6 / median "
                 "step time at each checkpoint.\n")
    lines.append("| model | window | books | prefill 0–2k | 2–8k | 8–16k | 16–32k | decode @1.9k | @7.9k | @32.5k |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for w in WINDOWS:
            runs = {kv: load("stream", w, m, kv) for kv in KVS}
            if not all(runs.values()):
                continue
            cells = [str(len(runs["f32kv"]["books"]))]
            for lo, hi in BANDS:
                vals = []
                for kv in KVS:
                    toks = us = 0
                    for book in runs[kv]["books"]:
                        for start, n, t in book["prefill_chunks"]:
                            if lo <= start < hi:
                                toks, us = toks + n, us + t
                    vals.append(toks / us * 1e6 if us else None)
                cells.append("–" if vals[0] is None else f"{vals[0]:.0f} → {vals[1]:.0f} ({vals[1] / vals[0]:.2f}×)")
            dec = []
            for d in (1900, 7900, 32500):
                v = []
                for kv in KVS:
                    pd = per_depth(runs[kv])
                    v.append(1e6 / st.median(pd[d]["step_us"]) if d in pd else None)
                dec.append("–" if v[0] is None else f"{v[0]:.1f} → {v[1]:.1f} ({v[1] / v[0]:.2f}×)")
            lines.append(f"| {LABEL[m]} | {w} | " + " | ".join(cells + dec) + " |")
    lines.append("\n## Memory on the phone (MiB), fp32 cache → int8 cache\n")
    lines.append("| model | window | after load | peak |")
    lines.append("|---|---|---|---|")
    for m in MODELS:
        for w in WINDOWS:
            runs = {kv: load("stream", w, m, kv) for kv in KVS}
            if not all(runs.values()):
                continue
            after = [st.mean(b["after_load"]["rss_kb"] for b in runs[kv]["books"]) / 1024 for kv in KVS]
            peak = [max(b["end"]["hwm_kb"] for b in runs[kv]["books"]) / 1024 for kv in KVS]
            lines.append(f"| {LABEL[m]} | {w} | {after[0]:,.0f} → {after[1]:,.0f} | {peak[0]:,.0f} → {peak[1]:,.0f} |")
    lines.append("")


def main() -> None:
    lines = ["# KV cache study: fp32 vs int8 cache on CPU (XNNPACK), measured on the phone\n"]
    for part in (quality, needles, app, speed):
        part(lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
