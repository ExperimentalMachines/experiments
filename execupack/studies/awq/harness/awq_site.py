"""Writes the 4-bit rounding study's numbers (RTN, GPTQ, AWQ, AWQ + GPTQ and the published file) as a TypeScript data
module for the Experimental Machines research post (experimentalmachines.org: lib/quant-data.ts).

    HF_TOKEN=... python scripts/awq_site.py work/awq ../kvstudy-data <site>/lib/quant-data.ts

Reads work/awq/8k/<model>/<arm>/: result-<method>.json (publish gate, app questions), rows-<method>.json (per question
and per reply) and books-<method>.json (scripts/awq_books.py: the KV study's long-text measurements on the same 8k
file), with the KV study's book data (bookN.json: the fp32 model's perplexity and the needles' answers) to score them.
Also reads the first round's arms (work/awq/<model>/<arm>/, 2k files solved on Modal) for their solve times, measured
one method per 8-core sandbox, and to set their app-question results beside this round's.

Intervals are paired percentile bootstraps against GPTQ calibrated the same way (4,000 resamples, seed 0): per run of
16 neighbouring positions within a book for the long text (the KV study's blocks), per question for the app's first
tokens, and per reply for the reply positions.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]).resolve()
DATA = Path(sys.argv[2]).resolve()
OUT = Path(sys.argv[3]).resolve()

MODELS = [
    # key, name, KV-study data dir, HF id and revision (for the needles' tokenizer), the token that opens a tool call
    (
        "lfm25-1.2b",
        "LFM2.5 1.2B",
        "lfm",
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
        10,
    ),
    (
        "llama32-1b",
        "Llama 3.2 1B",
        "llama",
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
        5018,
    ),
    ("qwen3-1.7b", "Qwen3 1.7B", "qwen", "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", 151657),
]
# This round's arm directories -> (method, calibration). Qwen3 is the one reasoning model: its comparison uses every
# method calibrated with thinking off, as the app sends a turn; "gptq" there is the thinking-on solve, for contrast.
ARMS = {
    "rtn": ("rtn", None),
    "published": ("published", None),
    "gptq": ("gptq", "on"),
    "awq": ("awq", "on"),
    "awq_gptq": ("awq_gptq", "on"),
    "gptq-nothink": ("gptq", "off"),
    # The KV-cache study's own GPTQ file of each model (same calibration text, solved and built separately): a second
    # GPTQ solve, to show how far two solves of one method land apart.
    "gptq-kv": ("published", "kv"),
    "awq-nothink": ("awq", "off"),
    "awq_gptq-nothink": ("awq_gptq", "off"),
}
BLOCK = 16
TOP = 64
REF_ROW = [("t", "<i4"), ("tid", "<i4"), ("tlp", "<f4"), ("ids", "<i4", (TOP,)), ("lp", "<f4", (TOP,))]


def boot(diffs, n=4000, seed=0):
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [sum(diffs) / k, means[int(0.025 * n)], means[int(0.975 * n)]]


def boot_groups(groups, n=4000, seed=0):
    rng = random.Random(seed)
    flat = [v for g in groups for v in g]
    means = []
    for _ in range(n):
        vals = [v for _ in groups for v in groups[rng.randrange(len(groups))]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return [sum(flat) / len(flat), means[int(0.025 * n)], means[int(0.975 * n)]]


def boot_ratio(pairs, n=4000, seed=0, alpha=0.05):
    """pairs: groups (lists) of (file, gptq) values, resampled whole. Returns the change in the mean against GPTQ's,
    as a fraction of GPTQ's (mean(file) / mean(gptq) - 1), with its 95% percentile interval."""
    rng = random.Random(seed)

    def ratio(gs):
        a = sum(x for g in gs for x, _ in g)
        b = sum(y for g in gs for _, y in g)
        return a / b - 1

    stats = sorted(ratio([pairs[rng.randrange(len(pairs))] for _ in pairs]) for _ in range(n))
    return [ratio(pairs), stats[int(alpha / 2 * n)], stats[int((1 - alpha / 2) * n) - 1]]


def r(x, d=4):
    return None if x is None else round(x, d)


def pct(xs, q):
    """The q-th percentile (0-100) by linear interpolation, as numpy's default."""
    xs = sorted(xs)
    at = (len(xs) - 1) * q / 100
    lo = int(at)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (at - lo)


def tails(kl):
    """llama.cpp's KL statistics (tools/perplexity, --kl-divergence): the mean hides the few positions a file gets
    badly wrong, so the median, the 99th percentile and the maximum beside it."""
    return {"median": r(pct(kl, 50)), "p90": r(pct(kl, 90)), "p99": r(pct(kl, 99)), "max": r(max(kl))}


def delta_p(dp):
    """The change in the true next token's probability against the original model (llama.cpp's Δp): the mean says
    whether the file predicts the text better or worse; RMS is the noise; a 1st percentile much further from zero than
    the 99th means the file mostly loses probability rather than scattering it."""
    return {
        "mean": r(sum(dp) / len(dp)),
        "rms": r(math.sqrt(sum(x * x for x in dp) / len(dp))),
        "p1": r(pct(dp, 1)),
        "p99": r(pct(dp, 99)),
    }


def gen_summary(g):
    if not g:
        return None
    rs = g["replies"]
    n = len(rs)
    return {
        "replies": n,
        "same": sum(x["same"] for x in rs),
        "prefix": r(sum(x["prefix"] for x in rs) / n, 2),
        "refLen": r(sum(len(x["want"]) for x in rs) / n, 2),
        # Replies that matched the original's for at least its first 8 tokens.
        "prefix8": sum(x["prefix"] >= min(8, len(x["want"])) for x in rs),
        "lcsF1": r(sum(x["lcsF1"] for x in rs) / n, 3),
        # The original's replies that ran to the 48-token cap rather than ending.
        "capped": sum(len(x["want"]) >= 48 for x in rs),
    }


def pooled(depths):
    if not depths:
        return None
    kl = [x for v in depths.values() for x in v["kl"]]
    nll = [x for v in depths.values() for x in v["nll"]]
    top1 = [x for v in depths.values() for x in v["top1"]]
    hf = [x for v in depths.values() for x in v["hf_nll"]]  # one per book and depth, each over the same 128 positions
    return {
        "kl": r(sum(kl) / len(kl)),
        "ppl": r(math.exp(sum(nll) / len(nll)), 2),
        "hfPpl": r(math.exp(sum(hf) / len(hf)), 2),
        "top1": r(sum(top1) / len(top1), 3),
        "positions": len(kl),
        "klTails": tails(kl),
        "dp": delta_p([x for v in depths.values() for x in v["dp"]]),
    }


def load(d: Path, method: str):
    f = lambda kind: d / f"{kind}-{method}.json"  # noqa: E731
    if not f("result").exists():
        return None
    return {k: json.loads(f(k).read_text()) if f(k).exists() else None for k in ("result", "rows", "books", "gen")}


def main() -> None:
    from transformers import AutoTokenizer

    models = []
    for key, name, data, hf_id, rev, call in MODELS:
        tok = AutoTokenizer.from_pretrained(hf_id, revision=rev)
        metas = {
            p.stem.removeprefix("book"): json.loads(p.read_text()) for p in sorted((DATA / data).glob("book*.json"))
        }
        # The original model's log-probability of each true next token (awq_books.REF_ROW's tlp), by book.
        refs = {
            book: np.fromfile(DATA / data / f"book{book}.ref.bin", dtype=np.dtype(REF_ROW))[["t", "tlp"]]
            for book in metas
        }
        arms, raw = {}, {}
        for arm, (method, calib) in ARMS.items():
            got = load(ROOT / "8k" / key / arm, method)
            if not got:
                continue
            res, rows, bk = got["result"], got["rows"], got["books"]
            per = rows["per_row"]
            calls = [x for x in per if x["fp32_top"] == call]
            depths, needles = {}, {"found": 0, "of": 0, "hf": 0, "hits": [], "hfHits": []}
            for b in (bk or {}).get("books", []):
                meta = metas[str(b["book"])]
                for c in b["checkpoints"]:
                    dd = depths.setdefault(str(c["d"]), {"kl": [], "nll": [], "top1": [], "hf_nll": [], "dp": []})
                    dd["kl"] += c["kl"]
                    dd["nll"] += c["nll"]
                    dd["top1"] += c["top1"]
                    dd["hf_nll"].append(meta["hf_nll"][str(c["d"])])
                    # The scored positions are the reference rows from t = d - scored on (awq_books.books).
                    ref = refs[str(b["book"])]
                    row = int(np.searchsorted(ref["t"], c["d"] - meta["scored"]))
                    hf_p = np.exp(ref["tlp"][row : row + len(c["nll"])].astype(np.float64))
                    dd["dp"] += [float(math.exp(-n) - h) for n, h in zip(c["nll"], hf_p, strict=True)]
                qs = {q["key"]: q for q in meta["questions"][str(bk["window"])]}
                for nd in b["needles"]:
                    q = qs[nd["key"]]
                    hit = q["number"] in tok.decode(nd["ids"], skip_special_tokens=True)
                    needles["of"] += 1
                    needles["found"] += hit
                    needles["hf"] += bool(q["hf_correct"])
                    # In question order, book by book: the figure lays them out one book to a row.
                    needles["hits"].append(int(hit))
                    needles["hfHits"].append(int(bool(q["hf_correct"])))
            # Flips (Dutta et al., "Accuracy is Not All You Need"): questions the original answered and the file did
            # not, and the reverse.
            needles["lost"] = sum(h and not f for f, h in zip(needles["hits"], needles["hfHits"], strict=True))
            needles["gained"] = sum(f and not h for f, h in zip(needles["hits"], needles["hfHits"], strict=True))
            arms[arm] = {
                "method": method,
                "calibration": calib,
                "books": {
                    d: {
                        "kl": r(sum(v["kl"]) / len(v["kl"])),
                        "ppl": r(math.exp(sum(v["nll"]) / len(v["nll"])), 2),
                        "hfPpl": r(math.exp(sum(v["hf_nll"]) / len(v["hf_nll"])), 2),
                        "top1": r(sum(v["top1"]) / len(v["top1"]), 3),
                        "positions": len(v["kl"]),
                    }
                    for d, v in depths.items()
                },
                "booksAll": pooled(depths),
                "needles": needles,
                "app": {
                    "kl": r(res["heldout"]["mean_kl"]),
                    "top1": r(res["heldout"]["top1_agree"], 3),
                    "medianKl": r(sorted(x["kl"] for x in per)[len(per) // 2]),
                    "klTails": tails([x["kl"] for x in per]),
                    # First tokens that changed from the original's: the app's own flips.
                    "flips": sum(x["top"] != x["fp32_top"] for x in per),
                    "replyKl": r(res["replies"]["mean_kl"]),
                    "replyTop1": r(res["replies"]["top1_agree"], 3),
                },
                "calls": {
                    "of": len(calls),
                    "kept": sum(x["top"] == call for x in calls),
                    "extra": sum(x["top"] == call and x["fp32_top"] != call for x in per),
                },
                "gate": {
                    "passed": bool(res["gate"].get("passed")),
                    "agreed": res["gate"].get("agreed"),
                    "graded": res["gate"].get("graded"),
                    "fp32Mean": r(res["gate"].get("fp32_mean")),
                    "exportMean": r(res["gate"].get("export_mean")),
                },
                "bytes": res.get("bytes"),
                # Free-running replies against the original's greedy ones (awq_compare.py generate).
                "gen": gen_summary(got["gen"]),
            }
            raw[arm] = got
            # The first round (2k files solved on Modal, one method per 8-core sandbox): solve time and app result.
            first = load(ROOT / key / arm, method)
            if first:
                fr = first["result"]
                arms[arm]["firstRound"] = {
                    "solveS": fr["timings_s"].get("solve"),
                    "appKl": r(fr["heldout"]["mean_kl"]),
                    "replyKl": r(fr["replies"]["mean_kl"]),
                    "gatePassed": bool(fr["gate"].get("passed")),
                    # The same questions' first tokens in the 2k build against this round's 8k build. For RTN the
                    # 4-bit values are identical, so every change is the rebuild alone.
                    "topChanged": sum(
                        a["top"] != b["top"] for a, b in zip(first["rows"]["per_row"], rows["per_row"], strict=True)
                    ),
                }
        # Paired against GPTQ calibrated the same way.
        vs = {}
        for arm, a in arms.items():
            # Qwen3's files are compared with GPTQ calibrated with thinking off, except the second solve, which is set
            # against the solve it repeats (thinking on): the same rows, solved again.
            base = "gptq-nothink" if (key == "qwen3-1.7b" and arm not in ("gptq", "gptq-kv")) else "gptq"
            if arm == base or base not in raw:
                continue
            x, y = raw[arm], raw[base]
            entry = {"against": base}
            entry["app"] = [
                r(v)
                for v in boot_ratio(
                    [[(p["kl"], q["kl"])] for p, q in zip(x["rows"]["per_row"], y["rows"]["per_row"], strict=True)]
                )
            ]
            entry["reply"] = [
                r(v)
                for v in boot_ratio(
                    [
                        list(zip([p["kl"] for p in ra["positions"]], [q["kl"] for q in rb["positions"]], strict=True))
                        for ra, rb in zip(x["rows"]["per_reply"], y["rows"]["per_reply"], strict=True)
                    ]
                )
            ]
            if x["gen"] and y["gen"]:
                diffs = [a["lcsF1"] - b["lcsF1"] for a, b in zip(x["gen"]["replies"], y["gen"]["replies"], strict=True)]
                entry["genLcs"] = [r(v) for v in boot(diffs)]
            if x["books"] and y["books"]:
                every = []
                for d in a["books"]:
                    blocks = []
                    for bx, by in zip(x["books"]["books"], y["books"]["books"], strict=True):
                        assert bx["book"] == by["book"]
                        cx = next(c for c in bx["checkpoints"] if str(c["d"]) == d)
                        cy = next(c for c in by["checkpoints"] if str(c["d"]) == d)
                        pairs = list(zip(cx["kl"], cy["kl"], strict=True))
                        blocks += [pairs[i : i + BLOCK] for i in range(0, len(pairs), BLOCK)]
                    entry[f"book{d}"] = [r(v) for v in boot_ratio(blocks)]
                    every += blocks
                entry["books"] = [r(v) for v in boot_ratio(every)]
                # Family-wise: Bonferroni over the twelve long-text comparisons with GPTQ (RTN, AWQ, AWQ + GPTQ and the
                # published file, on three models), with more resamples for the narrower tails.
                entry["booksFw"] = [r(v) for v in boot_ratio(every, n=20000, alpha=0.05 / 12)]
            vs[arm] = entry
        models.append(
            {
                "key": key,
                "name": name,
                "callToken": tok.decode([call]),
                "heldoutRows": len(next(iter(raw.values()))["rows"]["per_row"]) if raw else 0,
                "arms": arms,
                "vsGptq": vs,
                # The first round's solve times (Modal, each method alone on an 8-core sandbox; Qwen3 calibrated
                # with thinking on): the one like-for-like timing of the three methods.
                "solveS": {
                    m: (load(ROOT / key / m, m) or {}).get("result", {}).get("timings_s", {}).get("solve")
                    for m in ("gptq", "awq", "awq_gptq")
                },
                # The first round's first-token divergence on the app's questions (2k files, Modal; Qwen3 thinking on).
                "firstAppKl": {
                    m: r((load(ROOT / key / m, m) or {}).get("result", {}).get("heldout", {}).get("mean_kl"))
                    for m in ("rtn", "gptq", "awq", "awq_gptq")
                },
            }
        )
    head = (
        "// Generated by scripts/awq_site.py in the ExecuPack AWQ-study tree (branch awq-compare) from every arm's\n"
        "// results. Do not edit by hand; rerun the script.\n\n"
    )
    OUT.write_text(head + "export const quant = " + json.dumps({"models": models}, indent=1) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for m in models:
        for arm, a in m["arms"].items():
            print(
                f"{m['key']:11} {arm:17} books={ {d: (v['kl'], v['ppl'], v['top1']) for d, v in a['books'].items()} } "
                f"needles={a['needles']['found']}/{a['needles']['of']} app={a['app']['kl']}",
                f"vs={m['vsGptq'].get(arm, {}).get('books')}",
            )


main()
