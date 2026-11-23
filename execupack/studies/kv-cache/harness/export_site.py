"""Writes the KV study's numbers as a TypeScript data module for the Experimental Machines research post
(experimentalmachines.org: lib/kv-cache-data.ts), so the page never retypes a number.

    HF_TOKEN=... python scripts/kvstudy/export_site.py work/kvstudy <site>/lib/kv-cache-data.ts

Every aggregate comes from report.py's collect() and gpu_passes(), the same code that builds the reviewed report
page (work/kvstudy/report/index.html), so the two cannot disagree. The few extras the post adds (top-token
agreement between the two caches' files, one book's divergence by depth, the 32k repeat, the published-file check)
are computed here from the same raw results.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.argv = [sys.argv[0], str(ROOT / "phone-out"), str(ROOT / "data"), str(ROOT / "report" / "narrative.html")]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import report as R  # noqa: E402

LABEL = {"llama": "Llama 3.2 1B", "lfm": "LFM2.5 1.2B", "qwen": "Qwen3 1.7B", "gemma": "Gemma 3 1B"}


def r4(x: float | None, d: int = 4):
    return None if x is None else round(x, d)


def quality_rows(rows: list[dict]) -> list[dict]:
    return [{
        "model": LABEL[r["model"]], "key": r["model"], "window": r["window"], "phone": r.get("phone"), "books": r["books"], "depth": r["depth"],
        "klA": r4(r["kl_f"]), "klB": r4(r["kl_q"]), "change": r4(r["d"], 5), "low": r4(r["lo"], 5), "high": r4(r["hi"], 5),
        "direct": r4(r["direct"][0], 5) if r["direct"] else None, "directMax": r4(r["direct"][1]) if r["direct"] else None,
        "pplA": r4(r["ppl_f"], 1), "pplB": r4(r["ppl_q"], 1), "pplHf": r4(r["ppl_hf"], 1), "top1A": r4(r["t1_f"], 3), "top1B": r4(r["t1_q"], 3),
    } for r in rows]


def speed_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        clk = r.get("clk") or {}
        out.append({
            "model": LABEL[r["model"]], "key": r["model"], "window": r["window"], "phone": r.get("phone"), "books": r["books"],
            "decode": [{"depth": d, "a": r4(a, 4), "b": r4(b, 4), "levelA": (clk.get(d) or (None, None))[0], "levelB": (clk.get(d) or (None, None))[1]}
                       for d, (a, b) in sorted(r["decode"].items())],
            "prefill": [[r4(a, 1), r4(b, 1)] for a, b in r["prefill"]],
        })
    return out


def memory_rows(rows: list[dict]) -> list[dict]:
    return [{
        "model": LABEL[r["model"]], "key": r["model"], "window": r["window"], "phone": r.get("phone"),
        "afterA": r4(r["after"][0], 1), "afterB": r4(r["after"][1], 1), "peakA": r4(r["peak"][0], 1), "peakB": r4(r["peak"][1], 1),
        "gpuA": r4(r["gpu"][0], 1) if r["gpu"] else None, "gpuB": r4(r["gpu"][1], 1) if r["gpu"] else None,
    } for r in rows]


def needle_rows(rows: list[dict]) -> list[dict]:
    return [{
        "model": LABEL[r["model"]], "key": r["model"], "window": r["window"], "phone": r.get("phone"), "books": r["books"],
        "hf": r["hf"][0], "of": r["n"], "a": r["f"], "b": r["q"], "both": r["both"], "onlyA": r["only_f"], "onlyB": r["only_q"], "neither": r["neither"],
    } for r in rows]


def app_rows(rows: list[dict]) -> list[dict]:
    return [{
        "model": LABEL[r["model"]], "key": r["model"], "phone": r.get("phone"),
        "heldA": r4(r["ho_f"], 4), "heldB": r4(r["ho_q"], 4), "heldChange": r4(r["hd"], 4), "heldLow": r4(r["hci"][0], 4), "heldHigh": r4(r["hci"][1], 4),
        "top1A": r["t1_f"], "top1B": r["t1_q"], "rows": r["rows"], "worse": r["worse"], "better": r["better"],
        "replyA": r4(r["rk_f"], 4), "replyB": r4(r["rk_q"], 4), "replyChange": r4(r["rd"], 4), "replyLow": r4(r["rci"][0], 4), "replyHigh": r4(r["rci"][1], 4),
    } for r in rows]


def top_differs(out_dir: Path, pair: tuple[str, str], models) -> list[dict]:
    """How often the two files' likeliest next token differ, over the reference's 64 tokens (each file's argmax
    among them), per model, window and depth."""
    rows = []
    for m in models:
        for w in R.WIN_ORDER:
            a, b = (out_dir / f"stream-{w}-{m}-{kv}.json" for kv in pair)
            if not a.exists() or not b.exists():
                continue
            pa, pb = A.per_depth(json.loads(a.read_text())), A.per_depth(json.loads(b.read_text()))
            for d in sorted(pa):
                la, lb = pa[d]["ref_lp"], pb.get(d, {}).get("ref_lp", [])
                if not la or not lb:
                    continue
                diff = sum(max(range(len(x)), key=x.__getitem__) != max(range(len(y)), key=y.__getitem__) for x, y in zip(la, lb))
                rows.append({"model": LABEL[m], "key": m, "window": w, "depth": d, "differ": diff, "of": len(la)})
    return rows


def depth_curve() -> dict:
    """Llama 3.2 1B on book 10321 through its 32k files: divergence from the fp32 model and agreement with its top
    token at each depth, for the CPU fp32 and int8 files (first phone) and the GPU fp32 and fp16 files (third phone)."""
    def run(path: Path) -> dict:
        b = next(x for x in json.loads(path.read_text())["books"] if str(x["book"]) == "10321")
        return {str(c["d"]): {"kl": r4(st.mean(c["kl"])), "top1": r4(st.mean(c["top1"]), 3),
                              "tail": r4(st.mean(1 - sum(math.exp(v) for v in lp) for lp in c["ref_lp"]), 3) if c.get("ref_lp") else None}
                for c in b["checkpoints"]}
    return {
        "cpuA": run(ROOT / "phone-out" / "stream-32k-llama-f32kv.json"), "cpuB": run(ROOT / "phone-out" / "stream-32k-llama-q8kv.json"),
        "gpuA": run(ROOT / "gpu-out4" / "stream-32k-llama-f32kv.json"), "gpuB": run(ROOT / "gpu-out4" / "stream-32k-llama-f16kv.json"),
    }


def breakdown() -> dict:
    """The 32.5k checkpoint of Llama's GPU fp16 file against its fp32 file, position by position."""
    a = json.loads((ROOT / "gpu-out4" / "stream-32k-llama-f32kv.json").read_text())["books"][0]
    b = json.loads((ROOT / "gpu-out4" / "stream-32k-llama-f16kv.json").read_text())["books"][0]
    ca = next(c for c in a["checkpoints"] if c["d"] == 32500)
    cb = next(c for c in b["checkpoints"] if c["d"] == 32500)
    direct = [R.direct_kl(x, y) for x, y in zip(ca["ref_lp"], cb["ref_lp"])]
    rep = json.loads((ROOT / "gpu-out4" / "stream-32k-llama-f16kv.b10321.rep.json").read_text())["books"][0]
    same = all(c1["kl"] == c2["kl"] for c1, c2 in zip(b["checkpoints"], rep["checkpoints"]))
    return {"positions": len(direct), "higher": sum(y > x for x, y in zip(ca["kl"], cb["kl"])), "directMin": r4(min(direct), 3),
            "directMean": r4(st.mean(direct), 3), "directMax": r4(max(direct), 2), "repeatIdentical": same}


def repeat_rows() -> list[dict]:
    o4 = ROOT / "gpu-out4"
    starts = R.starts(o4 / "progress.log")
    rows = []
    for tag, suffix, idx in (("first", "", 0), ("repeat", ".rep", 1)):
        for kv in ("f32kv", "f16kv"):
            b = json.loads((o4 / f"stream-32k-llama-{kv}.b10321{suffix}.json").read_text())["books"][0]
            pc = b["prefill_chunks"]
            t0 = starts[f"stream 32k llama {kv} book 10321"][idx]
            cps = {c["d"]: c for c in b["checkpoints"]}
            # step 0 is the prefill's output, not a decode step (as in analyze.per_depth)
            rows.append({"run": tag, "cache": R.CACHE[kv], "startC": t0[1], "prefillS": r4(sum(c[2] for c in pc) / 1e6, 1),
                         "prefillTokens": sum(c[1] for c in pc),
                         "decode": [r4(1e6 / st.median(cps[d]["step_us"][1:]), 4) for d in R.DEPTHS],
                         "kl32k": r4(st.mean(cps[32500]["kl"])), "gpuMiB": round((b["end"].get("kgsl") or {}).get("kernel_max", 0) / 2**20)})
    return rows


def published() -> list[dict]:
    log = (ROOT / "gpu-out4" / "progress.log").read_text().splitlines()
    rows = []
    for w in ("16k", "32k"):
        end = [ln for ln in log if f"end pub {w} llama" in ln]
        full = (ROOT / "gpu-out4" / f"pub-{w}-llama.logcat-full").read_text(errors="replace")
        m = re.search(r"Abort message: '([^']*)'", full)
        rows.append({"window": w, "rc": int(end[-1].split("rc=")[-1]) if end else None, "abort": m.group(1) if m else None,
                     "ran": (ROOT / "gpu-out4" / f"pub-{w}-llama.json").exists()})
    return rows


def main() -> None:
    cpu = R.collect(ROOT / "phone-out", ("f32kv", "q8kv"), A.MODELS)
    rerun = R.collect(ROOT / "phone-out2", ("f32kv", "q8kv"), A.MODELS)
    gpu, gpu3 = R.gpu_passes(ROOT)
    data = {
        "cpu": {"quality": quality_rows(cpu["quality"]), "speed": speed_rows(cpu["speed"]), "memory": memory_rows(cpu["memory"]),
                "needles": needle_rows(cpu["needles"]), "app": app_rows(cpu["app"])},
        "rerun": {"quality": quality_rows(rerun["quality"]), "needles": needle_rows(rerun["needles"]), "speed": speed_rows(rerun["speed"]),
                  "topDiffers": top_differs(ROOT / "phone-out2", ("f32kv", "q8kv"), A.MODELS)},
        "gpu": {"quality": quality_rows(gpu["quality"]), "speed": speed_rows(gpu["speed"]), "memory": memory_rows(gpu["memory"]),
                "needles": needle_rows(gpu["needles"]), "app": app_rows(gpu["app"]),
                "topDiffers": top_differs(ROOT / "gpu-out", ("f32kv", "f16kv"), ("llama", "qwen")) + top_differs(ROOT / "gpu-out4", ("f32kv", "f16kv"), ("llama", "qwen"))},
        "depth": depth_curve(), "breakdown": breakdown(), "repeat": repeat_rows(), "published": published(),
    }
    head = ("// Generated by scripts/kvstudy/export_site.py in the ExecuPack KV-study tree (branch kv-study) from the phones' raw\n"
            "// results, through the same code as the reviewed report. Do not edit by hand; rerun the script.\n\n")
    OUT.write_text(head + "export const kv = " + json.dumps(data, indent=1) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


main()
