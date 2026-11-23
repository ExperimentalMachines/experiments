"""int8 KV cache against fp32 KV cache on GPTQ files: same codes, same window, only the cache differs.

    python scripts/kv_compare.py MODEL --revision SHA --ref REF_DIR --text TEXT --out OUT --work WORK [--keep DIR]

One GPTQ solve on the reference's calibration rows (scripts/awq_compare.py reference), then
the 2k window exported twice from those codes through the pipeline's own exporter (smoke test
included): once as published today, once with ``model.quantize_kv_cache`` (KV_CACHE_INT8=1 in
export_xnnpack.export_llm_config). Each file is measured exactly as awq_compare's arms are
(publish gate, held-out first-token KL and agreement, teacher-forced KL along fp32's replies),
and the two files against each other on raw text at three depths of the window.
An experiment, not a pipeline stage.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import awq_compare as ac  # noqa: E402

PREFIXES = (64, 512, 1800)
STEPS = 128


def text_kl(a: Path, b: Path, tokenizer_path: Path, tokenizer, text: str, scratch: Path) -> dict:
    """KL(fp32-cache file || int8-cache file) over 128 teacher-forced tokens after each prefix."""
    import torch

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < max(PREFIXES) + STEPS:
        ids = ids + ids
    vocab = len(tokenizer)
    out = {}
    for prefix in PREFIXES:
        runners = [ac._runner(p, tokenizer_path, vocab, scratch) for p in (a, b)]
        logits = [[ac._last(r.forward(torch.tensor([ids[:prefix]]), torch.tensor([0])))] for r in runners]
        for s in range(STEPS - 1):
            pos = prefix + s
            for r, sink in zip(runners, logits):
                sink.append(ac._last(r.forward(torch.tensor([[ids[pos]]]), torch.tensor([pos]))))
        base, other = (torch.stack(x).log_softmax(-1) for x in logits)
        kl = (base.exp() * (base - other)).sum(-1)
        out[f"{prefix}-{prefix + STEPS}"] = {
            "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()),
            "top1_agree": float((base.argmax(-1) == other.argmax(-1)).float().mean()),
        }
        print("text kl", prefix, out[f"{prefix}-{prefix + STEPS}"], flush=True)
        del runners
        gc.collect()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--revision", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--keep", help="a directory (on the Volume) that receives the codes and both .pte files")
    p.add_argument("--codes", help="GPTQ codes from an earlier solve on the same rows; skips the solve")
    args = p.parse_args()

    import torch
    from transformers import AutoTokenizer

    from pipeline import export_xnnpack, gate, solve

    out, work, ref_dir = Path(args.out).resolve(), Path(args.work).resolve(), Path(args.ref).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    result: dict = {"model_id": args.model, "revision": args.revision, "window": ac.WINDOW, "files": {}}
    timings: dict = {}

    t = time.time()
    source, plan, src_dir, checkpoint = ac._prepare(args.model, args.revision, work / "solve")
    timings["download_convert"] = time.time() - t
    keep = Path(args.keep) if args.keep else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    if args.codes:
        codes_path = Path(args.codes)
        result["codes"] = f"reused {codes_path}"
    else:
        rows = torch.load(ref_dir / "rows.pt", weights_only=True)
        t = time.time()
        model = solve.eager_model(plan.model_class, plan.params, checkpoint, work / "solve", lfm3=plan.lfm3)
        codes = solve.solve_model(model, rows)
        timings["solve"] = time.time() - t
        codes_path = work / "codes-gptq.pt"
        torch.save(codes, codes_path)
        del model, codes
        gc.collect()
        if keep:  # saved at once, so a later failure never costs the solve again
            shutil.copyfile(codes_path, keep / "codes-gptq.pt")

    def save() -> None:
        result["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
        (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        (out / "rows.json").write_text(json.dumps(per_row) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
    tokenizer_path = src_dir / "tokenizer.json"
    ref = json.loads((ref_dir / "gate.json").read_text(encoding="utf-8"))
    evalref = torch.load(ref_dir / "evalref.pt", weights_only=True)
    ptes: dict[str, Path] = {}
    per_row: dict = {}
    # Directory names must not contain "int8": ExecuTorch 1.5.1's llama/model.py takes any
    # checkpoint path containing it for an int8 weight-only checkpoint and fails on an import.
    for kv, tag in (("fp32", "f32kv"), ("int8", "q8kv")):
        os.environ["KV_CACHE_INT8"] = "1" if kv == "int8" else "0"
        r: dict = {}
        result["files"][kv] = r
        try:
            t = time.time()
            report = export_xnnpack.run(
                args.model, args.revision, work / f"export-{tag}", work / f"build-{tag}", context=ac.WINDOW,
                codes=codes_path,
            )
            timings[f"export_and_smoke_{kv}"] = time.time() - t
            pte = sorted((work / f"export-{tag}").rglob("*.pte"))[0]
            ptes[kv] = pte
            r["pte"] = pte.name
            r["bytes"] = pte.stat().st_size
            r["smoke"] = {k: report.get("smoke", {}).get(k) for k in ("passed", "reply", "problems")}
            for f in (work / f"export-{tag}").rglob("export-report-*.json"):
                shutil.copyfile(f, out / f"export-report-{kv}.json")
            if keep:
                shutil.copyfile(pte, keep / f"{tag}-{pte.name}")
            t = time.time()
            r["gate"] = gate.check(ref, gate.measure(pte, tokenizer_path, tokenizer, ref))
            timings[f"gate_{kv}"] = time.time() - t
            print(kv, "gate", {k: r["gate"].get(k) for k in ("passed", "fp32_mean", "export_mean", "agreed")}, flush=True)
            save()
            scratch = work / f"eval-{tag}"
            scratch.mkdir(parents=True, exist_ok=True)
            t = time.time()
            ev = ac.evaluate(pte, tokenizer_path, tokenizer, evalref, scratch)
            timings[f"eval_{kv}"] = time.time() - t
            r.update({k: ev[k] for k in ("heldout", "replies")})
            per_row[kv] = {"per_row": ev["per_row"], "per_reply": ev["per_reply"]}
            print(kv, "heldout", ev["heldout"], "replies", ev["replies"], flush=True)
        except Exception as error:  # one file failing is a result; the other still gets measured
            r["error"] = f"{type(error).__name__}: {str(error)[-600:]}"
            print(kv, "FAILED", r["error"], flush=True)
        save()
    os.environ.pop("KV_CACHE_INT8", None)
    if len(ptes) < 2:
        shutil.copyfile(tokenizer_path, out / "tokenizer.json")
        return 1

    t = time.time()
    scratch = work / "eval-text"
    scratch.mkdir(parents=True, exist_ok=True)
    result["text_kl_int8_vs_fp32_cache"] = text_kl(
        ptes["fp32"], ptes["int8"], tokenizer_path, tokenizer, Path(args.text).read_text(encoding="utf-8"), scratch
    )
    timings["text_kl"] = time.time() - t
    save()
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    print(json.dumps({k: v for k, v in result.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
