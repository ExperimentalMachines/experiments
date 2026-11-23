"""One XNNPACK file from saved GPTQ codes, with an fp32, fp16 or int8 KV cache, for the phone speed test.

    python scripts/kv_export.py MODEL --revision SHA --codes CODES --kv fp32|fp16|int8 --context N
        --work WORK --out OUT --keep DIR [--ref REF_DIR]

The pipeline's own exporter and smoke test (export_xnnpack.run), as scripts/kv_compare.py uses them; only the
window and the cache differ. fp16 builds the whole model in fp16 (dtype_override), the only way ExecuTorch has to a
16-bit cache. A failed smoke test is recorded, not fatal: whether this x86 host runs an fp16 file is itself a
result. With --ref (a 2k awq_compare reference), the file also gets the publish gate and the held-out evaluation
when it runs here. The file lands in DIR as f32kv-, f16kv- or q8kv-<name>. An experiment, not a pipeline stage.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TAGS = {"fp32": "f32kv", "fp16": "f16kv", "int8": "q8kv"}


def host() -> dict:
    flags = set()
    model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return {"cpu": model, **{f: f in flags for f in ("avx512fp16", "avx512_fp16", "f16c", "avx512f", "amx_fp16")}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--revision", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--kv", choices=tuple(TAGS), required=True)
    p.add_argument("--backend", choices=("xnnpack", "vulkan"), default="xnnpack",
                   help="vulkan: the GPU delegate; there --kv fp16 means Vulkan's force_fp16, not an fp16 model")
    p.add_argument("--context", type=int, required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep", required=True)
    p.add_argument("--ref", help="awq_compare reference dir: also run the gate and the held-out evaluation")
    args = p.parse_args()

    from pipeline import export_xnnpack

    # No "int8" in any path: ExecuTorch 1.5.1's llama/model.py takes such a checkpoint path for int8 weight-only.
    tag = TAGS[args.kv]
    os.environ["KV_CACHE_INT8"] = "1" if args.kv == "int8" else "0"
    vulkan = args.backend == "vulkan"
    os.environ["DTYPE_OVERRIDE"] = "fp16" if args.kv == "fp16" and not vulkan else "fp32"
    os.environ["VULKAN_FORCE_FP16"] = "1" if args.kv == "fp16" and vulkan else "0"
    os.environ["KV_STUDY_NO_GATE_PROXY"] = "1" if vulkan else "0"
    work, out, keep = Path(args.work), Path(args.out), Path(args.keep)
    for d in (work, out, keep):
        d.mkdir(parents=True, exist_ok=True)
    summary: dict = {"model": args.model, "kv": args.kv, "context": args.context, "host": host()}

    def save() -> None:
        (out / f"export-{tag}-{args.context}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    t = time.time()
    try:
        report = export_xnnpack.run(
            args.model,
            args.revision,
            work / f"export-{tag}",
            work / f"build-{tag}",
            context=args.context,
            codes=Path(args.codes), backend=args.backend,
            # The evaluation reads the tokenizer from the export's download, which run() deletes.
            keep_work=bool(args.ref),
        )
        summary["smoke"] = {k: report.get("smoke", {}).get(k) for k in ("passed", "reply", "problems")}
    except Exception as error:  # the smoke test raises after the .pte is written; keep the file and the reason
        summary["smoke"] = {"passed": False, "error": f"{type(error).__name__}: {str(error)[-800:]}"}
    summary["seconds"] = round(time.time() - t)
    ptes = sorted((work / f"export-{tag}").rglob("*.pte"))
    if not ptes:
        summary["error"] = "no .pte written"
        save()
        print(json.dumps(summary), flush=True)
        return 1
    pte = ptes[0]
    shutil.copyfile(pte, keep / f"{tag}-{pte.name}")
    summary.update({"pte": f"{tag}-{pte.name}", "bytes": pte.stat().st_size})
    save()
    print(json.dumps(summary), flush=True)

    if args.ref and summary["smoke"].get("passed"):
        import awq_compare as ac
        import torch
        from transformers import AutoTokenizer

        from pipeline import gate

        ref_dir = Path(args.ref)
        src_dir = work / f"build-{tag}" / "source"  # export_xnnpack.run downloads the HF repo here
        tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
        tokenizer_path = src_dir / "tokenizer.json"
        ref = json.loads((ref_dir / "gate.json").read_text(encoding="utf-8"))
        t = time.time()
        summary["gate"] = gate.check(ref, gate.measure(pte, tokenizer_path, tokenizer, ref))
        print("gate", {k: summary["gate"].get(k) for k in ("passed", "fp32_mean", "export_mean", "agreed")}, flush=True)
        save()
        scratch = work / f"eval-{tag}"
        scratch.mkdir(parents=True, exist_ok=True)
        evalref = torch.load(ref_dir / "evalref.pt", weights_only=True)
        ev = ac.evaluate(pte, tokenizer_path, tokenizer, evalref, scratch)
        summary.update({k: ev[k] for k in ("heldout", "replies")})
        summary["eval_seconds"] = round(time.time() - t)
        (out / f"rows-{tag}.json").write_text(
            json.dumps({"per_row": ev["per_row"], "per_reply": ev["per_reply"]}) + "\n", encoding="utf-8"
        )
        print("heldout", ev["heldout"], "replies", ev["replies"], flush=True)
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
