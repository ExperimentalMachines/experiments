"""Publish gate and held-out evaluation of one existing .pte, measured as scripts/kv_compare.py measures its files.

    python scripts/kv_eval.py MODEL --revision SHA --pte FILE --ref REF_DIR --out OUT --work WORK

The tokenizer comes from the same Hub revision the export used (tokenizer and config files only). An experiment,
not a pipeline stage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--revision", required=True)
    p.add_argument("--pte", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", required=True)
    args = p.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    import awq_compare as ac
    from pipeline import gate

    pte, ref_dir, out, work = Path(args.pte), Path(args.ref), Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(snapshot_download(
        args.model, revision=args.revision, local_dir=str(work / "source"),
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
    ))
    tokenizer = AutoTokenizer.from_pretrained(str(src))
    tokenizer_path = src / "tokenizer.json"
    summary: dict = {"model": args.model, "pte": pte.name}

    def save() -> None:
        (out / f"eval-{pte.stem}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    t = time.time()
    ref = json.loads((ref_dir / "gate.json").read_text(encoding="utf-8"))
    summary["gate"] = gate.check(ref, gate.measure(pte, tokenizer_path, tokenizer, ref))
    summary["gate_seconds"] = round(time.time() - t)
    print("gate", {k: summary["gate"].get(k) for k in ("passed", "fp32_mean", "export_mean", "agreed")}, flush=True)
    save()
    scratch = work / "eval"
    scratch.mkdir(parents=True, exist_ok=True)
    t = time.time()
    evalref = torch.load(ref_dir / "evalref.pt", weights_only=True)
    ev = ac.evaluate(pte, tokenizer_path, tokenizer, evalref, scratch)
    summary.update({k: ev[k] for k in ("heldout", "replies")})
    summary["eval_seconds"] = round(time.time() - t)
    (out / f"rows-{pte.stem}.json").write_text(
        json.dumps({"per_row": ev["per_row"], "per_reply": ev["per_reply"]}) + "\n", encoding="utf-8"
    )
    print("heldout", ev["heldout"], "replies", ev["replies"], flush=True)
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
