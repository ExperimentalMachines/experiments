"""Exported Gemma 3 files against Hugging Face fp32 past the sliding window, through the pybindings.

    python scripts/gemma3_pte_check.py SOURCE_DIR TEXT PTE... [--prefix 1024] [--steps 128]

Each file prefills ``prefix`` tokens of TEXT (BOS first) in chunks of get_max_seq_len - 1, the
most its graph accepts, then is teacher-forced ``steps`` tokens one at a time, the decode path
the app uses. KL(HF || file), top-1 agreement and the file's NLL over those
positions; HF is one fp32 forward with eager attention (which applies the sliding window).
For a prefix past 512 every sliding layer's ring has wrapped, so a ring or local-RoPE error in
the lowered graph (or in the int8 ring cache, upstream PR #24473) shows up here.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source")
    p.add_argument("text")
    p.add_argument("ptes", nargs="+")
    p.add_argument("--prefix", type=int, default=1024)
    p.add_argument("--steps", type=int, default=128)
    args = p.parse_args()

    import awq_compare as ac
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline import smoke

    src = Path(args.source)
    tok = AutoTokenizer.from_pretrained(str(src))
    ids = [tok.bos_token_id] + tok(Path(args.text).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    ids = ids[: args.prefix + args.steps]
    hf = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0, args.prefix - 1 : args.prefix + args.steps - 1].float()
    del hf
    targets = torch.tensor(ids[args.prefix : args.prefix + args.steps])
    results = {}
    for pte in map(Path, args.ptes):
        # ExecuTorch 1.5.1 bounds the graph's token dimension at get_max_seq_len - 1
        # (examples/models/llama/model.py, dynamic shapes), one less than TextLLMRunner sends.
        chunk = int(smoke.read_metadata(pte).get("get_max_seq_len")) - 1
        with tempfile.TemporaryDirectory() as scratch:
            runner = ac._runner(pte, src / "tokenizer.json", len(tok), Path(scratch))
            logits = None
            for start in range(0, args.prefix, chunk):
                piece = ids[start : min(start + chunk, args.prefix)]
                logits = runner.forward(torch.tensor([piece]), torch.tensor([start]))
            got = [ac._last(logits)]
            for s in range(args.steps - 1):
                pos = args.prefix + s
                got.append(ac._last(runner.forward(torch.tensor([[ids[pos]]]), torch.tensor([pos]))))
            del runner
        got = torch.stack(got)
        a, b = ref.log_softmax(-1), got.log_softmax(-1)
        kl = (a.exp() * (a - b)).sum(-1)
        results[pte.name] = {
            "positions": f"{args.prefix}-{args.prefix + args.steps}",
            "prefill_chunk": chunk,
            "kl_mean": float(kl.mean()),
            "kl_max": float(kl.max()),
            "top1": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            "nll_file": float(-b.gather(-1, targets[:, None]).mean()),
            "nll_hf": float(-a.gather(-1, targets[:, None]).mean()),
            "finite": bool(torch.isfinite(got).all()),
        }
        print(pte.name, json.dumps(results[pte.name]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
