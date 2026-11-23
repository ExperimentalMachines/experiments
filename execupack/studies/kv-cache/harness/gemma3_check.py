"""Gemma 3 through pipeline/gemma3_model.py against Hugging Face, fp32 on CPU, before any export.

    python scripts/gemma3_check.py WORK [--model google/gemma-3-1b-it] [--revision SHA] [--tokens 1800]

Two eager ExecuTorch models from the converted checkpoint, each against one fp32
Gemma3ForCausalLM pass (eager attention, which applies the sliding window) over the same tokens:

* the KV-cache path the export lowers: prompt fed in 512-token chunks (the plan's prefill
  chunk) through the cache, the sliding layers' caches turned into rings exactly as
  export_llm's local_global_attention does, all logits kept (generate_full_logits);
* the solve's path (solve.eager_model's configuration: no cache, one pass with the banded
  sliding mask) over the first 1,024 tokens.

Reported per 512-position segment: mean and max KL(HF || ExecuTorch) and top-1 agreement, so a
sliding-window or local-RoPE error shows up past position 512. Text: a PG-19 test book
(storage.googleapis.com/deepmind-gutenberg/test/25773.txt), BOS first. An experiment check.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BOOK = "https://storage.googleapis.com/deepmind-gutenberg/test/25773.txt"


def kl_table(ref, other, segment: int = 512) -> list[dict]:
    rows = []
    for start in range(0, ref.shape[0], segment):
        a = ref[start : start + segment].log_softmax(-1)
        b = other[start : start + segment].log_softmax(-1)
        kl = (a.exp() * (a - b)).sum(-1)
        rows.append(
            {
                "positions": f"{start}-{start + a.shape[0]}",
                "kl_mean": float(kl.mean()),
                "kl_max": float(kl.max()),
                "top1": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            }
        )
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("work")
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument("--revision", default="dcc83ea841ab6100d6b47a070329e1ba4cf78752")
    p.add_argument("--tokens", type=int, default=1800)
    args = p.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline import convert, families, solve

    torch.manual_seed(0)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    src = Path(snapshot_download(args.model, revision=args.revision, local_dir=str(work / "source")))
    config = json.loads((src / "config.json").read_text())
    book = work / "25773.txt"
    if not book.exists():
        urllib.request.urlretrieve(BOOK, book)
    tok = AutoTokenizer.from_pretrained(str(src))
    ids = [tok.bos_token_id] + tok(book.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    ids = ids[: args.tokens]
    family = families.family_for(config)
    plan = families.xnnpack_plan(family, config)
    print("plan", plan.model_class, plan.converter, "chunk", plan.prefill_chunk, flush=True)
    checkpoint = work / "gemma3-fp32.pth"
    if not checkpoint.exists():
        print("convert", convert.convert(src, checkpoint, plan.converter, families.text_config(config)), flush=True)

    hf = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        ref = hf(torch.tensor([ids])).logits[0].float()
    del hf
    result: dict = {"model": args.model, "revision": args.revision, "tokens": len(ids)}

    # The KV-cache path, as export_llm builds it before lowering.
    from executorch.examples.models.llama.model import Llama2Model
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
        replace_kv_cache_with_ring_kv_cache,
    )
    from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType

    from pipeline import gemma3_model

    gemma3_model.apply()
    params_path = work / "params.json"
    params_path.write_text(json.dumps(plan.params))
    cfg = LlmConfig()
    cfg.base.model_class = ModelType(plan.model_class)
    cfg.base.params = str(params_path)
    cfg.base.checkpoint = str(checkpoint)
    cfg.model.use_kv_cache = True
    cfg.model.use_sdpa_with_kv_cache = False
    cfg.model.enable_dynamic_shape = True
    cfg.model.dtype_override = "fp32"
    cfg.debug.generate_full_logits = True
    cfg.export.max_seq_length = plan.prefill_chunk
    cfg.export.max_context_length = 2048
    model = Llama2Model(cfg).get_eager_model().to(torch.float32).eval()
    replace_kv_cache_with_ring_kv_cache(model, list(plan.local_global_attention))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(ids), plan.prefill_chunk):
            piece = torch.tensor([ids[start : start + plan.prefill_chunk]])
            out = model(piece, {"input_pos": torch.tensor([start])})
            out = out[0] if isinstance(out, tuple) else out
            chunks.append(out[0].float())
    kv = torch.cat(chunks)
    result["kv_cache_chunked"] = kl_table(ref, kv)
    print("kv-cache path, 512-token chunks:", json.dumps(result["kv_cache_chunked"], indent=1), flush=True)
    del model

    # The solve's path: solve.eager_model's configuration (no cache, one pass), with every
    # position's logits kept, over two windows so the band is exercised.
    cfg = LlmConfig()
    cfg.base.model_class = ModelType(plan.model_class)
    cfg.base.params = str(params_path)
    cfg.base.checkpoint = str(checkpoint)
    cfg.model.use_kv_cache = False
    cfg.model.use_sdpa_with_kv_cache = False
    cfg.model.dtype_override = "fp32"
    cfg.debug.generate_full_logits = True
    cfg.export.max_seq_length = 1024
    cfg.export.max_context_length = 1024
    eager = Llama2Model(cfg).get_eager_model().to(torch.float32).eval()
    assert solve.int4_linears(eager), "no linears for the solve"
    with torch.no_grad():
        out = eager(torch.tensor([ids[:1024]]))
        out = out[0] if isinstance(out, tuple) else out
    result["solve_path_1024"] = kl_table(ref[:1024], out[0].float())
    print("solve path, 1024 tokens:", json.dumps(result["solve_path_1024"]), flush=True)
    (work / "gemma3_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
