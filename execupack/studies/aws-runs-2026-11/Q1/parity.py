"""Is ExecuTorch's Phi-4-mini, built from execupack's params and converter, HF's Phi-4-mini?

    python parity.py <execupack checkout> <work dir> <out.json>

Independent of ExecuTorch's own RoPE code (codex review, 2026-11-09: the unit test computes
its expectation with the function it tests): the reference is transformers' Phi3ForCausalLM
in fp32, on the decision gate's own prompts (the app's system turn and tool schema).

1. HF fp32, its own dynamic LongRoPE: short factors at these lengths (<= 4096 tokens).
2. HF fp32 with the long factors forced, what HF does past 4096 tokens.
3. ExecuTorch eager (llama_transformer from the generated params and converted weights) at
   a 2k window (short factors) and an 8k window (long factors, families._phi4_mini_plan).

ET-2k against HF-short and ET-8k against HF-long say whether the recipe is the same function
(max logit difference, top-1, KL). ET-8k against HF-short is the measured cost of LongRoPE's
long factors on short prompts, graded by gate.check as the publish gate would.
"""

import gc
import json
import sys
from pathlib import Path

import torch

checkout, work, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(checkout))
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from pipeline import convert, families, gate  # noqa: E402

ID, REV = "microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083"
work.mkdir(parents=True, exist_ok=True)
model_dir = Path(snapshot_download(ID, revision=REV, local_dir=work / "hf"))
config = json.loads((model_dir / "config.json").read_text())
plan = families.xnnpack_plan(families.family_for(config), config)
tok = AutoTokenizer.from_pretrained(model_dir)
prompts = gate.prompts(tok)
order = list(prompts)
ids = {q: tok(prompts[q], add_special_tokens=False)["input_ids"] for q in order}
print("prompt tokens", min(map(len, ids.values())), "to", max(map(len, ids.values())), flush=True)
assert max(map(len, ids.values())) <= 4096
assert all("<|tool|>" in prompts[q] for q in order), "gate prompts lack the tool schema"


def stack(fn):
    with torch.no_grad():
        return torch.stack([fn(torch.tensor([ids[q]])).double() for q in order])


hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
rot = hf.model.rotary_emb
hf_short = stack(lambda x: hf(x).logits[0, -1])
dim = int(plan.params["head_dim"] * plan.params["partial_rotary_factor"])
base = 1.0 / (plan.params["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
long_inv = base / torch.tensor(plan.params["rope_scaling_long_factor"], dtype=torch.float32)
names = [n for n in ("inv_freq", "original_inv_freq") if hasattr(rot, n)]
print("HF rotary", type(rot).__name__, "buffers", names, "attention_scaling", getattr(rot, "attention_scaling", None))
for n in names:
    setattr(rot, n, long_inv.clone())
hf_long = stack(lambda x: hf(x).logits[0, -1])
del hf, rot
gc.collect()

ckpt = work / "phi4-converted.pth"
if not ckpt.exists():
    print("convert", convert.convert(model_dir, ckpt, plan.converter, config), flush=True)
state = torch.load(ckpt, mmap=True, weights_only=True)

from executorch.examples.models.llama.llama_transformer import construct_transformer  # noqa: E402
from executorch.examples.models.llama.model_args import ModelArgs  # noqa: E402


def executorch_logits(window):
    args = ModelArgs(**plan.params, max_context_len=window, max_seq_len=window, use_kv_cache=False)
    model = construct_transformer(args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "freqs" not in k and "mask" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    model = model.float().eval()
    print(f"ET {window}: rope table {tuple(model.rope.freqs_cos.shape)}", flush=True)

    def last(x):
        y = model(tokens=x)
        y = y[0] if isinstance(y, tuple) else y
        return y.reshape(-1, y.shape[-1])[-1]

    result = stack(last)
    del model
    gc.collect()
    return result


et_2k = executorch_logits(2048)
et_8k = executorch_logits(8192)


def compare(a, b):
    pa, pb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    return {
        "max_abs_logit_diff": float((a - b).abs().max()),
        "top1_agree": int((a.argmax(-1) == b.argmax(-1)).sum()),
        "rows": len(order),
        "kl_max": float(kl.max()),
        "kl_mean": float(kl.mean()),
    }


def graded(ref_logits, got_logits):
    pr, pg = torch.softmax(ref_logits, -1), torch.softmax(got_logits, -1)
    ref = {"rows": {}}
    measured = {}
    for i, q in enumerate(order):
        t = int(pr[i].argmax())
        ref["rows"][q] = {"token": t, "p": float(pr[i, t])}
        measured[q] = {"top": int(pg[i].argmax()), "p": float(pg[i, t])}
    verdict = gate.check(ref, measured)
    verdict.pop("rows")
    return verdict


p_short, p_8k = torch.softmax(hf_short, -1), torch.softmax(et_8k, -1)
result = {
    "params": {k: v for k, v in plan.params.items() if not k.startswith("rope_scaling")},
    "et_2k_vs_hf_short": compare(hf_short, et_2k),
    "et_8k_vs_hf_long": compare(hf_long, et_8k),
    "hf_long_vs_hf_short": compare(hf_short, hf_long),
    "et_8k_vs_hf_short": compare(hf_short, et_8k),
    "gate_et_8k_against_hf_short": graded(hf_short, et_8k),
    "gate_hf_long_against_hf_short": graded(hf_short, hf_long),
    "gate_et_2k_against_hf_short": graded(hf_short, et_2k),
    "rows": [
        {
            "q": q,
            "hf_short": [tok.decode([int(p_short[i].argmax())]), round(float(p_short[i].max()), 4)],
            "et_8k": [tok.decode([int(p_8k[i].argmax())]), round(float(p_8k[i].max()), 4)],
        }
        for i, q in enumerate(order)
    ],
}
out_path.write_text(json.dumps(result, indent=1))
print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "params")}, indent=1))
for r in result["rows"]:
    print(r)
