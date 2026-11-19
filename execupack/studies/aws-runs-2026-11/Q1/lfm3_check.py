"""Check pipeline/lfm3_model.py against the checkpoint's own modeling_lfm3.py, on fp32.

    python lfm3_check.py <work dir> [--pte <fp32 .pte>]

1. state_dict keys of the ExecuTorch module == converted checkpoint keys (load is strict=False upstream)
2. no-cache module (what the GPTQ solve sees): full logits vs HF over a >256-token prompt
3. KV-cache module with export_llm's custom-op transforms, small prefill chunks (ring wraps), then a
   second prompt at position 0 (state reset)
4. optional: an exported .pte through the C++ runtime (pybindings), same comparison
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.home() / "execupack"))
from pipeline import calibration, convert, families, hub, lfm3_model, solve  # noqa: E402

MODEL = "LiquidAI/LFM3-1.7B"
work = Path(sys.argv[1])
pte = Path(sys.argv[sys.argv.index("--pte") + 1]) if "--pte" in sys.argv else None
torch.manual_seed(0)
torch.set_grad_enabled(False)

source = hub.fetch(MODEL)
plan = families.xnnpack_plan(families.family_for(source.config), source.config)
src = work / "source"
ckpt = work / "checkpoint" / "consolidated.pth"
if not ckpt.exists():
    hub.download(source, src)
    print(convert.convert(src, ckpt, plan.converter, families.text_config(source.config)))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(src))
hf = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.float32, trust_remote_code=True).eval()
prompt_a = calibration.render_app(tok, "What is Concord the capital of?\n\n" + calibration.TRAILER.format(subject="Concord"))
prompt_b = calibration.render(tok, "Write three sentences about the sea.")
ids_a = tok(prompt_a, add_special_tokens=False)["input_ids"]
ids_b = tok(prompt_b, add_special_tokens=False)["input_ids"]
print(f"prompt a {len(ids_a)} tokens, prompt b {len(ids_b)} tokens")
ref_a = hf(torch.tensor([ids_a])).logits[0].float()
ref_b = hf(torch.tensor([ids_b])).logits[0].float()
print("HF top token a:", repr(tok.decode([int(ref_a[-1].argmax())])), "b:", repr(tok.decode([int(ref_b[-1].argmax())])))
del hf


def compare(name, got, ref):
    got = got.float()
    diff = (got - ref).abs().max().item()
    agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    p = torch.softmax(ref, -1)
    q = torch.log_softmax(got, -1)
    kl = (p * (torch.log(p.clamp_min(1e-30)) - q)).sum(-1).max().item()
    print(f"{name}: max|dlogit| {diff:.4g}, argmax agreement {agree:.4f}, max KL {kl:.3g}")
    return diff, agree, kl


results = {}
# 1 + 2: no cache, the solve's module
model = solve.eager_model(plan.model_class, plan.params, ckpt, work, lfm3=True, max_seq_len=4096)
state = torch.load(ckpt, map_location="cpu", weights_only=True)
RUNTIME = ("conv_state", "k_cache", "v_cache", "cache_positions")
model_keys = {k for k in model.state_dict() if not k.endswith(RUNTIME)}
ckpt_keys = set(state.keys())
print("keys only in model:", sorted(model_keys - ckpt_keys)[:10])
print("keys only in checkpoint:", sorted(ckpt_keys - model_keys)[:10])
assert model_keys == ckpt_keys, "converted checkpoint and LFM3 module disagree on tensor names"
model.generate_full_logits = True
results["nocache_a"] = compare("no-cache a", model(torch.tensor([ids_a]))[0], ref_a)
results["nocache_b"] = compare("no-cache b", model(torch.tensor([ids_b]))[0], ref_b)
del model

# 3: KV cache + custom ops, chunked prefill and decode
from executorch.examples.models.llama.model import Llama2Model  # noqa: E402
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (  # noqa: E402
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import replace_sdpa_with_custom_op  # noqa: E402
from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType  # noqa: E402

lfm3_model.apply()
params_path = work / "kv-params.json"
params_path.write_text(json.dumps(plan.params))
chunk = 96
cfg = LlmConfig()
cfg.base.model_class = ModelType(plan.model_class)
cfg.base.params = str(params_path)
cfg.base.checkpoint = str(ckpt)
cfg.model.use_kv_cache = True
cfg.model.use_sdpa_with_kv_cache = True
cfg.model.enable_dynamic_shape = True
cfg.model.dtype_override = "fp32"
cfg.export.max_seq_length = chunk
cfg.export.max_context_length = 2048
kv = Llama2Model(cfg).get_eager_model().to(torch.float32).eval()
replace_kv_cache_with_custom_kv_cache(kv)
replace_sdpa_with_custom_op(kv)
kv.generate_full_logits = True


def run_cached(ids, chunk, decode_from):
    outs = []
    pos = 0
    while pos < decode_from:
        n = min(chunk, decode_from - pos)
        out = kv(torch.tensor([ids[pos : pos + n]]), {"input_pos": torch.tensor([pos])})
        outs.append(out[0] if out.dim() == 3 else out)
        pos += n
    while pos < len(ids):
        out = kv(torch.tensor([[ids[pos]]]), {"input_pos": torch.tensor([pos])})
        outs.append(out[0] if out.dim() == 3 else out)
        pos += 1
    return torch.cat(outs, 0)


for name, ids, ref in (("a", ids_a, ref_a), ("b", ids_b, ref_b), ("a again", ids_a, ref_a)):
    got = run_cached(ids, chunk, len(ids) - 40)
    results[f"kv_{name}"] = compare(f"kv-cache {name} (chunk {chunk}, 40 decode steps)", got, ref)
del kv

# 4: the exported program through the runtime
if pte is not None:
    from executorch.extension.pybindings.portable_lib import _load_for_executorch  # noqa: E402
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: E402, F401
    from executorch.kernels import quantized  # noqa: E402, F401

    program = _load_for_executorch(str(pte))
    meta = {m: program.run_method(m, ())[0] for m in ("get_max_seq_len", "get_max_context_len")}
    print("pte", meta)
    step = int(meta["get_max_seq_len"])

    def run_pte(ids, decode_from):
        outs, pos = [], 0
        while pos < len(ids):
            n = min(step, decode_from - pos) if pos < decode_from else 1
            out = program.forward((torch.tensor([ids[pos : pos + n]], dtype=torch.long), torch.tensor([pos], dtype=torch.long)))[0]
            outs.append(out.reshape(-1, out.shape[-1])[-1:])
            pos += n
        return torch.cat(outs, 0)

    for name, ids, ref in (("a", ids_a, ref_a), ("b", ids_b, ref_b), ("a again", ids_a, ref_a)):
        decode_from = len(ids) - 40
        got = run_pte(ids, decode_from)
        # last logits of each prefill chunk, then every decode step
        rows = []
        pos = 0
        while pos < len(ids):
            n = min(step, decode_from - pos) if pos < decode_from else 1
            rows.append(pos + n - 1)
            pos += n
        results[f"pte_{name}"] = compare(f"pte {name}", got, ref[rows])

print(json.dumps({k: [round(x, 6) for x in v] for k, v in results.items()}, indent=1))
