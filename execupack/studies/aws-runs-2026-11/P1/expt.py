"""Which part of Phi-4-mini's 8da4w recipe costs its tool calls? Eager, against fp32's gate.

    python expt.py <execupack checkout> <runs dir>

ExecuTorch's eager transformer from execupack's params and converter (exact against HF,
parity.py), quantised the way export_llm does it (int8 per-channel embeddings, then
Int8DynamicActivationIntxWeightConfig on every nn.Linear, the solve's GPTQ codes injected
through pipeline.export_with_codes.inject), with the output head varied:

  V0 int4 GPTQ everywhere, the published recipe (sanity: should match the .pte's gate)
  V1 output head left in fp32
  V2 output head int8 per channel (XNNPACK's qd8-f32-qc8w)
  V3 every linear int8 per channel (8da8w), the 8-bit ceiling
Each is graded by gate.check against the solve's fp32 reference (gate.json).
"""

import gc
import json
import sys
from pathlib import Path

import torch

checkout, runs = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(checkout))
from huggingface_hub import snapshot_download  # noqa: E402
from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_  # noqa: E402
from torchao.quantization.granularity import PerAxis, PerGroup  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from executorch.examples.models.llama.llama_transformer import construct_transformer  # noqa: E402
from executorch.examples.models.llama.model_args import ModelArgs  # noqa: E402
from executorch.examples.models.llama.source_transformation.quantize import get_quant_embedding_transform  # noqa: E402
from pipeline import convert, families, gate  # noqa: E402
from pipeline.export_with_codes import inject  # noqa: E402

ID, REV = "microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083"
model_dir = Path(snapshot_download(ID, revision=REV))
config = json.loads((model_dir / "config.json").read_text())
plan = families.xnnpack_plan(families.family_for(config), config)
tok = AutoTokenizer.from_pretrained(model_dir)
prompts = gate.prompts(tok)
order = list(prompts)
ids = {q: tok(prompts[q], add_special_tokens=False)["input_ids"] for q in order}
ref = json.loads((runs / "gate.json").read_text())
work = runs / "expt"
work.mkdir(exist_ok=True)
ckpt = work / "phi4-converted.pth"
if not ckpt.exists():
    convert.convert(model_dir, ckpt, plan.converter, config)
state = torch.load(ckpt, mmap=True, weights_only=True)
codes = torch.load(runs / "codes.pt", map_location="cpu", weights_only=True)
print("codes for", len(codes), "linears; output in codes:", "output" in codes, flush=True)


def linear_filter(skip):
    def keep(m, fqn):
        return isinstance(m, torch.nn.Linear) and fqn not in skip and m.weight.shape[1] % 32 == 0

    return keep


INT4 = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4, weight_granularity=PerGroup(32))
INT8 = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int8, weight_granularity=PerAxis(0))


def build(output_mode):
    args = ModelArgs(**plan.params, max_context_len=2048, max_seq_len=2048, use_kv_cache=False)
    model = construct_transformer(args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [k for k in missing if "freqs" not in k and "mask" not in k] and not unexpected
    model = model.float().eval()
    model = get_quant_embedding_transform("8,0")(model)
    with torch.no_grad():
        if output_mode == "all8":
            quantize_(model, INT8, filter_fn=linear_filter(set()))
            return model
        quantize_(model, INT4, filter_fn=linear_filter({"output"}))
        if output_mode == "int4":
            quantize_(model, INT4, filter_fn=lambda m, fqn: fqn == "output")
        elif output_mode == "int8":
            quantize_(model, INT8, filter_fn=lambda m, fqn: fqn == "output")
        use = dict(codes) if output_mode == "int4" else {k: v for k, v in codes.items() if k != "output"}
        injected, left = inject(model, use, log=lambda s: None)
        assert injected == len(use) and left == ([] if output_mode == "int4" or output_mode == "fp32" else ["output"]), (
            injected, len(use), left)
    return model


results = {}
for name, mode in (("V2 output int8", "int8"), ("V3 all int8", "all8")):
    model = build(mode)
    measured = {}
    with torch.no_grad():
        for q in order:
            y = model(tokens=torch.tensor([ids[q]]))
            y = (y[0] if isinstance(y, tuple) else y).reshape(-1, plan.params["vocab_size"])[-1]
            p = torch.softmax(y.double(), -1)
            t = ref["rows"][q]["token"]
            measured[q] = {"top": int(p.argmax()), "p": float(p[t])}
    verdict = gate.check(ref, measured)
    rows = {q: (round(ref["rows"][q]["p"], 3), round(measured[q]["p"], 3), tok.decode([measured[q]["top"]])) for q in order}
    verdict.pop("rows")
    results[name] = {"verdict": verdict, "rows": rows}
    print(name, {k: verdict[k] for k in ("passed", "agreed", "graded", "fp32_mean", "export_mean",
                                         "quiet_rows_answered_like_fp32", "quiet_rows")}, verdict["problems"], flush=True)
    del model
    gc.collect()
(work / "expt2.json").write_text(json.dumps(results, indent=1))
for name, r in results.items():
    print("==", name)
    for q, row in r["rows"].items():
        print("  ", row, q)
