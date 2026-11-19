import json, sys, torch
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from pathlib import Path
from torch import nn
from pipeline import families, hub, solve
from awq_compare import _kl
model_id, rev, work, ref, res = sys.argv[1:6]
work, ref = Path(work), Path(ref)
source = hub.fetch(model_id, rev)
plan = families.xnnpack_plan(families.family_for(source.config), source.config)
model = solve.eager_model(plan.model_class, plan.params, work / "solve" / "checkpoint" / "consolidated.pth", work / "solve", lfm3=plan.lfm3)
codes = torch.load(next(work.glob("codes-*.pt")), weights_only=True)
fp32 = codes.pop("__fp32__", {})
params = dict(model.named_parameters())
for k, v in fp32.items():
    params[k].data.copy_(v)
mods = dict(model.named_modules())
for k, e in codes.items():
    if k == "output" and plan.output_bits == 8:
        continue
    m = mods[k]
    m.weight.data = solve.dequantise(e["qdata"], e["scale"])
    m.register_forward_pre_hook(lambda mod, a: (solve.fake_act(a[0]),))
print("overrides", len(fp32), "codes", len(codes))
evalref = torch.load(ref / "evalref.pt", weights_only=True)
measured = json.load(open(res))["per_row"]
with torch.no_grad():
    for r, m in zip(evalref["heldout"], measured):
        logits = model(torch.tensor([r["ids"]]))
        last = (logits[0, -1] if logits.dim() == 3 else logits[-1]).float()
        kl = _kl(r["top_ids"], r["top_logp"], torch.log_softmax(last, -1))
        print(f"eager-sim KL {kl:.4f}   .pte KL {m['kl']:.4f}")
