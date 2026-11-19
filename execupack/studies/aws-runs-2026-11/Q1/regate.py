import json, shutil, sys
from pathlib import Path
import torch
sys.path.insert(0, "/home/ubuntu/execupack")
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline import gate
src = "/home/ubuntu/runs/lfm3/solve-work/source"
out = Path("/home/ubuntu/runs/lfm3/gate.json")
if not Path("/home/ubuntu/runs/lfm3/gate-thinking.json").exists():
    shutil.copy(out, "/home/ubuntu/runs/lfm3/gate-thinking.json")
tok = AutoTokenizer.from_pretrained(src)
print(repr(gate.prompts(tok)["What is Concord the capital of?"][-120:]))
m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, trust_remote_code=True, local_files_only=True).eval()
ref = gate.reference(m, tok)
out.write_text(json.dumps(ref, indent=2) + "\n")
print("confident", ref["confident"], "of", len(ref["rows"]))
for q, r in ref["rows"].items():
    print("%.3f %-16r %s" % (r["p"], r["text"], q))
