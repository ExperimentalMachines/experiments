import json, sys
from pathlib import Path
import torch
sys.path.insert(0, "/home/ubuntu/execupack")
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline import gate, hub
source = hub.fetch("LiquidAI/LFM3-1.7B-Base")
src = Path("/home/ubuntu/runs/lfm3-base/probe/source")
hub.download(source, src)
print("sha", source.sha, "remote code ok", hub.remote_code_allowed(source))
tok = AutoTokenizer.from_pretrained(str(src))
m = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.float32, trust_remote_code=True, local_files_only=True).eval()
ref = gate.reference(m, tok)
print("confident", ref["confident"], "of", len(ref["rows"]))
for q, r in ref["rows"].items():
    print("%.3f %-16r %s" % (r["p"], r["text"], q))
