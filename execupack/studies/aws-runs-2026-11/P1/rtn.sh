cd ~/execupack-main && . ~/execupack/.venv/bin/activate && set -a && . ~/.config/execupack/hf.env && set +a
R=/home/ubuntu/runs/phi4-mini
rm -rf $R/rtn-out $R/rtn-work
python -m pipeline export-xnnpack microsoft/Phi-4-mini-instruct --revision cfbefacb99257ffa30c83adab238a50856ac3083 --out $R/rtn-out --work $R/rtn-work --context 2048 --skip-smoke > $R/rtn-export.log 2>&1 || { echo EXPORT-FAILED; exit 1; }
python - <<"PY"
import json
from pathlib import Path
from transformers import AutoTokenizer
from pipeline import gate
R = Path("/home/ubuntu/runs/phi4-mini")
ref = json.loads((R / "gate.json").read_text())
tok = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", revision="cfbefacb99257ffa30c83adab238a50856ac3083")
pte = next((R / "rtn-out/xnnpack").glob("*.pte"))
m = gate.measure(pte, R / "rtn-out/tokenizer.json", tok, ref)
v = gate.check(ref, m)
print("RTN", pte.name, {k: v[k] for k in ("passed", "agreed", "graded", "fp32_mean", "export_mean", "quiet_rows_answered_like_fp32")}, v["problems"])
for q, r in m.items():
    print("  ", round(ref["rows"][q]["p"], 3), "->", round(r["p"], 3), "top", repr(tok.decode([r["top"]])), q)
PY
