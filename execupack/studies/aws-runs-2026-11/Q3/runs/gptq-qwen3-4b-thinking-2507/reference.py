# The gate reference exactly as pipeline.solve.run computes it (teacher, then gate.reference),
# without the calibration that run does first: the reference does not depend on it.
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from pipeline import gate
from pipeline.solve import teacher

src, out = Path(sys.argv[1]), Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(str(src))
fp32, _ = teacher(src, trust_remote_code=False)
ref = gate.reference(fp32, tok)
out.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
print("fp32 is confident on", ref["confident"], "of", len(ref["rows"]), "gate rows")
