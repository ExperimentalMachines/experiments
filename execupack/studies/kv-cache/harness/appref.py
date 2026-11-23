"""The 141 held-out app questions' fp32 reference (scripts/awq_compare.py's evalref.pt) as JSON for the phone.

    python scripts/kvstudy/appref.py work/kvstudy/refs/<key>/ref/evalref.pt work/kvstudy/data/<key>/app.json

Same rows and replies, plain lists: StudyBench's app mode scores them as awq_compare.evaluate() does.
"""

import json
import sys

import torch

ref = torch.load(sys.argv[1], weights_only=True)
out = {
    "heldout": [
        {"ids": list(r["ids"]), "top_ids": r["top_ids"].tolist(), "top_logp": [round(x, 7) for x in r["top_logp"].tolist()]}
        for r in ref["heldout"]
    ],
    "replies": [
        {
            "prompt": list(r["prompt"]),
            "reply": list(r["reply"]),
            "top_ids": r["top_ids"].tolist(),
            "top_logp": [[round(x, 7) for x in row] for row in r["top_logp"].tolist()],
        }
        for r in ref["replies"]
    ],
}
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"))
print(sys.argv[2], len(out["heldout"]), "rows", len(out["replies"]), "replies")
