"""GPTQ codes for the KV-cache study, solved on an awq_compare reference's calibration rows.

    python scripts/kv_solve.py MODEL --revision SHA --ref REF_DIR --work WORK --codes OUT_CODES

The solve scripts/kv_compare.py runs (solve.eager_model, solve.solve_model on the reference's
rows.pt), on its own, so the codes land on the Volume before any export starts. Prints the
rows' lengths, which matter for a sliding-window model (pipeline/gemma3_model.py bands the
sliding layers' mask in this no-cache pass). An experiment, not a pipeline stage.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--revision", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--codes", required=True, help="where the codes go (on the Volume)")
    args = p.parse_args()

    import awq_compare as ac
    import torch

    from pipeline import solve

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t = time.time()
    _, plan, _, checkpoint = ac._prepare(args.model, args.revision, work / "solve")
    rows = torch.load(Path(args.ref) / "rows.pt", weights_only=True)
    lengths = sorted(len(ids) for ids, _ in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "min_len": lengths[0],
                "median_len": lengths[len(lengths) // 2],
                "max_len": lengths[-1],
                "window": plan.params.get("sliding_window"),
            }
        ),
        flush=True,
    )
    max_seq_len = max(2048, lengths[-1])
    model = solve.eager_model(plan.model_class, plan.params, checkpoint, work / "solve", max_seq_len=max_seq_len)
    prepared = time.time() - t
    t = time.time()
    codes = solve.solve_model(model, rows)
    solved = time.time() - t
    local = work / "codes-gptq.pt"
    torch.save(codes, local)
    Path(args.codes).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, args.codes)
    print(
        json.dumps({"codes": args.codes, "linears": len(codes), "prepare_s": round(prepared), "solve_s": round(solved)})
    )
    del model, codes
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
