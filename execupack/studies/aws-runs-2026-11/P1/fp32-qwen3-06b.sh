#!/usr/bin/env bash
# Qwen3-0.6B with fp32 linears on XNNPACK: export, gate against the solve's fp32 reference,
# check the bytes and the report, publish. One window per call.
set -uo pipefail
W=$1; R=$HOME/runs/fp32-qwen3-06b; G=$HOME/runs/gptq-qwen3-06b/gate.json
mkdir -p $R; LOG=$R/fp32-$W.log; exec >> $LOG 2>&1
cd $HOME/execupack-main2 && . $HOME/execupack/.venv/bin/activate && set -a && . $HOME/.config/execupack/hf.env && set +a
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
echo "$(date -u +%FT%TZ) START $W"
rm -rf $R/out-$W $R/work-$W $R/vwork-$W
python -m pipeline export-xnnpack Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --out $R/out-$W --work $R/work-$W --context $W --qmode fp32 --skip-smoke || { echo "EXPORT FAILED rc=$?"; exit 1; }
rm -rf $R/work-$W
python -m pipeline verify $R/out-$W --backend xnnpack --work $R/vwork-$W --gate $G || { echo "VERIFY FAILED"; exit 10; }
rm -rf $R/vwork-$W
python - $R/out-$W $G <<'PY' || { echo "CHECK FAILED"; exit 12; }
import hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path.home() / "execupack-main2/codecheck"))
from ptecodes import load_pte, xnnpack_int4, vulkan_int4
out, gate = pathlib.Path(sys.argv[1], "xnnpack"), pathlib.Path(sys.argv[2])
r = json.loads(next(out.glob("export-report-*.json")).read_text())
pte = out / pathlib.Path(r["files"][0]["path"]).name
p, named = load_pte(pte)
int4 = len(xnnpack_int4(p, named)) + len(vulkan_int4(p, named))
g, rec = r.get("gate") or {}, r["recipe"]
ok = (int4 == 0 and rec.get("qmode") == "fp32" and rec.get("int4_codes") is None and g.get("passed") is True
      and not g.get("override") and g.get("reference_sha256") == hashlib.sha256(gate.read_bytes()).hexdigest()
      and hashlib.sha256(pte.read_bytes()).hexdigest() == r["files"][0]["sha256"])
print(json.dumps({"file": r["files"][0]["path"], "sha256": r["files"][0]["sha256"], "int4_linears": int4, "qmode": rec.get("qmode"),
                  "gate": {k: g.get(k) for k in ("passed", "graded", "agreed", "fp32_mean", "export_mean", "quiet_rows_answered_like_fp32")}, "ok": ok}))
sys.exit(0 if ok else 1)
PY
for a in 1 2 3; do python -m pipeline publish-hf $R/out-$W --backend xnnpack && { echo "$(date -u +%FT%TZ) PUBLISHED $W"; rm -f $R/out-$W/xnnpack/*.pte; exit 0; }; sleep 60; done
echo "PUBLISH FAILED"; exit 20
