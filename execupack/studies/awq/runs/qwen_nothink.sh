#!/bin/bash
# Qwen3-1.7B: calibration rows with thinking off, then GPTQ on them, measured against the same reference.
set -u
cd /Users/alpha/execupack-awq
export MODEL_ID=Qwen/Qwen3-1.7B REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
D=work/awq/qwen3-1.7b
if [ ! -f "$D/calib-nothink/rows.pt" ]; then
  .venv/bin/python -m pipeline modal-run --memory-mib 40960 --cpu 8 --get "$D/calib-nothink" \
    --script 'python scripts/awq_compare.py calib "$MODEL_ID" --revision "$REVISION" --thinking off --out "$OUT" --work "$WORK"' \
    > "$D/calib-nothink.log" 2>&1 || { echo "calib FAILED"; exit 1; }
fi
echo "calib done"
rm -rf "$D/ref-nothink"; mkdir -p "$D/ref-nothink"
cp "$D/ref/gate.json" "$D/ref/evalref.pt" "$D/ref/reference.json" "$D/ref-nothink/"
cp "$D/calib-nothink/rows.pt" "$D/ref-nothink/rows.pt"
.venv/bin/python -m pipeline modal-run --memory-mib 40960 --cpu 8 --put "$D/ref-nothink" --get "$D/gptq-nothink" \
  --script 'python scripts/awq_compare.py arm "$MODEL_ID" --revision "$REVISION" --method gptq --ref "$IN/ref-nothink" --out "$OUT" --work "$WORK"' \
  > "$D/gptq-nothink.log" 2>&1 && echo "gptq-nothink done" || echo "gptq-nothink FAILED"
