#!/bin/bash
# usage: eval16.sh NAME MODEL REVISION PTE_ON_VOLUME: gate + held-out eval of an fp16 file already on the Volume.
set -u
G=/private/tmp/claude-501/-Users-alpha-execupack/269b336d-a21c-4e52-8bb8-87f1d7e4a1c0/scratchpad/gq
PY=$G/../kvq/venv/bin/python
cd $G/src
export PYTHONPATH=$G/src HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $G/logs16 $G/res16
$PY -m pipeline modal-run --memory-mib 24576 --cpu 8 --put $G/refs/$1/ref --get $G/res16/$1-eval \
  --env MODEL_ID=$2 --env REVISION=$3 --env PTE=$4 \
  --script 'python scripts/kv_eval.py "$MODEL_ID" --revision "$REVISION" --pte "$PTE" --ref "$IN/ref" --out "$OUT" --work "$WORK"' \
  > $G/logs16/$1-eval.log 2>&1 && echo "$(date +%H:%M:%S) $1 eval DONE" >> $G/status16.log || echo "$(date +%H:%M:%S) $1 eval FAILED" >> $G/status16.log
