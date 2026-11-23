#!/bin/bash
# 8k GPTQ files for the phone: each model's saved codes, exported with the fp32 and the int8 KV cache (6 sandboxes).
set -u
G=/private/tmp/claude-501/-Users-alpha-execupack/269b336d-a21c-4e52-8bb8-87f1d7e4a1c0/scratchpad/gq
PY=$G/../kvq/venv/bin/python
cd $G/src
export PYTHONPATH=$G/src HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $G/logs8k $G/res8k
for spec in "llama meta-llama/Llama-3.2-1B-Instruct 9213176726f574b556790deb65791e0c5aa438b6 /vol/local-1-d257dc0a/keep/codes-gptq.pt" \
            "lfm LiquidAI/LFM2.5-1.2B-Instruct 0f604ada3f766f9f257460c4c9f0b5d6f69d431b /vol/local-1-bbcb08eb/keep/codes-gptq.pt" \
            "qwen Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e /vol/local-1-84529e26/keep/codes-gptq.pt"; do
  set -- $spec
  for kv in fp32 int8; do
    ( $PY -m pipeline modal-run --memory-mib 32768 --cpu 8 --get $G/res8k/$1-$kv \
        --env MODEL_ID=$2 --env REVISION=$3 --env CODES=$4 --env KV=$kv \
        --script 'python scripts/kv_export.py "$MODEL_ID" --revision "$REVISION" --codes "$CODES" --kv "$KV" --context 8192 --work "$WORK" --out "$OUT" --keep /vol/phone-8k' \
        > $G/logs8k/$1-$kv.log 2>&1 && echo "$(date +%H:%M:%S) $1 $kv DONE" || echo "$(date +%H:%M:%S) $1 $kv FAILED ($?)" ) >> $G/status8k.log &
  done
done
wait
echo "$(date +%H:%M:%S) ALL DONE" >> $G/status8k.log
