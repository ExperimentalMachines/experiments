#!/bin/bash
# fp16 GPTQ files for the phone: each model's saved codes at 2k (plus gate and held-out eval when x86 runs it) and 8k.
set -u
G=/private/tmp/claude-501/-Users-alpha-execupack/269b336d-a21c-4e52-8bb8-87f1d7e4a1c0/scratchpad/gq
PY=$G/../kvq/venv/bin/python
cd $G/src
export PYTHONPATH=$G/src HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p $G/logs16 $G/res16
for spec in "llama meta-llama/Llama-3.2-1B-Instruct 9213176726f574b556790deb65791e0c5aa438b6 /vol/local-1-d257dc0a/keep/codes-gptq.pt" \
            "lfm LiquidAI/LFM2.5-1.2B-Instruct 0f604ada3f766f9f257460c4c9f0b5d6f69d431b /vol/local-1-bbcb08eb/keep/codes-gptq.pt" \
            "qwen Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e /vol/local-1-84529e26/keep/codes-gptq.pt"; do
  set -- $spec
  for ctx in 2048 8192; do
    if [ $ctx = 2048 ]; then puts="--put $G/refs/$1/ref"; ref='--ref "$IN/ref"'; dest=/vol/phone-2k; else puts=""; ref=""; dest=/vol/phone-8k; fi
    ( $PY -m pipeline modal-run --memory-mib 32768 --cpu 8 $puts --get $G/res16/$1-$ctx \
        --env MODEL_ID=$2 --env REVISION=$3 --env CODES=$4 --env CTX=$ctx --env DEST=$dest \
        --script "python scripts/kv_export.py \"\$MODEL_ID\" --revision \"\$REVISION\" --codes \"\$CODES\" --kv fp16 --context \"\$CTX\" --work \"\$WORK\" --out \"\$OUT\" --keep \"\$DEST\" $ref" \
        > $G/logs16/$1-$ctx.log 2>&1 && echo "$(date +%H:%M:%S) $1 $ctx DONE" || echo "$(date +%H:%M:%S) $1 $ctx FAILED ($?)" ) >> $G/status16.log &
  done
done
wait
echo "$(date +%H:%M:%S) ALL DONE" >> $G/status16.log
