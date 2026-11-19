#!/bin/bash
# One model: the fp32 reference on Modal, then every arm in parallel on Modal.
# Usage: chain.sh NAME MODEL_ID REVISION PUBLISHED_FILE
set -u
NAME=$1; export MODEL_ID=$2 REVISION=$3; PUB=$4
cd /Users/alpha/execupack-awq
D=work/awq/$NAME; mkdir -p "$D"
export HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
if [ ! -f "$D/ref/evalref.pt" ]; then
  PUTS=(--put work/awq/heldout.json); RESUME_ENV=()
  if [ -f "$D/ref/rows.pt" ] && [ -f "$D/ref/gate.json" ]; then
    rm -rf "$D/resume"; mv "$D/ref" "$D/resume"; PUTS+=(--put "$D/resume"); RESUME_ENV=(--env RESUME=1)
  elif [ -f "$D/resume/rows.pt" ]; then
    PUTS+=(--put "$D/resume"); RESUME_ENV=(--env RESUME=1)
  fi
  .venv/bin/python -m pipeline modal-run --memory-mib 40960 --cpu 8 "${PUTS[@]}" ${RESUME_ENV[@]+"${RESUME_ENV[@]}"} --get "$D/ref" \
    --script 'python scripts/awq_compare.py reference "$MODEL_ID" --revision "$REVISION" --heldout "$IN/heldout.json" ${RESUME:+--resume "$IN/resume"} --out "$OUT" --work "$WORK"' \
    > "$D/ref.log" 2>&1 || { echo "$NAME reference FAILED"; exit 1; }
fi
echo "$NAME reference done"
for method in ${METHODS:-rtn gptq awq awq_gptq published}; do
  .venv/bin/python -m pipeline modal-run --memory-mib 40960 --cpu 8 --put "$D/ref" --get "$D/$method" \
    --env METHOD=$method --env PUB="$PUB" \
    --script 'python scripts/awq_compare.py arm "$MODEL_ID" --revision "$REVISION" --method "$METHOD" --ref "$IN/ref" --published-file "$PUB" --out "$OUT" --work "$WORK"' \
    > "$D/$method.log" 2>&1 && echo "$NAME $method done" || echo "$NAME $method FAILED" &
done
wait
