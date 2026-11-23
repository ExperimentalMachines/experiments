#!/bin/bash
# KV study builds on Modal: per model, one GPTQ solve (scripts/kv_solve.py) and then every window x cache exported
# from those codes in parallel (scripts/kv_export.py): 2k, 8k, 32k x fp32 cache, int8 cache.
# usage: scripts/kvstudy/build.sh [MODEL_KEY...]   (default: llama lfm qwen; Gemma is built by its own port)
# Layout on the execupack-runs Volume: /vol/study/codes/<key>-codes-gptq.pt, /vol/study/<window>/<f32kv|q8kv>-<name>.pte
# Logs and downloaded summaries: work/kvstudy/{logs,res}; one line per finished step in work/kvstudy/status.log.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
W=$R/work/kvstudy
PY=${PY:-/private/tmp/claude-501/-Users-alpha-execupack/269b336d-a21c-4e52-8bb8-87f1d7e4a1c0/scratchpad/kvq/venv/bin/python}
cd "$R"
export PYTHONPATH=$R HF_TOKEN=$(cat ~/.cache/huggingface/token) HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$W/logs" "$W/res"
log() { echo "$(date +%H:%M:%S) $*" >> "$W/status.log"; }

# key -> "model revision 32k-export-memory-MiB" (macOS bash 3.2 has no associative arrays). The 32k memory is
# twice each model's estimated export peak (export reports: Llama 27.8, LFM 16.3, Qwen 48.2 GB).
spec() {
  case $1 in
    llama) echo "meta-llama/Llama-3.2-1B-Instruct 9213176726f574b556790deb65791e0c5aa438b6 65536" ;;
    lfm) echo "LiquidAI/LFM2.5-1.2B-Instruct 0f604ada3f766f9f257460c4c9f0b5d6f69d431b 49152" ;;
    qwen) echo "Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e 98304" ;;
  esac
}

model_chain() {
  local k=$1 codes=/vol/study/codes/$1-codes-gptq.pt
  local model rev mem32
  read -r model rev mem32 <<< "$(spec "$k")"
  if ! $PY -m pipeline modal-run --memory-mib 40960 --cpu 8 --put "$W/refs/$k/ref" --get "$W/res/$k-solve" \
      --env MODEL_ID=$model --env REVISION=$rev --env CODES=$codes \
      --script 'python scripts/kv_solve.py "$MODEL_ID" --revision "$REVISION" --ref "$IN/ref" --work "$WORK" --out "$OUT" --codes "$CODES"' \
      > "$W/logs/$k-solve.log" 2>&1; then
    log "$k solve FAILED"; return 1
  fi
  log "$k solve DONE"
  local pids=()
  for ctx in 2048 8192 32768; do
    local win=$((ctx / 1024))k mem=32768
    [ $ctx = 32768 ] && mem=$mem32
    for kv in fp32 int8; do
      ( $PY -m pipeline modal-run --memory-mib $mem --cpu 8 --timeout-minutes 240 --get "$W/res/$k-$win-$kv" \
          --env MODEL_ID=$model --env REVISION=$rev --env CODES=$codes --env KV=$kv --env CTX=$ctx --env DEST=/vol/study/$win \
          --script 'python scripts/kv_export.py "$MODEL_ID" --revision "$REVISION" --codes "$CODES" --kv "$KV" --context "$CTX" --work "$WORK" --out "$OUT" --keep "$DEST"' \
          > "$W/logs/$k-$win-$kv.log" 2>&1 && log "$k $win $kv DONE" || log "$k $win $kv FAILED" ) &
      pids+=($!)
    done
  done
  wait "${pids[@]}"
}

keys=${*:-llama lfm qwen}
for k in $keys; do model_chain "$k" & done
wait
log "ALL DONE"
