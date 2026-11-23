#!/bin/bash
# The KV study's int8-cache exports on the Colab VM, beside colab_job.py, whose own int8 exports fail: its work
# directories were named after the cache ("...-int8-2k"), and ExecuTorch 1.5.1's llama/model.py takes any
# checkpoint path containing "int8" for an int8 weight-only checkpoint. Here every path says q8kv.
# Order: Qwen 2k, 8k (clean tree, Qwen's codes) -> Gemma 2k, 8k once its codes exist (Gemma tree) -> Gemma 32k
# once colab_job.py's Gemma 32k fp32 export has finished (one 32k export at a time, for memory).
# Announces each file as "READY <window>/<name>" in /content/int8.log, the line colab_ship.sh ships from.
set -u
C=/content
PY=$C/venv/bin/python
log() { echo "$(date +%H:%M:%S) $*" >> $C/int8.log; }

export_q8() {  # tree model revision codes context [extra env]
  local tree=$1 model=$2 rev=$3 codes=$4 ctx=$5 win=$(($5 / 1024))k
  local tag=$(basename "$model")-q8kv-$win
  if (cd $C/$tree && env PYTHONPATH=$C/$tree ${6:-} $PY scripts/kv_export.py "$model" --revision "$rev" --codes "$codes" \
      --kv int8 --context $ctx --work $C/work/$tag --out $C/res/$tag --keep $C/out/$win > $C/logs/export-$tag.log 2>&1); then
    local line
    line=$($PY -c "import json,glob; d=json.load(open(sorted(glob.glob('$C/res/$tag/export-*.json'))[0])); print(('READY' if d['smoke'].get('passed') else 'SMOKE-FAILED'), '$win/' + str(d.get('pte')), d.get('seconds'))")
    log "$line"
  else
    log "FAILED $tag: $(tail -2 $C/logs/export-$tag.log | tr '\n' ' ' | cut -c1-300)"
  fi
  rm -rf $C/work/$tag
}

QWEN="Qwen/Qwen3-1.7B 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
GEMMA="google/gemma-3-1b-it dcc83ea841ab6100d6b47a070329e1ba4cf78752"
log "int8 lane started"
for ctx in 2048 8192; do export_q8 execupack-kvbuild $QWEN $C/codes/qwen-codes-gptq.pt $ctx; done
until [ -f $C/codes/gemma-codes-gptq.pt ]; do sleep 30; done
sleep 20  # the solve writes the file with torch.save; give it time to close
for ctx in 2048 8192; do export_q8 execupack-kvstudy $GEMMA $C/codes/gemma-codes-gptq.pt $ctx; done
until ls $C/res/gemma-3-1b-it-fp32-32k/export-*.json > /dev/null 2>&1; do sleep 60; done
export_q8 execupack-kvstudy $GEMMA $C/codes/gemma-codes-gptq.pt 32768 KV_STUDY_HOST_BUDGET_BYTES=400000000000
log "int8 lane done"
