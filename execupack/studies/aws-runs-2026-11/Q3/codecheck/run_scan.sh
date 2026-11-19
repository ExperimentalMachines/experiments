#!/usr/bin/env bash
# python make_tasks + N parallel scan.py workers on this box. Usage: run_scan.sh TAG N [make_tasks filter args]
set -uo pipefail
TAG=$1 N=$2; shift 2
cd ~/codecheck && . ~/execupack/.venv/bin/activate && set -a && . ~/.config/execupack/hf.env && set +a
export EXECUPACK=$HOME/execupack-main2 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
python make_tasks.py experimentalmachines "$N" tasks/$TAG "$@" > logs/$TAG-tasks.log 2>&1 || exit 1
for i in $(seq 0 $((N - 1))); do
  python scan.py tasks/$TAG/tasks-$i.json results/$TAG-$i.jsonl dl/$TAG-$i work/$TAG-$i heretic-overrides.json > logs/$TAG-$i.log 2>&1 &
done
wait
touch results/$TAG.done
