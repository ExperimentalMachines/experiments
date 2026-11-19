#!/usr/bin/env bash
# Unattended XNNPACK re-export of one model on this machine, then power off.
#
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct REV=<sha> RUNS=~/runs/qwen25-15b \
#     nohup bash ~/run-export.sh > ~/runs/qwen25-15b/queue.out 2>&1 &
#
# 1. GPTQ solve once (codes.pt + the gate's fp32 reference gate.json).
# 2. Every window, at most $PARALLEL at a time: export with the solved codes, verify
#    (smoke test through the runner + decision gate against fp32), publish to Hugging Face.
#    A window that fails its gate is never published.
# 3. Lists what is on the Hub, writes summary.txt, stops the machine.
#
# Progress: $RUNS/queue.log. Code: $CODE (origin/main), toolchain: ~/execupack/.venv.
set -uo pipefail

: "${MODEL:?}" "${REV:?}" "${RUNS:?}"
CODE=${CODE:-$HOME/execupack-main}
read -r -a WINDOWS <<< "${WINDOWS:-32768 16384 8192 4096 2048}"
PARALLEL=${PARALLEL:-3}
LOG=$RUNS/queue.log
mkdir -p "$RUNS"
# Dead-man stop: whatever happens below, compute billing ends within 8 hours.
sudo shutdown -h +480 "execupack dead-man stop" 2> /dev/null
cd "$CODE" || exit 1
# shellcheck disable=SC1091
. "$HOME/execupack/.venv/bin/activate"
set -a
# shellcheck disable=SC1090
. "$HOME/.config/execupack/hf.env"
set +a
export PYTHONUNBUFFERED=1

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

solve() {
  for attempt in 1 2; do
    if [ -s "$RUNS/codes.pt" ] && [ -s "$RUNS/gate.json" ]; then
      log "SOLVED codes.pt $(stat -c %s "$RUNS/codes.pt") B, gate.json present"
      return 0
    fi
    log "SOLVE attempt $attempt"
    rm -rf "$RUNS/solve-work"
    python -m pipeline solve "$MODEL" --revision "$REV" --out "$RUNS/codes.pt" --work "$RUNS/solve-work" \
      > "$RUNS/solve-$attempt.log" 2>&1
    log "SOLVE attempt $attempt exited $?"
  done
  [ -s "$RUNS/codes.pt" ] && [ -s "$RUNS/gate.json" ]
}

window() {
  local w=$1 out=$RUNS/out-$1 work=$RUNS/work-$1 rc=1
  if [ -f "$RUNS/published-$w" ]; then
    log "SKIP $w already published"
    return 0
  fi
  for attempt in 1 2; do
    rm -rf "$out" "$work"
    log "EXPORT $w attempt $attempt"
    python -m pipeline export-xnnpack "$MODEL" --revision "$REV" --out "$out" --work "$work" \
      --context "$w" --codes "$RUNS/codes.pt" --skip-smoke > "$RUNS/export-$w-$attempt.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    log "EXPORT-FAILED $w attempt $attempt (exit $rc)"
  done
  if [ $rc -ne 0 ]; then
    log "GIVE-UP $w at export"
    return 1
  fi
  rm -rf "$work"
  log "EXPORTED $w $(ls -l "$out"/xnnpack/*.pte | awk '{print $5, $9}')"
  for attempt in 1 2; do
    rm -rf "$RUNS/vwork-$w"
    python -m pipeline verify "$out" --backend xnnpack --work "$RUNS/vwork-$w" --gate "$RUNS/gate.json" \
      > "$RUNS/verify-$w-$attempt.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    log "VERIFY-FAILED $w attempt $attempt (exit $rc)"
    # A smoke or gate failure is a property of the file; only a crash is worth a second run.
    if grep -q "smoke test failed\|decision gate failed" "$RUNS/verify-$w-$attempt.log"; then
      break
    fi
  done
  if [ $rc -ne 0 ]; then
    log "GIVE-UP $w at verify: $(grep -h "failed" "$RUNS/verify-$w-$attempt.log" | tail -1)"
    return 1
  fi
  log "VERIFIED $w $(python - "$out" <<'EOF'
import json, sys, pathlib
r = json.loads(next(pathlib.Path(sys.argv[1], "xnnpack").glob("export-report-*.json")).read_text())
g = r.get("gate") or {}
print(f"smoke={r['smoke']['passed']} reply={r['smoke'].get('reply', '')[:60]!r} gate={g.get('passed')} "
      f"agreed={g.get('agreed')}/{g.get('graded')} quiet={g.get('quiet_rows_answered_like_fp32')}/{g.get('quiet_rows')} "
      f"fp32={g.get('fp32_mean')} export={g.get('export_mean')}")
EOF
)"
  if [ -n "${NO_PUBLISH:-}" ]; then
    log "HOLD $w verified, not published (NO_PUBLISH)"
    return 0
  fi
  for attempt in 1 2 3 4; do
    if python -m pipeline publish-hf "$out" --backend xnnpack > "$RUNS/publish-$w-$attempt.log" 2>&1; then
      touch "$RUNS/published-$w"
      log "PUBLISHED $w"
      # The .pte is on the Hub; keep the reports and logs, free the disk.
      rm -f "$out"/xnnpack/*.pte
      return 0
    fi
    log "PUBLISH-FAILED $w attempt $attempt"
    sleep 120
  done
  log "GIVE-UP $w at publish"
  return 1
}

log "QUEUE START $MODEL@$REV windows ${WINDOWS[*]} code $(cat "$CODE/.commit" 2> /dev/null)"
if ! solve; then
  log "SOLVE FAILED twice; nothing exported"
else
  for w in "${WINDOWS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 30; done
    window "$w" &
    sleep 20
  done
  wait
fi

python - "$MODEL" > "$RUNS/summary.txt" 2>&1 <<'EOF'
import sys
from huggingface_hub import HfApi
repo = "experimentalmachines/" + sys.argv[1].rsplit("/", 1)[-1] + "-ExecuTorch"
try:
    info = HfApi().model_info(repo, files_metadata=True)
    for s in info.siblings:
        print(f"{s.size or 0:>14,}  {s.rfilename}")
    print("repo", repo, "sha", info.sha)
except Exception as error:  # noqa: BLE001
    print("could not list", repo, error)
EOF
log "HUB $(grep -c '\.pte$' "$RUNS/summary.txt") .pte files in the repo"
log "QUEUE DONE published: $(ls "$RUNS" | grep -c '^published-') of ${#WINDOWS[@]}"
df -h / | tail -1 >> "$LOG"
# The HF token leaves the machine with the job.
shred -u "$HOME/.config/execupack/hf.env" 2> /dev/null
log "TOKEN shredded"
# Compute billing ends here; the instance's shutdown behaviour is stop.
[ -z "${NO_SHUTDOWN:-}" ] && sudo shutdown -h +3 "execupack queue finished"
