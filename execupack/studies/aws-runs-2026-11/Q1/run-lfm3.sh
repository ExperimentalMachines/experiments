#!/usr/bin/env bash
# Unattended LFM3 build on this machine, then power off.
#
#   nohup bash ~/run-lfm3.sh > ~/runs/lfm3/queue.out 2>&1 &
#
# 1. GPTQ solve once (codes.pt + the gate's fp32 reference gate.json); waits for a solve
#    already running, reruns it once if it left nothing.
# 2. Every window, at most $PARALLEL at a time: export with the solved codes, verify
#    (smoke test through the runner + decision gate against fp32), publish to Hugging Face.
#    A window that fails its gate is never published.
# 3. Checks what is on the Hub, writes summary.txt, stops the machine.
#
# Progress: ~/runs/lfm3/queue.log. The machine also has a dead-man stop 20 h after boot.
set -uo pipefail

MODEL=${MODEL:-LiquidAI/LFM3-1.7B}
REV=${REV:-ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7}
WINDOWS=(32768 16384 8192 4096 2048)
PARALLEL=3
RUNS=${RUNS:-$HOME/runs/lfm3}
# --allow-base for a base checkpoint asked for by name; SHUTDOWN=0 leaves the machine on.
EXTRA=${EXTRA:-}
SHUTDOWN=${SHUTDOWN:-1}
LOG=$RUNS/queue.log
mkdir -p "$RUNS"
cd "$HOME/execupack" || exit 1
# shellcheck disable=SC1091
. .venv/bin/activate
set -a
# shellcheck disable=SC1090
. "$HOME/.config/execupack/hf.env"
set +a
export PYTHONUNBUFFERED=1

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

solve() {
  # A solve started by hand before this script: wait for it rather than run a second one.
  while pgrep -f "pipeline solve $MODEL" > /dev/null; do sleep 60; done
  for attempt in 1 2; do
    if [ -s "$RUNS/codes.pt" ] && [ -s "$RUNS/gate.json" ]; then
      log "SOLVED codes.pt $(stat -c %s "$RUNS/codes.pt") B, gate.json present"
      return 0
    fi
    log "SOLVE attempt $attempt"
    rm -rf "$RUNS/solve-work"
    python -m pipeline solve "$MODEL" --revision "$REV" --out "$RUNS/codes.pt" --work "$RUNS/solve-work" $EXTRA \
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
  if ls "$out"/xnnpack/*.pte > /dev/null 2>&1 && ls "$out"/xnnpack/export-report-*.json > /dev/null 2>&1; then
    # Exported by an earlier run of this queue: keep the file, drop any earlier verdict so
    # verify runs the smoke test and the gate again (it leaves a report with a smoke result alone).
    python - "$out" <<'EOF2'
import json, pathlib, sys
for path in pathlib.Path(sys.argv[1], "xnnpack").glob("export-report-*.json"):
    report = json.loads(path.read_text())
    for key in ("smoke", "gate", "measuring_gate"):
        report[key] = None if key == "smoke" else report.get(key)
        if key != "smoke":
            report.pop(key, None)
    path.write_text(json.dumps(report, indent=2) + "\n")
EOF2
    log "REUSE $w $(ls -l "$out"/xnnpack/*.pte | awk '{print $5, $9}')"
    rc=0
  fi
  for attempt in 1 2; do
    [ $rc -eq 0 ] && break
    rm -rf "$out" "$work"
    log "EXPORT $w attempt $attempt"
    python -m pipeline export-xnnpack "$MODEL" --revision "$REV" --out "$out" --work "$work" \
      --context "$w" --codes "$RUNS/codes.pt" --skip-smoke $EXTRA > "$RUNS/export-$w-$attempt.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    log "EXPORT-FAILED $w attempt $attempt (exit $rc)"
  done
  if [ $rc -ne 0 ]; then
    log "GIVE-UP $w at export"
    return 1
  fi
  log "READY $w $(ls -l "$out"/xnnpack/*.pte | awk '{print $5, $9}')"
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
      f"agreed={g.get('agreed')}/{g.get('graded')} fp32={g.get('fp32_mean')} export={g.get('export_mean')}")
EOF
)"
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

log "QUEUE START $MODEL@$REV windows ${WINDOWS[*]}"
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
# Compute billing ends here; the instance's shutdown behaviour is stop.
if [ "$SHUTDOWN" = 1 ]; then
  sudo shutdown -h +3 "execupack-lfm3 queue finished"
fi
