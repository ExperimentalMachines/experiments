#!/usr/bin/env bash
# One export job on this box, started by the Mac's driver.py:
#
#   nohup bash ~/job.sh <id> <kind> <model> <rev> <runs> [window] &
#
# kind: setup | solve | xnn | vk | parity | fetch <solver-private-ip>. Writes
# $RUNS/jobs/<id>.rc (the exit code) when it ends, and nothing is published unless the
# export passed its own checks (verify's smoke test and decision gate for XNNPACK, the
# structural check inside export-vulkan for Vulkan).
set -uo pipefail
id=$1 kind=$2 MODEL=$3 REV=$4 RUNS=$5 W=${6:-}
mkdir -p "$RUNS/jobs"
LOG=$RUNS/jobs/$id.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
finish() { echo "$1" > "$RUNS/jobs/$id.rc"; log "END rc=$1"; exit 0; }
log "START $kind $MODEL $W"
export PATH=$HOME/.local/bin:$PATH PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1

if [ "$kind" = setup ]; then
  # A fresh Ubuntu 24.04 box: uv, Python 3.11 and the XNNPACK/Vulkan toolchain, as
  # .github/actions/setup-export installs it (torch from the CPU index first).
  {
    command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    cd "$HOME" && mkdir -p execupack && cd execupack
    uv venv --python 3.11 .venv && . .venv/bin/activate
    uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    uv pip install -r "$HOME/execupack-main/requirements/export-xnnpack.txt"
    python -c "import torch, executorch, transformers; print(torch.__version__, executorch.__file__, transformers.__version__)"
  } >> "$RUNS/setup.log" 2>&1
  finish $?
fi

cd "$HOME/execupack-main" || finish 1
# shellcheck disable=SC1091
. "$HOME/execupack/.venv/bin/activate"
set -a
# shellcheck disable=SC1090
. "$HOME/.config/execupack/hf.env"
set +a

case $kind in
  fetch)
    # The solve's codes and fp32 gate reference, from the box that solved them.
    src=$W
    for f in codes.pt gate.json; do
      scp -q -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/boxkey" "ubuntu@$src:$RUNS/$f" "$RUNS/$f.part" \
        && mv "$RUNS/$f.part" "$RUNS/$f" || finish 1
    done
    log "fetched codes.pt $(stat -c %s "$RUNS/codes.pt") B"
    finish 0
    ;;
  solve)
    for attempt in 1 2; do
      [ -s "$RUNS/codes.pt" ] && [ -s "$RUNS/gate.json" ] && finish 0
      rm -rf "$RUNS/solve-work"
      python -m pipeline solve "$MODEL" --revision "$REV" --out "$RUNS/codes.pt" --work "$RUNS/solve-work" \
        > "$RUNS/solve-$attempt.log" 2>&1
      log "solve attempt $attempt exited $?"
    done
    [ -s "$RUNS/codes.pt" ] && [ -s "$RUNS/gate.json" ] && finish 0
    finish 1
    ;;
  parity)
    python "$HOME/parity.py" "$HOME/execupack-main" "$RUNS/parity" "$RUNS/parity.json" > "$RUNS/parity.log" 2>&1
    finish $?
    ;;
  xnn | vk)
    backend=$([ "$kind" = xnn ] && echo xnnpack || echo vulkan)
    out=$RUNS/out-$backend-$W work=$RUNS/work-$backend-$W
    [ -f "$RUNS/published-$backend-$W" ] && finish 0
    rc=1
    for attempt in 1 2; do
      rm -rf "$out" "$work"
      if [ "$kind" = xnn ]; then
        python -m pipeline export-xnnpack "$MODEL" --revision "$REV" --out "$out" --work "$work" \
          --context "$W" --codes "$RUNS/codes.pt" --skip-smoke > "$RUNS/export-$backend-$W-$attempt.log" 2>&1
      else
        python -m pipeline export-vulkan "$MODEL" --revision "$REV" --out "$out" --work "$work" \
          --context "$W" > "$RUNS/export-$backend-$W-$attempt.log" 2>&1
      fi
      rc=$?
      log "export attempt $attempt exited $rc"
      [ $rc -eq 0 ] && break
      [ $rc -eq 4 ] && finish 4  # the export's own size gate: this window cannot be built
      grep -q "structural check\|exported metadata disagrees" "$RUNS/export-$backend-$W-$attempt.log" && break
    done
    [ $rc -ne 0 ] && finish $rc
    rm -rf "$work"
    log "exported $(ls -l "$out"/"$backend"/*.pte | awk '{print $5, $9}')"
    if [ "$kind" = xnn ]; then
      for attempt in 1 2; do
        rm -rf "$RUNS/vwork-$W"
        python -m pipeline verify "$out" --backend xnnpack --work "$RUNS/vwork-$W" --gate "$RUNS/gate.json" \
          > "$RUNS/verify-$W-$attempt.log" 2>&1
        rc=$?
        log "verify attempt $attempt exited $rc"
        [ $rc -eq 0 ] && break
        grep -q "smoke test failed\|decision gate failed" "$RUNS/verify-$W-$attempt.log" && break
      done
      [ $rc -ne 0 ] && finish 10
      rm -rf "$RUNS/vwork-$W"
    fi
    log "REPORT $(python - "$out" "$backend" <<'PY'
import json, pathlib, sys
r = json.loads(next(pathlib.Path(sys.argv[1], sys.argv[2]).glob("export-report-*.json")).read_text())
g, s = r.get("gate") or {}, r.get("smoke") or {}
print(json.dumps({"file": r["files"][0]["path"], "bytes": r["files"][0]["bytes"], "smoke": s.get("passed"),
                  "reply": (s.get("reply") or "")[:80], "gate": g.get("passed"), "agreed": g.get("agreed"),
                  "graded": g.get("graded"), "quiet": g.get("quiet_rows_answered_like_fp32"),
                  "fp32_mean": g.get("fp32_mean"), "export_mean": g.get("export_mean"),
                  "tool_calling": g.get("measures_tool_calling"), "fits_phone": r.get("estimates", {}).get("fits_phone_budget")}))
PY
)"
    [ -n "${NO_PUBLISH:-}" ] && finish 0
    for attempt in 1 2 3 4; do
      if python -m pipeline publish-hf "$out" --backend "$backend" > "$RUNS/publish-$backend-$W-$attempt.log" 2>&1; then
        touch "$RUNS/published-$backend-$W"
        log "published"
        rm -f "$out"/"$backend"/*.pte
        finish 0
      fi
      log "publish attempt $attempt failed"
      sleep 90
    done
    finish 20
    ;;
esac
finish 2
