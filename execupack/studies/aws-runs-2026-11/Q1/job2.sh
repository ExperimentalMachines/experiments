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
    uv venv --allow-existing --python 3.11 .venv && . .venv/bin/activate
    uv pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    uv pip install -r "$HOME/execupack-main2/requirements/export-xnnpack.txt"
    python -c "import torch, executorch, transformers; print(torch.__version__, executorch.__file__, transformers.__version__)"
    # The pre-publish byte check needs ExecuTorch's flatc and must import on this box.
    python -c "import sys; sys.path.insert(0, '$HOME/execupack-main2/codecheck'); import codesmatch, ptecodes; from executorch.exir._serialize._flatbuffer import _get_flatc_path; print('flatc', _get_flatc_path())"
  } >> "$RUNS/setup.log" 2>&1
  finish $?
fi

cd "$HOME/execupack-main2" || finish 1
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
    ssh -q -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/boxkey" "ubuntu@$src" "test -f $RUNS/solve.ok" || finish 1
    for f in codes.pt gate.json; do
      scp -q -o StrictHostKeyChecking=accept-new -i "$HOME/.ssh/boxkey" "ubuntu@$src:$RUNS/$f" "$RUNS/$f.part" \
        && mv "$RUNS/$f.part" "$RUNS/$f" || finish 1
    done
    log "fetched codes.pt $(stat -c %s "$RUNS/codes.pt") B"
    finish 0
    ;;
  solve)
    # Success is the solve's own exit code, recorded as solve.ok: a killed solve can leave a
    # partial codes.pt behind, and files merely existing proved nothing (codex review).
    [ -f "$RUNS/solve.ok" ] && finish 0
    for attempt in 1 2; do
      rm -rf "$RUNS/solve-work" "$RUNS/codes.pt" "$RUNS/gate.json"
      python -m pipeline solve "$MODEL" --revision "$REV" --out "$RUNS/codes.pt" --work "$RUNS/solve-work" \
        > "$RUNS/solve-$attempt.log" 2>&1
      rc=$?
      log "solve attempt $attempt exited $rc"
      if [ $rc -eq 0 ] && [ -s "$RUNS/codes.pt" ] && [ -s "$RUNS/gate.json" ]; then
        echo "$MODEL $REV" > "$RUNS/solve.ok"
        finish 0
      fi
    done
    finish 1
    ;;
  parity)
    python "$HOME/parity.py" "$HOME/execupack-main" "$RUNS/parity" "$RUNS/parity.json" > "$RUNS/parity.log" 2>&1
    finish $?
    ;;
  xnn | vk | vkc)
    # vkc: Vulkan with the solve's codes (published only after the XNNPACK file from the same
    # codes passed its gate; the driver orders that).
    backend=$([ "$kind" = xnn ] && echo xnnpack || echo vulkan)
    out=$RUNS/out-$backend-$W work=$RUNS/work-$backend-$W
    # No shortcut on an earlier run's published-* marker: that run's codes, recipe or report
    # may not be this one's (codex review); a retried window is rebuilt and checked again.
    rc=1
    for attempt in 1 2; do
      rm -rf "$out" "$work"
      if [ "$kind" = xnn ]; then
        python -m pipeline export-xnnpack "$MODEL" --revision "$REV" --out "$out" --work "$work" \
          --context "$W" --codes "$RUNS/codes.pt" --skip-smoke > "$RUNS/export-$backend-$W-$attempt.log" 2>&1
      elif [ "$kind" = vkc ]; then
        proxy=$(ls "$RUNS/out-xnnpack-$W/xnnpack/"export-report-*.json 2>/dev/null | head -1)
        python -m pipeline export-vulkan "$MODEL" --revision "$REV" --out "$out" --work "$work" \
          --context "$W" --codes "$RUNS/codes.pt" --gate-proxy "$proxy" > "$RUNS/export-$backend-$W-$attempt.log" 2>&1
      else
        python -m pipeline export-vulkan "$MODEL" --revision "$REV" --out "$out" --work "$work" \
          --context "$W" > "$RUNS/export-$backend-$W-$attempt.log" 2>&1
      fi
      rc=$?
      log "export attempt $attempt exited $rc"
      [ $rc -eq 0 ] && break
      [ $rc -eq 4 ] && finish 4  # the export's own size gate: this window cannot be built
      grep -q "structural check\|exported metadata disagrees\|gate proxy\|needs --gate-proxy" "$RUNS/export-$backend-$W-$attempt.log" && break
    done
    [ $rc -ne 0 ] && finish $rc
    rm -rf "$work"
    log "exported $(ls -l "$out"/"$backend"/*.pte | awk '{print $5, $9}')"
    if [ "$kind" = xnn ]; then
      for attempt in 1 2; do
        rm -rf "$RUNS/vwork-$W"
        python -m pipeline verify "$out" --backend xnnpack --work "$RUNS/vwork-$W" --gate "$RUNS/gate.json" \
          ${ACCEPT_GATE:+--accept-gate "$ACCEPT_GATE"} > "$RUNS/verify-$W-$attempt.log" 2>&1
        rc=$?
        log "verify attempt $attempt exited $rc"
        [ $rc -eq 0 ] && break
        grep -q "smoke test failed\|decision gate failed" "$RUNS/verify-$W-$attempt.log" && break
      done
      [ $rc -ne 0 ] && finish 10
      rm -rf "$RUNS/vwork-$W"
      # Belt and braces: an XNNPACK file is published only with a passing gate in its report.
      python - "$out" <<'PY' || finish 11
import json, pathlib, sys
r = json.loads(next(pathlib.Path(sys.argv[1], "xnnpack").glob("export-report-*.json")).read_text())
g = r.get("gate") or {}
# An override covers what the operator accepted: one tool call short and nothing else wrong.
accepted = (g.get("override") or {}).get("reason") and len(g.get("problems") or []) == 1 \
    and (g.get("agreed") or 0) >= (g.get("graded") or 0) - 1
sys.exit(0 if (r.get("smoke") or {}).get("passed") and (g.get("passed") or accepted) else 1)
PY
    fi
    # Byte check before anything is published: every int4 linear in the .pte must be exactly
    # the solve's codes and scales (read from the file, matched by content), every codes.pt
    # entry must be in it, and the report must name these codes by their sha256. A file that
    # says GPTQ is GPTQ, down to the last nibble.
    pte_file=$(ls "$out"/"$backend"/*.pte | head -1)
    python "$HOME/execupack-main2/codecheck/codesmatch.py" "$pte_file" "$RUNS/codes.pt" \
      ${ALLOW_UNUSED:+--allow-unused "$ALLOW_UNUSED"} > "$RUNS/codesmatch-$backend-$W.json" 2>"$RUNS/codesmatch-$backend-$W.err" \
      || { log "codesmatch FAILED: $(tr -d '\n' < "$RUNS/codesmatch-$backend-$W.json" | cut -c1-400)"; finish 12; }
    python - "$out" "$backend" "$RUNS/codes.pt" <<'PY' || { log "report does not name these codes"; finish 13; }
import hashlib, json, pathlib, sys
r = json.loads(next(pathlib.Path(sys.argv[1], sys.argv[2]).glob("export-report-*.json")).read_text())
h = hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest()
rec = r["recipe"]
ok = rec.get("int4_codes") == "gptq" and rec.get("codes_sha256") == h
gate = r.get("gate") if sys.argv[2] == "xnnpack" else ((r.get("gate_proxy") or {}).get("gate"))
# This rollout publishes only measured passes: no operator override stands in for a gate here.
ok = ok and (gate or {}).get("passed") is True and not (gate or {}).get("override")
sys.exit(0 if ok else 1)
PY
    log "codesmatch ok: $(python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['matched_exactly'], 'of', d['int4_linears'], 'int4 linears exact')" "$RUNS/codesmatch-$backend-$W.json")"
    log "REPORT $(python - "$out" "$backend" <<'PY'
import json, pathlib, sys
r = json.loads(next(pathlib.Path(sys.argv[1], sys.argv[2]).glob("export-report-*.json")).read_text())
g, s = r.get("gate") or {}, r.get("smoke") or {}
print(json.dumps({"file": r["files"][0]["path"], "bytes": r["files"][0]["bytes"], "recipe": r["recipe"].get("label"),
                  "smoke": s.get("passed"),
                  "reply": (s.get("reply") or "")[:80], "gate": g.get("passed"), "agreed": g.get("agreed"),
                  "graded": g.get("graded"), "quiet": g.get("quiet_rows_answered_like_fp32"),
                  "fp32_mean": g.get("fp32_mean"), "export_mean": g.get("export_mean"),
                  "tool_calling": g.get("measures_tool_calling"), "override": (g.get("override") or {}).get("reason"),
                  "proxy": (r.get("gate_proxy") or {}).get("gate"), "fits_phone": r.get("estimates", {}).get("fits_phone_budget")}))
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
