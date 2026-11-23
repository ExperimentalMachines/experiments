#!/bin/bash
# Fourth round: ships each file the Colab job announced (READY lines in work/kvstudy/logs/colab-r4.log) to the third
# phone's gpu/ directory (gfetch.sh), with the sha256 computed on the VM. Each hash is also checked against the one the
# same file had before (32k-pc256.sha256, round 3's urls-r3gpu-*.txt): a match means the rebuild is byte-identical.
# Idempotent: shipped files are listed in work/kvstudy/colab-r4-shipped.txt.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); W=$R/work/kvstudy
LOG=$W/logs/colab-r4.log; DONE=$W/colab-r4-shipped.txt
touch "$DONE"
base=$(grep -o 'TUNNEL https://[^ ]*' "$LOG" | tail -1 | cut -d' ' -f2)
[ -z "$base" ] && { echo "no tunnel yet"; exit 1; }
gpu=$W/urls-r4gpu-$(date +%H%M%S).txt; : > "$gpu"
grep -o 'READY [0-9]*k/[^ ]*\.pte' "$LOG" | cut -d' ' -f2 | sort -u | while read -r wf; do
  grep -qx "$wf" "$DONE" && continue
  sum=$(perl -e 'alarm 120; exec @ARGV' colab ssh -s kvr4 -- "sha256sum /content/out/$wf" < /dev/null 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  [ -z "$sum" ] && { echo "no hash for $wf yet" >&2; continue; }
  old=$( (cat "$W/32k-pc256.sha256"; awk '{print $3, $2}' "$W"/urls-r3gpu-*.txt) 2>/dev/null | awk -v n="${wf#*/}" '{f=$2; sub(".*/", "", f)} f == n {print $1}' | sort -u)
  if [ -z "$old" ]; then echo "$wf: no earlier hash"; elif [ "$old" = "$sum" ]; then echo "$wf: same bytes as before"; else echo "$wf: DIFFERS from before ($old)"; fi
  echo "${wf%%/*} $base/${wf#*/} $sum" >> "$gpu"
  echo "$wf" >> "$DONE"
done
[ -s "$gpu" ] || { rm -f "$gpu"; echo "nothing new"; exit 0; }
adb push "$gpu" /data/local/tmp/kvq/gpu/ > /dev/null
perl -e 'alarm 20; exec @ARGV' adb shell "cd /data/local/tmp/kvq/gpu && setsid sh gfetch.sh $(basename "$gpu") > /dev/null 2>&1 < /dev/null &" 2>/dev/null || true
echo "shipped $(grep -c . "$gpu") file(s)"
