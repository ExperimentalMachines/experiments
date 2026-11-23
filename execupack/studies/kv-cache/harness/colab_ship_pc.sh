#!/bin/bash
# Third round: ships each file the Colab job announced (READY lines in work/kvstudy/logs/colab-r3.log) to the phone,
# with the sha256 computed on the VM. Vulkan files go to study's sibling gpu/ (gfetch.sh), CPU files to study/
# (sfetch.sh). Idempotent: shipped files are listed in work/kvstudy/colab-r3-shipped.txt.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); W=$R/work/kvstudy
LOG=$W/logs/colab-pc256.log; DONE=$W/colab-pc-shipped.txt
touch "$DONE"
base=$(grep -o 'TUNNEL https://[^ ]*' "$LOG" | tail -1 | cut -d' ' -f2)
[ -z "$base" ] && { echo "no tunnel yet"; exit 1; }
stamp=$(date +%H%M%S); cpu=$W/urls-pccpu-$stamp.txt; gpu=$W/urls-pcgpu-$stamp.txt; : > "$cpu"; : > "$gpu"
cat "$LOG" | grep -o 'READY [0-9]*k/[^ ]*\.pte' | cut -d' ' -f2 | sort -u | while read -r wf; do
  grep -qx "$wf" "$DONE" && continue
  sum=$(perl -e 'alarm 120; exec @ARGV' colab ssh -s kvpc -- "sha256sum /content/out/$wf" < /dev/null 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  [ -z "$sum" ] && { echo "no hash for $wf yet" >&2; continue; }
  case "$wf" in *) echo "${wf%%/*} $base/${wf#*/} $sum" >> "$gpu" ;; *) echo "${wf%%/*} $base/${wf#*/} $sum" >> "$cpu" ;; esac
  echo "$wf" >> "$DONE"
done
n=0
for pair in "$cpu study sfetch.sh" "$gpu gpu gfetch.sh"; do
  set -- $pair
  [ -s "$1" ] || { rm -f "$1"; continue; }
  adb push "$1" /data/local/tmp/kvq/$2/ > /dev/null
  perl -e 'alarm 20; exec @ARGV' adb shell "cd /data/local/tmp/kvq/$2 && setsid sh $3 $(basename "$1") > /dev/null 2>&1 < /dev/null &" 2>/dev/null || true
  n=$((n + $(grep -c . "$1")))
done
echo "shipped $n file(s)"
