#!/bin/bash
# Ships files the Colab job has announced (READY lines in work/kvstudy/logs/colab.log, and in colab-int8.log, a
# copy of the VM's /content/int8.log from colab_int8_lane.sh) to the phone: one
# "WINDOW URL SHA256" line per new file (the hash computed on the VM over colab ssh), pushed with study/sfetch.sh detached on the phone. Idempotent: what it shipped is
# listed in work/kvstudy/colab-shipped.txt, so run it whenever new READY lines appear.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); W=$R/work/kvstudy
LOG=$W/logs/colab-gpu.log; DONE=$W/colab-gpu-shipped.txt
touch "$DONE"
# work/kvstudy/tunnel-base.txt overrides the job's first tunnel: the quick tunnel's default QUIC transport ran at
# ~4 KB/s per stream from Colab, a second one with --protocol http2 at 10-23 MB/s.
base=$(grep -o 'TUNNEL https://[^ ]*' "$LOG" | tail -1 | cut -d' ' -f2)
[ -z "$base" ] && { echo "no tunnel yet"; exit 1; }
batch=$W/urls-colab-$(date +%H%M%S).txt
cat "$LOG" "$W/logs/colab-vk16.log" 2>/dev/null | grep -o 'READY [0-9]*k/[^ ]*\.pte' | cut -d' ' -f2 | sort -u | while read -r wf; do
  grep -qx "$wf" "$DONE" && continue
  sum=$(perl -e 'alarm 120; exec @ARGV' colab ssh -s kvgpu -- "sha256sum /content/out/$wf" < /dev/null 2>/dev/null | grep -o '^[0-9a-f]\{64\}')
  [ -z "$sum" ] && { echo "no hash for $wf yet" >&2; continue; }
  echo "${wf%%/*} $base/${wf#*/} $sum"
  echo "$wf" >> "$DONE"
done > "$batch"
n=$(grep -c . "$batch")
[ "$n" -eq 0 ] && { rm -f "$batch"; echo "nothing new"; exit 0; }
adb push "$batch" /data/local/tmp/kvq/gpu/ > /dev/null
perl -e 'alarm 20; exec @ARGV' adb shell "cd /data/local/tmp/kvq/gpu && setsid sh gfetch.sh $(basename "$batch") > /dev/null 2>&1 < /dev/null &" 2>/dev/null || true
echo "shipped $n file(s): $(cut -d' ' -f2 "$batch" | sed 's#.*/##' | tr '\n' ' ')"
