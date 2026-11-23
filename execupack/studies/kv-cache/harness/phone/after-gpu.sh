#!/system/bin/sh
# Starts the CPU supplement once the GPU pass has finished (its PASS DONE line), so runs never overlap.
cd /data/local/tmp/kvq/study || exit 1
until grep -q "PASS DONE" /data/local/tmp/kvq/gpu/out/progress.log 2>/dev/null; do sleep 60; done
while pgrep -f StudyBench > /dev/null; do sleep 30; done
WAIT=0 sh cpu2.sh "2k 8k" "lfm llama qwen gemma"
