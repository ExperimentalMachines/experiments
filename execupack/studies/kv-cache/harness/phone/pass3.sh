#!/system/bin/sh
# Waits until no study pass and no StudyBench run is active, then runs every remaining file of all four models,
# waiting for downloads (WAIT=1). Runs never overlap, so none shares the CPU with another.
cd /data/local/tmp/kvq/study || exit 1
while pgrep -f "sh study.sh" > /dev/null || pgrep -f "StudyBench" > /dev/null; do sleep 30; done
WAIT=1 sh study.sh "2k 8k 32k" "lfm llama qwen gemma"
