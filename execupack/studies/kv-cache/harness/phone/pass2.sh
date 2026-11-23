#!/system/bin/sh
# Waits for any running study.sh pass to finish, then runs every remaining file of all four models, waiting for
# downloads (WAIT=1). Two passes never overlap, so no two runs share the CPU.
cd /data/local/tmp/kvq/study || exit 1
while pgrep -f "sh study.sh" > /dev/null; do sleep 30; done
WAIT=1 sh study.sh "2k 8k 32k" "lfm llama qwen gemma"
