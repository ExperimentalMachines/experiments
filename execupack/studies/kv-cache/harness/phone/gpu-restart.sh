#!/system/bin/sh
# Stops the GPU pass, removes results of runs that failed, and starts it again (skips runs with a result).
for p in "sh gpu.sh" StudyBench; do pkill -9 -f "$p"; done
cd /data/local/tmp/kvq/gpu || exit 1
for f in out/*.stdout; do b=${f%.stdout}; [ -f "$b.json" ] || rm -f "$b.stdout" "$b.logcat"; done
WAIT=1 MAXWAIT=60 setsid sh gpu.sh "2k 8k 32k" "llama qwen" > out/launch.log 2>&1 < /dev/null &
