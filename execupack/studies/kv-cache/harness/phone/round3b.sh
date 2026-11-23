#!/system/bin/sh
# Round 3, GPU pass again with gpu.sh's thermal gate (the first attempt ran throttled; see out/throttled).
D=/data/local/tmp/kvq
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "2k 8k" "llama qwen"
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "32k" "llama"
echo "$(date +%H:%M:%S) ROUND3 DONE" >> $D/gpu/out/progress.log
