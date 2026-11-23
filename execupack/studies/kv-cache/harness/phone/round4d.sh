#!/system/bin/sh
# Round 4, last part (replaces round4c.sh after Qwen's app questions): the Llama 32k pair repeated in reverse order,
# each run admitted at <= 40 C (35 C was not reached in 16 minutes idle), then the published-file check. Qwen 2k's
# stream replication is dropped: the third phone's fp16 app run already matched the second phone's on all 141 rows.
D=/data/local/tmp/kvq
cd $D/gpu && REP=rep FIRST=f16kv COOLTEMP=40000 COOLMAX=30 sh gpu2.sh "32k" "llama"
cd $D/gpu && sh pubcheck.sh
echo "$(date +%H:%M:%S) ROUND4 DONE" >> $D/gpu/out/progress.log
