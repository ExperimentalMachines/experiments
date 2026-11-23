#!/system/bin/sh
# Round 4, continued (replaces round4.sh's queue after its Llama 32k pair; gpu.sh is pushed as gpu2.sh so the copy
# still running is never edited under it). The 32k fp16 run started at a GPU temperature of 48 C against fp32's 30 C,
# and a 4.5-minute 32k run throttles, so the 32k pair is repeated in reverse order (fp16 first), both from <= 35 C.
D=/data/local/tmp/kvq
while ps -A -o ARGS | grep -q "^sh gpu.sh"; do sleep 10; done
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu2.sh "8k" "qwen"
cd $D/gpu && WAIT=1 MAXWAIT=120 SKIPSTREAM=1 sh gpu2.sh "2k" "qwen"
cd $D/gpu && REP=rep FIRST=f16kv COOLTEMP=35000 COOLMAX=40 sh gpu2.sh "32k" "llama"
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu2.sh "2k" "qwen"
echo "$(date +%H:%M:%S) ROUND4 DONE" >> $D/gpu/out/progress.log
