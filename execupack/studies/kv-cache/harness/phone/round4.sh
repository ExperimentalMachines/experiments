#!/system/bin/sh
# Round 4 (third QDC phone, 2026-11-21): the GPU runs the second phone never finished, each from gpu.sh's thermal gate.
# Llama 32k first (its 256-token-chunk files are on the phone first), then Qwen 8k, then Qwen's 2k app questions
# (both caches, on one phone), then Qwen 2k streams again as a third-phone replication if the session lasts.
D=/data/local/tmp/kvq
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "32k" "llama"
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "8k" "qwen"
cd $D/gpu && WAIT=1 MAXWAIT=120 SKIPSTREAM=1 sh gpu.sh "2k" "qwen"
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "2k" "qwen"
echo "$(date +%H:%M:%S) ROUND4 DONE" >> $D/gpu/out/progress.log
