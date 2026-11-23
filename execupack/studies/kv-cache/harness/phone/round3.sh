#!/system/bin/sh
# Third round on the second QDC phone: the CPU supplement (2k and 8k pairs with per-position log-probs; LFM2 needles
# from a fresh model), then the GPU pass (Llama and Qwen at 2k and 8k, Llama at 32k). One run at a time.
D=/data/local/tmp/kvq
cd $D/study && WAIT=1 sh cpu2.sh "2k 8k" "lfm llama qwen gemma"
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "2k 8k" "llama qwen"
cd $D/gpu && WAIT=1 MAXWAIT=120 sh gpu.sh "32k" "llama"
echo "$(date +%H:%M:%S) ROUND3 DONE" >> $D/gpu/out/progress.log
