#!/system/bin/sh
# Stops every study run and download, then deletes every model file that came through a Colab tunnel (two fetches
# wrote some of them at once), keeping the 7 served from Modal before any overlap, and the one result measured on a
# suspect file.
for pat in "sh study.sh" StudyBench sfetch.sh "curl -sS -f"; do pkill -9 -f "$pat"; done
sleep 2
echo "left: $(ps -A -o ARGS | grep -cE '[s]fetch|[c]url -sS|[S]tudyBench|[s]tudy\.sh')"
cd /data/local/tmp/kvq/study || exit 1
rm -f out/stream-2k-lfm-q8kv.*
for f in pte/*/*.pte; do
  case $f in
    pte/2k/f32kv-LFM2.5*|pte/2k/f32kv-Llama-3.2*|pte/32k/f32kv-LFM2.5*|pte/32k/f32kv-Llama-3.2*|pte/32k/q8kv-LFM2.5*|pte/8k/f32kv-Llama-3.2*|pte/8k/q8kv-LFM2.5*) ;;
    *) rm -f "$f" "$f.ok" ;;
  esac
done
ls pte/*/*.pte
