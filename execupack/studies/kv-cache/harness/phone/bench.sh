#!/system/bin/sh
# Runs on the phone, detached from adb (setsid), so a dropped tunnel does not stop it.
# Every .pte twice, fp32 cache and int8 cache alternated (A B, then B A) so drift in clocks or heat hits both;
# the 2k GPTQ files first, then the 8k RTN files. One fresh process per run. Progress in out/progress.log.
D=/data/local/tmp/kvq
cd $D || exit 1
mkdir -p out
log() { echo "$(date +%H:%M:%S) $*" >> out/progress.log; }
therm() { dumpsys thermalservice 2>/dev/null | sed -n '/Current temperatures from HAL/,/Current cooling devices/p' | grep -E 'mValue' | sed 's/.*mValue=\([0-9.]*\).*mName=\([^,]*\).*/\2=\1/' | tr '\n' ' '; }

run() {  # window model kv rep
  w=$1; m=$2; kv=$3; rep=$4
  pte=$(ls pte/$kv-*-$w.pte 2>/dev/null | grep -i "$(echo $m | sed 's/lfm/LFM/;s/llama/Llama/;s/qwen/Qwen/')" | head -1)
  out=out/$w-$m-$kv-$rep.json
  [ -f "$out" ] && return
  [ -z "$pte" ] && { log "MISSING $w $m $kv"; return; }
  seq=2048; [ "$w" = 8k ] && seq=8192
  log "start $w $m $kv rep$rep $(therm)"
  logcat -c
  CLASSPATH=$D/progs.dex:$D/kvbench.dex LD_LIBRARY_PATH=$D/lib app_process -Djava.library.path=$D/lib /system/bin \
    KvBench "$pte" "tok/$m.json" "prompts/$m/phases-$w.txt" $seq "$out" > "out/$w-$m-$kv-$rep.stdout" 2>&1
  rc=$?
  logcat -d -s ExecuTorch:* > "out/$w-$m-$kv-$rep.logcat" 2>/dev/null
  log "end $w $m $kv rep$rep rc=$rc"
  sleep 20
}

for w in 2k 8k; do
  # the 8k files arrive in a second download; wait for it (at most 2 h)
  if [ "$w" = 8k ]; then n=0; while [ ! -f pte/8k.ready ] && [ $n -lt 240 ]; do sleep 30; n=$((n+1)); done; fi
  for m in llama lfm qwen; do
    run $w $m f32kv 1; run $w $m q8kv 1
    run $w $m q8kv 2; run $w $m f32kv 2
  done
done
log "ALL DONE"
