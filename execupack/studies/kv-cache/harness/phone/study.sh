#!/system/bin/sh
# KV study on the phone, detached from adb (launch with setsid) so a dropped tunnel does not stop it.
# usage: [WAIT=0] sh study.sh "2k 8k 32k" "llama lfm qwen gemma"
# WAIT=0 runs only the files already downloaded and skips the rest; relaunch later for those.
# Files: study/pte/<window>/<f32kv|q8kv>-<name>.pte (each with a .ok marker once fully downloaded), study/data/<key>/
# (reference.py's books and appref.py's app.json). One fresh process per run; results in study/out/, progress in
# study/out/progress.log. A run whose result exists is skipped, so the script can be relaunched after a failure.
D=/data/local/tmp/kvq
S=$D/study
cd $D || exit 1
mkdir -p $S/out
BOOKS=10321,10356,10762
# 32k for Llama and Qwen on the first book only (user's call, 2026-11-20): measured on LFM, a 32k run takes ~7.5 min
# per book, and Llama and Qwen are ~2.5x and ~7x that; LFM and Gemma keep all three books at 32k.
books_for() { case "$1/$2" in 32k/llama|32k/qwen) echo 10321 ;; *) echo $BOOKS ;; esac; }
log() { echo "$(date +%H:%M:%S) $*" >> $S/out/progress.log; }
therm() { dumpsys thermalservice 2>/dev/null | sed -n '/Current temperatures from HAL/,/Current cooling devices/p' | grep -E 'mValue' | sed 's/.*mValue=\([0-9.]*\).*mName=\([^,]*\).*/\2=\1/' | grep -E '^(CPU[0-9]|skin|battery)=' | tr '\n' ' '; }
pattern() { case $1 in llama) echo Llama ;; lfm) echo LFM ;; qwen) echo Qwen ;; gemma) echo gemma ;; esac; }

pte_for() {  # window key kv: prints the fully downloaded file's path; with WAIT=1 waits up to 3 h for it
  n=0
  while :; do
    f=$(ls $S/pte/$1/$3-*.pte 2>/dev/null | grep -i "$(pattern $2)" | head -1)
    [ -n "$f" ] && [ -f "$f.ok" ] && { echo "$f"; return 0; }
    [ "${WAIT:-1}" = 0 ] && return 1
    [ $n -ge 360 ] && return 1
    sleep 30; n=$((n + 1))
  done
}

run() {  # mode window key kv
  mode=$1; w=$2; m=$3; kv=$4
  out=$S/out/$mode-$w-$m-$kv.json
  [ -f "$out" ] && return
  pte=$(pte_for $w $m $kv) || { log "not here yet: $w $m $kv"; return; }
  case $w in 2k) win=2048; to=1800 ;; 8k) win=8192; to=3600 ;; *) win=32768; to=10800 ;; esac
  [ $mode = app ] && to=3600
  log "start $mode $w $m $kv $(therm)"
  logcat -c
  if [ $mode = app ]; then args="app $pte $S/data/$m/app.json $out"; else args="stream $pte $S/data/$m $win $out $(books_for $w $m)"; fi
  timeout $to env CLASSPATH=$D/progs.dex:$S/studybench.dex LD_LIBRARY_PATH=$D/lib \
    app_process -Djava.library.path=$D/lib /system/bin StudyBench $args > $S/out/$mode-$w-$m-$kv.stdout 2>&1
  rc=$?
  logcat -d -s ExecuTorch:E > $S/out/$mode-$w-$m-$kv.logcat 2>/dev/null
  log "end $mode $w $m $kv rc=$rc"
  sleep 30
}

i=0
for w in $1; do
  for m in $2; do
    # Alternate which cache goes first from model to model, so drift in clocks or heat lands on both.
    if [ $((i % 2)) -eq 0 ]; then a=f32kv; b=q8kv; else a=q8kv; b=f32kv; fi
    i=$((i + 1))
    run stream $w $m $a; run stream $w $m $b
    if [ $w = 2k ]; then run app $w $m $b; run app $w $m $a; fi
  done
done
log "PASS DONE $1 / $2"
