#!/system/bin/sh
# KV study, GPU (Vulkan) pass, on the phone, through org.pytorch:executorch-android-vulkan:1.5.1's library (lib-vk):
# the plain executorch-android AAR has no Vulkan backend, detached from adb (launch with setsid) so a dropped tunnel does not stop it.
# usage: [WAIT=0] sh study.sh "2k 8k 32k" "llama lfm qwen gemma"
# WAIT=0 runs only the files already downloaded and skips the rest; relaunch later for those.
# Files: study/pte/<window>/<f32kv|q8kv>-<name>.pte (each with a .ok marker once fully downloaded), study/data/<key>/
# (reference.py's books and appref.py's app.json). One fresh process per run; results in study/out/, progress in
# study/out/progress.log. A run whose result exists is skipped, so the script can be relaunched after a failure.
D=/data/local/tmp/kvq
S=$D/gpu
cd $D || exit 1
mkdir -p $S/out
BOOKS=10321,10356,10762
# 32k for Llama and Qwen on the first book only (user's call, 2026-11-20): measured on LFM, a 32k run takes ~7.5 min
# per book, and Llama and Qwen are ~2.5x and ~7x that; LFM and Gemma keep all three books at 32k.
# GPU pass (270-minute reservation): every model on the first book at 32k.
books_for() { case "$1" in 32k) echo 10321 ;; *) echo $BOOKS ;; esac; }
log() { echo "$(date +%H:%M:%S) $*" >> $S/out/progress.log; }
therm() { dumpsys thermalservice 2>/dev/null | sed -n '/Current temperatures from HAL/,/Current cooling devices/p' | grep -E 'mValue' | sed 's/.*mValue=\([0-9.]*\).*mName=\([^,]*\).*/\2=\1/' | grep -E '^(CPU[0-9]|skin|battery)=' | tr '\n' ' '; }
G=/sys/class/kgsl/kgsl-3d0
# Thermal gate (added after the second phone's first GPU runs ran at thermal_pwrlevel 12, 636 MHz, thermal status 3,
# right after 90 min of CPU runs: Llama decode 16.7 tok/s against 47 on the first phone). Before each run, wait until
# the GPU's thermal cap is lifted (thermal_pwrlevel 0) and the device thermal status is at most 1, up to COOLMAX
# minutes; the run's clock is sampled every 5 s into <run>.therm and its worst cap logged at the end.
tstatus() { dumpsys thermalservice 2>/dev/null | sed -n 's/^Thermal Status: //p' | head -1; }
cool() {  # also waits for the GPU to cool to COOLTEMP (milli-degC): a book started at level 0 but still hot throttled
  # within seconds (Qwen 2k fp16, third book, level 8); idle on this rack the GPU settles near 48 C
  sleep 30; n=0
  while :; do
    lv=$(cat $G/thermal_pwrlevel 2>/dev/null); st=$(tstatus); t=$(cat $G/temp 2>/dev/null)
    # A missing reading counts as hot (Codex review 3: defaults of 0 let an unreadable sensor pass the gate).
    [ "${lv:-99}" -eq 0 ] && [ "${st:-9}" -le 1 ] && [ "${t:-999999}" -le ${COOLTEMP:-50000} ] && { log "cool after $((n * 15 + 30)) s: pwrlevel=$lv status=$st gpu_temp=$t"; return 0; }
    [ $n -ge $(( ${COOLMAX:-30} * 4 )) ] && { log "ADMITTED ON TIMEOUT, still hot after $((n * 15 + 30)) s: pwrlevel=$lv status=$st gpu_temp=$t"; return 0; }
    sleep 15; n=$((n + 1))
  done
}
pattern() { case $1 in llama) echo Llama ;; lfm) echo LFM ;; qwen) echo Qwen ;; gemma) echo gemma ;; esac; }
# A Vulkan run that cannot load still costs a slot: give up on a file after one failed run (no retry loop).

pte_for() {  # window key kv: prints the fully downloaded file's path; with WAIT=1 waits up to 3 h for it
  n=0
  while :; do
    f=$(ls $S/pte/$1/$3-*.pte 2>/dev/null | grep -i "$(pattern $2)" | head -1)
    [ -n "$f" ] && [ -f "$f.ok" ] && { echo "$f"; return 0; }
    [ "${WAIT:-1}" = 0 ] && return 1
    [ $n -ge ${MAXWAIT:-360} ] && return 1
    sleep 30; n=$((n + 1))
  done
}

run() {  # mode window key kv
  mode=$1; w=$2; m=$3; kv=$4
  [ -f "$S/out/$mode-$w-$m-$kv.json" ] && return
  [ $mode = stream ] && [ "${SKIPSTREAM:-0}" = 1 ] && return  # round 4: app questions only
  pte=$(pte_for $w $m $kv) || { log "not here yet: $w $m $kv"; return; }
  case $w in 2k) win=2048; to=1800 ;; 8k) win=8192; to=3600 ;; *) win=32768; to=10800 ;; esac
  if [ $mode = app ]; then
    one app "$S/out/app-$w-$m-$kv" 3600 "app $pte $S/data/$m/app.json $S/out/app-$w-$m-$kv.json"
  else
    # One process per book, each from a cool GPU (a three-book Qwen run throttled from 1,450 to 636 MHz within ~20 s
    # and its third book decoded at 21.7 tok/s against 30). The harness reloads the model for every book anyway.
    # Results: stream-<w>-<m>-<kv>.b<book>.json, merged on the host.
    for bk in $(books_for $w $m | tr "," " "); do  # not "b": the main loop holds the second cache in b
      base=$S/out/stream-$w-$m-$kv.b$bk${REP:+.$REP}  # REP: a repeat run, kept apart (merge_books.py skips it)
      [ -f "$base.json" ] && continue
      cool
      one "stream $w $m $kv book $bk" "$base" $to "stream $pte $S/data/$m $win $base.json $bk"
    done
  fi
}

one() {  # label base timeout harness-args: one harness process, its GPU clock sampled every 5 s into <base>.therm
  log "start $1 $(therm)"
  ( while :; do echo "$(date +%s) $(cat $G/thermal_pwrlevel) $(cat $G/gpuclk) $(cat $G/gpu_busy_percentage | tr -d ' %')"; sleep 5; done ) > $2.therm 2>/dev/null &
  sp=$!
  logcat -c
  timeout $3 env CLASSPATH=$D/progs.dex:$S/studybench.dex LD_LIBRARY_PATH=$D/lib-vk \
    app_process -Djava.library.path=$D/lib-vk /system/bin StudyBench $4 > $2.stdout 2>&1
  rc=$?
  kill $sp 2>/dev/null
  logcat -d -s ExecuTorch:E > $2.logcat 2>/dev/null
  # A failed run keeps the whole log (round 3's 32k abort, rc 134, left only an empty ExecuTorch:E filter).
  [ $rc -ne 0 ] && logcat -d -t 4000 > $2.logcat-full 2>/dev/null
  log "end $1 rc=$rc worst_pwrlevel=$(awk 'BEGIN{m=0} $2>m{m=$2} END{print m}' $2.therm) min_clk=$(awk 'NR==1||$3<m{m=$3} END{print m}' $2.therm)"
  sleep 10
}

i=0
for w in $1; do
  for m in $2; do
    # Alternate which cache goes first from model to model, so drift in clocks or heat lands on both.
    # On Vulkan the int8 cache does not export in ExecuTorch 1.5.1 (data-dependent guard in the Vulkan passes), so
    # the GPU pair is fp32 against fp16 (Vulkan force_fp16, which keeps the KV cache in fp16 on the GPU).
    if [ $((i % 2)) -eq 0 ]; then a=f32kv; b=f16kv; else a=f16kv; b=f32kv; fi
    [ "${FIRST:-}" = f16kv ] && { a=f16kv; b=f32kv; }  # round 4's 32k repeat runs the pair in reverse order
    i=$((i + 1))
    run stream $w $m $a; run stream $w $m $b
    if [ $w = 2k ]; then run app $w $m $b; run app $w $m $a; fi
  done
done
log "PASS DONE $1 / $2"
