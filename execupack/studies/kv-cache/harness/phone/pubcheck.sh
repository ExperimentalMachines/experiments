#!/system/bin/sh
# After round 4: do ExecuPack's published Llama-3.2-1B Vulkan files at 16k and 32k load on this phone? They are built
# like the study's 32k file that aborted at load in round 3 (ExecuTorch 1.5.1, 2,048-token prefill chunk; Vulkan sizes
# two attention temporaries heads x chunk x window). Each file is fetched from the Hub (sha256 = its LFS oid), run once
# through the study harness on book 10321 from a cool GPU, and a failure keeps the full logcat.
D=/data/local/tmp/kvq; S=$D/gpu; P=$S/pub; G=/sys/class/kgsl/kgsl-3d0
mkdir -p $P
while ps -A -o ARGS | grep -q "^sh round4c.sh"; do sleep 20; done
log() { echo "$(date +%H:%M:%S) $*" >> $S/out/progress.log; }
tstatus() { dumpsys thermalservice 2>/dev/null | sed -n 's/^Thermal Status: //p' | head -1; }
cool() {
  sleep 30; n=0
  while :; do
    lv=$(cat $G/thermal_pwrlevel 2>/dev/null); st=$(tstatus); t=$(cat $G/temp 2>/dev/null)
    [ "${lv:-99}" -eq 0 ] && [ "${st:-9}" -le 1 ] && [ "${t:-999999}" -le 50000 ] && { log "cool after $((n * 15 + 30)) s: pwrlevel=$lv status=$st gpu_temp=$t"; return 0; }
    [ $n -ge 120 ] && { log "ADMITTED ON TIMEOUT: pwrlevel=$lv status=$st gpu_temp=$t"; return 0; }
    sleep 15; n=$((n + 1))
  done
}
H=https://huggingface.co/experimentalmachines/Llama-3.2-1B-Instruct-ExecuTorch/resolve/main/vulkan
for spec in "32k 32768 1f75dc6134d579b033f6d112c157dfcba5c728a134b92519aa688ce72494a633" \
            "16k 16384 6e982a131e3a1fd4c9828fcb32cce6fb06372dcb5063c1299edae42c7daaae6a"; do
  set -- $spec; w=$1; win=$2; sum=$3
  f=$P/Llama-3.2-1B-Instruct-vulkan-8da4w-gptq-$w.pte; base=$S/out/pub-$w-llama
  [ -f "$base.json" ] && continue
  for i in 1 2 3; do
    [ "$(sha256sum $f 2>/dev/null | cut -d' ' -f1)" = "$sum" ] && break
    curl -sS -f -L -o $f "$H/${f##*/}"
  done
  [ "$(sha256sum $f | cut -d' ' -f1)" = "$sum" ] || { log "pub $w: download failed"; continue; }
  cool
  log "start pub $w llama (published file)"
  ( while :; do echo "$(date +%s) $(cat $G/thermal_pwrlevel) $(cat $G/gpuclk) $(cat $G/gpu_busy_percentage | tr -d ' %')"; sleep 5; done ) > $base.therm 2>/dev/null &
  sp=$!
  logcat -c
  timeout 3600 env CLASSPATH=$D/progs.dex:$S/studybench.dex LD_LIBRARY_PATH=$D/lib-vk \
    app_process -Djava.library.path=$D/lib-vk /system/bin StudyBench stream $f $S/data/llama $win $base.json 10321 > $base.stdout 2>&1
  rc=$?
  kill $sp 2>/dev/null
  logcat -d -s ExecuTorch:E > $base.logcat 2>/dev/null
  [ $rc -ne 0 ] && logcat -d -t 4000 > $base.logcat-full 2>/dev/null
  log "end pub $w llama rc=$rc"
  sleep 10
done
log "PUBCHECK DONE"
