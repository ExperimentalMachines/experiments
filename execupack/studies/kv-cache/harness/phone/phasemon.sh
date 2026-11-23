#!/system/bin/sh
# Every 2 s: epoch, GPU thermal_pwrlevel, gpuclk (MHz), the newest harness stdout in gpu/out, its line count and
# last line. Lets each checkpoint's decode be matched to the GPU clock it ran at (the harness has no wall times).
O=/data/local/tmp/kvq/gpu/out; G=/sys/class/kgsl/kgsl-3d0
while :; do
  f=$(ls -t $O/*.stdout 2>/dev/null | head -1)
  echo "$(date +%s) $(cat $G/thermal_pwrlevel) $(( $(cat $G/gpuclk) / 1000000 )) $(basename $f) $(wc -l < $f) $(grep -v -E 'regex|Registering' $f | tail -1 | tr ' ' '_')" >> $O/phase.log
  sleep 2
done
