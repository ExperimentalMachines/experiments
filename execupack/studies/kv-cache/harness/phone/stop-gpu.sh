#!/system/bin/sh
# Stops the round chains, the phase monitor and the GPU pass (script file, so pkill patterns do not match this shell's own command line).
for p in $(ps -A -o PID,ARGS | grep -E "sh (round[0-9][a-z]*|gpu[0-9]*|phasemon)\.sh" | grep -v grep | awk '{print $1}'); do kill -9 $p; done
for p in $(ps -A -o PID,ARGS | grep "StudyBench" | grep -v grep | awk '{print $1}'); do kill -9 $p; done
sleep 2; ps -A -o PID,ARGS | grep -E "round[0-9]|phasemon|gpu\.sh|StudyBench" | grep -v grep
echo stopped
