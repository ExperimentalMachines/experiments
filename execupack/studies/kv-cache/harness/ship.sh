#!/bin/bash
# usage: ship.sh BATCH_NAME: serves every finished study export not shipped yet (smoke passed, per work/kvstudy/res)
# from a short-lived Modal sandbox and starts the phone's detached download of them (study/sfetch.sh).
# Prints the sandbox id; stop it with serve.py --stop ID once the phone's progress log says "fetched".
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd); W=$R/work/kvstudy
PY=${PY:-/private/tmp/claude-501/-Users-alpha-execupack/269b336d-a21c-4e52-8bb8-87f1d7e4a1c0/scratchpad/kvq/venv/bin/python}
touch "$W/shipped.txt"
paths=$(python3 - "$W" <<'PY'
import json, sys
from pathlib import Path
w = Path(sys.argv[1]); shipped = set((w / "shipped.txt").read_text().split())
for f in sorted(w.glob("res/*-*k-*/export-*.json")):
    d = json.loads(f.read_text())
    if d["smoke"].get("passed") and d.get("pte"):
        win = f.parent.name.split("-")[1]
        p = f"study/{win}/{d['pte']}"
        if p not in shipped:
            print(p)
PY
)
[ -z "$paths" ] && { echo "nothing new to ship"; exit 0; }
out=$("$PY" "$R/scripts/kvstudy/phone/serve.py" $paths)
echo "$out" | grep sandbox
echo "$out" | grep https | while read -r url; do
  name=${url##*/}; win=$(echo "$name" | sed -E 's/.*-([0-9]+k)\.pte$/\1/'); echo "$win $url"
done > "$W/urls-$1.txt"
adb push "$W/urls-$1.txt" /data/local/tmp/kvq/study/ > /dev/null
perl -e 'alarm 20; exec @ARGV' adb shell "cd /data/local/tmp/kvq/study && setsid sh sfetch.sh urls-$1.txt > /dev/null 2>&1 < /dev/null &" || true
echo "$paths" >> "$W/shipped.txt"
wc -l < "$W/urls-$1.txt"
