"""One line per finished study export: file, size, export time, smoke result (work/kvstudy/res/*/export-*.json)."""
import json
import sys
from pathlib import Path

for f in sorted(Path(sys.argv[1]).glob("*-*k-*/export-*.json")):
    d = json.loads(f.read_text())
    s = d["smoke"]
    tail = (s.get("reply") or s.get("error") or "")[-40:]
    print(f"{d.get('pte', '?'):58} {d.get('bytes', 0) / 1e6:6.0f} MB {d['seconds']:5}s smoke={s.get('passed')} {tail!r}")
