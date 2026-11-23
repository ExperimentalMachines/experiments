"""Prompt files and phase lists for KvBench, cut from the same text and token ranges as kvq/bench.py.

2k files (GPTQ, get_max_seq_len == get_max_context_len == 2048): a short chat (100 tokens, 128 new), then a fresh
context with one 1,800-token prompt and 64 new, so decoding runs at about 1.9k positions.
8k files (RTN, same kernels): the Mac benchmark exactly: 100 + 128, then four 1,900-token turns with 64 new each.

usage: mkprompts.py TEXT OUTDIR MODEL=TOKENIZER...
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

text, out = Path(sys.argv[1]), Path(sys.argv[2])
dev = "/data/local/tmp/kvq/prompts"
for spec in sys.argv[3:]:
    name, tok_path = spec.split("=", 1)
    t = Tokenizer.from_file(tok_path)
    ids = t.encode(text.read_text(encoding="utf-8"), add_special_tokens=False).ids
    while len(ids) < 100 + 4 * 1900:
        ids = ids + ids
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    pieces = {"short": ids[:100], "deep": ids[100:1900]}
    pieces.update({f"fill{i}": ids[100 + i * 1900 : 100 + (i + 1) * 1900] for i in range(4)})
    for k, v in pieces.items():
        (d / f"{k}.txt").write_text(t.decode(v), encoding="utf-8")
    rows = {
        "2k": [("short", 128, 0), ("deep", 64, 1)],
        "8k": [("short", 128, 0)] + [(f"fill{i}", 64, 0) for i in range(4)],
    }
    for w, phases in rows.items():
        (d / f"phases-{w}.txt").write_text(
            "".join(f"{p}\t{n}\t{dev}/{name}/{p}.txt\t{r}\n" for p, n, r in phases), encoding="utf-8"
        )
    print(name, {k: len(v) for k, v in pieces.items()})
