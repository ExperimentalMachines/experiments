"""Long-text measurements for the AWQ study, the KV-cache study's on x86: scripts/kvstudy/phone/StudyBench.java's
stream mode ported to ExecuTorch's Python runtime (the runner awq_compare.evaluate uses).

For each PG-19 book (the KV study's reference.py files: bookN.stream.i32, bookN.ref.bin, bookN.json), feed the stream
in chunks of the program's prefill length, and at each checkpoint the window holds (1,900 and 7,900 tokens for an 8k
file) score 128 teacher-forced next-token predictions against the fp32 Hugging Face model: KL over its top 64 tokens
plus one bucket, the true next token's negative log-likelihood (perplexity), and whether the file's likeliest token is
the reference's. Then every needle question of the window, answered greedily: after rewinding to the stream's end
minus 4 tokens, or, for a model with convolution state (LFM2, whose state a rewind does not restore), from a freshly
loaded file fed the whole prefix.
"""

from __future__ import annotations

import json
from pathlib import Path

TOP = 64
LENGTH = {2048: 1900, 8192: 7900, 32768: 32500}
REF_ROW = [("t", "<i4"), ("tid", "<i4"), ("tlp", "<f4"), ("ids", "<i4", (TOP,)), ("lp", "<f4", (TOP,))]


def _kl(ids, lp, lq) -> float:
    import numpy as np

    p = np.exp(lp.astype(np.float64))
    q = lq[ids]
    kl = float((p * (lp - q)).sum())
    pt, qt = max(1 - p.sum(), 1e-12), max(1 - float(np.exp(q).sum()), 1e-12)
    return kl + pt * (np.log(pt) - np.log(qt))


def books(runner_factory, data: Path, window: int, book_ids: list[str], fresh: bool, prefill: int) -> list[dict]:
    """``runner_factory()`` returns a fresh runner with forward(tokens [1, n], input_pos [1]) -> logits."""
    import numpy as np
    import torch

    length = LENGTH[window]

    def feed(runner, ids, start: int):
        logits = None
        for at in range(0, len(ids), prefill):
            chunk = ids[at : at + prefill]
            logits = runner.forward(torch.tensor([list(map(int, chunk))]), torch.tensor([start + at]))
        last = logits[0, -1] if logits.dim() == 3 else logits[-1]
        return torch.log_softmax(last.double(), -1).numpy()

    out = []
    for book in book_ids:
        meta = json.loads((data / f"book{book}.json").read_text())
        s = np.fromfile(data / f"book{book}.stream.i32", dtype="<i4")
        ref = np.fromfile(data / f"book{book}.ref.bin", dtype=np.dtype(REF_ROW))
        scored = meta["scored"]
        runner = runner_factory()
        pos, row, cps = 0, 0, []
        for d in meta["checkpoints"]:
            if d > length:
                break
            while row < len(ref) and ref[row]["t"] < d - scored:
                row += 1
            lq = feed(runner, s[pos : d - scored], pos) if pos < d - scored else None
            kl, nll, top1, arg = [], [], [], []
            for k in range(scored):
                if k > 0:  # decode: feed target t-1, predict t
                    lq = feed(runner, s[d - scored + k - 1 : d - scored + k], d - scored + k - 1)
                r = ref[row + k]
                kl.append(round(_kl(r["ids"], r["lp"], lq), 7))
                nll.append(round(float(-lq[r["tid"]]), 6))
                a = int(lq.argmax())
                arg.append(a)
                top1.append(int(a == int(r["ids"][0])))
            cps.append({"d": d, "kl": kl, "nll": nll, "top1": top1, "argmax": arg})
            pos = d - 1
        needles = []
        for q in meta["questions"][str(window)]:
            ids = [int(x) for x in s[length - 4 : length]] + list(q["tail"])
            if fresh:  # a new file, the whole prefix, then the question: nothing carried over
                del runner
                runner = runner_factory()
                feed(runner, s[: length - 4], 0)
            lq = feed(runner, ids, length - 4)
            gen = []
            for g in range(meta["answer_tokens"]):
                gen.append(int(lq.argmax()))
                if g + 1 < meta["answer_tokens"]:
                    lq = feed(runner, [gen[-1]], length - 4 + len(ids) + g)
            needles.append({"key": q["key"], "ids": gen})
        del runner
        out.append({"book": book, "checkpoints": cps, "needles": needles})
    return out
