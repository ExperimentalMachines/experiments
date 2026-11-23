"""The AWQ study's long-text round on a Colab CPU machine: every method's file built at an 8k window and measured the
way the KV-cache study measured its files (scripts/awq_books.py: KL to the fp32 model, perplexity, top-1 and needles
on three PG-19 books at 1,900 and 7,900 tokens), beside the publish gate and the app's 141 held-out questions, all on
the same file.

    colab exec -s SESSION --timeout 0 --env HF_TOKEN=... -f scripts/awq_full.py

Reads /content/plan.json: {"setup": bool, "lanes": [[run, ...], ...]}, each run {"name", "model", "revision", "method",
"ref", "work", "out", "published": file or null, "books": data dir, "fresh": bool, "reuse": bool}. Lanes run in
parallel, runs within a lane one after another. With "setup" it first unpacks /content/awq.tgz, /content/books.tgz and
/content/refs.tgz and installs the 1.6.0 toolchain (torch 2.15.0); without it, it uses the session's existing venv.
Prints START, RESULT and FAILED lines.
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

C = Path("/content")
PY = C / "venv-awq" / "bin" / "python"
TREE = C / "execupack-awq"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
lock = threading.Lock()


def say(*parts) -> None:
    with lock:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sh(cmd: str, log: str, env: dict | None = None) -> bool:
    (C / "logs").mkdir(exist_ok=True)
    with open(C / "logs" / f"{log}.log", "w") as f:
        # Before the tree is unpacked (the setup's own tar commands), run from /content.
        rc = subprocess.call(
            ["bash", "-c", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=TREE if TREE.exists() else C,
            env={**os.environ, "PYTHONPATH": str(TREE), **(env or {})},
        )
    if rc != 0:
        tail = (C / "logs" / f"{log}.log").read_text(errors="replace").strip().splitlines()[-3:]
        say("FAILED", log, "rc", rc, "|", " / ".join(t[:240] for t in tail))
    return rc == 0


def run(r: dict) -> None:
    say("START", r["name"])
    cmd = (
        f'{PY} scripts/awq_compare.py arm "{r["model"]}" --revision "{r["revision"]}" --method {r["method"]} '
        f"--ref {r['ref']} --out {r['out']} --work {r['work']} --keep-pte --books {r['books']}"
    )
    if r.get("published"):
        cmd += f" --published-file {r['published']}"
    if r.get("fresh"):
        cmd += " --fresh-needles"
    if r.get("reuse"):
        cmd += " --reuse-codes"
    if not sh(cmd, f"run-{r['name']}", env={"AWQ_WINDOW": "8192"}):
        return
    res = Path(r["out"]) / f"result-{r['method']}.json"
    d = json.loads(res.read_text())
    bk = json.loads((Path(r["out"]) / f"books-{r['method']}.json").read_text())["books"]
    kl = {}
    for b in bk:
        for c in b["checkpoints"]:
            kl.setdefault(c["d"], []).extend(c["kl"])
    say(
        "RESULT",
        r["name"],
        f"gate={'PASS' if d['gate'].get('passed') else 'FAIL'}",
        f"app_kl={d['heldout']['mean_kl']:.4f}",
        " ".join(f"book_kl@{k}={sum(v) / len(v):.4f}" for k, v in sorted(kl.items())),
        f"solve={d['timings_s'].get('solve')}s books={d['timings_s'].get('books')}s",
    )


def main() -> None:
    plan = json.loads((C / "plan.json").read_text())
    if plan.get("setup"):
        for name in ("awq", "books", "refs"):
            sh(f"tar xzf /content/{name}.tgz -C /content", f"untar-{name}")
        say("setup: installing", f"cpus={os.cpu_count()}")
        if not sh(
            f"pip install -q uv && uv venv -q -p 3.11 {C / 'venv-awq'} && "
            f"uv pip install -q --python {PY} torch==2.15.0 --index-url {CPU_INDEX} && "
            f"uv pip install -q --python {PY} -r {TREE}/requirements/export-xnnpack.txt",
            "install",
        ):
            return
        say("setup: done")
    else:
        for name in ("awq", "books", "refs"):
            if (C / f"{name}.tgz").exists():
                sh(f"tar xzf /content/{name}.tgz -C /content", f"untar-{name}")
    threads = [threading.Thread(target=lambda lane=lane: [run(r) for r in lane]) for lane in plan["lanes"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("ALL DONE")


main()
