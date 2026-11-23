"""The AWQ study's long-text round as a work queue on one Colab CPU machine, steerable without a shell on it (the Colab
CLI's ssh reaches only one of several sessions; upload and download reach each).

    colab upload -s S queue.json /content/queue.json
    colab exec -s S --timeout 0 --env HF_TOKEN=... -f scripts/awq_queue.py

/content/queue.json: {"lanes": N, "runs": [run, ...]}, each run as scripts/awq_full.py's (name, model, revision, method,
ref, work, out, published, books, fresh, reuse). N lanes run in parallel; a free lane re-reads the queue and takes the
first run it has not started, so uploading a new queue.json changes what runs next. Names listed in /content/kill.json
are stopped (checked every 30 s) and never started. Prints START, RESULT, FAILED and, when the queue is empty, ALL DONE.
"""

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

C = Path("/content")
PY = C / "venv-awq" / "bin" / "python"
TREE = C / "execupack-awq"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
lock = threading.Lock()
started: set[str] = set()
procs: dict[str, subprocess.Popen] = {}


def say(*parts) -> None:
    with lock:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def read(name: str, default):
    try:
        return json.loads((C / name).read_text())
    except (OSError, ValueError):
        return default


def sh(cmd: str, log: str, env: dict | None = None, name: str | None = None) -> bool:
    (C / "logs").mkdir(exist_ok=True)
    with open(C / "logs" / f"{log}.log", "w") as f:
        p = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=TREE if TREE.exists() else C,
            env={**os.environ, "PYTHONPATH": str(TREE), **(env or {})},
            start_new_session=True,
        )
        if name:
            procs[name] = p
        rc = p.wait()
        procs.pop(name, None)
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
    if not sh(cmd, f"run-{r['name']}", env={"AWQ_WINDOW": "8192"}, name=r["name"]):
        return
    d = json.loads((Path(r["out"]) / f"result-{r['method']}.json").read_text())
    bk = json.loads((Path(r["out"]) / f"books-{r['method']}.json").read_text())["books"]
    kl: dict = {}
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


def lane(i: int) -> None:
    while True:
        with lock:
            kill = set(read("kill.json", []))
            todo = [
                r for r in read("queue.json", {}).get("runs", []) if r["name"] not in started and r["name"] not in kill
            ]
            if not todo:
                return
            started.add(todo[0]["name"])
        try:
            run(todo[0])
        except Exception as e:  # a broken result file must not end the lane
            say("FAILED", todo[0]["name"], repr(e)[:240])


def killer() -> None:
    while True:
        time.sleep(30)
        for name in set(read("kill.json", [])) & set(procs):
            p = procs.get(name)
            if p and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                say("KILLED", name)


def main() -> None:
    if not PY.exists():
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
    else:  # a machine already set up: refresh the tree, add the long-text and reference data if missing
        for name in ("awq", "books", "refs"):
            if (C / f"{name}.tgz").exists() and (name == "awq" or not (C / name).exists()):
                sh(f"tar xzf /content/{name}.tgz -C /content", f"untar-{name}")
    threading.Thread(target=killer, daemon=True).start()
    threads = [threading.Thread(target=lane, args=(i,)) for i in range(int(read("queue.json", {}).get("lanes", 2)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("ALL DONE")


main()
