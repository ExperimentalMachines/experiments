"""The AWQ study's last runs, on a Colab CPU machine (Modal's credit ran out): Qwen3-1.7B with every method calibrated
thinking-off, and the same codes built with ExecuTorch 1.5.1 for the phone check.

    colab exec -s awq --timeout 0 --env HF_TOKEN=... -f scripts/awq_colab.py

Inputs uploaded beside this script: /content/awq.tgz (this tree), /content/kvbuild-awq.tgz (the 1.5.1 build tree with
export_with_codes' fp32 overrides from b4177ee, and kv_export accepting no codes for the RTN file), and
/content/ref-nothink.tgz (work/awq/qwen3-1.7b/ref-nothink: thinking-off calibration rows, with the gate and held-out
reference of the original run).

Lane 1 solves one method at a time with this tree (ExecuTorch 1.6.0, torch 2.15.0), as on Modal: awq_compare.py arm
with --keep-pte, so each method is measured exactly as the earlier arms were. Lane 2 builds each method's codes with
the 1.5.1 tree for the phone (the apps' runtime; 1.6.0 has no Android library) and measures that file on x86 too, so
phone and x86 can be compared on the same file. Prints RESULT, BUILT and READY lines; serves the phone files over an
http2 tunnel until /content/serve.done exists.
"""

import json
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

C = Path("/content")
OUT = C / "out" / "2k"
MODEL = "Qwen/Qwen3-1.7B"
REV = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
VENV = {"awq": C / "venv-awq", "151": C / "venv-151"}
TREE = {"awq": C / "execupack-awq", "151": C / "execupack-kvbuild"}
METHODS = ("gptq", "awq", "awq_gptq")
lock = threading.Lock()


def say(*parts) -> None:
    with lock:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sh(cmd: str, log: str, cwd: Path | None = None, env: dict | None = None) -> bool:
    (C / "logs").mkdir(exist_ok=True)
    with open(C / "logs" / f"{log}.log", "w") as f:
        rc = subprocess.call(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env={**os.environ, **(env or {})})
    if rc != 0:
        tail = (C / "logs" / f"{log}.log").read_text(errors="replace").strip().splitlines()[-3:]
        say("FAILED", log, "rc", rc, "|", " / ".join(t[:220] for t in tail))
    return rc == 0


def install(key: str, torch_version: str) -> bool:
    py = VENV[key] / "bin" / "python"
    return sh(
        f"pip install -q uv && uv venv -q -p 3.11 {VENV[key]} && "
        f"uv pip install -q --python {py} torch=={torch_version} --index-url {CPU_INDEX} && "
        f"uv pip install -q --python {py} -r {TREE[key]}/requirements/export-xnnpack.txt",
        f"install-{key}",
    )


def arm(method: str) -> dict | None:
    """One method, as the earlier arms ran: solve, export, gate and held-out evaluation, the .pte kept."""
    tree, py = TREE["awq"], VENV["awq"] / "bin" / "python"
    ok = sh(
        f'{py} scripts/awq_compare.py arm "{MODEL}" --revision "{REV}" --method {method} --ref {C}/ref-nothink '
        f"--out {C}/res/{method} --work {C}/work/{method} --keep-pte",
        f"arm-{method}", cwd=tree, env={"PYTHONPATH": str(tree)},
    )
    res = C / "res" / method / f"result-{method}.json"
    if not ok or not res.exists():
        return None
    r = json.loads(res.read_text())
    g = r["gate"]
    say("RESULT", method, f"solve={r['timings_s'].get('solve')}s",
        f"gate={'PASS' if g.get('passed') else 'FAIL'} {g.get('agreed')}/{g.get('graded')} {g.get('export_mean')}",
        f"heldout_kl={r['heldout']['mean_kl']:.4f} top1={r['heldout']['top1_agree']:.3f}",
        f"reply_kl={r['replies']['mean_kl']:.4f}")
    return r


def phone_build(method: str) -> None:
    """The method's codes (none for RTN) built with ExecuTorch 1.5.1, measured on x86, and kept for the phone."""
    tree, py = TREE["151"], VENV["151"] / "bin" / "python"
    codes = "" if method == "rtn" else str(C / "work" / method / f"codes-{method}.pt")
    keep = C / "keep151" / method
    ok = sh(
        f'{py} scripts/kv_export.py "{MODEL}" --revision "{REV}" --codes "{codes}" --kv fp32 --context 2048 '
        f"--work {C}/w151/{method} --out {C}/r151/{method} --keep {keep} --ref {C}/ref-nothink",
        f"build151-{method}", cwd=tree, env={"PYTHONPATH": f"{tree}:{tree}/scripts"},
    )
    ptes = sorted(keep.glob("*.pte")) if keep.exists() else []
    summary = C / "r151" / method / "export-f32kv-2048.json"
    if not ok or not ptes:
        say("NO-FILE", method)
        return
    name = f"{method}-Qwen3-1.7B-8da4w-2k.pte"
    shutil.copyfile(ptes[0], OUT / name)
    s = json.loads(summary.read_text()) if summary.exists() else {}
    say("BUILT", method, f"heldout_kl={s.get('heldout', {}).get('mean_kl')}", f"reply_kl={s.get('replies', {}).get('mean_kl')}",
        f"gate={(s.get('gate') or {}).get('passed')}")
    say("READY", f"2k/{name}")


def serve() -> None:
    server = r'''
import http.server, os, sys
TOKEN, ROOT = sys.argv[1], sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parts = self.path.split("/")
        if len(parts) != 3 or parts[1] != TOKEN or "/" in parts[2] or parts[2].startswith("."):
            return self.send_error(404)
        path = os.path.join(ROOT, parts[2])
        if not os.path.isfile(path):
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()
        with open(path, "rb") as f:
            while chunk := f.read(4 << 20):
                self.wfile.write(chunk)
    def log_message(self, *a):
        pass
http.server.ThreadingHTTPServer(("127.0.0.1", 8000), H).serve_forever()
'''
    token = secrets.token_hex(16)
    (C / "server.py").write_text(server)
    subprocess.Popen(["python3", str(C / "server.py"), token, str(OUT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh("curl -sSfL -o /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/"
       "cloudflared-linux-amd64 && chmod +x /content/cloudflared", "cloudflared")
    log = open(C / "logs" / "tunnel.log", "w")
    # --protocol http2: the default QUIC transport ran at ~4 KB/s per stream out of Colab (2026-11-20).
    subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:8000"],
                     stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        words = (C / "logs" / "tunnel.log").read_text(errors="replace").split()
        urls = [w.rstrip("|") for w in words if w.startswith("https://") and w.rstrip("|").endswith("trycloudflare.com")]
        if urls:
            say("TUNNEL", urls[0] + "/" + token)
            return
    say("FAILED tunnel")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (C / "logs").mkdir(exist_ok=True)
    for name in ("awq", "kvbuild-awq", "ref-nothink"):
        sh(f"tar xzf /content/{name}.tgz -C /content", f"untar-{name}")
    say("setup: installing", f"cpus={os.cpu_count()}")
    t = [threading.Thread(target=lambda: install("awq", "2.15.0")), threading.Thread(target=lambda: install("151", "2.14.0"))]
    for x in t:
        x.start()
    for x in t:
        x.join()
    if not all((VENV[k] / "bin" / "python").exists() for k in VENV):
        say("FAILED install")
        return
    serve()
    say("setup: done")
    # Lane 2 starts with RTN (no solve) while lane 1 solves; each solved method is handed to lane 2 when done.
    queue: list[str] = ["rtn"]
    done = threading.Event()

    def lane2() -> None:
        while True:
            with lock:
                job = queue.pop(0) if queue else None
            if job is None:
                if done.is_set():
                    return
                time.sleep(20)
                continue
            phone_build(job)

    t2 = threading.Thread(target=lane2)
    t2.start()
    arm("rtn")  # the RTN arm again, on this machine: a control for Modal against Colab numerics
    for method in METHODS:
        if arm(method):
            with lock:
                queue.append(method)
    done.set()
    t2.join()
    say("ALL DONE; serving until /content/serve.done")
    while not (C / "serve.done").exists():
        time.sleep(60)


main()
