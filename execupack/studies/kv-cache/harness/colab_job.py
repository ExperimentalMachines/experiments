"""KV study builds on a Colab VM, after both Modal workspaces hit their spend limits.

Run in the session's kernel (so the VM counts as busy) with the secrets as kernel env, never in code:

    colab exec -s kv --timeout 0 --env HF_TOKEN=... --env MODAL_TOKEN_ID=... --env MODAL_TOKEN_SECRET=... \
        -f scripts/kvstudy/colab_job.py

Expects /content/kvbuild.tgz (the clean tree that built Llama, LFM and Qwen) and /content/kvstudy.tgz (the tree
with the Gemma 3 port and Gemma's calibration reference). Steps, each printed as one line:
1. the export toolchain in a Python 3.11 venv, as the Modal image has it (torch from the CPU index first);
2. from the execupack-runs Volume (reads still work past the spend limit): Qwen's GPTQ codes and the built files
   the phone does not have yet;
3. a file server behind a Cloudflare quick tunnel, every file under an unguessable token path; prints TUNNEL;
4. two lanes: Qwen's three missing exports (clean tree, Qwen's own codes) | Gemma's GPTQ solve and its 2k and 8k
   exports (Gemma tree); then Gemma's 32k exports once Qwen's lane is done (memory). READY lines name each file
   the phone can fetch;
5. new files back to the Volume (best effort), then serving until /content/serve.done exists.
"""

import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

C = Path("/content")
OUT = C / "out"
VENV = C / "venv"
PY = str(VENV / "bin" / "python")
MODAL = str(VENV / "bin" / "modal")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
VOLUME_FILES = [  # built on Modal, not yet complete on the phone
    "study/2k/q8kv-LFM2.5-1.2B-Instruct-8da4w-gptq-2k.pte",
    "study/8k/f32kv-LFM2.5-1.2B-Instruct-8da4w-gptq-8k.pte",
    "study/2k/q8kv-Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte",
    "study/8k/q8kv-Llama-3.2-1B-Instruct-8da4w-gptq-8k.pte",
    "study/32k/q8kv-Llama-3.2-1B-Instruct-8da4w-gptq-32k.pte",
    "study/2k/f32kv-Qwen3-1.7B-8da4w-gptq-2k.pte",
    "study/32k/f32kv-Qwen3-1.7B-8da4w-gptq-32k.pte",
    "study/32k/q8kv-Qwen3-1.7B-8da4w-gptq-32k.pte",
]
QWEN = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
GEMMA = ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752")
lock = threading.Lock()


def say(*parts) -> None:
    with lock:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def sh(cmd: str, log: str, cwd: Path | None = None, env: dict | None = None) -> bool:
    """Runs one step, its output to logs/<log>.log; True on success."""
    (C / "logs").mkdir(exist_ok=True)
    with open(C / "logs" / f"{log}.log", "w") as f:
        rc = subprocess.call(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, cwd=cwd,
                             env={**os.environ, **(env or {})})
    if rc != 0:
        tail = (C / "logs" / f"{log}.log").read_text(errors="replace").strip().splitlines()[-3:]
        say("FAILED", log, "rc", rc, "|", " / ".join(t[:200] for t in tail))
    return rc == 0


def export(tree: str, model: tuple, codes: str, kv: str, ctx: int, extra_env: dict | None = None) -> None:
    win = f"{ctx // 1024}k"
    tag = f"{model[0].split('/')[1]}-{kv}-{win}"
    ok = sh(
        f'{PY} scripts/kv_export.py "{model[0]}" --revision "{model[1]}" --codes "{codes}" --kv {kv} '
        f"--context {ctx} --work /content/work/{tag} --out /content/res/{tag} --keep {OUT / win}",
        f"export-{tag}", cwd=C / tree, env={"PYTHONPATH": str(C / tree), **(extra_env or {})},
    )
    subprocess.call(["rm", "-rf", f"/content/work/{tag}"])
    if ok:
        res = sorted((C / "res" / tag).glob("export-*.json"))
        import json

        d = json.loads(res[0].read_text()) if res else {}
        smoke = d.get("smoke", {})
        say("READY" if smoke.get("passed") else "SMOKE-FAILED", f"{win}/{d.get('pte')}",
            f"{d.get('bytes', 0) / 1e6:.0f}MB", f"{d.get('seconds')}s", repr((smoke.get("reply") or smoke.get("error") or "")[-50:]))


SERVER = r'''
import http.server, os, sys
TOKEN, ROOT = sys.argv[1], sys.argv[2]
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parts = self.path.split("/")
        if len(parts) != 3 or parts[1] != TOKEN or "/" in parts[2] or parts[2].startswith("."):
            return self.send_error(404)
        path = next((os.path.join(ROOT, w, parts[2]) for w in ("2k", "8k", "32k")
                     if os.path.isfile(os.path.join(ROOT, w, parts[2]))), None)
        if not path:
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


def serve() -> None:
    token = secrets.token_hex(16)
    (C / "server.py").write_text(SERVER)
    subprocess.Popen([PY, str(C / "server.py"), token, str(OUT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh("curl -sSfL -o /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/"
       "cloudflared-linux-amd64 && chmod +x /content/cloudflared", "cloudflared")
    log = open(C / "logs" / "tunnel.log", "w")
    subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8000"],
                     stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        text = (C / "logs" / "tunnel.log").read_text(errors="replace")
        urls = [w for w in text.split() if w.startswith("https://") and w.rstrip("|").endswith("trycloudflare.com")]
        if urls:
            say("TUNNEL", urls[0].rstrip("|") + "/" + token)
            return
    say("FAILED tunnel: no trycloudflare URL")


def main() -> None:
    for w in ("2k", "8k", "32k"):
        (OUT / w).mkdir(parents=True, exist_ok=True)
    for name in ("kvbuild", "kvstudy"):
        sh(f"tar xzf /content/{name}.tgz -C /content", f"untar-{name}")
    say("setup: installing the toolchain")
    if not sh(
        f"pip install -q uv && uv venv -q -p 3.11 {VENV} && "
        f"uv pip install -q --python {PY} torch==2.14.0 --index-url {CPU_INDEX} && "
        f"uv pip install -q --python {PY} -r /content/execupack-kvbuild/requirements/export-xnnpack.txt modal==1.6.1",
        "install",
    ):
        return
    say("setup: done;", subprocess.run([PY, "-c", "import executorch, torch; print(torch.__version__)"],
                                       capture_output=True, text=True).stdout.strip())
    serve()
    (C / "codes").mkdir(exist_ok=True)
    sh(f"{MODAL} volume get --force execupack-runs study/codes/qwen-codes-gptq.pt /content/codes/", "get-qwen-codes")
    for p in VOLUME_FILES:
        w = p.split("/")[1]
        if sh(f"{MODAL} volume get --force execupack-runs {p} {OUT / w}/", "get-" + p.split("/")[-1]):
            say("READY", f"{w}/{p.split('/')[-1]}", "from the Volume")
    qwen_codes = "/content/codes/qwen-codes-gptq.pt"
    gemma_codes = "/content/codes/gemma-codes-gptq.pt"

    def qwen_lane() -> None:
        for kv, ctx in (("int8", 2048), ("fp32", 8192), ("int8", 8192)):
            export("execupack-kvbuild", QWEN, qwen_codes, kv, ctx)

    def gemma_lane() -> None:
        ok = sh(
            f'{PY} scripts/kv_solve.py "{GEMMA[0]}" --revision "{GEMMA[1]}" --ref work/kvstudy/refs/gemma/ref '
            f"--work /content/work/gemma-solve --codes {gemma_codes}",
            "gemma-solve", cwd=C / "execupack-kvstudy", env={"PYTHONPATH": str(C / "execupack-kvstudy")},
        )
        say("gemma solve", "DONE" if ok else "FAILED")
        if ok:
            for ctx in (2048, 8192):
                for kv in ("fp32", "int8"):
                    export("execupack-kvstudy", GEMMA, gemma_codes, kv, ctx)

    a = threading.Thread(target=qwen_lane)
    b = threading.Thread(target=gemma_lane)
    a.start(); b.start(); a.join(); b.join()
    if Path(gemma_codes).exists():
        for kv in ("fp32", "int8"):
            export("execupack-kvstudy", GEMMA, gemma_codes, kv, 32768,
                   extra_env={"KV_STUDY_HOST_BUDGET_BYTES": str(400 * 10**9)})
    say("builds done; saving the new files to the Volume")
    for w in ("2k", "8k", "32k"):
        for f in sorted((OUT / w).glob("*.pte")):
            if "Qwen" in f.name and w != "32k" or "gemma" in f.name.lower():
                sh(f"{MODAL} volume put --force execupack-runs {f} study/{w}/{f.name}", f"put-{f.name}")
    if Path(gemma_codes).exists():
        sh(f"{MODAL} volume put --force execupack-runs {gemma_codes} study/codes/gemma-codes-gptq.pt", "put-gemma-codes")
    say("ALL BUILT; serving until /content/serve.done")
    while not (C / "serve.done").exists():
        time.sleep(60)
        say("serving")
    say("EXIT")


main()
