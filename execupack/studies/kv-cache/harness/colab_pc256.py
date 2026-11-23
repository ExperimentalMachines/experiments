"""KV study, GPU pass, Llama 3.2 1B at 32k on Vulkan with a 256-token prefill chunk (fp32 and fp16 cache).

The 2,048-token-chunk 32k files abort at load on the phone: the Vulkan delegate sizes its attention buffers heads x
prefill chunk x window up front (the 8k fp32 file already peaks at 5.6 GiB of GPU memory, 2k at 2.4 GiB), about 16 GiB
at 32k. Same codes and recipe as every other GPU file; only export.prefill_chunk differs (KV_STUDY_PREFILL_CHUNK).
Files are renamed with a -pc256 suffix. Prints TUNNEL once and READY 32k/<file> per file; serves until serve.done.

    colab exec -s kvpc --timeout 0 --env HF_TOKEN=... --env MODAL_TOKEN_ID=... --env MODAL_TOKEN_SECRET=... -f colab_pc256.py
"""

import json
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path

C = Path("/content")
OUT = C / "out"
PY = str(C / "venv" / "bin" / "python")
MODAL = str(C / "venv" / "bin" / "modal")
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
MODELS = {
    "llama": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", "execupack-kvbuild"),
    "qwen": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "execupack-kvbuild"),
    "gemma": ("google/gemma-3-1b-it", "dcc83ea841ab6100d6b47a070329e1ba4cf78752", "execupack-kvstudy"),
}
TAG = {"fp32": "f32kv", "int8": "q8kv", "fp16": "f16kv"}
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


def export(key: str, kv: str, ctx: int, env: dict | None = None) -> None:
    model, rev, tree = MODELS[key]
    win = f"{ctx // 1024}k"
    tag = f"vk-{key}-{TAG[kv]}-{win}"
    ok = sh(
        f'{PY} scripts/kv_export.py "{model}" --revision "{rev}" --codes /content/codes/{key}-codes-gptq.pt '
        f"--kv {kv} --backend vulkan --context {ctx} --work /content/work/{tag} --out /content/res/{tag} --keep {OUT / win}",
        f"export-{tag}", cwd=C / tree, env={"PYTHONPATH": str(C / tree), **(env or {})},
    )
    subprocess.call(["rm", "-rf", f"/content/work/{tag}"])
    res = sorted((C / "res" / tag).glob("export-*.json"))
    d = json.loads(res[0].read_text()) if res else {}
    if ok and d.get("pte"):
        smoke = d.get("smoke", {})
        state = "BUILT" if smoke.get("passed") else "CHECK-FAILED"
        say(state, f"{win}/{d['pte']}", f"{d.get('bytes', 0) / 1e6:.0f}MB", f"{d.get('seconds')}s",
            repr((smoke.get("problems") or smoke.get("error") or "")[:160]))
    else:
        say("NO-FILE", tag, repr(d.get("smoke", {}).get("error", "")[-300:]))


def serve() -> None:
    server = r'''
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
    token = secrets.token_hex(16)
    (C / "server.py").write_text(server)
    subprocess.Popen([PY, str(C / "server.py"), token, str(OUT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh("curl -sSfL -o /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/"
       "cloudflared-linux-amd64 && chmod +x /content/cloudflared", "cloudflared")
    log = open(C / "logs" / "tunnel.log", "w")
    # --protocol http2: the default QUIC transport ran at ~4 KB/s per stream out of Colab (2026-11-20).
    subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--protocol", "http2", "--url",
                      "http://127.0.0.1:8000"], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        words = (C / "logs" / "tunnel.log").read_text(errors="replace").split()
        urls = [w.rstrip("|") for w in words if w.startswith("https://") and w.rstrip("|").endswith("trycloudflare.com")]
        if urls:
            say("TUNNEL", urls[0] + "/" + token)
            return
    say("FAILED tunnel")


def main() -> None:
    (OUT / "32k").mkdir(parents=True, exist_ok=True)
    for name in ("kvbuild", "kvstudy"):
        sh(f"tar xzf /content/{name}.tgz -C /content", f"untar-{name}")
    say("setup: installing")
    if not sh(f"pip install -q uv && uv venv -q -p 3.11 {C / 'venv'} && "
              f"uv pip install -q --python {PY} torch==2.14.0 --index-url {CPU_INDEX} && "
              f"uv pip install -q --python {PY} -r /content/execupack-kvbuild/requirements/export-xnnpack.txt modal==1.6.1",
              "install"):
        return
    serve()
    (C / "codes").mkdir(exist_ok=True)
    sh(f"{MODAL} volume get --force execupack-runs study/codes/llama-codes-gptq.pt /content/codes/", "get-llama-codes")
    say("setup: done")
    env = {"KV_STUDY_PREFILL_CHUNK": "256", "KV_STUDY_HOST_BUDGET_BYTES": "400000000000"}
    for kv in ("fp16", "fp32"):  # one at a time: a 32k export beside another was OOM-killed on this 52 GB VM
        before = set((OUT / "32k").glob("*.pte"))
        export("llama", kv, 32768, env)
        for p in set((OUT / "32k").glob("*.pte")) - before:
            q = p.with_name(p.stem + "-pc256.pte")
            p.rename(q)
            say("READY", f"32k/{q.name}", "(renamed)")
    say("ALL BUILT; serving until /content/serve.done")
    while not (C / "serve.done").exists():
        time.sleep(30)


main()
