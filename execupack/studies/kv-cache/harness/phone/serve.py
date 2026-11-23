"""Serve chosen files from the execupack-runs Volume to the phone over HTTPS, for one hour at most.

Only the listed volume paths are served, each under an unguessable token; anything else is a 404 and
there is no directory listing. Prints one URL per file, then exits; the sandbox keeps serving until it
is terminated (python serve.py --stop SANDBOX_ID) or its timeout.

usage: serve.py VOLUME_PATH...    |    serve.py --stop SANDBOX_ID
"""

import secrets
import sys

import modal

if sys.argv[1] == "--stop":
    modal.Sandbox.from_id(sys.argv[2]).terminate()
    print("terminated", sys.argv[2])
    sys.exit(0)

paths = sys.argv[1:]
token = secrets.token_hex(16)
files = {f"/{token}/{p.rsplit('/', 1)[-1]}": f"/vol/{p}" for p in paths}
server = f"""
import http.server, os, shutil
FILES = {files!r}
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = FILES.get(self.path)
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 4 << 20)
    def log_message(self, *a):
        pass
http.server.ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
"""
app = modal.App.lookup("execupack-phone-serve", create_if_missing=True)
sb = modal.Sandbox.create(
    "python", "-c", server,
    app=app,
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/vol": modal.Volume.from_name("execupack-runs")},
    encrypted_ports=[8000],
    timeout=3600,
    cpu=2,
)
url = sb.tunnels()[8000].url
print("sandbox", sb.object_id)
for k in files:
    print(url + k)
