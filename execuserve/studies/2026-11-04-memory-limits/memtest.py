#!/usr/bin/env python3
"""Run a model in ExecuServe's foreground service with an Android 17 style memory cap, and record
what happened. Run with ADB_SERVER_SOCKET set. A finished run goes to results.jsonl; one in progress
is kept in results.jsonl.partial.

    memtest.py <model-id> <cap-mib|none> [--swap-mib N] [--requests N] [--tokens N]

The cap is what the Memory Limiter's config sets for a not-visible process: memory.high = cap and
memory.swap.max = cap, on the app's uid cgroup, before the model loads. Page cache is dropped first
so the model's file pages are charged to the app, as they would be for a real load.
"""
import argparse, json, os, re, secrets, subprocess, sys, time

S = os.environ["SER"]  # the phone's adb serial
PKG = "org.experimentalmachines.execuserve"
SVC = f"{PKG}/{PKG}.app.serve.ServeService"
ACT = f"{PKG}.action"
PROMPT = "Explain how a refrigerator keeps food cold. /no_think"
LONG = "Summarize this log in one sentence.\n" + "\n".join(
    f"Entry {i}: the compressor pushed refrigerant through coil {i % 7}, the fan ran at step {i % 5}, and the door stayed shut."
    for i in range(1, 121))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(["adb", "-s", S, "shell", cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ADB_TIMEOUT"


def pid():
    return sh(f"pidof {PKG}").split(" ")[0]


def cg(uid):
    return f"/sys/fs/cgroup/apps/uid_{uid}"


def rd(uid, name):
    return sh(f"cat {cg(uid)}/{name} 2>/dev/null")


def num(text):
    return int(text) if text.isdigit() else 0


def save(rec):
    rec["saved"] = time.strftime("%H:%M:%S")
    with open(OUT + ".partial", "w") as f:
        f.write(json.dumps(rec))


def kv(text):
    return {p[0]: int(p[1]) for p in (l.split() for l in text.splitlines()) if len(p) == 2 and p[1].isdigit()}


def snapshot(uid, p):
    stat = kv(rd(uid, "memory.stat"))
    rollup = {}
    if p:
        for l in sh(f"cat /proc/{p}/smaps_rollup 2>/dev/null").splitlines():
            m = re.match(r"(\w+):\s+(\d+) kB", l)
            if m:
                rollup[m.group(1)] = int(m.group(2)) // 1024
    return {
        "pid": p,
        "current_mib": num(rd(uid, "memory.current")) // 2**20,
        "peak_mib": num(rd(uid, "memory.peak")) // 2**20,
        "swap_mib": num(rd(uid, "memory.swap.current")) // 2**20,
        "swap_peak_mib": num(rd(uid, "memory.swap.peak")) // 2**20,
        "events": kv(rd(uid, "memory.events")),
        "swap_events": kv(rd(uid, "memory.swap.events")),
        "anon_mib": stat.get("anon", 0) // 2**20,
        "file_mib": stat.get("file", 0) // 2**20,
        "file_mapped_mib": stat.get("file_mapped", 0) // 2**20,
        "pgmajfault": stat.get("pgmajfault", 0),
        "refault_file": stat.get("workingset_refault_file", 0),
        "refault_anon": stat.get("workingset_refault_anon", 0),
        "rollup_mib": {k: rollup.get(k) for k in ("Rss", "Pss", "Pss_Anon", "Pss_File", "Anonymous", "Swap")},
    }


def proc_state():
    out = sh(f"dumpsys activity processes {PKG} | grep -E 'curProcState|procState=|mCurProcState|curAdj|setAdj' | head -4")
    return re.sub(r"\s+", " ", out)[:240]


def api(key, method, path, body=None, timeout=300):
    data = ""
    if body is not None:
        tmp = OUT + ".req"
        with open(tmp, "w") as f:
            json.dump(body, f)
        subprocess.run(["adb", "-s", S, "push", tmp, "/data/local/tmp/es_req.json"], capture_output=True, timeout=60)
        data = "-H 'Content-Type: application/json' -d @/data/local/tmp/es_req.json"
    t = time.time()
    out = sh(f"curl -sS -m {timeout} -X {method} -H 'Authorization: Bearer {key}' {data} -w '\\n%{{http_code}}' http://127.0.0.1:8080{path} 2>&1", timeout + 20)
    head, _, code = out.rpartition("\n")
    return code.strip(), head, round(time.time() - t, 1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("model")
    a.add_argument("cap")
    a.add_argument("--swap-mib", type=int)
    a.add_argument("--requests", type=int, default=2)
    a.add_argument("--tokens", type=int, default=64)
    a.add_argument("--timeout", type=int, default=180)
    o = a.parse_args()

    uid = re.search(r"uid:(\d+)", sh(f"pm list packages -U {PKG}")).group(1)
    key = "es-memtest-" + secrets.token_hex(20)
    rec = {"model": o.model, "cap_mib": o.cap, "started": time.strftime("%H:%M:%S"), "uid": uid}

    # A clean process and clean file cache for every run. Only this app is stopped.
    sh(f"am force-stop {PKG}")
    time.sleep(1)
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"am start-foreground-service -n {SVC} -a {ACT}.KEEP_ALIVE")
    for _ in range(20):
        if pid() and sh(f"[ -f {cg(uid)}/memory.high ] && echo y") == "y":
            break
        time.sleep(0.5)
    p0 = pid()
    if o.cap != "none":
        swap = o.swap_mib if o.swap_mib is not None else int(o.cap)
        sh(f"echo {int(o.cap) * 2**20} > {cg(uid)}/memory.high; echo {swap * 2**20} > {cg(uid)}/memory.swap.max")
    else:
        sh(f"echo max > {cg(uid)}/memory.high; echo max > {cg(uid)}/memory.swap.max")
    rec["limits"] = {"high": rd(uid, "memory.high"), "swap_max": rd(uid, "memory.swap.max")}
    rec["before"] = snapshot(uid, p0)

    # Start serving without loading a model: the load below is what gets timed.
    t_start = time.time()
    sh(f"am start-foreground-service -n {SVC} -a {ACT}.START --es model {o.model} --ei port 8080 --es bind loopback --es key {key} --ei threads -1")
    for _ in range(60):
        if sh("curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health") == "200":
            break
        time.sleep(1)
    code, body, secs = api(key, "POST", f"/v1/execuserve/models/{o.model}/load", timeout=o.timeout)
    rec["load"] = {"http": code, "seconds": round(time.time() - t_start, 1), "body": body[:200]}
    rec["after_load"] = snapshot(uid, pid())
    rec["proc_state"] = proc_state()
    save(rec)

    rec["requests"] = []
    for i in range(o.requests + 1):
        content = LONG if i == o.requests else PROMPT
        code, body, secs = api(key, "POST", "/v1/chat/completions", {
            "model": o.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": o.tokens, "temperature": 0, "stream": False,
        }, timeout=o.timeout)
        r = {"http": code, "seconds": secs}
        try:
            j = json.loads(body)
            r["usage"] = j.get("usage")
            r["finish"] = j["choices"][0]["finish_reason"]
            r["text"] = j["choices"][0]["message"]["content"][:80]
        except Exception:
            r["body"] = body[:300]
        r["after"] = snapshot(uid, pid())
        rec["requests"].append(r)
        save(rec)
        print(f"  req{i}: http={code} {secs}s pid={r['after']['pid']}", flush=True)
        if code != "200":
            break

    code, body, _ = api(key, "GET", "/v1/execuserve/runs?format=csv", timeout=30)
    rec["runs_csv"] = body[-3000:]
    rec["final_pid"] = pid()
    rec["same_process"] = rec["final_pid"] == p0
    rec["exit_info"] = sh(f"dumpsys activity exit-info {PKG} | grep -E 'timestamp=|reason=|description=|pss=|rss=' | head -8")
    rec["finished"] = time.strftime("%H:%M:%S")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if os.path.exists(OUT + ".partial"):
        os.remove(OUT + ".partial")
    al = rec["after_load"]
    print(json.dumps({"model": o.model, "cap": o.cap, "load": rec["load"]["seconds"], "same_process": rec["same_process"],
                      "anon": al["anon_mib"], "file": al["file_mib"], "current": al["current_mib"], "swap": al["swap_mib"],
                      "high_events": al["events"].get("high"), "rollup": al["rollup_mib"]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
