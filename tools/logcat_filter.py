"""Keeps the lines of a device logcat that belong to the app under test, and nothing else.

A cloud lab's full logcat carries every process on the phone: test accounts, network names and
addresses, other apps' tokens. The measurement is the app's own lines. This keeps, in order:

  * every line of a process the app started (`Start proc <pid>:<package>`) or that logged under
    one of the app's own tags;
  * every line, from any process, that names the app's package (installs, kills, crashes);
  * every line of a crash dump whose header names the package (`>>> <package> <<<`);
  * the buffer markers (`--------- beginning of ...`).

The redactions that follow it in the import config are a second net for what an app line can
still carry, such as an address in an error message.
"""

import re

PACKAGE = "io.github.alpharomercoma.openweights"
LINE = re.compile(r"^\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+(\d+)\s+(\d+)\s+[VDIWEFA]\s+(.*?)\s*: ")
APP_TAGS = re.compile(r"^(TestRunner|MonitoringInstr|ExecuTorch|LlamaCpp\w*|OpenWeights\w*|ow\w*|kv-log|.*\.engine\.test)$")
START = re.compile(r"Start proc (\d+):" + re.escape(PACKAGE))
CRASH = re.compile(r"pid: (\d+), tid: \d+, name: .*>>> " + re.escape(PACKAGE))


def keep_app_lines(text: str) -> str:
    lines = text.split("\n")
    pids = set()
    for line in lines:
        m = LINE.match(line)
        if m and APP_TAGS.match(m.group(3)):
            pids.add(m.group(1))
        s = START.search(line)
        if s:
            pids.add(s.group(1))
    dumpers = set()
    for line in lines:
        m = LINE.match(line)
        if m and CRASH.search(line):
            dumpers.add(m.group(1))
    out = []
    for line in lines:
        m = LINE.match(line)
        if line.startswith("--------- beginning of") or PACKAGE in line:
            out.append(line)
        elif m and (m.group(1) in pids or m.group(1) in dumpers):
            out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


# The second net, for what an app line can still carry.
REDACT = [
    (re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"), "<mac>"),
    (re.compile(r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|169\.254)"
                r"\.\d{1,3}\.\d{1,3}\b"), "<lan-ip>"),
    (re.compile(r"\bfe80:(?::?[0-9A-Fa-f]{1,4}){1,7}"), "<ipv6-ll>"),
    (re.compile(r"(?i)(\bB?SSID\b\W{0,3})\"[^\"\n]*\""), r'\1"<ssid>"'),
]


def filter_xz_logcat(path: str, data: bytes, *hooks) -> bytes:
    """An xz-compressed logcat, cut to the app's lines, redacted, and compressed again."""
    import lzma
    text = keep_app_lines(lzma.decompress(data).decode("utf-8", "surrogateescape"))
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    for hook in hooks:
        text = hook(text)
    return lzma.compress(text.encode("utf-8", "surrogateescape"), preset=9)


def main() -> None:
    """`python3 tools/logcat_filter.py <full.logcat[.xz]> > <run>/app.logcat`: the same cut and
    redactions the imports applied, for a log you are about to commit. Serials still need their
    pseudonyms (see docs/adding-a-run.md)."""
    import lzma
    import sys
    if len(sys.argv) != 2:
        raise SystemExit(main.__doc__)
    data = open(sys.argv[1], "rb").read()
    if sys.argv[1].endswith(".xz"):
        data = lzma.decompress(data)
    text = keep_app_lines(data.decode("utf-8", "surrogateescape"))
    for pattern, replacement in REDACT:
        text = pattern.sub(replacement, text)
    sys.stdout.buffer.write(text.encode("utf-8", "surrogateescape"))


if __name__ == "__main__":
    main()
