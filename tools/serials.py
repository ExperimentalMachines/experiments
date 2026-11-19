"""Device serials and cloud project ids, found in a source at its pinned commit and replaced by
stable pseudonyms.

A run record names the unit it ran on (`"unit": "<serial>"`, `unit <serial>` in a chain log, an
`adb devices` listing) because results are compared per unit. The serial itself is an identifier,
so each one becomes `unit-` and six hex digits of its SHA-256: the same unit keeps the same name
in every file and every rerun, and no serial is written into this repository, not even here.
A Google Cloud project id or number (`projects/<id>/...`, `project=<number>`) becomes
`gcp-project-` and six hex digits the same way, and a cloud login address in a run's notes or
logs becomes `account-` and six hex digits.
"""

import hashlib
import re
import subprocess
from pathlib import Path

# A serial is 7 to 16 letters and digits with at least one digit; the contexts keep words out.
CONTEXTS = [
    r'"unit"\s*:\s*"([A-Za-z0-9]{7,16})"',
    r"\bunit[= ]([A-Za-z0-9]{7,16})\b",
    r"(?m)^([A-Za-z0-9]{7,16})\s+device\b",
]
SERIAL = re.compile(r"^(?=.*\d)[A-Za-z0-9]{7,16}$")


PROJECT_CONTEXTS = [r"projects/([a-z][a-z0-9-]{4,28}[a-z0-9])/", r"projects/(\d{6,})/", r"[?&]project=(\d{6,})\b"]
PROJECT = re.compile(r"^(?:[a-z][a-z0-9-]{4,28}[a-z0-9]|\d{6,})$")
# An address in a run's prose or logs (not in a dataset row). Library names like
# `android.hardware...@1.0.so` and documentation placeholders are not accounts.
ACCOUNT_CONTEXTS = [r"(?<![\w.@\\])([A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[a-z]{2,})\b"]
ACCOUNT = re.compile(r"^(?!.*\.so$)(?!.*@example\.(?:com|org)$)(?!.*qualcomm\.com$)[^@\s]+@[^@\s]+$")


def grep(checkout: Path, commit: str, paths: list[str], context: str) -> list[str]:
    run = subprocess.run(["git", "-C", str(checkout), "grep", "-h", "-o", "-P", context, commit, "--", *paths],
                         capture_output=True, text=True)
    # git grep exits 1 when nothing matches; anything else is a failed search, and a failed
    # search must not pass for a clean one, or nothing would be replaced.
    if run.returncode not in (0, 1):
        raise SystemExit(f"git grep failed in {checkout}: {run.stderr.strip()}")
    return run.stdout.splitlines()


def discover(checkout: Path, commit: str, paths: list[str], contexts=None, valid=None) -> set[str]:
    found = set()
    for context in contexts or CONTEXTS:
        for line in grep(checkout, commit, paths + [":!*.json", ":!*.jsonl"] if contexts is ACCOUNT_CONTEXTS
                         else paths, context):
            m = re.search(context, line.split(":", 2)[-1] if line.startswith(commit) else line)
            if m and (valid or SERIAL).match(m.group(1)):
                found.add(m.group(1))
    return found


def discover_projects(checkout: Path, commit: str, paths: list[str]) -> set[str]:
    return discover(checkout, commit, paths, PROJECT_CONTEXTS, PROJECT)


def discover_accounts(checkout: Path, commit: str, paths: list[str]) -> set[str]:
    return discover(checkout, commit, paths, ACCOUNT_CONTEXTS, ACCOUNT)


def discover_files(paths) -> set[str]:
    """The same search over files on disk (work that was never committed)."""
    found = set()
    for path in paths:
        text = Path(path).read_text(errors="ignore")
        for context in CONTEXTS:
            found |= {m for m in re.findall(context, text) if SERIAL.match(m)}
    return found


def pseudonym(serial: str) -> str:
    return "unit-" + hashlib.sha256(serial.encode()).hexdigest()[:6]


def replacer(serials: set[str], name=None, token=r"A-Za-z0-9"):
    """A function that replaces every listed value, as a whole token, by its pseudonym."""
    if not serials:
        return lambda text: text
    name = name or pseudonym
    pattern = re.compile(rf"(?<![{token}])(" + "|".join(map(re.escape, sorted(serials, key=len, reverse=True)))
                         + rf")(?![{token}])")
    return lambda text: pattern.sub(lambda m: name(m.group(1)), text)


def account_replacer(accounts: set[str]):
    return replacer(accounts, lambda a: "account-" + hashlib.sha256(a.encode()).hexdigest()[:6], r"A-Za-z0-9._%+@-")


def project_replacer(projects: set[str]):
    return replacer(projects, lambda p: "gcp-project-" + hashlib.sha256(p.encode()).hexdigest()[:6], r"A-Za-z0-9-")
