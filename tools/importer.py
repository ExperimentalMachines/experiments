"""Copies one project's experiment material into this repository, from pinned sources.

Each project has a small config module beside this one (import_openweights.py and so on).
A config lists one or more sources. A source is either a git checkout read at a pinned
commit, or a plain folder (gitignored results, an archive) read as it is on disk. For each
it names which paths come across, where they land, and the few exact edits a moved script
needs to find its files again. Running a config:

  * reads every file of a git source from the commit itself (`git cat-file`), never from
    the working tree, so an uncommitted edit cannot leak in and a rerun gives the same bytes;
  * copies the working-tree files a source names explicitly (untracked work, gitignored
    results), and nothing else from the working tree;
  * rewrites moved paths in prose and code (never in data files), applies the edits, each of
    which must match exactly as often as the config says or the run stops, then applies the
    source's redactions to every text file;
  * writes `<project>/PROVENANCE.tsv`: for every file, the source, the source path and
    blob, the blob written here, and whether the bytes were changed on the way.

It only writes the paths it imports, plus PROVENANCE.tsv, so the hand-written READMEs that
sit beside the imported folders survive a rerun. `--check` compares instead of writing.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parent.parent

# Rewriting a path inside a data file would change a measurement record, so moves are only
# rewritten in prose and code. Extensionless files count as code when they start with a shebang.
REWRITE_SUFFIXES = {".md", ".sh", ".py", ".cpp", ".h", ".c", ".kt", ".kts", ".toml", ".yml", ".yaml"}


@dataclass
class Edit:
    old: str
    new: str
    count: int = 1
    regex: bool = False


@dataclass
class Source:
    label: str                     # how PROVENANCE names it, e.g. "execupack@executorch-1.6"
    checkout: Path                 # a git checkout, or a plain folder when commit is None
    commit: str | None             # pinned commit (full sha); None reads the folder as it is
    roots: list[str]               # paths (files or folders) under the checkout that come across
    moves: list[tuple[str, str]]   # (source prefix, destination prefix), most specific first
    worktree: list[str] = field(default_factory=list)      # working-tree files or folders to copy too
    exclude: dict[str, str] = field(default_factory=dict)  # prefix or glob -> reason
    edits: dict[str, list[Edit]] = field(default_factory=dict)  # source path -> edits
    redact: list[tuple[str, str, str]] = field(default_factory=list)  # (path glob, regex, replacement)
    max_bytes: int | None = None   # plain folders only: larger files are listed as excluded
    suffixes: set[str] | None = None  # plain folders only: the file types that come across
    patches: dict[str, list[str]] = field(default_factory=dict)  # destination -> git args whose output is the file
    # "all": moved paths are rewritten in prose and code; "prose": in Markdown only, so a run
    # script stays exactly as it ran; "none": the files are records and stay as they were.
    rewrite: str = "all"
    url: str | None = None         # the source's GitHub address, for pinned links to what stayed
    # Text that names another project's path (`openweights docs/research/...` inside ExecuPack)
    # and must not be read as one of this source's moves.
    protect: list[str] = field(default_factory=list)
    # Records whose words are evidence (review transcripts): no path or link is rewritten in
    # them, only the redactions apply.
    verbatim: list[str] = field(default_factory=lambda: ["*/codex/*"])
    # (path glob, function of the bytes) applied before anything else: a full device logcat
    # is cut down to the app's own lines here.
    transforms: list = field(default_factory=list)
    # Functions of the text applied to every text file last, such as the serial pseudonyms.
    hooks: list = field(default_factory=list)


# Compiled Python is rebuilt by Python and ignored by git, so it never comes across.
ALWAYS_EXCLUDE = {"*/__pycache__/*": "compiled Python", "*.pyc": "compiled Python"}


@dataclass
class Project:
    name: str                      # folder in this repository
    sources: list[Source]


def git(src: Source, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", str(src.checkout), *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def destination(src: Source, path: str) -> str:
    for old, new in src.moves:
        if under(path, old):
            return new + path[len(old):]
    raise SystemExit(f"{src.label}: no move covers {path}")


def rewrite_pattern(prefix: str) -> re.Pattern:
    # A path token: not preceded by a path character (so `$ROOT/tools/eval` and
    # `foo-tools/eval` are left to the explicit edits), and ended by a separator.
    return re.compile(r"(?<![A-Za-z0-9_./$-])" + re.escape(prefix) + r"(?=[/\s\"'`)\]:,;*}>|#]|$)")


def is_text(path: str, data: bytes) -> bool:
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def rewritable(path: str, data: bytes) -> bool:
    suffix = Path(path).suffix
    if suffix in REWRITE_SUFFIXES:
        return is_text(path, data)
    return suffix == "" and data.startswith(b"#!")


def excluded(src: Source, path: str) -> str | None:
    for pattern, reason in {**ALWAYS_EXCLUDE, **src.exclude}.items():
        if under(path, pattern) or fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
            return reason
    return None


def git_files(src: Source) -> dict[str, tuple[str, str]]:
    listing = git(src, "ls-tree", "-r", "-z", "--full-tree", src.commit, "--", *src.roots)
    files = {}
    for entry in filter(None, listing.split("\0")):
        meta, path = entry.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob":
            raise SystemExit(f"{src.label}: {path} is a {kind}, not a file")
        files[path] = (mode, sha)
    return files


def walk(base: Path, items: list[str], label: str) -> list[str]:
    found = []
    for item in items:
        full = base / item
        if full.is_dir():
            for root, dirs, names in os.walk(full):
                dirs.sort()
                for name in sorted(names):
                    found.append(str(Path(root, name).relative_to(base)))
        elif full.is_file():
            found.append(item)
        else:
            raise SystemExit(f"{label}: {item} does not exist under {base}")
    return found


def last_touch(src: Source) -> dict[str, str]:
    """The last commit that changed each path, as `sha date`."""
    log = git(src, "log", "--format=@%H %ad", "--date=short", "--name-only", src.commit, "--", *src.roots)
    seen, current = {}, None
    for line in log.splitlines():
        if line.startswith("@"):
            current = line[1:9] + line[41:]
        elif line and line not in seen:
            seen[line] = current
    return seen


MD_LINK = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")


def relink(src: Source, path: str, dest: str, text: str, imported: dict[str, str], kinds: dict) -> str:
    """Re-points the relative links of a Markdown file that moved.

    A link to something that moved with it gets the new relative path. A link to something
    that stayed in the source becomes a link pinned to the source commit. A link that was
    already broken in the source is left as it was.
    """
    def kind(target: str) -> str | None:
        if target not in kinds:
            out = subprocess.run(["git", "-C", str(src.checkout), "cat-file", "-t", f"{src.commit}:{target}"],
                                 capture_output=True, text=True)
            kinds[target] = out.stdout.strip() if out.returncode == 0 else None
        return kinds[target]

    def fix(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return m.group(0)
        raw, hashmark, frag = target.partition("#")
        if not raw:
            return m.group(0)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(raw)))
        if resolved.startswith(".."):
            return m.group(0)
        tail = hashmark + frag
        if resolved in imported or any(p.startswith(resolved + "/") for p in imported):
            new = imported.get(resolved) or destination(src, resolved)
            rel = posixpath.relpath(new, posixpath.dirname(dest))
            return m.group(1) + rel + ("/" if raw.endswith("/") else "") + tail + m.group(3)
        if src.commit and src.url and kind(resolved) in ("blob", "tree"):
            return f"{m.group(1)}{src.url}/{kind(resolved)}/{src.commit}/{resolved}{tail}{m.group(3)}"
        return m.group(0)

    return MD_LINK.sub(fix, text)


def transform(src: Source, path: str, dest: str, data: bytes, moves: list[tuple[str, str]],
              imported: dict[str, str], kinds: dict) -> bytes:
    for glob, fn in src.transforms:
        if fnmatch.fnmatch(path, glob):
            data = fn(path, data)
    edits = src.edits.get(path, [])
    if not is_text(path, data):
        if edits:
            raise SystemExit(f"{src.label}: edits listed for {path}, which is not text")
        return data
    text = data.decode("utf-8")
    for edit in edits:
        found = len(re.findall(edit.old, text)) if edit.regex else text.count(edit.old)
        if found != edit.count:
            raise SystemExit(f"{src.label}: {path}: expected {edit.count} of {edit.old!r}, found {found}")
        text = re.sub(edit.old, edit.new, text) if edit.regex else text.replace(edit.old, edit.new)
    is_md = Path(path).suffix == ".md"
    record = any(fnmatch.fnmatch(path, g) for g in src.verbatim)
    if src.rewrite != "none" and is_md and not record:
        text = relink(src, path, dest, text, imported, kinds)
    if not record and ((src.rewrite == "all" and rewritable(path, data)) or (src.rewrite == "prose" and is_md)):
        saved = []

        def keep(m: re.Match) -> str:
            saved.append(m.group(0))
            return f"\0{len(saved) - 1}\0"

        for pattern in src.protect:
            text = re.sub(pattern, keep, text)
        for old, new in moves:
            text = rewrite_pattern(old).sub(new, text)
        text = re.sub(r"\0(\d+)\0", lambda m: saved[int(m.group(1))], text)
    for glob, pattern, replacement in src.redact:
        if fnmatch.fnmatch(path, glob):
            text = re.sub(pattern, replacement, text)
    for hook in src.hooks:
        text = hook(text)
    return text.encode("utf-8")


def listing(src: Source):
    """Every candidate file as (path, mode, loader, source_blob, state, last_changed, size)."""
    if src.commit:
        tracked = git_files(src)
        touched = last_touch(src)
        for path in sorted(tracked):
            mode, sha = tracked[path]
            yield (path, mode, lambda sha=sha: git(src, "cat-file", "blob", sha, binary=True),
                   sha, src.commit[:12], touched.get(path, "-"), None)
        extra = walk(src.checkout, src.worktree, src.label)
        for path in extra:
            if path in tracked:
                raise SystemExit(f"{src.label}: {path} is listed as working-tree but is in {src.commit[:8]}")
    else:
        extra = walk(src.checkout, src.roots + src.worktree, src.label)
    for path in extra:
        full = src.checkout / path
        if src.commit is None and src.suffixes is not None and not wanted_type(src, path):
            yield path, None, None, None, "skip", "not a text record of the run (file type)", full.stat().st_size
            continue
        mode = "100755" if os.access(full, os.X_OK) else "100644"
        yield path, mode, full.read_bytes, None, "worktree", "-", full.stat().st_size


def wanted_type(src: Source, path: str) -> bool:
    name = Path(path).name
    return Path(path).suffix in src.suffixes or name in src.suffixes or any(
        "*" in pattern and fnmatch.fnmatch(name, pattern) for pattern in src.suffixes)


def emit(dest: str, data: bytes, mode: str, check: bool) -> bool:
    """Writes one file, or with check, says whether what is here already matches."""
    target = REPO / dest
    if check:
        same = target.is_file() and target.read_bytes() == data
        if not same:
            print(f"differs: {dest}")
        elif bool(target.stat().st_mode & 0o100) != (mode == "100755"):
            print(f"mode differs: {dest}")
            same = False
        return same
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    target.chmod(0o755 if mode == "100755" else 0o644)
    return True


def run(project: Project, check: bool) -> int:
    rows, skipped, problems, written = [], [], 0, {}
    moves = [m for s in project.sources if s.rewrite != "none" for m in s.moves]
    plan = []
    for src in project.sources:
        imported = {}
        for path, mode, load, sha, state, touched, size in listing(src):
            reason = touched if state == "skip" else excluded(src, path)
            if not reason and src.max_bytes and size is not None and size > src.max_bytes:
                reason = f"larger than {src.max_bytes // 1_000_000} MB"
            if reason:
                skipped.append((src.label, path, reason))
                continue
            dest = destination(src, path)
            if dest in written:
                raise SystemExit(f"{dest} would be written by both {written[dest]} and {src.label}")
            written[dest] = src.label
            imported[path] = dest
            plan.append((src, path, dest, mode, load, sha, state, touched))
        unused = set(src.edits) - set(imported)
        if unused:
            raise SystemExit(f"{src.label}: edits for paths that were not imported: {sorted(unused)}")
        src._imported = imported
    kinds = {}
    for src, path, dest, mode, load, sha, state, touched in plan:
        data = load()
        sha = sha or blob_id(data)
        out = transform(src, path, dest, data, moves, src._imported, kinds.setdefault(src.label, {}))
        problems += not emit(dest, out, mode, check)
        rows.append((dest, src.label, path, state, touched, sha, blob_id(out),
                     "changed" if out != data else "verbatim"))
    for src in project.sources:
        for dest, args in src.patches.items():
            data = git(src, *args, binary=True)
            if not data.strip():
                raise SystemExit(f"{src.label}: `git {' '.join(args)}` gave nothing for {dest}")
            # A patch is text the hooks have to see too: its headers name the commits' authors.
            if src.hooks and is_text(dest, data):
                text = data.decode("utf-8")
                for hook in src.hooks:
                    text = hook(text)
                data = text.encode("utf-8")
            if dest in written:
                raise SystemExit(f"{dest} would be written twice")
            written[dest] = src.label
            problems += not emit(dest, data, "100644", check)
            rows.append((dest, src.label, "git " + " ".join(args), "generated", "-", "-", blob_id(data),
                         "changed" if src.hooks else "verbatim"))
    lines = [f"# {project.name}: every file here, where it came from, and whether its bytes changed on the way.",
             "# source_state is the commit the bytes were read from, `worktree` for a file read from disk,",
             "# or `generated` for a patch written by the git command in source_path.",
             "\t".join(("path", "source", "source_path", "source_state", "last_changed",
                        "source_blob", "blob", "bytes"))]
    lines += ["\t".join(r) for r in sorted(rows)]
    lines += [f"# excluded\t{label}\t{p}\t{r}" for label, p, r in skipped]
    provenance = "\n".join(lines) + "\n"
    prov = (project.name + "/PROVENANCE.tsv")
    if check:
        problems += not emit(prov, provenance.encode(), "100644", True)
        problems += audit_tree(project.name, set(written) | {prov})
        print(f"{project.name}: {len(rows)} files checked, {problems} differ, {len(skipped)} excluded")
        return 1 if problems else 0
    emit(prov, provenance.encode(), "100644", False)
    changed = sum(1 for r in rows if r[-1] == "changed")
    print(f"{project.name}: {len(rows)} files ({changed} changed on the way), {len(skipped)} excluded")
    return 0


def audit_tree(name: str, expected: set[str]) -> int:
    """Files under the project that no source accounts for, and imported files git would drop.

    Only a README.md may be in the tree without a PROVENANCE row: those are written by hand.
    """
    problems = 0
    on_disk = {str(p.relative_to(REPO)) for p in (REPO / name).rglob("*") if p.is_file()}
    for path in sorted(on_disk - expected):
        if Path(path).name != "README.md":
            print(f"no provenance: {path}")
            problems += 1
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", name],
                             capture_output=True, text=True).stdout.split("\0")
    # Once the repository has a commit, every imported file must be tracked, even when none is.
    has_commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", "-q", "HEAD"],
                                capture_output=True).returncode == 0
    if has_commit:
        for path in sorted(expected - set(tracked)):
            print(f"not in git: {path}")
            problems += 1
    return problems


def main(project: Project) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with what is here instead of writing")
    sys.exit(run(project, ap.parse_args().check))
