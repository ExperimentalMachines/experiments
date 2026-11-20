#!/usr/bin/env python3
"""Every relative link in this repository's Markdown resolves to a file or folder that is here,
and every #anchor in a hand-written page names a heading that exists.

A link into a project's own repository (an app file, a write-up that stayed there) has to be a
pinned https link instead; a relative one would point at nothing once it is read here.

  python3 tools/check_links.py            # lists the broken links, exits 1 if there are any
"""

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
# A chart is shown with <picture>: its <source srcset> and <img src> must resolve too.
HTML_SRC = re.compile(r'\b(?:src|srcset)="([^"]+)"')


def imported() -> set[Path]:
    """Files an import script wrote; their anchors are the source project's business."""
    rows = set()
    for tsv in REPO.glob("*/PROVENANCE.tsv"):
        for line in tsv.read_text().splitlines():
            if line and not line.startswith("#"):
                rows.add(REPO / line.split("\t", 1)[0])
    return rows


def anchors(md: Path) -> set[str]:
    """The ids GitHub gives a file's headings: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"```.*?```", "", md.read_text(errors="replace"), flags=re.S)
    out = set()
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*#*$", text):
        heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "")
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        n, base = 1, slug
        while slug in out:
            slug, n = f"{base}-{n}", n + 1
        out.add(slug)
    return out


def broken():
    copied = imported()
    for md in sorted(REPO.rglob("*.md")):
        if ".git" in md.parts:
            continue
        text = md.read_text(errors="replace")
        # Links inside fenced code are examples, not links.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        targets = [m.group(1) or m.group(2) for m in LINK.finditer(text)] + HTML_SRC.findall(text)
        for target in targets:
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            path, _, anchor = unquote(target).partition("#")
            # `file.py:73` is a citation in a review transcript, written from the source
            # repository's root, not a relative link.
            if re.search(r":\d+$", path):
                continue
            resolved = md if not path else (REPO / path.lstrip("/")) if path.startswith("/") else (md.parent / path)
            if not resolved.exists():
                yield md.relative_to(REPO), target
            elif anchor and md not in copied and resolved.suffix == ".md" and anchor not in anchors(resolved):
                yield md.relative_to(REPO), target


def main():
    found = list(broken())
    for md, target in found:
        print(f"{md}: {target}")
    print(f"{len(found)} broken links", file=sys.stderr)
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
