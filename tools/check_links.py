#!/usr/bin/env python3
"""Every relative link in this repository's Markdown resolves to a file or folder that is here.

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


def broken():
    for md in sorted(REPO.rglob("*.md")):
        if ".git" in md.parts:
            continue
        text = md.read_text(errors="replace")
        # Links inside fenced code are examples, not links.
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        for m in LINK.finditer(text):
            target = m.group(1) or m.group(2)
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = unquote(target.split("#", 1)[0])
            # `file.py:73` is a citation in a review transcript, written from the source
            # repository's root, not a relative link.
            if not path or re.search(r":\d+$", path):
                continue
            resolved = (REPO / path.lstrip("/")) if path.startswith("/") else (md.parent / path)
            if not resolved.exists():
                yield md.relative_to(REPO), target


def main():
    found = list(broken())
    for md, target in found:
        print(f"{md}: {target}")
    print(f"{len(found)} broken links", file=sys.stderr)
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
