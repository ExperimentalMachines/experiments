"""Checks the members of a tar against the text hooks, so a snapshot cannot carry an identifier.

A tar whose members the hooks leave unchanged is returned exactly as it came in, which is every
tar imported so far. Rebuilding one would not keep every format detail (GNU headers, custom PAX
records), so a tar that needs a change stops the import instead: handle it on purpose.
"""

import io
import tarfile


def rewrite_tar(path: str, data: bytes, *hooks) -> bytes:
    if not data:
        return data
    with tarfile.open(fileobj=io.BytesIO(data)) as src:
        for info in src.getmembers():
            if not info.isfile():
                continue
            body = src.extractfile(info).read()
            if b"\0" in body[:8192]:
                continue
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                continue
            new = text
            for hook in hooks:
                new = hook(new)
            if new != text:
                raise SystemExit(f"{path}: member {info.name} holds an identifier; decide how to handle this tar")
    return data
