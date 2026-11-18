"""Hashes that miners, validators and the owner must all compute identically."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

CHUNK: Final[int] = 1 << 20
SHORT_DIGEST_CHARS: Final[int] = 32
HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """The one JSON encoding every signature and digest is computed over."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def tree_digest(files: Mapping[str, tuple[str, int]]) -> str:
    """Digest of a file tree from its (relative path -> (sha256, size)) listing.

    One line per file, sorted by POSIX path: ``path<TAB>sha256<TAB>size<LF>``.
    The listing is hashed rather than the files concatenated, so a validator
    can check a package before downloading it and two packages that differ
    only in file layout never collide.
    """
    lines = []
    for path in sorted(files):
        sha, size = files[path]
        if "\t" in path or "\n" in path:
            raise ValueError(f"file path {path!r} contains a tab or newline")
        lines.append(f"{path}\t{sha}\t{int(size)}\n")
    return sha256_bytes("".join(lines).encode("utf-8"))


def short_digest(digest: str, chars: int = SHORT_DIGEST_CHARS) -> str:
    body = digest.strip().lower()
    if len(body) < chars or not set(body) <= HEX_DIGITS:
        raise ValueError(f"malformed digest {digest!r}")
    return body[:chars]


def is_hex(value: str, length: int | None = None) -> bool:
    body = value.strip().lower()
    if length is not None and len(body) != length:
        return False
    return bool(body) and set(body) <= HEX_DIGITS


def listing(paths: Iterable[tuple[str, Path]]) -> dict[str, tuple[str, int]]:
    """Hash local files into the listing tree_digest takes."""
    return {rel: (sha256_file(local), local.stat().st_size) for rel, local in paths}
