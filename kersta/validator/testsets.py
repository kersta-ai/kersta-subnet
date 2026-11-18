"""Fetching and checking a competition's test set.

A test set is accepted only when all of these hold:
  1. its manifest is signed by the subnet owner's hotkey;
  2. the owner's commitment, read at the competition's deadline block, is the
     lock ``kc1|<index>|<deadline>|<reveal>|<end>|<digest32>`` and matches the
     manifest digest, proving the set was fixed before miners' packages froze;
  3. every listed file matches its sha256.

The store is a base URL (or a local folder for offline runs) holding
``competition-<n>/manifest.json`` and the files it lists.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import httpx

from kersta.chain.wallet import verify
from kersta.core.hashing import canonical_json, sha256_file
from kersta.core.testset import MANIFEST_FILE, Manifest, TestsetError, verify_files

log = logging.getLogger("kersta.validator.testsets")
TIMEOUT = httpx.Timeout(60.0, connect=15.0)


def folder_name(index: int) -> str:
    return f"competition-{index}"


class TestsetStore:
    __test__ = False  # not a pytest test class, despite the name

    def __init__(self, base: str, cache: Path, *, token: str = "", allow_dev: bool = False) -> None:
        self.base = base.rstrip("/")
        self.cache = cache
        self.token = token
        self.allow_dev = allow_dev

    @property
    def remote(self) -> bool:
        return self.base.startswith(("http://", "https://"))

    def _get(self, rel: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not self.remote:
            shutil.copyfile(Path(self.base) / rel, dest)
            return
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        partial = dest.with_name(dest.name + ".part")
        with httpx.stream("GET", f"{self.base}/{rel}", headers=headers, timeout=TIMEOUT,
                          follow_redirects=True) as response:
            response.raise_for_status()
            with open(partial, "wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
        partial.replace(dest)

    def manifest(self, index: int, *, owner: str) -> Manifest:
        """Download and check only the signed manifest (its notes, its owner)."""
        path = self.cache / folder_name(index) / MANIFEST_FILE
        self._get(f"{folder_name(index)}/{MANIFEST_FILE}", path)
        manifest = Manifest.load(path)
        if manifest.competition != index:
            raise TestsetError(f"the manifest is for competition {manifest.competition}, not {index}")
        if not owner or manifest.owner_hotkey != owner:
            raise TestsetError("the manifest is not issued by the subnet owner hotkey")
        if not verify(owner, canonical_json(manifest.body()), manifest.signature,
                      allow_dev=self.allow_dev):
            raise TestsetError("the manifest signature does not verify")
        return manifest

    def fetch(self, index: int, *, owner: str, locked_digest: str | None) -> tuple[Manifest, Path]:
        """Download (or reuse) competition ``index``, check it, and return its folder."""
        if locked_digest is None:  # nothing is downloaded for an unlocked competition
            raise TestsetError("the owner did not lock this competition's test set before its deadline")
        manifest = self.manifest(index, owner=owner)
        if not manifest.digest.startswith(locked_digest):
            raise TestsetError("the manifest does not match the digest locked on chain")
        root = self.cache / folder_name(index)
        for rel, entry in manifest.files.items():
            path = root / rel
            cached = path.is_file() and path.stat().st_size == int(entry["size"])
            if cached and sha256_file(path) == entry["sha256"]:
                continue
            self._get(f"{folder_name(index)}/{rel}", path)
        verify_files(root, manifest)
        log.info("competition %d test set verified: %d files", index, len(manifest.files))
        return manifest, root
