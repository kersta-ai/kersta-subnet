"""The weekly test set: files, answer keys and the manifest that locks them.

On disk, one competition's test set looks like::

    competition-<n>/
      manifest.json             signed by the owner hotkey; its digest is on chain
      image/answers.jsonl       one Answer per line
      image/media/<id>.png      anonymised names; a name says nothing about the label
      image/masks/<id>.png      true fake region of partial edits (white = fake)
      audio/answers.jsonl
      audio/media/<id>.wav
      ...

The manifest lists every file with its sha256 and size. Its digest (sha256
of the canonical JSON, signature excluded) goes on chain before the
deadline, so not one file can change after miners commit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from kersta.core.hashing import canonical_json, sha256_bytes, sha256_file
from kersta.core.lanes import Lane

FORMAT: Final[str] = "kersta-testset/1"
MANIFEST_FILE: Final[str] = "manifest.json"
ANSWERS_FILE: Final[str] = "answers.jsonl"
REAL: Final[int] = 0
FAKE: Final[int] = 1
CLEAN: Final[str] = "clean"
DAMAGED: Final[str] = "damaged"
UNSEEN: Final[str] = "unseen"
SLICES: Final[tuple[str, ...]] = (CLEAN, DAMAGED, UNSEEN)


class TestsetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Answer:
    id: str
    file: str  # relative to the competition folder
    label: int  # 0 real, 1 fake
    slices: tuple[str, ...]
    generator: str | None = None  # tool family that made a fake
    seen: bool = True  # False when the generator is new in this competition
    partial: bool = False  # a partial edit with a location answer
    mask: str | None = None  # image partial edits: mask file
    segments: tuple[tuple[float, float], ...] | None = None  # audio/video partial edits
    damage: tuple[str, ...] = ()  # damage operations applied, in order
    seconds: float | None = None  # clip length for audio and video

    def __post_init__(self) -> None:
        if self.label not in (REAL, FAKE):
            raise TestsetError(f"{self.id}: label must be 0 or 1")
        if any(s not in SLICES for s in self.slices):
            raise TestsetError(f"{self.id}: unknown slice in {self.slices}")
        if self.partial and self.label != FAKE:
            raise TestsetError(f"{self.id}: only a fake can be a partial edit")

    @property
    def is_fake(self) -> bool:
        return self.label == FAKE

    @property
    def damaged(self) -> bool:
        return bool(self.damage)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["slices"] = list(self.slices)
        out["damage"] = list(self.damage)
        if self.segments is not None:
            out["segments"] = [list(s) for s in self.segments]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Answer:
        segments = data.get("segments")
        return cls(
            id=str(data["id"]),
            file=str(data["file"]),
            label=int(data["label"]),
            slices=tuple(str(s) for s in data.get("slices") or ()),
            generator=data.get("generator"),
            seen=bool(data.get("seen", True)),
            partial=bool(data.get("partial", False)),
            mask=data.get("mask"),
            segments=None
            if segments is None
            else tuple((float(a), float(b)) for a, b in segments),
            damage=tuple(str(d) for d in data.get("damage") or ()),
            seconds=None if data.get("seconds") is None else float(data["seconds"]),
        )


def load_answers(path: str | Path) -> list[Answer]:
    answers = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            answers.append(Answer.from_dict(json.loads(line)))
        except (KeyError, ValueError, TypeError) as exc:
            raise TestsetError(f"{path}:{number}: {exc}") from exc
    ids = [a.id for a in answers]
    if len(set(ids)) != len(ids):
        raise TestsetError(f"{path}: item ids repeat")
    return answers


def write_answers(path: str | Path, answers: Iterable[Answer]) -> None:
    lines = [json.dumps(a.to_dict(), sort_keys=True) for a in answers]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Manifest:
    competition: int
    created: str
    lanes: Mapping[str, Mapping[str, Any]]  # lane -> {"answers": path, "items": n}
    files: Mapping[str, Mapping[str, Any]]  # path -> {"sha256": ..., "size": ...}
    owner_hotkey: str = ""
    signature: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def body(self) -> dict[str, Any]:
        """Everything the digest and the owner's signature cover."""
        return {
            "format": FORMAT,
            "competition": self.competition,
            "created": self.created,
            "lanes": {k: dict(v) for k, v in self.lanes.items()},
            "files": {k: dict(v) for k, v in self.files.items()},
            "owner_hotkey": self.owner_hotkey,
            "notes": dict(self.notes),
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(self.body()))

    def owned_by(self, hotkey: str) -> Manifest:
        return replace(self, owner_hotkey=hotkey, signature="")

    def signed(self, signature: str) -> Manifest:
        return replace(self, signature=signature)

    @property
    def lane_ids(self) -> list[Lane]:
        return [Lane.parse(name) for name in self.lanes]

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "signature": self.signature}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        if data.get("format") != FORMAT:
            raise TestsetError(f"unknown test set format {data.get('format')!r}")
        return cls(
            competition=int(data["competition"]),
            created=str(data.get("created", "")),
            lanes=dict(data.get("lanes") or {}),
            files=dict(data.get("files") or {}),
            owner_hotkey=str(data.get("owner_hotkey") or ""),
            signature=str(data.get("signature") or ""),
            notes=dict(data.get("notes") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        Path(path).write_text(text + "\n", encoding="utf-8")


def build_manifest(
    root: Path, competition: int, created: str, notes: Mapping[str, Any] | None = None
) -> Manifest:
    """Hash every file under a competition folder into an unsigned manifest."""
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILE:
            continue
        rel = path.relative_to(root).as_posix()
        files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    lanes: dict[str, dict[str, Any]] = {}
    for lane in Lane:
        answers = root / lane.value / ANSWERS_FILE
        if answers.is_file():
            lanes[lane.value] = {
                "answers": f"{lane.value}/{ANSWERS_FILE}",
                "items": len(load_answers(answers)),
            }
    if not lanes:
        raise TestsetError(f"{root} holds no lane answers")
    return Manifest(competition, created, lanes, files, notes=dict(notes or {}))


def verify_files(root: Path, manifest: Manifest) -> None:
    """Every listed file is present with the listed hash. Raises TestsetError."""
    for rel, entry in manifest.files.items():
        path = root / rel
        if not path.is_file():
            raise TestsetError(f"{rel} is missing")
        if path.stat().st_size != int(entry["size"]) or sha256_file(path) != entry["sha256"]:
            raise TestsetError(f"{rel} does not match the manifest")


def lane_answers(root: Path, manifest: Manifest, lane: Lane) -> list[Answer]:
    entry = manifest.lanes.get(lane.value)
    if entry is None:
        return []
    return load_answers(root / str(entry["answers"]))
