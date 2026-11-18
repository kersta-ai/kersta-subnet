"""The model package: what a miner submits (spec: "Interface 1").

A submission is one Hugging Face repository revision with one folder per
lane the miner enters::

    image/                      any subset of image/, video/, audio/
      model_config.yaml         name, version, lane, dtype, preprocessing
      model.py                  load_model(weights_path, config) -> detector
      model.safetensors         weights only; no pickle formats
    audio/
      ...

Only files inside the lane folders are part of the submission. Everything
else in the repository (README, .gitattributes) is ignored, never
downloaded, and not hashed.

The submission digest is ``tree_digest`` over every lane file, with paths
relative to the repository root (``image/model.py``). Its first 32 hex
characters go on chain.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Final

import yaml

from kersta.core.hashing import sha256_file, tree_digest
from kersta.core.lanes import LANES, Lane, spec_for
from kersta.media.preprocess import Prep, PreprocessError, parse_preprocessing

CONFIG_FILE: Final[str] = "model_config.yaml"
CODE_FILE: Final[str] = "model.py"
WEIGHTS_FILE: Final[str] = "model.safetensors"
ALLOWED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".safetensors"}
)
DTYPES: Final[frozenset[str]] = frozenset({"float32", "float16", "bfloat16"})
DTYPE_ALIASES: Final[dict[str, str]] = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
MAX_FILES_PER_LANE: Final[int] = 64
SAFETENSORS_HEADER_LIMIT: Final[int] = 100 * 1024 * 1024


class PackageError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    version: str
    lane: Lane
    dtype: str
    preprocessing: Prep
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, data: Any, lane: Lane) -> ModelConfig:
        if not isinstance(data, Mapping):
            raise PackageError(f"{CONFIG_FILE} must be a mapping")
        name = str(data.get("name") or "").strip()
        version = str(data.get("version") or "").strip()
        if not name or len(name) > 64:
            raise PackageError("name is required and at most 64 characters")
        if not version or len(version) > 32:
            raise PackageError("version is required and at most 32 characters")
        # "modality" is accepted as another name for "lane" (plain classifier packages).
        declared = Lane.parse(str(data.get("lane") or data.get("modality") or ""))
        if declared is not lane:
            raise PackageError(f"config says lane {declared.value!r} but sits in {lane.value}/")
        dtype = str(data.get("dtype") or "float32").strip().lower()
        dtype = DTYPE_ALIASES.get(dtype, dtype)
        if dtype not in DTYPES:
            raise PackageError(f"dtype must be one of {', '.join(sorted(DTYPES))}")
        block = data.get("preprocessing") or {}
        if not isinstance(block, Mapping):
            raise PackageError("preprocessing must be a mapping")
        try:
            prep = parse_preprocessing(lane, dict(block))
        except PreprocessError as exc:
            raise PackageError(f"preprocessing: {exc}") from exc
        return cls(name, version, lane, dtype, prep, dict(data))

    @classmethod
    def load(cls, path: Path, lane: Lane) -> ModelConfig:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PackageError(f"{CONFIG_FILE} is not valid YAML: {exc}") from exc
        return cls.parse(data, lane)


@dataclass(frozen=True, slots=True)
class LanePackage:
    lane: Lane
    root: Path  # the lane folder on disk
    files: Mapping[str, tuple[str, int]]  # repo relative path -> (sha256, size)
    config: ModelConfig

    @property
    def digest(self) -> str:
        """Digest of this lane alone, used to spot identical copies."""
        return tree_digest(self.files)

    @property
    def size(self) -> int:
        return sum(size for _, size in self.files.values())

    @property
    def weights_path(self) -> Path:
        return self.root / WEIGHTS_FILE


@dataclass(frozen=True, slots=True)
class Submission:
    root: Path
    lanes: Mapping[Lane, LanePackage]

    @property
    def files(self) -> dict[str, tuple[str, int]]:
        merged: dict[str, tuple[str, int]] = {}
        for package in self.lanes.values():
            merged.update(package.files)
        return merged

    @property
    def digest(self) -> str:
        return tree_digest(self.files)


def lane_of(path: str) -> Lane | None:
    """The lane a repository path belongs to, or None when it is not package content."""
    parts = PurePosixPath(path).parts
    if len(parts) < 2 or any(part.startswith(".") for part in parts):
        return None
    try:
        return Lane(parts[0])
    except ValueError:
        return None


def check_remote_listing(paths: Iterable[str], sizes: Mapping[str, int] | None = None) -> None:
    """Refuse a repository listing before downloading anything from it."""
    totals: dict[Lane, int] = {}
    counts: dict[Lane, int] = {}
    for path in paths:
        lane = lane_of(path)
        if lane is None:
            continue
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise PackageError(f"{path}: file type {suffix or '(none)'} is not allowed")
        counts[lane] = counts.get(lane, 0) + 1
        totals[lane] = totals.get(lane, 0) + int((sizes or {}).get(path, 0))
    if not counts:
        raise PackageError("the repository has no image/, video/ or audio/ package folder")
    for lane, count in counts.items():
        if count > MAX_FILES_PER_LANE:
            raise PackageError(f"{lane.value}/ has {count} files; at most {MAX_FILES_PER_LANE}")
        limit = spec_for(lane).max_package_bytes
        if totals[lane] > limit:
            raise PackageError(f"{lane.value}/ is {totals[lane]} bytes; the limit is {limit}")


def check_safetensors(path: Path) -> None:
    """Read a safetensors header without loading any tensor or running code."""
    import json
    import struct

    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise PackageError(f"{path.name} is too short to be safetensors")
        (length,) = struct.unpack("<Q", prefix)
        if length <= 0 or length > SAFETENSORS_HEADER_LIMIT:
            raise PackageError(f"{path.name} has an invalid safetensors header")
        try:
            header = json.loads(handle.read(length))
        except ValueError as exc:
            raise PackageError(f"{path.name} header is not JSON") from exc
    if not isinstance(header, dict) or not any(k != "__metadata__" for k in header):
        raise PackageError(f"{path.name} holds no tensors")


def scan_lane(root: Path, lane: Lane, *, check_code: bool = True) -> LanePackage:
    folder = root / lane.value
    files: dict[str, tuple[str, int]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if lane_of(rel) is None:
            continue  # hidden files are not package content
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise PackageError(f"{rel}: file type {suffix or '(none)'} is not allowed")
        if suffix == ".safetensors":
            check_safetensors(path)
        files[rel] = (sha256_file(path), path.stat().st_size)

    for required in (CONFIG_FILE, CODE_FILE, WEIGHTS_FILE):
        if f"{lane.value}/{required}" not in files:
            raise PackageError(f"{lane.value}/{required} is missing")
    if len(files) > MAX_FILES_PER_LANE:
        raise PackageError(f"{lane.value}/ has {len(files)} files; at most {MAX_FILES_PER_LANE}")
    size = sum(s for _, s in files.values())
    limit = spec_for(lane).max_package_bytes
    if size > limit:
        raise PackageError(f"{lane.value}/ is {size} bytes; the limit is {limit}")

    config = ModelConfig.load(folder / CONFIG_FILE, lane)
    if check_code:
        from kersta.sandbox.policy import check_package_code

        violations = check_package_code(folder)
        if violations:
            listed = "; ".join(str(v) for v in violations[:10])
            raise PackageError(f"{lane.value}/ code is not allowed: {listed}")
    return LanePackage(lane=lane, root=folder, files=files, config=config)


def scan_submission(root: str | Path, *, check_code: bool = True) -> Submission:
    """Validate a submission folder and hash it. Raises PackageError."""
    root = Path(root)
    if not root.is_dir():
        raise PackageError(f"{root} is not a folder")
    lanes = {
        lane: scan_lane(root, lane, check_code=check_code)
        for lane in LANES
        if (root / lane.value).is_dir()
    }
    if not lanes:
        raise PackageError("no image/, video/ or audio/ folder found")
    return Submission(root=root, lanes=lanes)
