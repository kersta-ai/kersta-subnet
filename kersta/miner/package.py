"""Starting and checking a submission."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kersta.core.lanes import Lane, spec_for
from kersta.core.package import Submission, scan_submission
from kersta.media.preprocess import AudioPrep, ImagePrep, Prep, VideoPrep
from kersta.sandbox.runner import LocalLauncher, run_model

STARTER_ROOT = Path(__file__).resolve().parent.parent / "starter"


def init(lane: Lane | str, dest: Path, *, force: bool = False) -> Path:
    """Copy the starter package for a lane into ``dest/<lane>/``."""
    lane = Lane.parse(lane)
    source = STARTER_ROOT / lane.value / "package"
    target = dest / lane.value
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to overwrite")
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, target / path.name)
    return target


def synthetic_item(lane: Lane, prep: Prep, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    """A random input of exactly the shape and dtype a validator would send."""
    rng = np.random.default_rng(seed)

    def pixels(prefix: tuple[int, ...], image: ImagePrep) -> np.ndarray:
        h, w = image.size
        shape = (*prefix, 3, h, w) if image.layout == "nchw" else (*prefix, h, w, 3)
        raw = rng.integers(0, 256, size=shape, dtype=np.uint8)
        return raw if image.scale == "uint8" else raw.astype(np.float32) / (
            255.0 if image.scale == "unit" else 1.0)

    if lane is Lane.IMAGE:
        assert isinstance(prep, ImagePrep)
        return {"inputs": pixels((), prep)}, {"width": prep.size[1], "height": prep.size[0]}
    if lane is Lane.AUDIO:
        assert isinstance(prep, AudioPrep)
        n = int(prep.seconds * prep.sample_rate)
        samples = (rng.standard_normal(n) * 0.1).astype(np.float32)
        return {"inputs": samples}, {"sample_rate": prep.sample_rate, "samples": n,
                                     "seconds": prep.seconds, "offset": 0.0}
    assert isinstance(prep, VideoPrep)
    arrays = {"inputs": pixels((prep.max_frames,), prep.image)}
    if prep.audio:
        arrays["audio"] = np.zeros(int(10 * prep.sample_rate), dtype=np.float32)
    meta = {"seconds": prep.max_frames / prep.fps, "frames": prep.max_frames,
            "frame_times": [round(i / prep.fps, 4) for i in range(prep.max_frames)]}
    return arrays, meta


@dataclass(slots=True)
class LaneCheck:
    lane: Lane
    digest: str
    size: int
    interface: str = ""
    load_seconds: float = 0.0
    p95_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(slots=True)
class CheckReport:
    submission: Submission
    lanes: list[LaneCheck]

    @property
    def ok(self) -> bool:
        return all(lane.ok for lane in self.lanes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.submission.digest,
            "ok": self.ok,
            "lanes": [
                {"lane": c.lane.value, "digest": c.digest, "size": c.size, "interface": c.interface,
                 "load_seconds": c.load_seconds, "p95_ms_smoke": round(c.p95_ms, 2),
                 "errors": c.errors}
                for c in self.lanes
            ],
        }


def check(root: Path, *, smoke: bool = True, items: int = 8) -> CheckReport:
    """Static checks, then (optionally) a smoke run on random inputs.

    The smoke run uses your own Python, not the validator's sandbox image, so
    a pass here means the code runs; ``kersta miner score`` on a released test
    set tells you how it scores.
    """
    submission = scan_submission(root)
    checks: list[LaneCheck] = []
    for lane, package in submission.lanes.items():
        result = LaneCheck(lane, package.digest, package.size)
        if smoke:
            prep = package.config.preprocessing
            loaders = [
                (f"smoke-{i}", lambda i=i, lane=lane, prep=prep: synthetic_item(lane, prep, i))
                for i in range(items)
            ]
            run = run_model(LocalLauncher(), package.root, lane, loaders, total=items,
                            layout=getattr(prep, "layout", "nchw"))
            result.interface, result.load_seconds = run.interface, run.load_seconds
            if run.error:
                result.errors.append(run.error)
            failed = [r for r in run.results.values() if r.error]
            result.errors.extend(f"{r.id}: {r.error}" for r in failed[:3])
            times = sorted(r.seconds * 1000 for r in run.results.values() if r.seconds and not r.error)
            if times:
                result.p95_ms = float(np.percentile(times, 95))
                if result.p95_ms > spec_for(lane).speed_limit_ms:
                    result.errors.append(
                        f"smoke p95 {result.p95_ms:.0f} ms is over the {spec_for(lane).speed_limit_ms:.0f} "
                        "ms limit on this machine")
        checks.append(result)
    return CheckReport(submission, checks)
