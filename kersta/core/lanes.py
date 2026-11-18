"""The three lanes and their limits (spec: "Lanes and limits")."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

GIB: Final[int] = 1024**3
MIB: Final[int] = 1024**2


class Lane(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

    @classmethod
    def parse(cls, value: str | Lane) -> Lane:
        try:
            return cls(str(value.value if isinstance(value, Lane) else value).strip().lower())
        except ValueError as exc:
            known = ", ".join(lane.value for lane in cls)
            raise ValueError(f"unknown lane {value!r}; known: {known}") from exc


@dataclass(frozen=True, slots=True)
class LaneSpec:
    lane: Lane
    # p95 time per item, measured in the sandbox on the reference GPU.
    speed_limit_ms: float
    # Total bytes of the lane's package folder (weights, code and config).
    max_package_bytes: int
    # "mask" for images (pixels), "segments" for audio and video (seconds).
    location: str
    # Longest clip the lane accepts; None for still images.
    max_seconds: float | None
    media_suffixes: tuple[str, ...]


LANE_SPECS: Final[dict[Lane, LaneSpec]] = {
    Lane.IMAGE: LaneSpec(
        lane=Lane.IMAGE,
        speed_limit_ms=100.0,
        max_package_bytes=2 * GIB,
        location="mask",
        max_seconds=None,
        media_suffixes=(".png", ".jpg", ".jpeg", ".webp"),
    ),
    Lane.VIDEO: LaneSpec(
        lane=Lane.VIDEO,
        speed_limit_ms=2000.0,
        max_package_bytes=4 * GIB,
        location="segments",
        max_seconds=10.0,
        media_suffixes=(".mp4", ".webm", ".mov"),
    ),
    Lane.AUDIO: LaneSpec(
        lane=Lane.AUDIO,
        speed_limit_ms=300.0,
        max_package_bytes=500 * MIB,
        location="segments",
        max_seconds=10.0,
        media_suffixes=(".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"),
    ),
}

LANES: Final[tuple[Lane, ...]] = (Lane.IMAGE, Lane.VIDEO, Lane.AUDIO)


def spec_for(lane: str | Lane) -> LaneSpec:
    return LANE_SPECS[Lane.parse(lane)]
