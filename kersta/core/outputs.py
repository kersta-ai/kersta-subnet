"""What a detector returns, and how the validator checks it.

predict(batch) returns one dict per item:

    p_fake    float in [0, 1]                         required, every lane
    mask      HxW array in [0, 1] (image lane)        optional, scored as location
    segments  [[start_s, end_s], ...] (audio, video)  optional, scored as location
    source    tool family name or "unknown"           optional until Phase 3

Anything malformed makes that item a failure, never an exception.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from PIL import Image

from kersta.core.lanes import Lane

MAX_SEGMENTS: Final[int] = 64
MAX_SOURCE_CHARS: Final[int] = 64
P_TOLERANCE: Final[float] = 1e-6


class OutputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Prediction:
    p_fake: float
    mask: np.ndarray | None = None  # bool, on the model's input grid
    segments: tuple[tuple[float, float], ...] | None = None
    source: str | None = None


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise OutputError("p_fake must be a number")
    p = float(value)
    if not math.isfinite(p) or p < -P_TOLERANCE or p > 1 + P_TOLERANCE:
        raise OutputError(f"p_fake must lie in [0, 1], got {p}")
    return min(1.0, max(0.0, p))


def _mask(value: Any, grid: tuple[int, int] | None) -> np.ndarray:
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.size == 0:
        raise OutputError("mask must be a 2D array")
    if array.dtype == np.uint8:
        binary = array >= 128  # the worker ships masks as round(m * 255)
    else:
        floats = array.astype(np.float64)
        if not np.all(np.isfinite(floats)):
            raise OutputError("mask holds a non finite value")
        binary = floats >= 0.5
    if grid is not None and binary.shape != grid:
        picture = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
        resized = picture.resize((grid[1], grid[0]), Image.Resampling.NEAREST)
        binary = np.asarray(resized) > 127
    return binary


def _segments(value: Any, duration: float | None) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise OutputError("segments must be a list of [start, end] pairs")
    if len(value) > MAX_SEGMENTS:
        raise OutputError(f"at most {MAX_SEGMENTS} segments")
    out: list[tuple[float, float]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise OutputError("each segment must be [start_seconds, end_seconds]")
        start, end = (float(v) for v in pair)
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise OutputError("a segment must have finite start <= end")
        if duration is not None:
            start, end = max(0.0, min(start, duration)), max(0.0, min(end, duration))
        if end > start:
            out.append((start, end))
    return tuple(out)


def validate(
    raw: Mapping[str, Any],
    lane: Lane,
    *,
    grid: tuple[int, int] | None = None,
    duration: float | None = None,
) -> Prediction:
    if not isinstance(raw, Mapping):
        raise OutputError("each prediction must be a dict")
    p_fake = _probability(raw.get("p_fake"))
    mask = segments = source = None
    if lane is Lane.IMAGE and raw.get("mask") is not None:
        mask = _mask(raw["mask"], grid)
    if lane is not Lane.IMAGE and raw.get("segments") is not None:
        segments = _segments(raw["segments"], duration)
    if raw.get("source") is not None:
        text = str(raw["source"]).strip()
        source = text[:MAX_SOURCE_CHARS] or None
    return Prediction(p_fake=p_fake, mask=mask, segments=segments, source=source)
