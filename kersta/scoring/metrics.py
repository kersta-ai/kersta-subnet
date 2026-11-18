"""The metric formulas, each small enough to check by hand (tests/test_metrics.py).

Spec, "Scoring details to implement exactly":
  * threshold 0.5 on p_fake: an item is called fake when p_fake >= 0.5
  * MCC rescaled to 0..1 as (MCC + 1) / 2
  * calibration = max(0, 1 - Brier / 0.25)
  * slice score = sqrt(rescaled MCC * calibration)
  * image location: IoU of the mask at 0.5 and the true mask
  * audio/video location: overlap of time segments (IoU of the covered time)
  * false alarm rate: share of real items with p_fake >= 0.5
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from kersta.core.constants import BRIER_REFERENCE, THRESHOLD


def _arrays(labels: Sequence[int], probs: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be equal length vectors")
    return y, p


def confusion(
    labels: Sequence[int], probs: Sequence[float], threshold: float = THRESHOLD
) -> tuple[int, int, int, int]:
    """(true positives, false positives, true negatives, false negatives); fake is positive."""
    y, p = _arrays(labels, probs)
    called = p >= threshold
    tp = int(np.sum(called & (y == 1)))
    fp = int(np.sum(called & (y == 0)))
    tn = int(np.sum(~called & (y == 0)))
    fn = int(np.sum(~called & (y == 1)))
    return tp, fp, tn, fn


def mcc(labels: Sequence[int], probs: Sequence[float], threshold: float = THRESHOLD) -> float:
    """Matthews correlation; 0 when undefined (e.g. everything called one class)."""
    tp, fp, tn, fn = confusion(labels, probs, threshold)
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0.0:
        return 0.0
    return (tp * tn - fp * fn) / denominator


def rescaled_mcc(labels: Sequence[int], probs: Sequence[float]) -> float:
    return (mcc(labels, probs) + 1.0) / 2.0


def brier(labels: Sequence[int], probs: Sequence[float]) -> float:
    y, p = _arrays(labels, probs)
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def calibration(labels: Sequence[int], probs: Sequence[float]) -> float:
    return max(0.0, 1.0 - brier(labels, probs) / BRIER_REFERENCE)


@dataclass(frozen=True, slots=True)
class SliceScore:
    n: int
    mcc: float
    mcc_rescaled: float
    brier: float
    calibration: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}


def slice_score(labels: Sequence[int], probs: Sequence[float]) -> SliceScore:
    m = mcc(labels, probs)
    rescaled = (m + 1.0) / 2.0
    b = brier(labels, probs)
    cal = max(0.0, 1.0 - b / BRIER_REFERENCE)
    return SliceScore(len(labels), m, rescaled, b, cal, math.sqrt(rescaled * cal))


def false_alarm_rate(labels: Sequence[int], probs: Sequence[float]) -> float:
    y, p = _arrays(labels, probs)
    real = y == 0
    if not np.any(real):
        return 0.0
    return float(np.mean(p[real] >= THRESHOLD))


def mask_iou(predicted: np.ndarray | None, true: np.ndarray) -> float:
    """Pixel IoU of two boolean masks on the same grid; 0 when no mask was given."""
    if predicted is None:
        return 0.0
    if predicted.shape != true.shape:
        raise ValueError(f"mask shapes differ: {predicted.shape} vs {true.shape}")
    union = int(np.sum(predicted | true))
    if union == 0:
        return 1.0
    return int(np.sum(predicted & true)) / union


def merge_intervals(segments: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in segments if float(b) > float(a))
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def covered(segments: Sequence[tuple[float, float]]) -> float:
    return sum(end - start for start, end in segments)


def interval_iou(
    predicted: Sequence[Sequence[float]] | None, true: Sequence[Sequence[float]]
) -> float:
    """IoU of the time covered by two segment lists; 0 when no segments were given."""
    if predicted is None:
        return 0.0
    a, b = merge_intervals(predicted), merge_intervals(true)
    union = covered(merge_intervals([*a, *b]))
    if union == 0.0:
        return 1.0
    inter, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        low, high = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        inter += max(0.0, high - low)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return inter / union


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation percentile (numpy's default), q in [0, 100]."""
    if not len(values):
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Correlation of two output vectors. Two constant vectors correlate 1 only if equal."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("vectors must be the same length")
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1.0, 1.0))
