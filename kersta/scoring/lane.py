"""One miner's score in one lane (spec: "The score, step by step").

    D = 0.25 clean + 0.50 damaged + 0.25 unseen      (slice scores)
    L = mean location overlap over partial edits      (0 for no location)
    S = 0.85 D + 0.15 L
    false alarms on real media above 2%  ->  S halved
    p95 time per item above the lane limit -> S = 0
    copy of an earlier model (copycheck)   -> S = 0

Two rules the spec leaves open, decided here and documented in README section 6 and docs/decisions.md:
  * a model that fails on more than 1% of items scores 0; a failed item
    below that counts as p_fake = 0.5 and no location;
  * an empty slice is left out of D and the other weights renormalise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kersta.core.constants import (
    DETECTION_WEIGHT,
    FAILED_ITEM_P_FAKE,
    FALSE_ALARM_FACTOR,
    FALSE_ALARM_LIMIT,
    LOCATION_WEIGHT,
    MAX_FAILURE_SHARE,
    SCORE_DIGITS,
    SLICE_WEIGHTS,
    THRESHOLD,
)
from kersta.core.lanes import Lane, spec_for
from kersta.core.testset import Answer
from kersta.scoring.metrics import (
    SliceScore,
    false_alarm_rate,
    interval_iou,
    mask_iou,
    percentile,
    slice_score,
)


@dataclass(slots=True)
class ItemResult:
    """What came back for one item. ``p_fake`` is None when the item failed."""

    id: str
    p_fake: float | None
    mask: np.ndarray | None = None
    segments: tuple[tuple[float, float], ...] | None = None
    source: str | None = None
    seconds: float | None = None
    error: str | None = None


@dataclass(slots=True)
class LaneScore:
    lane: Lane
    hotkey: str
    items: int
    failures: int
    slices: dict[str, SliceScore]
    detection: float
    location: float
    location_items: int
    raw: float
    false_alarm_rate: float
    p95_ms: float
    speed_limit_ms: float
    score: float
    reasons: list[str] = field(default_factory=list)
    copy_of: str | None = None
    breakdown: dict[str, Any] = field(default_factory=dict)

    @property
    def failure_share(self) -> float:
        return self.failures / self.items if self.items else 1.0

    def zero(self, reason: str) -> None:
        self.score = 0.0
        self.reasons.append(reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "hotkey": self.hotkey,
            "items": self.items,
            "failures": self.failures,
            "slices": {k: v.to_dict() for k, v in self.slices.items()},
            "detection": round(self.detection, SCORE_DIGITS),
            "location": round(self.location, SCORE_DIGITS),
            "location_items": self.location_items,
            "raw": round(self.raw, SCORE_DIGITS),
            "false_alarm_rate": round(self.false_alarm_rate, SCORE_DIGITS),
            "p95_ms": round(self.p95_ms, 3),
            "speed_limit_ms": self.speed_limit_ms,
            "score": round(self.score, SCORE_DIGITS),
            "reasons": list(self.reasons),
            "copy_of": self.copy_of,
            "breakdown": self.breakdown,
        }


def aligned_probabilities(answers: Sequence[Answer], results: Mapping[str, ItemResult]) -> np.ndarray:
    """p_fake in answer key order, failed items as the uninformative 0.5."""
    out = np.empty(len(answers), dtype=np.float64)
    for i, answer in enumerate(answers):
        result = results.get(answer.id)
        ok = result is not None and result.p_fake is not None and result.error is None
        out[i] = float(result.p_fake) if ok else FAILED_ITEM_P_FAKE  # type: ignore[union-attr]
    return out


def _failed(result: ItemResult | None) -> bool:
    return result is None or result.p_fake is None or result.error is not None


def _detection(answers: Sequence[Answer], probs: np.ndarray) -> tuple[float, dict[str, SliceScore]]:
    labels = np.asarray([a.label for a in answers], dtype=np.int64)
    slices: dict[str, SliceScore] = {}
    for name in SLICE_WEIGHTS:
        members = [i for i, a in enumerate(answers) if name in a.slices]
        if members:
            slices[name] = slice_score(labels[members].tolist(), probs[members].tolist())
    weight = sum(SLICE_WEIGHTS[name] for name in slices)
    if weight == 0.0:
        return 0.0, slices
    return sum(SLICE_WEIGHTS[n] * s.score for n, s in slices.items()) / weight, slices


def _location(
    lane: Lane,
    answers: Sequence[Answer],
    results: Mapping[str, ItemResult],
    true_mask: Callable[[Answer], np.ndarray] | None,
) -> tuple[float, int]:
    partial = [a for a in answers if a.partial]
    if not partial:
        return 0.0, 0
    total = 0.0
    for answer in partial:
        result = results.get(answer.id)
        if _failed(result):
            continue
        assert result is not None
        if lane is Lane.IMAGE:
            if result.mask is None or true_mask is None:
                continue
            total += mask_iou(result.mask, true_mask(answer))
        else:
            total += interval_iou(result.segments, answer.segments or ())
    return total / len(partial), len(partial)


def _breakdown(answers: Sequence[Answer], probs: np.ndarray) -> dict[str, Any]:
    """Where a model fails: by damage operation and by generator (for miner reports)."""
    by_damage: dict[str, list[int]] = {}
    by_generator: dict[str, list[int]] = {}
    for answer, p in zip(answers, probs, strict=True):
        correct = int((p >= THRESHOLD) == answer.is_fake)
        for op in answer.damage or ("none",):
            by_damage.setdefault(op.split(":")[0], []).append(correct)
        if answer.is_fake:
            name = answer.generator or "unknown"
            by_generator.setdefault(name, []).append(correct)

    def summary(groups: dict[str, list[int]]) -> dict[str, dict[str, float]]:
        return {
            key: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
            for key, v in sorted(groups.items())
        }

    return {"by_damage": summary(by_damage), "by_generator": summary(by_generator)}


def score_lane(
    lane: Lane,
    answers: Sequence[Answer],
    results: Mapping[str, ItemResult],
    *,
    hotkey: str = "",
    true_mask: Callable[[Answer], np.ndarray] | None = None,
    speed_factor: float = 1.0,
) -> LaneScore:
    """Score one model on one lane's answer key.

    ``true_mask`` returns an image item's true mask on the model's own input
    grid. ``speed_factor`` divides measured times; it is 1.0 unless the
    validator calibrated its GPU against the published reference timing.
    """
    lane = Lane.parse(lane)
    limit = spec_for(lane).speed_limit_ms
    probs = aligned_probabilities(answers, results)
    failures = sum(1 for a in answers if _failed(results.get(a.id)))
    detection, slices = _detection(answers, probs)
    location, location_items = _location(lane, answers, results, true_mask)
    raw = DETECTION_WEIGHT * detection + LOCATION_WEIGHT * location
    labels = [a.label for a in answers]
    far = false_alarm_rate(labels, probs.tolist())
    times = [
        r.seconds * 1000.0 / max(speed_factor, 1e-9)
        for a in answers
        if (r := results.get(a.id)) is not None and not _failed(r) and r.seconds is not None
    ]
    p95 = percentile(times, 95.0) if times else float("inf")

    out = LaneScore(
        lane=lane,
        hotkey=hotkey,
        items=len(answers),
        failures=failures,
        slices=slices,
        detection=detection,
        location=location,
        location_items=location_items,
        raw=raw,
        false_alarm_rate=far,
        p95_ms=p95,
        speed_limit_ms=limit,
        score=raw,
        breakdown=_breakdown(answers, probs),
    )
    if not answers:
        out.zero("the lane has no items")
        return out
    if out.failure_share > MAX_FAILURE_SHARE:
        out.zero(f"failed on {failures} of {len(answers)} items")
    if p95 > limit:
        out.zero(f"p95 {p95:.1f} ms is over the {limit:.0f} ms limit")
    if far > FALSE_ALARM_LIMIT and out.score > 0:
        out.score *= FALSE_ALARM_FACTOR
        out.reasons.append(f"false alarm rate {far:.2%} is over {FALSE_ALARM_LIMIT:.0%}: halved")
    out.score = round(max(0.0, out.score), SCORE_DIGITS)
    return out
