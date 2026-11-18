"""From lane scores to one weight per hotkey (spec: "Payout").

1. Rank each lane by score (ties: earlier commit block, then hotkey).
2. Pay the top 20; each rank earns 0.8 times the rank above, renormalised
   over the ranks actually paid (#1 ~20%, #5 ~8%, #10 ~3% with 20 paid).
3. Blend with the previous competition: 70% this competition's shares,
   30% the previous one's. If the previous competition paid nobody in the
   lane, this one takes all of it. If this one pays nobody, the previous
   competition's 30% is paid and the 70% burns.
4. Split across lanes, one third each. Whatever a lane cannot pay burns
   (goes to the subnet owner's hotkey), so emissions never reward work that
   did not clear the bar.

The weights returned here plus ``burned`` always sum to 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kersta.core.constants import (
    LANE_SHARES,
    PAID_RANKS,
    RANK_DECAY,
    SMOOTHING_PREVIOUS,
    SMOOTHING_THIS,
    WEIGHT_DIGITS,
)

Shares = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Standing:
    hotkey: str
    score: float
    commit_block: int


def rank_curve(count: int, paid: int = PAID_RANKS, decay: float = RANK_DECAY) -> list[float]:
    n = max(0, min(count, paid))
    raw = [decay**r for r in range(n)]
    total = math.fsum(raw)
    return [w / total for w in raw] if total > 0 else []


def ranked(standings: Sequence[Standing]) -> list[Standing]:
    eligible = [s for s in standings if s.score > 0.0]
    return sorted(eligible, key=lambda s: (-s.score, s.commit_block, s.hotkey))


def lane_shares(standings: Sequence[Standing], paid: int = PAID_RANKS) -> dict[str, float]:
    """This competition's shares within one lane; empty when nobody is eligible."""
    order = ranked(standings)[:paid]
    return {s.hotkey: w for s, w in zip(order, rank_curve(len(order), paid), strict=True)}


def blend(this: Shares, last: Shares | None) -> tuple[dict[str, float], float]:
    """(shares within the lane, fraction of the lane burned)."""
    if this and last:
        hotkeys = set(this) | set(last)
        mixed = {
            h: SMOOTHING_THIS * this.get(h, 0.0) + SMOOTHING_PREVIOUS * last.get(h, 0.0)
            for h in hotkeys
        }
        return {h: v for h, v in mixed.items() if v > 0.0}, 0.0
    if this:
        return dict(this), 0.0
    if last:
        return {h: SMOOTHING_PREVIOUS * v for h, v in last.items() if v > 0.0}, SMOOTHING_THIS
    return {}, 1.0


@dataclass(slots=True)
class Payout:
    weights: dict[str, float] = field(default_factory=dict)  # hotkey -> share of all emission
    burned: dict[str, float] = field(default_factory=dict)  # lane -> share burned
    lanes: dict[str, dict[str, float]] = field(default_factory=dict)  # lane -> blended shares

    @property
    def burned_total(self) -> float:
        return math.fsum(self.burned.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": dict(sorted(self.weights.items())),
            "burned": dict(self.burned),
            "burned_total": round(self.burned_total, WEIGHT_DIGITS),
            "lanes": {k: dict(sorted(v.items())) for k, v in self.lanes.items()},
        }


def combine(
    this: Mapping[str, Shares],
    previous: Mapping[str, Shares] | None = None,
    lane_split: Mapping[str, float] = LANE_SHARES,
) -> Payout:
    out = Payout()
    for lane, share in lane_split.items():
        blended, burned = blend(this.get(lane) or {}, (previous or {}).get(lane) or None)
        out.lanes[lane] = {h: round(v, WEIGHT_DIGITS) for h, v in blended.items()}
        if burned > 0.0:
            out.burned[lane] = round(share * burned, WEIGHT_DIGITS)
        for hotkey, value in blended.items():
            out.weights[hotkey] = out.weights.get(hotkey, 0.0) + share * value
    out.weights = {h: round(v, WEIGHT_DIGITS) for h, v in out.weights.items() if v > 0.0}
    return out
