"""The copy check (spec: "Outputs correlate above 0.98 with an earlier committed
model and do not beat it by 0.01: S = 0").

Order is by on chain commit block, then hotkey, so every validator agrees on
who came first even when two commits share a block. Each later model is
compared with every earlier one in the same lane, using the score each had
*before* the copy check, so copying a copy is caught too. A later model that
correlates above 0.98 but beats the earlier one by at least 0.01 keeps its
score: it is an improvement, not a copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kersta.core.constants import COPY_CORRELATION, COPY_MARGIN
from kersta.scoring.metrics import pearson


@dataclass(frozen=True, slots=True)
class Candidate:
    hotkey: str
    commit_block: int
    lane_digest: str
    score: float  # after every other penalty, before the copy check
    p_fake: np.ndarray  # aligned to the answer key order


def order(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda c: (c.commit_block, c.hotkey))


def find_copies(candidates: Sequence[Candidate]) -> dict[str, tuple[str, float]]:
    """Map each copying hotkey to (the earliest model it copies, the correlation)."""
    ranked = order(candidates)
    copies: dict[str, tuple[str, float]] = {}
    for j, later in enumerate(ranked):
        for earlier in ranked[:j]:
            same = later.lane_digest == earlier.lane_digest
            corr = 1.0 if same else pearson(later.p_fake, earlier.p_fake)
            if corr > COPY_CORRELATION and later.score < earlier.score + COPY_MARGIN:
                copies[later.hotkey] = (earlier.hotkey, round(corr, 6))
                break
    return copies
