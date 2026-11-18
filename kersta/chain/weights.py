"""Turning shares into the u16 vector every honest validator signs identically."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from kersta.core.constants import MECHANISM_VERSION

U16_MAX: Final[int] = 65535


@dataclass(frozen=True, slots=True)
class WeightVector:
    uids: tuple[int, ...]
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.uids) != len(self.values):
            raise ValueError("weight vector must pair one value with each uid")
        if len(set(self.uids)) != len(self.uids):
            raise ValueError("weight vector must not repeat a uid")
        if any(uid < 0 for uid in self.uids):
            raise ValueError("uids must not be negative")
        if any(not 0 <= v <= U16_MAX for v in self.values):
            raise ValueError("weights must fit in u16")

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def is_empty(self) -> bool:
        return not self.uids

    def as_dict(self) -> dict[int, int]:
        return dict(zip(self.uids, self.values, strict=True))


def version_key(version: str = MECHANISM_VERSION) -> int:
    """'1.2.3' -> 10203. Becomes the weights version key on chain."""
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"mechanism version {version!r} must look like major.minor.patch")
    major, minor, patch = (int(p) for p in parts)
    return major * 10_000 + minor * 100 + patch


def quantise(weights: Mapping[int, float], total: int = U16_MAX) -> WeightVector:
    """Largest remainder rounding to integers that sum to exactly ``total``.

    Ties go to the lower uid, so the result never depends on dict order.
    """
    positive = {int(uid): float(w) for uid, w in weights.items() if w > 0.0}
    if not positive:
        return WeightVector((), ())
    mass = sum(positive.values())
    uids = sorted(positive)
    exact = [positive[uid] / mass * total for uid in uids]
    floors = [int(v) for v in exact]
    order = sorted(range(len(uids)), key=lambda i: (-(exact[i] - floors[i]), uids[i]))
    for i in order[: total - sum(floors)]:
        floors[i] += 1
    kept = [(uids[i], floors[i]) for i in range(len(uids)) if floors[i] > 0]
    return WeightVector(tuple(u for u, _ in kept), tuple(v for _, v in kept))
