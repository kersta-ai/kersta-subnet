"""The one interface the rest of Kersta uses to reach the chain."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

from kersta.chain.commitments import OnChainCommitment
from kersta.chain.metagraph import MetagraphSnapshot
from kersta.chain.weights import WeightVector
from kersta.core.constants import CHAIN_ATTEMPTS, CHAIN_BACKOFF_SECONDS

log = logging.getLogger("kersta.chain")
T = TypeVar("T")


class ChainError(RuntimeError):
    pass


@runtime_checkable
class Chain(Protocol):
    @property
    def netuid(self) -> int: ...

    def block(self) -> int: ...

    def timestamp(self, block: int) -> int:
        """Unix seconds of a block, from the chain's own Timestamp pallet."""
        ...

    def snapshot(self, *, refresh: bool = False) -> MetagraphSnapshot: ...

    def commitments(self, at_block: int | None = None) -> list[OnChainCommitment]:
        """Every commitment on the subnet, as it stood after ``at_block`` (None = now)."""
        ...

    def commit(self, payload: str) -> None:
        """Write this wallet's hotkey commitment. Raises ChainError."""
        ...

    def owner_hotkey(self) -> str: ...

    def last_step_block(self) -> int | None:
        """The block of the subnet's last emission step, or None when unreadable."""
        ...

    def set_weights(self, vector: WeightVector) -> tuple[bool, str]: ...

    def close(self) -> None: ...


def with_retry(
    label: str,
    call: Callable[[], T],
    *,
    attempts: int = CHAIN_ATTEMPTS,
    backoff: float = CHAIN_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - every chain failure is retried alike
            last = exc
            if attempt == attempts:
                break
            log.warning("%s failed (attempt %d/%d): %s", label, attempt, attempts, exc)
            sleep(backoff * 2 ** (attempt - 1))
    raise ChainError(f"{label} failed after {attempts} attempts: {last}") from last
