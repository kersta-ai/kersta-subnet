"""Submitting weights and keeping them alive between competitions.

A competition produces one weight per hotkey. Here they become uids (a hotkey
that left the subnet has its share burned to the owner, not spread over the
rest), are quantised to u16, and are sent. The keeper re-sends the same
vector once per chain epoch until the next competition ends, so the validator
never goes inactive however far apart the owner schedules competitions.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

from kersta.chain.client import Chain, ChainError
from kersta.chain.metagraph import MetagraphSnapshot
from kersta.chain.weights import WeightVector, quantise, version_key
from kersta.core.constants import KEEPER_SECONDS, WEIGHTS_MIN_GAP_BLOCKS, WEIGHTS_REFRESH_BLOCKS
from kersta.validator.state import State

log = logging.getLogger("kersta.validator.weights")
LOCK = threading.Lock()


def vector_for(
    weights: Mapping[str, float], snapshot: MetagraphSnapshot, owner: str
) -> tuple[WeightVector, list[str]]:
    """(the vector, hotkeys that are no longer registered and were burned)."""
    uids = snapshot.uid_by_hotkey
    by_uid: dict[int, float] = {}
    gone = []
    for hotkey, weight in weights.items():
        uid = uids.get(hotkey)
        if uid is None:
            gone.append(hotkey)
            uid = uids.get(owner)
            if uid is None:
                continue
        by_uid[uid] = by_uid.get(uid, 0.0) + float(weight)
    return quantise(by_uid), sorted(gone)


def submit(chain: Chain, state: State, weights: Mapping[str, float], *, competition: int,
           owner: str, reason: str) -> tuple[bool, str]:
    with LOCK:
        return _submit(chain, state, dict(weights), competition, owner, reason)


def _submit(chain: Chain, state: State, weights: dict[str, float], competition: int, owner: str,
            reason: str) -> tuple[bool, str]:
    ok, message, block = False, "", 0
    vector, gone = WeightVector((), ()), []
    try:
        snapshot = chain.snapshot(refresh=True)
        vector, gone = vector_for(weights, snapshot, owner)
        ok, message = chain.set_weights(vector)
        block = chain.block()
    except ChainError as exc:
        message = str(exc)
    state.save_weights({
        "competition": competition, "weights": weights, "owner": owner,
        "sent_block": block if ok else 0, "ok": ok, "message": message,
    })
    state.log_weights({
        "at": int(time.time()), "block": block, "competition": competition, "reason": reason,
        "ok": ok, "message": message, "version_key": version_key(), "burned_unregistered": gone,
        "uids": list(vector.uids), "values": list(vector.values), "weights": weights,
    })
    if ok:
        log.info("competition %d weights sent at block %d (%s, %d uids)", competition, block,
                 reason, len(vector))
    else:
        log.error("competition %d weights not sent (%s): %s", competition, reason, message)
    return ok, message


def due(chain: Chain, block: int, sent_block: int) -> bool:
    if block - sent_block < WEIGHTS_MIN_GAP_BLOCKS:
        return False
    step = chain.last_step_block()
    if step is not None:
        return sent_block <= step  # the chain has run an epoch since we last sent
    return block - sent_block >= WEIGHTS_REFRESH_BLOCKS


def refresh(chain: Chain, state: State) -> bool:
    """Send the current weights again if a chain epoch has passed. True if sent."""
    with LOCK:
        current: dict[str, Any] | None = state.weights()
        if not current or not current.get("weights"):
            return False
        if not due(chain, chain.block(), int(current.get("sent_block") or 0)):
            return False
        ok, _ = _submit(chain, state, dict(current["weights"]), int(current["competition"]),
                        str(current.get("owner") or ""), "keeper")
        return ok


class Keeper(threading.Thread):
    def __init__(self, chain: Chain, state: State, interval: float = KEEPER_SECONDS) -> None:
        super().__init__(name="weights-keeper", daemon=True)
        self.chain, self.state, self.interval = chain, state, interval
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                refresh(self.chain, self.state)
            except Exception as exc:  # noqa: BLE001 - the keeper must never die
                log.error("weights keeper pass failed: %s", exc)
