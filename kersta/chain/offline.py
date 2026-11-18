"""A chain in a JSON file, for tests and full local competitions without TAO.

It keeps the history of every commitment, so reads "as of" a past block
behave like an archive node. Blocks advance with the wall clock (12 s) from
``genesis_ts`` unless a test pins ``block``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from kersta.chain.client import ChainError
from kersta.chain.commitments import OnChainCommitment
from kersta.chain.metagraph import MetagraphSnapshot, Neuron
from kersta.chain.weights import WeightVector
from kersta.core.constants import BLOCK_SECONDS


class OfflineChain:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        netuid: int = 456,
        hotkey: str = "",
        genesis_ts: int | None = None,
    ) -> None:
        self._path = Path(path) if path else None
        self._hotkey = hotkey
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {
            "netuid": netuid,
            "genesis_ts": int(time.time()) if genesis_ts is None else genesis_ts,
            "block": None,
            "owner": "",
            "neurons": [],
            "commitments": {},
            "weights": {},
            "last_step": None,
        }
        if self._path and self._path.is_file():
            self._state.update(json.loads(self._path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------ test controls

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_suffix(".part")
        partial.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        partial.replace(self._path)

    def as_hotkey(self, hotkey: str) -> OfflineChain:
        """The same chain, signing as another hotkey."""
        other = OfflineChain.__new__(OfflineChain)
        other._path, other._hotkey, other._lock, other._state = (
            self._path, hotkey, self._lock, self._state)
        return other

    def pin_block(self, block: int | None) -> None:
        with self._lock:
            self._state["block"] = block
            self._save()

    def register(self, hotkey: str, *, coldkey: str = "", stake: float = 0.0,
                 validator: bool = False) -> int:
        with self._lock:
            neurons = self._state["neurons"]
            for neuron in neurons:
                if neuron["hotkey"] == hotkey:
                    return int(neuron["uid"])
            uid = len(neurons)
            neurons.append({"uid": uid, "hotkey": hotkey, "coldkey": coldkey or hotkey,
                            "stake": stake, "validator_permit": validator})
            self._save()
            return uid

    def set_owner(self, hotkey: str) -> None:
        with self._lock:
            self._state["owner"] = hotkey
            self._save()

    def weights_of(self, hotkey: str) -> dict[int, int]:
        return {int(k): int(v) for k, v in (self._state["weights"].get(hotkey) or {}).items()}

    # ------------------------------------------------------------ Chain protocol

    @property
    def netuid(self) -> int:
        return int(self._state["netuid"])

    def block(self) -> int:
        pinned = self._state.get("block")
        if pinned is not None:
            return int(pinned)
        return max(0, int(time.time() - self._state["genesis_ts"]) // BLOCK_SECONDS)

    def timestamp(self, block: int) -> int:
        return int(self._state["genesis_ts"]) + int(block) * BLOCK_SECONDS

    def snapshot(self, *, refresh: bool = False) -> MetagraphSnapshot:
        neurons = tuple(
            Neuron(int(n["uid"]), n["hotkey"], n.get("coldkey", ""), float(n.get("stake", 0)),
                   bool(n.get("validator_permit", False)))
            for n in self._state["neurons"]
        )
        return MetagraphSnapshot(self.netuid, self.block(), neurons)

    def commitments(self, at_block: int | None = None) -> list[OnChainCommitment]:
        limit = self.block() if at_block is None else at_block
        found = []
        for hotkey, history in self._state["commitments"].items():
            written = [h for h in history if int(h["block"]) <= limit]
            if written:
                latest = written[-1]
                found.append(OnChainCommitment(hotkey, int(latest["block"]), latest["raw"]))
        return found

    def commit(self, payload: str) -> None:
        if not self._hotkey:
            raise ChainError("this offline chain client has no hotkey to sign with")
        if len(payload.encode("utf-8")) > 128:
            raise ChainError("commitment over 128 bytes")
        with self._lock:
            history = self._state["commitments"].setdefault(self._hotkey, [])
            history.append({"block": self.block(), "raw": payload})
            self._save()

    def owner_hotkey(self) -> str:
        return str(self._state.get("owner") or "")

    def last_step_block(self) -> int | None:
        step = self._state.get("last_step")
        return None if step is None else int(step)

    def set_weights(self, vector: WeightVector) -> tuple[bool, str]:
        if vector.is_empty:
            return False, "refusing to submit an empty weight vector"
        if not self._hotkey:
            return False, "no hotkey to sign with"
        with self._lock:
            self._state["weights"][self._hotkey] = {str(u): v for u, v in vector.as_dict().items()}
            self._save()
        return True, ""

    def close(self) -> None:
        return None
