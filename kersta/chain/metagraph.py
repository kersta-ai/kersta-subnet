"""A plain snapshot of the metagraph, whatever shape the SDK returned."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Neuron:
    uid: int
    hotkey: str
    coldkey: str = ""
    stake: float = 0.0
    validator_permit: bool = False


@dataclass(frozen=True, slots=True)
class MetagraphSnapshot:
    netuid: int
    block: int
    neurons: tuple[Neuron, ...] = ()

    def __post_init__(self) -> None:
        uids = [n.uid for n in self.neurons]
        if len(set(uids)) != len(uids):
            raise ValueError("duplicate uid in metagraph snapshot")

    def __len__(self) -> int:
        return len(self.neurons)

    @property
    def uid_by_hotkey(self) -> dict[str, int]:
        return {n.hotkey: n.uid for n in self.neurons}

    def find(self, hotkey: str) -> Neuron | None:
        return next((n for n in self.neurons if n.hotkey == hotkey), None)

    def is_registered(self, hotkey: str) -> bool:
        return self.find(hotkey) is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "netuid": self.netuid,
            "block": self.block,
            "neurons": [
                {"uid": n.uid, "hotkey": n.hotkey, "coldkey": n.coldkey, "stake": n.stake,
                 "validator_permit": n.validator_permit}
                for n in self.neurons
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetagraphSnapshot:
        return cls(
            netuid=int(data.get("netuid", 0)),
            block=int(data.get("block", 0)),
            neurons=tuple(
                Neuron(int(n["uid"]), str(n["hotkey"]), str(n.get("coldkey", "")),
                       float(n.get("stake", 0.0)), bool(n.get("validator_permit", False)))
                for n in data.get("neurons") or []
            ),
        )


def _number(value: Any) -> float:
    """Balance objects, tensors and plain numbers all become floats."""
    for attr in ("tao", "item"):
        inner = getattr(value, attr, None)
        if inner is not None:
            value = inner() if callable(inner) else inner
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _column(source: Any, name: str, length: int, default: Any) -> list[Any]:
    values = getattr(source, name, None)
    if values is None:
        return [default] * length
    try:
        listed = list(values)
    except TypeError:
        return [default] * length
    return (listed + [default] * length)[:length]


def snapshot_from(raw: Any, *, netuid: int, block: int = 0) -> MetagraphSnapshot:
    """Accept the v11 neuron list or the v9/v10 column metagraph."""
    hotkeys = getattr(raw, "hotkeys", None)
    if hotkeys is not None and getattr(raw, "uids", None) is not None:
        size = len(list(hotkeys))
        uids = [int(_number(u)) for u in _column(raw, "uids", size, 0)]
        coldkeys = _column(raw, "coldkeys", size, "")
        stakes = _column(raw, "S", size, 0.0) if hasattr(raw, "S") else _column(
            raw, "total_stake", size, 0.0
        )
        permits = _column(raw, "validator_permit", size, False)
        neurons = tuple(
            Neuron(uids[i], str(list(hotkeys)[i]), str(coldkeys[i]), _number(stakes[i]),
                   bool(_number(permits[i])))
            for i in range(size)
        )
        found_block = int(_number(getattr(raw, "block", block)) or block)
        return MetagraphSnapshot(netuid, found_block, neurons)

    records: Iterable[Any] = getattr(raw, "neurons", None) or raw
    validators = {str(getattr(v, "hotkey", v)) for v in getattr(raw, "validators", []) or []}
    neurons = []
    for record in records:
        hotkey = str(getattr(record, "hotkey", ""))
        if not hotkey:
            continue
        stake = getattr(record, "total_stake", None)
        if stake is None:
            stake = getattr(record, "stake", 0.0)
        permit = getattr(record, "validator_permit", None)
        neurons.append(
            Neuron(
                uid=int(_number(getattr(record, "uid", 0))),
                hotkey=hotkey,
                coldkey=str(getattr(record, "coldkey", "")),
                stake=_number(stake),
                validator_permit=bool(permit) if permit is not None else hotkey in validators,
            )
        )
    return MetagraphSnapshot(netuid, block, tuple(neurons))
