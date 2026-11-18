from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Final

from kersta.core.constants import NETUID, NETWORK

NETWORKS: Final[dict[str, str]] = {
    "finney": "wss://entrypoint-finney.opentensor.ai:443",
    "test": "wss://test.finney.opentensor.ai:443",
    "archive": "wss://archive.chain.opentensor.ai:443",
    "local": "ws://127.0.0.1:9944",
}
ENV_PREFIX: Final[str] = "KERSTA_"


@dataclass(frozen=True, slots=True)
class ChainConfig:
    netuid: int = NETUID
    network: str = NETWORK
    endpoint: str = ""
    # A node that keeps old state. Validators read commitments as they stood at
    # a competition's deadline block; a pruned node only keeps the last few hundred blocks.
    archive_endpoint: str = ""
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str = ""

    def __post_init__(self) -> None:
        if self.netuid < 0:
            raise ValueError("netuid must not be negative")
        if not self.endpoint and self.network not in NETWORKS and self.network != "offline":
            raise ValueError(f"unknown network {self.network!r}; known: {sorted(NETWORKS)}")

    @property
    def offline(self) -> bool:
        return self.network == "offline"

    @property
    def resolved_endpoint(self) -> str:
        return self.endpoint or NETWORKS.get(self.network, "")

    def with_overrides(self, **overrides: Any) -> ChainConfig:
        supplied = {k: v for k, v in overrides.items() if v is not None and v != ""}
        unknown = set(supplied) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"unknown chain settings: {sorted(unknown)}")
        return replace(self, **supplied)

    @classmethod
    def from_env(cls, **overrides: Any) -> ChainConfig:
        env: dict[str, Any] = {}
        for item in fields(cls):
            raw = os.environ.get(f"{ENV_PREFIX}{item.name.upper()}", "").strip()
            if raw:
                env[item.name] = int(raw) if item.name == "netuid" else raw
        return cls(**env).with_overrides(**overrides)
