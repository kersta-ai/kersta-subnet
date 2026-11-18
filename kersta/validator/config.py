from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from kersta.chain.config import ChainConfig
from kersta.core.constants import REPORT_URL, SANDBOX_IMAGE, TESTSET_URL


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


@dataclass(frozen=True, slots=True)
class ValidatorConfig:
    home: Path = field(default_factory=lambda: Path(_env("KERSTA_HOME", "~/.kersta")).expanduser()
                       / "validator")
    chain: ChainConfig = field(default_factory=ChainConfig.from_env)
    # Where competition test sets are published: an https base URL or a local folder.
    testset_url: str = TESTSET_URL
    testset_token: str = field(default_factory=lambda: _env("KERSTA_TESTSET_TOKEN"))
    # Optional: where signed competition reports are posted for the leaderboard.
    report_url: str = REPORT_URL
    # "docker" for every real validator; "local" runs miner code unsandboxed
    # and is refused on a real network.
    sandbox: str = field(default_factory=lambda: _env("KERSTA_SANDBOX", "docker"))
    sandbox_image: str = SANDBOX_IMAGE
    gpu: str | None = field(default_factory=lambda: _env("KERSTA_GPU", "0") or None)
    memory: str = field(default_factory=lambda: _env("KERSTA_SANDBOX_MEMORY", "48g"))
    cpus: str = field(default_factory=lambda: _env("KERSTA_SANDBOX_CPUS", "8"))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN"))
    poll_seconds: int = 60
    dry_run: bool = False
    # Debug only: score just the first N items per lane. Never set on a real network.
    item_limit: int | None = None

    @property
    def state_dir(self) -> Path:
        return self.home / "state"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    def check(self) -> None:
        if self.sandbox not in ("docker", "local"):
            raise ValueError("sandbox must be 'docker' or 'local'")
        if not self.chain.offline and self.sandbox != "docker":
            raise ValueError("a validator on a real network must run models in docker")
        if not self.chain.offline and self.item_limit is not None and not self.dry_run:
            raise ValueError("item_limit is only for dry runs")
        if not self.testset_url:
            raise ValueError("set KERSTA_TESTSET_URL to the test set store")
