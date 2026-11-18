"""Every number the mechanism depends on, in one place.

Scoring and payout constants are deliberately NOT read from the environment:
two validators with different values would score the same competition
differently and lose consensus. Only the network and service URLs can be
overridden. Competition timing is not a constant at all: the owner announces
each competition's blocks on chain (kersta.core.competition).
"""

from __future__ import annotations

import os
from typing import Final


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------- versions
# RELEASE_VERSION moves with every release, bug fixes included.
# MECHANISM_VERSION moves only when admission, scoring or payout changes. It
# becomes the on chain weights version key, so the owner can force every
# validator onto new rules by raising the subnet's WeightsVersionKey.
RELEASE_VERSION: Final[str] = "0.1.0"
MECHANISM_VERSION: Final[str] = "1.0.0"

# ---------------------------------------------------------------- network
NETUID: Final[int] = _env_int("KERSTA_NETUID", 456)
NETWORK: Final[str] = _env_str("KERSTA_NETWORK", "test")
BLOCK_SECONDS: Final[int] = 12

# ---------------------------------------------------------------- competitions
# Defaults the owner tools use when announcing a competition; the announced
# blocks on chain are what validators follow.
DEFAULT_UPLOAD_BLOCKS: Final[int] = 1800  # deadline -> reveal
DEFAULT_SCORING_BLOCKS: Final[int] = 3600  # reveal -> end
MIN_UPLOAD_BLOCKS: Final[int] = 300
MIN_SCORING_BLOCKS: Final[int] = 600

# ---------------------------------------------------------------- scoring (spec: "Scoring and rewards")
THRESHOLD: Final[float] = 0.5
SLICE_WEIGHTS: Final[dict[str, float]] = {"clean": 0.25, "damaged": 0.50, "unseen": 0.25}
DETECTION_WEIGHT: Final[float] = 0.85
LOCATION_WEIGHT: Final[float] = 0.15
BRIER_REFERENCE: Final[float] = 0.25
MASK_THRESHOLD: Final[float] = 0.5
FALSE_ALARM_LIMIT: Final[float] = 0.02
FALSE_ALARM_FACTOR: Final[float] = 0.5
COPY_CORRELATION: Final[float] = 0.98
COPY_MARGIN: Final[float] = 0.01
# Not in the spec: a model that fails on more than this share of items scores
# zero; below it, a failed item counts as an uninformative p_fake of 0.5.
MAX_FAILURE_SHARE: Final[float] = 0.01
FAILED_ITEM_P_FAKE: Final[float] = 0.5
SCORE_DIGITS: Final[int] = 6

# ---------------------------------------------------------------- payout
PAID_RANKS: Final[int] = 20
RANK_DECAY: Final[float] = 0.8
SMOOTHING_THIS: Final[float] = 0.7  # this competition's shares
SMOOTHING_PREVIOUS: Final[float] = 0.3  # the previous competition's shares
# One third each per the spec. A lane with no eligible miner burns its share.
LANE_SHARES: Final[dict[str, float]] = {"image": 1 / 3, "video": 1 / 3, "audio": 1 / 3}
WEIGHT_DIGITS: Final[int] = 9

# ---------------------------------------------------------------- test sets
ITEMS_PER_LANE: Final[int] = 4000
REAL_SHARE: Final[float] = 0.5
UNSEEN_SHARE: Final[float] = 0.2
PARTIAL_SHARE: Final[float] = 0.4
DAMAGE_SHARE: Final[float] = 0.7

# ---------------------------------------------------------------- chain
COMMITMENT_MAX_BYTES: Final[int] = 128
# The chain refuses weights more often than its rate limit, and a validator
# that stays silent past the activity cutoff goes inactive, so the keeper
# re-sends the current weights once per chain epoch (tempo).
WEIGHTS_MIN_GAP_BLOCKS: Final[int] = 100
WEIGHTS_REFRESH_BLOCKS: Final[int] = 360
KEEPER_SECONDS: Final[int] = 300
METAGRAPH_TTL_SECONDS: Final[int] = 300
CHAIN_ATTEMPTS: Final[int] = 4
CHAIN_BACKOFF_SECONDS: Final[float] = 2.0

# ---------------------------------------------------------------- services
TESTSET_URL: Final[str] = _env_str("KERSTA_TESTSET_URL", "")
REPORT_URL: Final[str] = _env_str("KERSTA_REPORT_URL", "")
SANDBOX_IMAGE: Final[str] = _env_str("KERSTA_SANDBOX_IMAGE", "ghcr.io/kersta-ai/sandbox:1")
