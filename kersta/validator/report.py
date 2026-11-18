"""Signed competition reports and the leaderboard built from them.

Every validator publishes the same facts for a competition: who was scored,
every lane score with its breakdown, the payout and the weights. Anyone can
recompute a report from the test set (published after the competition ends)
and the public packages; a validator that disagrees is visible.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from kersta.chain.wallet import Signer
from kersta.core.constants import MECHANISM_VERSION, RELEASE_VERSION
from kersta.core.hashing import canonical_json

log = logging.getLogger("kersta.validator.report")
FORMAT = "kersta-report/1"


def sign(body: Mapping[str, Any], signer: Signer) -> dict[str, Any]:
    out = dict(body)
    out["validator"] = signer.address
    out["signature"] = signer.sign(canonical_json({k: v for k, v in out.items()
                                                   if k != "signature"}))
    return out


def build(
    *,
    competition: int,
    netuid: int,
    deadline_block: int,
    testset_digest: str,
    lanes: Mapping[str, list[dict[str, Any]]],
    failures: Mapping[str, str],
    payout: Mapping[str, Any],
    weights: Mapping[str, float],
    burn_hotkey: str,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "competition": competition,
        "netuid": netuid,
        "mechanism_version": MECHANISM_VERSION,
        "release": RELEASE_VERSION,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "deadline_block": deadline_block,
        "testset_digest": testset_digest,
        "lanes": {k: list(v) for k, v in lanes.items()},
        "failures": dict(failures),
        "payout": dict(payout),
        "weights": dict(sorted(weights.items())),
        "burn_hotkey": burn_hotkey,
    }


def leaderboard(report: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The public table: per lane, rank order, the headline numbers only."""
    board: dict[str, list[dict[str, Any]]] = {}
    for lane, rows in (report.get("lanes") or {}).items():
        ranked = sorted(rows, key=lambda r: (r.get("rank") or 10**9, -float(r.get("score", 0))))
        board[lane] = [
            {
                "rank": r.get("rank"),
                "hotkey": r.get("hotkey"),
                "score": r.get("score"),
                "detection": r.get("detection"),
                "location": r.get("location"),
                "damaged": (r.get("slices") or {}).get("damaged", {}).get("score"),
                "unseen": (r.get("slices") or {}).get("unseen", {}).get("score"),
                "false_alarm_rate": r.get("false_alarm_rate"),
                "p95_ms": r.get("p95_ms"),
                "share": r.get("share"),
            }
            for r in ranked
        ]
    return board


def post(url: str, report: Mapping[str, Any]) -> bool:
    if not url:
        return False
    try:
        response = httpx.post(url, json=dict(report), timeout=30.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        log.warning("report not accepted by %s: %s", url, exc)
        return False
