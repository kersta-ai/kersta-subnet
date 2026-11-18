"""Scoring one whole competition. Deterministic given the chain and the test set.

    1. read every commitment as it stood just before the deadline block
    2. find the owner's lock in that same snapshot; fetch and verify the test set
    3. fetch and check every miner's package at the committed revision
    4. run each lane package in the sandbox on its lane, score it
    5. copy check per lane, rank, top 20 shares
    6. blend with the previous competition (70/30), split lanes, burn what nobody earned
    7. sign and save the report; the weights wait for the end block
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from kersta.chain.client import Chain
from kersta.chain.commitments import OnChainCommitment, announced, submissions
from kersta.chain.wallet import Signer
from kersta.core.competition import Competition
from kersta.core.package import Submission
from kersta.core.testset import TestsetError, lane_answers
from kersta.sandbox.runner import Launcher
from kersta.scoring import payout
from kersta.scoring.copycheck import Candidate, find_copies
from kersta.validator import report as reporting
from kersta.validator.config import ValidatorConfig
from kersta.validator.evaluate import Evaluation, evaluate, item_rows
from kersta.validator.fetcher import Fetcher, FetchError
from kersta.validator.inputs import InputCache
from kersta.validator.state import State
from kersta.validator.testsets import TestsetStore

log = logging.getLogger("kersta.validator.competition")


class ScoringError(RuntimeError):
    pass


@dataclass(slots=True)
class Context:
    config: ValidatorConfig
    chain: Chain
    signer: Signer
    state: State
    store: TestsetStore
    fetcher: Fetcher
    launcher: Launcher


@dataclass(slots=True)
class Result:
    competition: Competition
    weights: dict[str, float]
    owner: str
    report: dict[str, Any]
    evaluations: dict[str, list[Evaluation]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def roster(ctx: Context, competition: Competition) -> dict[str, Any]:
    """The commitment snapshot just before the deadline block, read once and kept."""
    kept = ctx.state.roster(competition.index)
    if kept is not None:
        return kept
    if ctx.chain.block() < competition.deadline:
        raise ScoringError(f"competition {competition.index} has not reached its deadline block")
    rows = ctx.chain.commitments(at_block=competition.deadline - 1)
    found = {
        "competition": competition.to_dict(),
        "owner": ctx.chain.owner_hotkey(),
        "rows": [{"hotkey": r.hotkey, "block": r.block, "raw": r.raw} for r in rows],
    }
    ctx.state.save_roster(competition.index, found)
    return found


def locked_digest(rows: list[OnChainCommitment], owner: str, competition: Competition) -> str | None:
    """The test set digest the owner locked for exactly this competition before its deadline."""
    lock = announced(rows, owner)
    if lock is None or not lock.locked or lock.competition != competition:
        return None
    return lock.digest


def score_competition(ctx: Context, competition: Competition) -> Result:
    index = competition.index
    snap = roster(ctx, competition)
    owner = str(snap["owner"])
    rows = [OnChainCommitment(r["hotkey"], int(r["block"]), r["raw"]) for r in snap["rows"]]
    try:
        manifest, root = ctx.store.fetch(index, owner=owner,
                                         locked_digest=locked_digest(rows, owner, competition))
    except (TestsetError, OSError) as exc:
        raise ScoringError(f"competition {index} test set unusable: {exc}") from exc

    registered = ctx.chain.snapshot(refresh=True).uid_by_hotkey
    entries = [e for e in submissions(rows, competition.deadline) if e.hotkey != owner]
    failures: dict[str, str] = {}
    packages: dict[str, tuple[Submission, int]] = {}
    for entry in entries:
        if entry.hotkey not in registered:
            failures[entry.hotkey] = "not registered on the subnet now"
            continue
        try:
            packages[entry.hotkey] = (ctx.fetcher.fetch(entry.commitment), entry.block)
        except (FetchError, OSError) as exc:
            failures[entry.hotkey] = str(exc)[:500]
            log.warning("%s: %s", entry.hotkey, exc)
    log.info("competition %d: %d commitments, %d packages ready", index, len(entries), len(packages))

    evaluations: dict[str, list[Evaluation]] = {}
    shares: dict[str, dict[str, float]] = {}
    lane_rows: dict[str, list[dict[str, Any]]] = {}
    for lane in manifest.lane_ids:
        answers = lane_answers(root, manifest, lane)
        if ctx.config.item_limit:
            answers = answers[: ctx.config.item_limit]
        inputs = InputCache(root, ctx.config.cache_dir / "inputs" / f"competition-{index}", lane)
        done: list[Evaluation] = []
        for hotkey, (submission, block) in sorted(packages.items(), key=lambda kv: (kv[1][1], kv[0])):
            package = submission.lanes.get(lane)
            if package is None:
                continue
            log.info("competition %d %s: running %s", index, lane.value, hotkey)
            evaluation = evaluate(ctx.launcher, package, answers, inputs, hotkey=hotkey,
                                  commit_block=block)
            ctx.state.save_items(index, hotkey, lane.value, item_rows(evaluation, answers))
            done.append(evaluation)

        copies = find_copies([
            Candidate(e.hotkey, e.commit_block, e.lane_digest, e.score.score, e.p_fake)
            for e in done if e.ran
        ])
        for evaluation in done:
            if evaluation.hotkey in copies:
                original, corr = copies[evaluation.hotkey]
                evaluation.score.copy_of = original
                evaluation.score.zero(f"copy of {original} (correlation {corr})")
        standings = [payout.Standing(e.hotkey, e.score.score, e.commit_block) for e in done]
        shares[lane.value] = payout.lane_shares(standings)
        ranks = {s.hotkey: i + 1 for i, s in enumerate(payout.ranked(standings))}
        lane_rows[lane.value] = [
            {**e.score.to_dict(), "uid": registered.get(e.hotkey), "rank": ranks.get(e.hotkey),
             "share": round(shares[lane.value].get(e.hotkey, 0.0), 9),
             "commit_block": e.commit_block, "lane_digest": e.lane_digest,
             "interface": e.run.interface, "load_seconds": e.run.load_seconds}
            for e in done
        ]
        evaluations[lane.value] = done

    ctx.state.save_shares(index, shares)
    previous = ctx.state.shares(index - 1) if index > 0 else None
    paid = payout.combine(shares, previous)
    weights = dict(paid.weights)
    if paid.burned_total > 0:
        if owner:
            weights[owner] = weights.get(owner, 0.0) + paid.burned_total
        else:
            log.error("the owner hotkey is unknown; %.4f of emission cannot be burned",
                      paid.burned_total)
    body = reporting.build(
        competition=index, netuid=ctx.chain.netuid, deadline_block=competition.deadline,
        testset_digest=manifest.digest, lanes=lane_rows, failures=failures,
        payout={**paid.to_dict(), "smoothed_with_previous": previous is not None},
        weights=weights, burn_hotkey=owner,
    )
    body["blocks"] = competition.to_dict()
    signed = reporting.sign(body, ctx.signer)
    ctx.state.save_scores(index, {"lanes": lane_rows, "failures": failures})
    ctx.state.save_report(index, signed)
    if not ctx.config.dry_run:
        reporting.post(ctx.config.report_url, signed)
    return Result(competition, weights, owner, signed, evaluations, failures)
