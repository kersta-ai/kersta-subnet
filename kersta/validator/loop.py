"""The long running validator loop. It follows block heights only.

Each pass:
  * read the owner's commitment and remember the competition it announces;
  * take the newest known competition whose reveal block has passed; if this
    validator has not scored it, score it (first scoring the previous
    competition when its shares are missing, so 70/30 smoothing matches
    every other validator);
  * once that competition's end block has passed, send its weights; until
    the next competition ends, the keeper re-sends them every chain epoch.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from kersta.chain.client import Chain
from kersta.chain.commitments import announced
from kersta.chain.wallet import DevSigner, Signer, WalletSigner
from kersta.core.competition import Competition, CompetitionError
from kersta.core.testset import TestsetError
from kersta.sandbox.runner import DockerLauncher, Launcher, LocalLauncher
from kersta.validator.competition import Context, ScoringError, score_competition
from kersta.validator.config import ValidatorConfig
from kersta.validator.fetcher import Fetcher, HuggingFaceSource, LocalSource
from kersta.validator.state import State, read_json, write_json
from kersta.validator.testsets import TestsetStore
from kersta.validator.weights import Keeper, submit

log = logging.getLogger("kersta.validator.loop")
PENDING = "pending_weights.json"


def build_context(config: ValidatorConfig, chain: Chain, *, wallet=None, signer: Signer | None = None,
                  local_packages: Path | None = None, launcher: Launcher | None = None) -> Context:
    config.check()
    if signer is None:
        signer = DevSigner(f"dev-validator-{chain.netuid}") if wallet is None else WalletSigner(wallet)
    source = LocalSource(local_packages) if local_packages else HuggingFaceSource(config.hf_token)
    if launcher is None:
        launcher = (DockerLauncher(config.sandbox_image, config.gpu, config.memory, config.cpus)
                    if config.sandbox == "docker" else LocalLauncher())
    return Context(
        config=config,
        chain=chain,
        signer=signer,
        state=State(config.state_dir),
        store=TestsetStore(config.testset_url, config.cache_dir / "testsets",
                           token=config.testset_token, allow_dev=config.chain.offline),
        fetcher=Fetcher(source, config.cache_dir / "packages"),
        launcher=launcher,
    )


def observe(ctx: Context) -> Competition | None:
    """Remember the competition the owner currently announces on chain."""
    found = announced(ctx.chain.commitments(), ctx.chain.owner_hotkey())
    if found is None:
        return None
    ctx.state.remember(found.competition)
    return found.competition


def previous_of(ctx: Context, competition: Competition) -> Competition | None:
    """The competition before this one, from local memory or the signed manifest."""
    known = ctx.state.competition(competition.index - 1)
    if known is not None or competition.index == 0:
        return known
    try:
        notes = ctx.store.manifest(competition.index, owner=ctx.chain.owner_hotkey()).notes
        previous = Competition.from_dict(notes["previous"])
    except (TestsetError, OSError, KeyError, TypeError, ValueError, CompetitionError):
        return None
    ctx.state.remember(previous)
    return previous


def tick(ctx: Context) -> None:
    block = ctx.chain.block()
    upcoming = observe(ctx)
    revealed = [c for c in ctx.state.known() if block >= c.reveal]
    if not revealed:
        if upcoming is None:
            log.info("block %d: no competition announced yet", block)
        else:
            log.info("block %d: competition %d is in its %s phase (deadline block %d)", block,
                     upcoming.index, upcoming.phase(block), upcoming.deadline)
        return
    target = revealed[-1]

    pending_path = ctx.state.root / PENDING
    if not ctx.state.scored(target.index):
        previous = previous_of(ctx, target)
        if previous is not None and ctx.state.shares(previous.index) is None:
            try:
                log.info("no shares for competition %d; scoring it first for smoothing",
                         previous.index)
                score_competition(ctx, previous)
            except ScoringError as exc:
                log.warning("competition %d could not be scored; smoothing starts fresh: %s",
                            previous.index, exc)
        result = score_competition(ctx, target)
        write_json(pending_path, {"competition": target.index, "end": target.end,
                                  "weights": result.weights, "owner": result.owner})

    pending = read_json(pending_path)
    if pending and int(pending["competition"]) == target.index:
        if block < target.end:
            log.info("competition %d scored; weights go out at block %d", target.index, target.end)
        elif ctx.config.dry_run:
            log.info("dry run: competition %d weights not sent", target.index)
            pending_path.unlink(missing_ok=True)
        else:
            ok, _ = submit(ctx.chain, ctx.state, pending["weights"], competition=target.index,
                           owner=str(pending.get("owner") or ""), reason="competition ended")
            if ok:
                pending_path.unlink(missing_ok=True)


def run(ctx: Context, *, once: bool = False) -> int:
    keeper = Keeper(ctx.chain, ctx.state)
    if not once and not ctx.config.dry_run:
        keeper.start()
    while True:
        try:
            tick(ctx)
        except ScoringError as exc:
            log.error("%s", exc)
        except Exception:  # noqa: BLE001 - log and keep the validator alive
            log.exception("validator pass failed")
        if once:
            return 0
        time.sleep(max(15, ctx.config.poll_seconds))
