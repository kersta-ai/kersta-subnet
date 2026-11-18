"""Whole competitions on the offline chain, timed in blocks only.

The owner builds a test set and locks it with the competition's blocks;
three miners commit (one copies another); validators score at the reveal
block, set weights at the end block, and carry 70/30 smoothing into the
next competition the owner announces.
"""

import dataclasses
import math
import random
import shutil
from pathlib import Path

import pytest

from kersta.chain.commitments import SubmissionCommitment
from kersta.chain.config import ChainConfig
from kersta.chain.offline import OfflineChain
from kersta.chain.wallet import DevSigner
from kersta.core.competition import Competition
from kersta.core.package import scan_submission
from kersta.owner.factory import FactoryConfig, build
from kersta.owner.publish import announce, lock, previous_on_chain, publish_to_folder, sign_competition
from kersta.validator import loop
from kersta.validator.competition import ScoringError, score_competition
from kersta.validator.config import ValidatorConfig
from tests.helpers import make_image_package, make_image_sources, random_hex

FIRST = Competition(0, deadline=1000, reveal=2800, end=6400)
SECOND = Competition(1, deadline=9000, reveal=10_800, end=14_400)


def _publish_package(packages: Path, repo: str, root: Path, rng: random.Random):
    revision = random_hex(rng, 40)
    shutil.copytree(root, packages / repo / revision)
    return revision, scan_submission(root).digest


def _lock_competition(tmp_path, chain, competition, sources, store):
    # Fixed seeds keep the small synthetic sets, and so the expected ranking, reproducible.
    seed = "secret" if competition.index == 0 else f"secret-{competition.index}"
    built, notes = build(sources, tmp_path / "built", FactoryConfig(
        competition=competition.index, seed=seed, items=60, allow_reuse=True))
    owner = chain.as_hotkey("owner")
    manifest = sign_competition(built, competition, DevSigner("owner"), {"lanes": notes["lanes"]},
                                previous_on_chain(owner, competition.index))
    lock(owner, competition, manifest)
    publish_to_folder(built, store)


def _validator(tmp_path, chain, store, packages, name="validator"):
    config = ValidatorConfig(home=tmp_path / name, chain=ChainConfig(network="offline"),
                             testset_url=str(store), sandbox="local")
    return loop.build_context(config, chain.as_hotkey(name), signer=DevSigner(name),
                              local_packages=packages)


@pytest.fixture()
def world(tmp_path):
    rng = random.Random(7)
    chain = OfflineChain(tmp_path / "chain.json")
    for hotkey in ("owner", "validator", "late-validator", "alice", "bob", "carol"):
        chain.register(hotkey, validator=hotkey in ("owner", "validator", "late-validator"))
    chain.set_owner("owner")
    sources = make_image_sources(tmp_path / "sources", per_pool=24)
    store = tmp_path / "store"

    # the owner announces competition 0 early, then locks its test set before the deadline
    chain.pin_block(5)
    announce(chain.as_hotkey("owner"), FIRST)
    chain.pin_block(10)
    _lock_competition(tmp_path, chain, FIRST, sources, store)

    # miners: alice is good, bob copies alice later, carol is weaker and gives no mask
    packages = tmp_path / "packages"
    good = make_image_package(tmp_path / "alice")
    weak = make_image_package(tmp_path / "carol", name="weak", centre=0.62, sharpness=6.0,
                              use_mask=False)
    for block, hotkey, root in ((20, "alice", good), (30, "carol", weak), (40, "bob", good)):
        revision, digest = _publish_package(packages, f"{hotkey}/kersta", root, rng)
        chain.pin_block(block)
        chain.as_hotkey(hotkey).commit(
            SubmissionCommitment.for_submission(f"{hotkey}/kersta", revision, digest).encode())

    # a commitment in or after the deadline block does not enter competition 0
    chain.pin_block(FIRST.deadline)
    chain.as_hotkey("carol").commit(
        SubmissionCommitment.for_submission("carol/other", "c" * 40, "d" * 64).encode())

    chain.pin_block(FIRST.reveal)
    ctx = _validator(tmp_path, chain, store, packages)
    return {"ctx": ctx, "chain": chain, "tmp": tmp_path, "sources": sources, "store": store,
            "packages": packages}


def test_a_competition_scores_flags_the_copy_and_burns_empty_lanes(world):
    ctx = world["ctx"]
    loop.observe(ctx)
    result = score_competition(ctx, FIRST)
    image = {row["hotkey"]: row for row in result.report["lanes"]["image"]}

    assert image["alice"]["rank"] == 1 and image["alice"]["score"] > image["carol"]["score"] > 0
    assert image["alice"]["location"] > 0.5 and image["carol"]["location"] == 0.0
    assert image["bob"]["copy_of"] == "alice" and image["bob"]["score"] == 0.0
    # carol's later commitment did not replace her entry for this competition
    assert image["carol"]["lane_digest"] != ""

    # image pays alice and carol; video and audio have no test set and burn to the owner
    assert set(result.weights) == {"alice", "carol", "owner"}
    assert result.weights["owner"] == pytest.approx(2 / 3, abs=1e-6)
    assert math.fsum(result.weights.values()) == pytest.approx(1.0)
    assert result.report["blocks"] == FIRST.to_dict()


def test_the_loop_scores_at_the_reveal_block_and_sets_weights_at_the_end_block(world):
    ctx, chain = world["ctx"], world["chain"]
    loop.tick(ctx)
    assert ctx.state.scored(0)
    assert chain.weights_of("validator") == {}
    chain.pin_block(FIRST.end)
    loop.tick(ctx)
    sent = chain.weights_of("validator")
    uid = chain.snapshot().uid_by_hotkey
    assert sum(sent.values()) == 65535
    assert sent[uid["alice"]] > sent[uid["carol"]] > 0
    assert uid["bob"] not in sent


def test_the_next_competition_is_smoothed_with_the_previous_one(world):
    ctx, chain = world["ctx"], world["chain"]
    loop.tick(ctx)
    chain.pin_block(FIRST.end)
    loop.tick(ctx)

    # the owner decides when the next competition runs, and locks it in blocks
    chain.pin_block(8000)
    _lock_competition(world["tmp"], chain, SECOND, world["sources"], world["store"])
    chain.pin_block(SECOND.reveal)
    loop.tick(ctx)
    report = ctx.state.report(1)
    assert report["payout"]["smoothed_with_previous"] is True

    # a validator that joins late rebuilds the previous competition from the signed manifest
    late = _validator(world["tmp"], chain, world["store"], world["packages"], "late-validator")
    loop.tick(late)
    assert late.state.scored(0) and late.state.scored(1)
    assert late.state.report(1)["weights"] == pytest.approx(report["weights"])


def test_a_competition_without_a_lock_before_its_deadline_is_not_scored(world, tmp_path):
    ctx, chain = world["ctx"], world["chain"]
    unlocked = Competition(5, deadline=20_000, reveal=21_800, end=25_400)
    chain.pin_block(19_000)
    announce(chain.as_hotkey("owner"), unlocked)
    chain.pin_block(unlocked.reveal)
    with pytest.raises(ScoringError, match="did not lock"):
        score_competition(ctx, unlocked)


def test_a_tampered_test_set_is_refused(world):
    ctx = world["ctx"]
    answers = world["store"] / "competition-0" / "image" / "answers.jsonl"
    answers.write_text(answers.read_text().replace('"label": 1', '"label": 0', 1))
    fresh = dataclasses.replace(ctx, store=type(ctx.store)(
        ctx.config.testset_url, ctx.config.cache_dir / "fresh", allow_dev=True))
    with pytest.raises(ScoringError, match="does not match"):
        score_competition(fresh, FIRST)
