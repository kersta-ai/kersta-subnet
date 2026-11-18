import pytest

from kersta.chain.commitments import CompetitionCommitment, OnChainCommitment, announced
from kersta.core.competition import Competition, CompetitionError, approx
from kersta.miner.submit import entry_line

DIGEST = "c" * 64


def test_phases_follow_the_announced_blocks():
    c = Competition(4, deadline=1000, reveal=2800, end=6400)
    assert c.phase(999) == "commit"
    assert c.phase(1000) == "upload"
    assert c.phase(2800) == "scoring"
    assert c.phase(6400) == "settled"


def test_a_commitment_enters_only_if_written_before_the_deadline_block():
    c = Competition(0, 1000, 2000, 3000)
    assert c.accepts(999) and not c.accepts(1000)


def test_blocks_must_be_in_order():
    for blocks in ((0, 10, 20), (10, 10, 20), (10, 30, 20)):
        with pytest.raises(CompetitionError):
            Competition(1, *blocks)


def test_announcement_and_lock_round_trip():
    c = Competition(7, 120_000, 121_800, 125_400)
    announced_only = CompetitionCommitment.announce(c)
    assert announced_only.encode() == "kc1|7|120000|121800|125400|-"
    assert not announced_only.locked
    lock = CompetitionCommitment.lock(c, DIGEST)
    text = lock.encode()
    assert text == f"kc1|7|120000|121800|125400|{'c' * 32}"
    assert len(text.encode()) <= 128
    decoded = CompetitionCommitment.decode(text)
    assert decoded == lock and decoded.covers(DIGEST)


@pytest.mark.parametrize("raw", ["", "kc1|7|1|2|3", "kc1|x|1|2|3|-", "kc1|1|30|20|10|-",
                                 "kc1|1|10|20|30|short"])
def test_malformed_competitions_decode_to_none(raw):
    assert CompetitionCommitment.decode(raw) is None


def test_announced_reads_only_the_owner_hotkey():
    c = Competition(1, 100, 200, 300)
    rows = [OnChainCommitment("someone", 5, CompetitionCommitment.announce(c).encode()),
            OnChainCommitment("owner", 6, CompetitionCommitment.lock(c, DIGEST).encode())]
    assert announced(rows, "owner").locked
    assert announced(rows, "nobody") is None


def test_entry_line_names_the_competition_a_commitment_enters():
    c = Competition(2, 1000, 2000, 3000)
    assert "enters competition 2" in entry_line(c, 500, 600)
    assert "next one" in entry_line(c, 1500, 1600)
    assert "next competition" in entry_line(None, 10, 10)


def test_approx_is_for_display_only():
    assert approx(5) == "about 1 min"
    assert approx(1800) == "about 6.0 h"
    assert approx(50_400) == "about 7.0 days"
