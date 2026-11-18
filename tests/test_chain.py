from types import SimpleNamespace

import pytest

from kersta.chain.commitments import (
    CommitmentError,
    OnChainCommitment,
    SubmissionCommitment,
    submissions,
)
from kersta.chain.metagraph import snapshot_from
from kersta.chain.offline import OfflineChain
from kersta.chain.sdk import decode_commitment_info
from kersta.chain.weights import U16_MAX, quantise, version_key

REV = "a" * 40
DIGEST = "b" * 64


def test_quantise_sums_to_full_scale_and_is_order_free():
    a = quantise({3: 0.5, 1: 0.25, 2: 0.25})
    b = quantise({2: 0.25, 3: 0.5, 1: 0.25})
    assert a == b
    assert sum(a.values) == U16_MAX
    assert quantise({1: 1, 2: 1, 3: 1}).values == (21845, 21845, 21845)


def test_quantise_drops_zero_and_negative():
    vector = quantise({1: 0.0, 2: -1.0, 3: 1.0})
    assert vector.uids == (3,) and vector.values == (U16_MAX,)
    assert quantise({}).is_empty


def test_version_key():
    assert version_key("1.0.0") == 10000
    assert version_key("2.3.4") == 20304
    with pytest.raises(ValueError):
        version_key("1.0")


def test_submission_commitment_round_trip_and_size():
    c = SubmissionCommitment.for_submission("alice/kersta-detectors", REV, DIGEST)
    text = c.encode()
    assert text == f"ks1|alice/kersta-detectors|{REV}|{'b' * 32}"
    assert len(text.encode()) <= 128
    assert SubmissionCommitment.decode(text) == c
    assert c.covers(DIGEST) and not c.covers("c" * 64)


def test_commitment_refuses_a_repo_too_long_for_the_chain():
    with pytest.raises(CommitmentError):
        SubmissionCommitment.for_submission("u" * 30 + "/" + "n" * 30, REV, DIGEST).encode()


def test_garbage_commitments_decode_to_none():
    for raw in ("", "hello", "ks1|bad repo|x|y", f"ks1|a/b|{REV}|short", "kc1|1|2|3|4|-"):
        assert SubmissionCommitment.decode(raw) is None


def test_submissions_keep_only_those_before_the_deadline_in_commit_order():
    ok = SubmissionCommitment.for_submission("a/b", REV, DIGEST).encode()
    rows = [
        OnChainCommitment("late", 150, ok),
        OnChainCommitment("z", 90, ok),
        OnChainCommitment("a", 90, ok),
        OnChainCommitment("junk", 10, "nonsense"),
    ]
    found = submissions(rows, before_block=100)
    assert [e.hotkey for e in found] == ["a", "z"]


@pytest.mark.parametrize("info", [
    {"fields": [{"Raw5": "0x68656c6c6f"}]},
    {"fields": [[{"Raw5": [104, 101, 108, 108, 111]}]]},
    {"fields": [{"Raw5": [[104, 101, 108, 108, 111]]}]},
    SimpleNamespace(value={"fields": [{"Raw5": b"hello"}]}),
])
def test_commitment_info_decodes_every_shape(info):
    assert decode_commitment_info(info) == "hello"


def test_offline_chain_reads_commitments_as_of_a_block():
    chain = OfflineChain(hotkey="m1")
    chain.pin_block(10)
    chain.commit("first")
    chain.pin_block(20)
    chain.commit("second")
    assert [c.raw for c in chain.commitments(at_block=15)] == ["first"]
    assert [c.raw for c in chain.commitments()] == ["second"]


def test_snapshot_from_v11_neuron_list_and_v10_columns():
    v11 = SimpleNamespace(
        neurons=[SimpleNamespace(uid=0, hotkey="h0", coldkey="c0",
                                 total_stake=SimpleNamespace(tao=12.5), validator_permit=True),
                 SimpleNamespace(uid=1, hotkey="h1", coldkey="c1", total_stake=0)],
        validators=[],
    )
    snap = snapshot_from(v11, netuid=456, block=5)
    assert snap.uid_by_hotkey == {"h0": 0, "h1": 1}
    assert snap.find("h0").stake == 12.5 and snap.find("h0").validator_permit

    v10 = SimpleNamespace(uids=[0, 1], hotkeys=["h0", "h1"], coldkeys=["c0", "c1"],
                          S=[1.0, 2.0], validator_permit=[False, True], block=9)
    snap = snapshot_from(v10, netuid=456)
    assert snap.block == 9 and snap.find("h1").validator_permit
