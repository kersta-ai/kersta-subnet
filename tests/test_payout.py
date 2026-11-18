import math

import pytest

from kersta.scoring import payout as p


def test_rank_curve_matches_the_spec_figures():
    curve = p.rank_curve(20)
    assert math.fsum(curve) == pytest.approx(1.0)
    assert curve[0] == pytest.approx(0.2023, abs=1e-3)  # "#1 about 20%"
    assert curve[4] == pytest.approx(0.0829, abs=1e-3)  # "#5 about 8%"
    assert curve[9] == pytest.approx(0.0272, abs=1e-3)  # "#10 about 3%"
    assert curve[1] / curve[0] == pytest.approx(0.8)


def test_rank_curve_renormalises_when_fewer_are_paid():
    assert p.rank_curve(3) == pytest.approx([1 / 2.44, 0.8 / 2.44, 0.64 / 2.44])
    assert p.rank_curve(0) == []
    assert len(p.rank_curve(50)) == 20


def test_only_positive_scores_are_paid_and_ties_go_to_the_earlier_commit():
    standings = [
        p.Standing("late", 0.8, commit_block=200),
        p.Standing("early", 0.8, commit_block=100),
        p.Standing("zero", 0.0, commit_block=1),
    ]
    shares = p.lane_shares(standings)
    assert list(shares) == ["early", "late"]
    assert shares["early"] == pytest.approx(1 / 1.8)


def test_blend_seventy_thirty():
    shares, burned = p.blend({"a": 1.0}, {"b": 1.0})
    assert shares == pytest.approx({"a": 0.7, "b": 0.3})
    assert burned == 0.0


def test_blend_without_history_pays_this_round_in_full():
    assert p.blend({"a": 1.0}, None) == ({"a": 1.0}, 0.0)


def test_blend_with_nobody_this_round_burns_seventy_percent():
    shares, burned = p.blend({}, {"b": 1.0})
    assert shares == pytest.approx({"b": 0.3})
    assert burned == pytest.approx(0.7)
    assert p.blend({}, None) == ({}, 1.0)


def test_combine_splits_lanes_and_burns_empty_ones():
    this = {"image": {"a": 1.0}, "audio": {"b": 0.6, "c": 0.4}}
    out = p.combine(this)
    assert out.weights["a"] == pytest.approx(1 / 3)
    assert out.weights["b"] == pytest.approx(0.2)
    assert out.burned == {"video": pytest.approx(1 / 3)}
    assert math.fsum(out.weights.values()) + out.burned_total == pytest.approx(1.0)


def test_combine_sums_to_one_with_smoothing():
    this = {"image": {"a": 0.6, "b": 0.4}, "video": {}, "audio": {"c": 1.0}}
    last = {"image": {"b": 1.0}, "video": {"d": 1.0}, "audio": {}}
    out = p.combine(this, last)
    assert out.weights["a"] == pytest.approx(0.7 * 0.6 / 3)
    assert out.weights["b"] == pytest.approx((0.7 * 0.4 + 0.3) / 3)
    assert out.weights["d"] == pytest.approx(0.3 / 3)
    assert out.burned["video"] == pytest.approx(0.7 / 3)
    assert math.fsum(out.weights.values()) + out.burned_total == pytest.approx(1.0)
