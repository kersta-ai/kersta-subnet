import math

import numpy as np
import pytest

from kersta.core.lanes import Lane
from kersta.core.testset import CLEAN, DAMAGED, UNSEEN, Answer
from kersta.scoring.lane import ItemResult, score_lane


def _key():
    """Ten items: 4 clean, 4 damaged, 2 unseen fakes; 1 real also in unseen."""
    rows = [
        Answer("r1", "x", 0, (CLEAN, UNSEEN)), Answer("r2", "x", 0, (CLEAN,)),
        Answer("f1", "x", 1, (CLEAN,), "gen_a"), Answer("f2", "x", 1, (CLEAN,), "gen_a"),
        Answer("r3", "x", 0, (DAMAGED,), damage=("jpeg:quality=40",)),
        Answer("r4", "x", 0, (DAMAGED,), damage=("resize:scale=0.5",)),
        Answer("f3", "x", 1, (DAMAGED,), "gen_b", damage=("jpeg:quality=40",)),
        Answer("f4", "x", 1, (DAMAGED,), "gen_b", damage=("resize:scale=0.5",)),
        Answer("u1", "x", 1, (UNSEEN,), "gen_new", seen=False),
        Answer("u2", "x", 1, (UNSEEN,), "gen_new", seen=False),
    ]
    return rows


def _results(answers, p_real=0.1, p_fake=0.9, seconds=0.01, **overrides):
    out = {}
    for a in answers:
        p = p_fake if a.is_fake else p_real
        out[a.id] = ItemResult(a.id, overrides.get(a.id, p), seconds=seconds)
    return out


def test_confident_correct_model_hand_example():
    answers = _key()
    s = score_lane(Lane.AUDIO, answers, _results(answers))
    # every slice: MCC 1 -> 1.0, Brier 0.01 -> calibration 0.96, slice sqrt(0.96)
    assert s.detection == pytest.approx(math.sqrt(0.96))
    assert s.location == 0.0  # no partial edits in this key
    assert s.score == pytest.approx(round(0.85 * math.sqrt(0.96), 6))
    assert s.reasons == []


def test_false_alarms_over_two_percent_halve_the_score():
    answers = _key()
    s = score_lane(Lane.AUDIO, answers, _results(answers, r3=0.7))  # 1 of 4 reals flagged
    assert s.false_alarm_rate == pytest.approx(0.25)
    assert s.score == pytest.approx(round(0.5 * 0.85 * s.detection, 6), abs=1e-6)


def test_slow_model_scores_zero():
    answers = _key()
    s = score_lane(Lane.IMAGE, answers, _results(answers, seconds=0.5))  # 500 ms > 100 ms
    assert s.score == 0.0
    assert any("p95" in r for r in s.reasons)


def test_too_many_failures_score_zero():
    answers = _key()
    results = _results(answers)
    results["f1"] = ItemResult("f1", None, error="crash")
    s = score_lane(Lane.AUDIO, answers, results)  # 1 of 10 = 10% > 1%
    assert s.score == 0.0
    assert s.failures == 1


def test_a_failed_item_under_the_limit_counts_as_half():
    answers = [Answer(f"r{i}", "x", 0, (CLEAN,)) for i in range(100)]
    answers += [Answer(f"f{i}", "x", 1, (CLEAN,)) for i in range(100)]
    results = _results(answers)
    results["f0"] = ItemResult("f0", None, error="crash")  # 1 of 200 = 0.5%
    s = score_lane(Lane.AUDIO, answers, results)
    assert s.score > 0
    assert s.failures == 1


def test_location_for_audio_uses_segment_overlap():
    answers = [
        Answer("r1", "x", 0, (CLEAN,)),
        Answer("p1", "x", 1, (CLEAN,), "gen", partial=True, segments=((1.0, 3.0),)),
    ]
    results = {
        "r1": ItemResult("r1", 0.0, seconds=0.01),
        "p1": ItemResult("p1", 1.0, segments=((2.0, 3.0),), seconds=0.01),
    }
    s = score_lane(Lane.AUDIO, answers, results)
    assert s.location == pytest.approx(0.5)
    assert s.location_items == 1
    assert s.score == pytest.approx(0.85 * 1.0 + 0.15 * 0.5)


def test_location_for_images_uses_the_true_mask_on_the_model_grid():
    answers = [
        Answer("r1", "x", 0, (CLEAN,)),
        Answer("p1", "x", 1, (CLEAN,), "gen", partial=True, mask="m.png"),
    ]
    true = np.zeros((4, 4), bool)
    true[:2] = True
    results = {
        "r1": ItemResult("r1", 0.0, seconds=0.01),
        "p1": ItemResult("p1", 1.0, mask=true.copy(), seconds=0.01),
    }
    s = score_lane(Lane.IMAGE, answers, results, true_mask=lambda a: true)
    assert s.location == 1.0


def test_an_empty_slice_is_left_out_and_weights_renormalise():
    answers = [Answer("r1", "x", 0, (CLEAN,)), Answer("f1", "x", 1, (CLEAN,))]
    s = score_lane(Lane.AUDIO, answers, _results(answers))
    assert set(s.slices) == {"clean"}
    assert s.detection == pytest.approx(math.sqrt(0.96))


def test_breakdown_reports_damage_and_generator_accuracy():
    answers = _key()
    s = score_lane(Lane.AUDIO, answers, _results(answers, f3=0.2))
    assert s.breakdown["by_damage"]["jpeg"]["accuracy"] == 0.5
    assert s.breakdown["by_generator"]["gen_b"]["accuracy"] == 0.5
