"""Every formula against a small example worked out by hand."""

import math

import numpy as np
import pytest

from kersta.scoring import metrics as m


def test_confusion_counts_p_at_threshold_as_fake():
    # p = 0.5 is called fake ("at or above 0.5")
    assert m.confusion([1, 0, 1, 0], [0.5, 0.5, 0.49, 0.1]) == (1, 1, 1, 1)


def test_mcc_balanced_mistakes_is_zero_and_rescales_to_half():
    labels, probs = [1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1]  # tp=1 fn=1 fp=1 tn=1
    assert m.mcc(labels, probs) == 0.0
    assert m.rescaled_mcc(labels, probs) == 0.5


def test_mcc_perfect_and_inverted():
    assert m.mcc([1, 0], [1.0, 0.0]) == 1.0
    assert m.mcc([1, 0], [0.0, 1.0]) == -1.0
    assert m.rescaled_mcc([1, 0], [0.0, 1.0]) == 0.0


def test_mcc_is_zero_when_everything_is_called_one_class():
    assert m.mcc([1, 0, 1], [0.9, 0.9, 0.9]) == 0.0


def test_mcc_hand_example():
    # tp=2 fp=1 tn=3 fn=0: (2*3 - 1*0) / sqrt(3*2*4*3) = 6 / sqrt(72)
    labels = [1, 1, 0, 0, 0, 0]
    probs = [0.8, 0.7, 0.6, 0.2, 0.1, 0.3]
    assert m.mcc(labels, probs) == pytest.approx(6 / math.sqrt(72))


def test_brier_and_calibration():
    # ((0.8-1)^2 + (0.3-0)^2) / 2 = 0.065 ; 1 - 0.065/0.25 = 0.74
    assert m.brier([1, 0], [0.8, 0.3]) == pytest.approx(0.065)
    assert m.calibration([1, 0], [0.8, 0.3]) == pytest.approx(0.74)


def test_calibration_floors_at_zero():
    assert m.calibration([1, 0], [0.0, 1.0]) == 0.0


def test_slice_score_is_sqrt_of_product():
    # MCC 1 -> 1.0 ; Brier (0.01+0.04+0.04+0.01)/4 = 0.025 -> calibration 0.9
    s = m.slice_score([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert s.mcc_rescaled == 1.0
    assert s.calibration == pytest.approx(0.9)
    assert s.score == pytest.approx(math.sqrt(0.9))


def test_false_alarm_rate_counts_reals_at_or_above_half():
    assert m.false_alarm_rate([0, 0, 0, 1], [0.5, 0.2, 0.1, 0.9]) == pytest.approx(1 / 3)
    assert m.false_alarm_rate([1, 1], [0.9, 0.9]) == 0.0


def test_mask_iou_hand_example():
    pred = np.zeros((4, 4), bool)
    pred[:2, :2] = True  # 4 px
    true = np.zeros((4, 4), bool)
    true[0, :] = True  # 4 px, 2 shared
    assert m.mask_iou(pred, true) == pytest.approx(2 / 6)
    assert m.mask_iou(None, true) == 0.0


def test_interval_iou_hand_examples():
    assert m.interval_iou([[0, 2]], [[1, 3]]) == pytest.approx(1 / 3)
    assert m.interval_iou([[0, 1], [0.5, 2]], [[0, 2]]) == pytest.approx(1.0)
    assert m.interval_iou([[0, 1], [2, 3]], [[0.5, 2.5]]) == pytest.approx(1.0 / 3.0)
    assert m.interval_iou(None, [[0, 1]]) == 0.0
    assert m.interval_iou([], [[0, 1]]) == 0.0


def test_percentile_matches_numpy_linear():
    values = [10, 20, 30, 40, 1000]
    assert m.percentile(values, 95) == pytest.approx(np.percentile(values, 95))


def test_pearson_edge_cases():
    assert m.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert m.pearson([0.5, 0.5], [0.5, 0.5]) == 1.0
    assert m.pearson([0.5, 0.5], [0.1, 0.9]) == 0.0
