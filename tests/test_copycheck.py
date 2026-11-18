import numpy as np

from kersta.scoring.copycheck import Candidate, find_copies

rng = np.random.default_rng(1)
BASE = rng.random(200)


def _c(hotkey, block, score, p, digest=None):
    return Candidate(hotkey, block, digest or hotkey, score, np.asarray(p))


def test_a_later_identical_model_is_a_copy():
    copies = find_copies([_c("orig", 10, 0.8, BASE), _c("copy", 20, 0.8, BASE)])
    assert copies == {"copy": ("orig", 1.0)}


def test_identical_package_digest_is_a_copy_even_with_noise():
    copies = find_copies([_c("orig", 10, 0.8, BASE, "d"),
                          _c("copy", 20, 0.805, BASE + 0.001, "d")])
    assert "copy" in copies


def test_a_correlated_model_that_beats_the_original_by_the_margin_keeps_its_score():
    near = BASE + rng.normal(0, 0.005, BASE.shape)
    assert find_copies([_c("orig", 10, 0.80, BASE), _c("better", 20, 0.811, near)]) == {}


def test_a_correlated_model_that_does_not_beat_by_the_margin_is_a_copy():
    near = BASE + rng.normal(0, 0.005, BASE.shape)
    assert "same" in find_copies([_c("orig", 10, 0.80, BASE), _c("same", 20, 0.805, near)])


def test_an_independent_model_is_not_a_copy():
    assert find_copies([_c("a", 10, 0.8, BASE), _c("b", 20, 0.7, rng.random(200))]) == {}


def test_the_earlier_commit_is_the_original_whatever_the_list_order():
    copies = find_copies([_c("copy", 20, 0.8, BASE), _c("orig", 10, 0.8, BASE)])
    assert copies == {"copy": ("orig", 1.0)}


def test_same_block_ties_break_by_hotkey():
    copies = find_copies([_c("bbb", 10, 0.8, BASE), _c("aaa", 10, 0.8, BASE)])
    assert copies == {"bbb": ("aaa", 1.0)}
