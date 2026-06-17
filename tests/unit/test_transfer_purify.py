"""Unit tests for the pure helpers of analysis.visualize.mnist_transfer_purify.

These target the prediction-array logic only (transfer filtering + outcome
categorisation); no model, no GPU, so they run fast and unmarked.
"""

import numpy as np

from analysis.visualize.mnist_transfer_purify import _categorize, _transfer_mask


def test_transfer_mask_keeps_only_fool_both_with_clean_correct():
    labels = np.array([0, 1, 2, 3, 4])
    src_clean = np.array([0, 1, 2, 9, 4])  # idx 3: source misclassifies clean -> exclude
    tgt_clean = np.array([0, 1, 2, 3, 9])  # idx 4: target misclassifies clean -> exclude
    src_adv = np.array([7, 1, 8, 7, 7])    # idx 1: source still correct on adv -> exclude
    tgt_adv = np.array([7, 7, 8, 7, 7])
    mask = _transfer_mask(src_clean, tgt_clean, src_adv, tgt_adv, labels)
    # Only idx 0 and 2 satisfy: both clean-correct AND both adv-wrong.
    assert mask.tolist() == [True, False, True, False, False]


def test_transfer_mask_all_excluded_when_clean_wrong():
    labels = np.array([5, 6])
    # Source never classifies clean correctly -> nothing transfers.
    mask = _transfer_mask(np.array([0, 0]), labels, np.array([0, 0]), np.array([0, 0]), labels)
    assert mask.tolist() == [False, False]


def test_categorize_all_four_outcomes():
    src = np.array([True, True, False, False])
    tgt = np.array([True, False, True, False])
    cats = _categorize(src, tgt)
    assert cats.tolist() == ["both", "source_only", "target_only", "neither"]


def test_categorize_accepts_python_lists():
    cats = _categorize([True, False], [False, False])
    assert cats.tolist() == ["source_only", "neither"]
