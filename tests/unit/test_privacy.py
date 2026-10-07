"""Membership inference attack (analysis/privacy.py)."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from bm4tc.analysis.privacy import FEATURES, features, membership_inference, worst_case_threshold


def test_features_of_known_probabilities():
    probs = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    labels = torch.tensor([0, 1])
    f = dict(zip(FEATURES, features(probs, labels).T))

    np.testing.assert_allclose(f["max_prob"], [0.9, 0.5], atol=1e-6)
    np.testing.assert_allclose(f["correct_prob"], [0.9, 0.5], atol=1e-6)
    np.testing.assert_allclose(f["loss"], [-np.log(0.9), -np.log(0.5)], atol=1e-6)
    np.testing.assert_allclose(f["margin"], [0.8, 0.0], atol=1e-6)
    h = -(0.9 * np.log(0.9) + 0.1 * np.log(0.1))
    np.testing.assert_allclose(f["entropy"], [h, np.log(2)], atol=1e-6)
    np.testing.assert_allclose(f["modified_entropy"], [1 - h / np.log(2), 0.0], atol=1e-6)


def test_worst_case_threshold_separable_and_identical():
    rng = np.random.default_rng(0)
    members = rng.uniform(0.6, 1.0, size=(50, len(FEATURES)))
    non_members = rng.uniform(0.0, 0.4, size=(50, len(FEATURES)))
    wc = worst_case_threshold(members, non_members)
    # Higher-is-member features separate perfectly; for loss/entropy the sign flip
    # turns the same data into the opposite ordering, which the attack cannot exploit.
    for name in FEATURES:
        expected = 0.5 if name in ("loss", "entropy") else 1.0
        assert wc[name] == pytest.approx(expected, abs=0.02), name

    same = np.ones((10, len(FEATURES)))
    assert all(v == pytest.approx(0.5) for v in worst_case_threshold(same, same).values())


@pytest.mark.slow
def test_membership_inference_runs_end_to_end(cbm):
    torch.manual_seed(0)
    lo, hi = cbm.input_range

    def loader(n):
        x = lo + (hi - lo) * torch.rand(n, 4)
        return DataLoader(TensorDataset(x, torch.randint(0, 2, (n,))), batch_size=16)

    result = membership_inference(cbm, loader(48), loader(32), "cpu", adv_eps_abs=0.05,
                                  adv_num_steps=2)
    assert 0.0 <= result.attack_accuracy <= 1.0
    assert 0.0 <= result.auc_roc <= 1.0
    assert set(result.worst_case) == set(FEATURES)
    assert set(result.adversarial_worst_case) == set(FEATURES)
    assert all(0.5 <= v <= 1.0 for v in result.worst_case.values())
