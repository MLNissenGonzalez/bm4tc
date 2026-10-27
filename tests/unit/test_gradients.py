"""The first-order steepness (D169): a linear model whose margin gradient is known, and
the MPS through the model interface."""
import torch

from bm4tc.analysis.gradients import margin_sensitivity


class _Linear:
    """log p(x, c) = x · W[:, c]: the margin of class 0 against class 1 has gradient
    W[:, 0] − W[:, 1]."""
    input_range = (-1.0, 1.0)
    out_dim = 2

    def __init__(self, w):
        self.w = w

    def log_joint(self, x):
        return x @ self.w

    def log_normalizer(self):
        return 0.0

    def reset(self):
        pass


def test_gain_counts_features_at_the_ends_only_inwards():
    # g = W[:, 0] − W[:, 1] = (2, −3, 5): feature 0 at lo can move up (g > 0 raises the
    # margin: no gain), feature 1 at hi can move down (g < 0: moving down raises it: no
    # gain), feature 2 inside counts |g| = 5. Swap the ends and both count.
    w = torch.tensor([[2.0, 0.0], [0.0, 3.0], [5.0, 0.0]])
    x = torch.tensor([[-1.0, 1.0, 0.2], [1.0, -1.0, 0.2]])
    s = margin_sensitivity(_Linear(w), x, torch.zeros(2, dtype=torch.long), 1, "cpu")
    torch.testing.assert_close(s.margin, x @ (w[:, 0] - w[:, 1]))
    torch.testing.assert_close(s.gain, torch.tensor([5.0, 10.0]))
    torch.testing.assert_close(s.gain_ends, torch.tensor([0.0, 5.0]))


def test_mps_gain_matches_a_finite_difference(cbm):
    """Inside the range the gain is the margin's fall under the worst L∞ step, to first
    order: compare with the step itself at a small ε."""
    torch.manual_seed(0)
    lo, hi = cbm.input_range
    x = lo + (hi - lo) * (0.25 + 0.5 * torch.rand(4, 4))     # away from the ends
    y = torch.randint(0, cbm.out_dim, (4,))
    s = margin_sensitivity(cbm, x, y, 2, "cpu")

    def margin(z):
        lj = cbm.log_joint(z)
        return lj.gather(1, y[:, None]).squeeze(1) - lj.scatter(
            1, y[:, None], float("-inf")).max(1).values

    z = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(margin(z).sum(), z)
    eps = 1e-3 * (hi - lo)
    with torch.no_grad():
        fall = s.margin - margin(x - eps * g.sign())
    torch.testing.assert_close(fall / eps, s.gain, rtol=2e-2, atol=1e-3)
    assert torch.equal(s.gain_ends, torch.zeros(4))
