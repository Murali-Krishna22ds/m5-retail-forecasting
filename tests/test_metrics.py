import numpy as np, pytest
from m5fc import metrics as M


def test_wape_known():
    assert M.wape([10, 20, 30], [12, 18, 30]) == pytest.approx(4 / 60)
    assert M.wape([0, 0], [0, 0]) == 0.0


def test_rmsse_scale_starts_at_first_sale():
    tr = np.array([0, 0, 0, 2, 4, 4, 8])        # diffs after first sale: 2,0,4 -> mean sq = (4+0+16)/3
    assert M.rmsse_scale(tr) == pytest.approx(20 / 3)
    assert np.isnan(M.rmsse_scale(np.zeros(10)))


def test_rmsse_known():
    s = 4.0
    assert M.rmsse([1, 2, 3], [1, 2, 3], s) == 0.0
    assert M.rmsse([0, 0], [2, 2], s) == pytest.approx(1.0)       # sqrt(4/4)


def test_pinball_and_coverage():
    assert M.pinball([10], [8], 0.9) == pytest.approx(1.8)
    assert M.pinball([10], [12], 0.9) == pytest.approx(0.2)
    assert M.coverage([1, 5, 9], [0, 0, 0], [2, 6, 8]) == pytest.approx(2 / 3)


def test_pinball_minimised_by_true_quantile():
    rng = np.random.default_rng(0); y = rng.poisson(3, 20000)
    q90 = np.quantile(y, 0.9)
    assert M.pinball(y, q90, 0.9) < M.pinball(y, q90 + 2, 0.9)
    assert M.pinball(y, q90, 0.9) < M.pinball(y, q90 - 2, 0.9)
