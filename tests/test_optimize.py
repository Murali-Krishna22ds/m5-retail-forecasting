import numpy as np, pandas as pd, pytest
from m5fc import optimize as O


def _params(n, rng):
    price = rng.uniform(1, 8, n)
    P = pd.DataFrame({"price": price, "cost": price * 0.7, "pack": rng.choice([3, 4, 6], n), "vol": rng.choice([1.0, 2.0], n)})
    P["hold"] = P.cost * 0.06; P["short"] = price * 0.4
    return P


def _demand(n, K, rng):
    mu = rng.uniform(2, 30, n)[:, None]
    return rng.poisson(mu, (n, K)).astype(float) * rng.uniform(0.8, 1.2, (n, K))


def test_fast_milp_equals_explicit_formulation():
    rng = np.random.default_rng(0); n, K = 8, 40
    P, D, I0 = _params(n, rng), _demand(n, K, rng), np.zeros(n)
    b, c = O.store_limits(P, D, I0, 0.9, 1.5)
    q1, i1 = O.solve_store(P, D, I0, b, c); q2, i2 = O.solve_store_explicit(P, D, I0, b, c)
    assert i1["obj"] == pytest.approx(i2["obj"], rel=1e-6)


def test_constraints_and_pack_multiples():
    rng = np.random.default_rng(1); n, K = 40, 100
    P, D = _params(n, rng), _demand(n, K, rng); I0 = rng.integers(0, 5, n).astype(float)
    b, c = O.store_limits(P, D, I0, 0.8, 1.2)
    q, info = O.solve_store(P, D, I0, b, c)
    assert info["status"] in ("OPTIMAL", "FEASIBLE")
    assert (P.cost.values * q).sum() <= b + 1e-6
    assert (P.vol.values * (I0 + q)).sum() <= c + 1e-6
    assert np.allclose(q % P.pack.values, 0) and (q >= 0).all()


def test_loosening_constraints_cannot_increase_cost():
    rng = np.random.default_rng(2); n, K = 30, 100
    P, D, I0 = _params(n, rng), _demand(n, K, rng), np.zeros(n)
    objs = []
    for bf in (0.6, 0.9, 1.5, 50):
        b, c = O.store_limits(P, D, I0, bf, 50)
        objs.append(O.solve_store(P, D, I0, b, c)[1]["obj"])
    assert all(objs[i] >= objs[i + 1] - 1e-9 for i in range(len(objs) - 1))


def test_unconstrained_solution_is_newsvendor_quantile():
    """Single SKU, pack=1, no limits -> optimal order is the critical-ratio quantile g/(g+h) of demand."""
    rng = np.random.default_rng(3); K = 20000
    D = rng.poisson(20, (1, K)).astype(float)
    P = pd.DataFrame({"price": [5.0], "cost": [3.5], "pack": [1], "vol": [1.0], "hold": [0.5], "short": [2.0]})
    q, _ = O.solve_store(P, D, np.zeros(1), 1e9, 1e9)
    cr = 2.0 / (2.0 + 0.5)
    assert abs(q[0] - np.quantile(D[0], cr)) <= 1


def test_on_hand_inventory_reduces_order():
    rng = np.random.default_rng(4); n, K = 10, 200
    P, D = _params(n, rng), _demand(n, K, rng)
    q0, _ = O.solve_store(P, D, np.zeros(n), 1e9, 1e9)
    q1, _ = O.solve_store(P, D, np.full(n, 10.0), 1e9, 1e9)
    assert q1.sum() < q0.sum()


def test_inverse_cdf_hits_quantile_knots_and_is_monotone():
    qv = np.array([[0, 1, 3, 6, 9.0]])
    for tau, v in zip([0.025, 0.1, 0.5, 0.9, 0.975], qv[0]):
        assert O._inv_cdf(qv, np.array([[tau + 1e-12]]))[0, 0] == pytest.approx(v, abs=1e-6)
    u = np.linspace(0.001, 0.994, 200)[None, :]
    x = O._inv_cdf(qv, u)[0]
    assert (np.diff(x) >= -1e-12).all() and x.min() >= 0


def test_scenarios_reproduce_forecast_quantiles(data):
    from m5fc.store import load
    _, _, S, h = data
    fc = load("lgbm_mint", "test"); sku = [h.sku.iloc[0]]
    D = O.cycle_scenarios(fc, sku, 1914, 1914, K=20000, rho=0.0)       # 1 day => marginal quantiles
    row = fc[(fc.node == sku[0]) & (fc.d == 1914)].iloc[0]
    assert np.quantile(D, 0.5) == pytest.approx(row["q0.5"], abs=0.1)
    assert np.quantile(D, 0.9) == pytest.approx(row["q0.9"], abs=0.15)


def test_milp_beats_heuristics_in_expected_cost():
    rng = np.random.default_rng(5); n, K = 40, 400
    P, D, I0 = _params(n, rng), _demand(n, K, rng), np.zeros(n)
    b, c = O.store_limits(P, D, I0, 1.3, 2.0)
    q, info = O.solve_store(P, D, I0, b, c)
    exp_cost = lambda q_: np.mean([O.cost_of(P, q_, I0, D[:, k])[0] for k in range(K)])
    qm = O.heuristic_order(P, D.mean(1), 0 * D.mean(1), I0, 0, b, c)
    assert exp_cost(q) <= exp_cost(qm) + 1e-9
