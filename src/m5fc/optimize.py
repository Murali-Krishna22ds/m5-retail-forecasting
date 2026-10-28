"""Inventory optimisation with OR-Tools: scenario-based newsvendor MILP per store.

For every SKU i and scenario s (cycle demand D_is drawn from the forecast quantiles):
    min  sum_i [ h_i * mean_s over_is  +  g_i * mean_s under_is ]
    s.t. I0_i + pack_i * n_i - D_is = over_is - under_is        (n_i integer >= 0)
         sum_i c_i * pack_i * n_i  <= Budget_store               (purchasing budget)
         sum_i v_i * (I0_i + pack_i * n_i) <= Capacity_store      (shelf / storage volume)
The LP relaxation with sample-average costs is the classical newsvendor (order ~ critical-ratio quantile);
budget + capacity + pack-size integrality make it a genuine MILP that SCIP solves in well under a second per store."""
from __future__ import annotations
import zlib
import numpy as np, pandas as pd
from scipy.stats import norm
from ortools.linear_solver import pywraplp
from . import config as C

PACK = {"FOODS": 6, "HOUSEHOLD": 4, "HOBBIES": 3}          # case-pack sizes (assumption)
VOL = {"FOODS": 1.0, "HOUSEHOLD": 2.0, "HOBBIES": 1.5}     # shelf volume per unit (assumption)
TAUS = np.array([0.0, 0.025, 0.1, 0.5, 0.9, 0.975, 0.995])


def item_params(h: pd.DataFrame, price_now: pd.Series, cycle_days: int = C.CYCLE_DAYS, hold_rate=C.HOLD_RATE_WEEK,
                cost_frac=C.COST_FRAC, goodwill=C.GOODWILL_FRAC, margin_mult=C.STOCKOUT_MARGIN_MULT) -> pd.DataFrame:
    p = h.set_index("sku").copy()
    p["price"] = price_now.reindex(p.index)
    p["cost"] = p.price * cost_frac
    p["hold"] = p.cost * hold_rate * cycle_days / 7                        # per unit left over at cycle end
    p["short"] = (p.price - p.cost) * margin_mult + goodwill * p.price     # per unit unmet
    p["pack"] = p.cat_id.map(PACK); p["vol"] = p.cat_id.map(VOL)
    return p


def _inv_cdf(qv: np.ndarray, u: np.ndarray) -> np.ndarray:
    """qv: (n, 5) quantiles at [.025,.1,.5,.9,.975]; u: (n, ...) uniforms. Piecewise-linear inverse CDF,
    support floored at 0 and a linear tail beyond q97.5."""
    n = qv.shape[0]
    tail = qv[:, 4] + (qv[:, 4] - qv[:, 3]) / 0.075 * 0.02
    knots = np.column_stack([np.zeros(n), qv, tail])            # (n, 7)
    u = np.clip(u, 0, TAUS[-1] - 1e-9)
    idx = np.clip(np.searchsorted(TAUS, u.reshape(n, -1), side="right") - 1, 0, len(TAUS) - 2)
    t0, t1 = TAUS[idx], TAUS[idx + 1]
    k0 = np.take_along_axis(knots, idx, 1); k1 = np.take_along_axis(knots, idx + 1, 1)
    w = (u.reshape(n, -1) - t0) / (t1 - t0)
    return (k0 + w * (k1 - k0)).reshape(u.shape)


def cycle_scenarios(qfc: pd.DataFrame, skus: list[str], d0: int, d1: int, K: int = 200, rho: float = 0.5, seed: int = 0):
    """Cycle-demand scenarios (n_sku x K) = sum over days of F_t^-1(u_t), u_t from a one-factor Gaussian copula."""
    rng = np.random.default_rng(seed)
    days = list(range(d0, d1 + 1)); T = len(days)
    qcols = [f"q{q}" for q in C.QUANTILES]
    x = qfc[qfc.node.isin(skus) & qfc.d.between(d0, d1)].set_index(["node", "d"])[qcols]
    x = x.reindex(pd.MultiIndex.from_product([skus, days]), fill_value=0.0)
    Q = x.values.reshape(len(skus), T, 5)
    z_common = rng.standard_normal((len(skus), 1, K)); z_idio = rng.standard_normal((len(skus), T, K))
    z = np.sqrt(rho) * z_common + np.sqrt(1 - rho) * z_idio
    u = norm.cdf(z)
    D = np.zeros((len(skus), K))
    for t in range(T):
        D += _inv_cdf(Q[:, t, :], u[:, t, :])
    return D


def calibrate_rho(qfc_cal: pd.DataFrame, S_all: pd.DataFrame, skus: list[str], grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), K=300):
    """Pick the copula correlation so that 7-day-sum 80% intervals have ~80% coverage on the calibration window."""
    res = {}
    for rho in grid:
        hits = []
        for w in range(4):
            d0 = C.CAL_START + 7 * w; d1 = d0 + 6
            D = cycle_scenarios(qfc_cal, skus, d0, d1, K, rho, seed=w)
            y = S_all.loc[d0:d1, skus].sum().values
            lo, hi = np.quantile(D, 0.1, axis=1), np.quantile(D, 0.9, axis=1)
            hits.append(((y >= lo) & (y <= hi)).mean())
        res[rho] = float(np.mean(hits))
    best = min(res, key=lambda r: abs(res[r] - 0.8))
    return best, res


def solve_store_explicit(P: pd.DataFrame, D: np.ndarray, I0: np.ndarray, budget: float, capacity: float,
                time_limit_s: float = 20.0, integer: bool = True):
    """Reference formulation with one over/under variable per (SKU, scenario). Exact but slow (O(n*K) vars);
    used in tests to verify solve_store."""
    n, K = D.shape
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(int(time_limit_s * 1000))
    ub = np.ceil(np.maximum(np.quantile(D, 0.999, axis=1) - I0, 0) / P.pack.values) + 1
    nvar = [s.IntVar(0, float(ub[i]), f"n{i}") if integer else s.NumVar(0, float(ub[i]), f"n{i}") for i in range(n)]
    obj = s.Objective()
    for i in range(n):
        for k in range(K):
            o = s.NumVar(0, s.infinity(), ""); u = s.NumVar(0, s.infinity(), "")
            # I0 + pack*n - D = over - under
            ct = s.Constraint(float(D[i, k] - I0[i]), float(D[i, k] - I0[i]))
            ct.SetCoefficient(nvar[i], float(P.pack.values[i])); ct.SetCoefficient(o, -1.0); ct.SetCoefficient(u, 1.0)
            obj.SetCoefficient(o, float(P.hold.values[i]) / K); obj.SetCoefficient(u, float(P.short.values[i]) / K)
    obj.SetMinimization()
    cb = s.Constraint(0.0, float(budget)); cc = s.Constraint(-s.infinity(), float(capacity - (P.vol.values * I0).sum()))
    for i in range(n):
        cb.SetCoefficient(nvar[i], float(P.cost.values[i] * P.pack.values[i]))
        cc.SetCoefficient(nvar[i], float(P.vol.values[i] * P.pack.values[i]))
    status = s.Solve()
    ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    q = np.array([nvar[i].solution_value() * P.pack.values[i] for i in range(n)]) if ok else np.zeros(n)
    return q, {"status": {0: "OPTIMAL", 1: "FEASIBLE"}.get(status, str(status)), "obj": s.Objective().Value() if ok else np.nan,
               "budget_used": float((P.cost.values * q).sum()), "cap_used": float((P.vol.values * (I0 + q)).sum()),
               "solve_s": s.WallTime() / 1000}


def expected_cost_table(P: pd.DataFrame, D: np.ndarray, I0: np.ndarray, ub: np.ndarray):
    """cost[i][k] = sample-average holding+shortage cost of SKU i if it orders k packs (k = 0..ub_i)."""
    tab = []
    for i in range(D.shape[0]):
        k = np.arange(int(ub[i]) + 1)
        after = I0[i] + P.pack.values[i] * k[:, None] - D[i][None, :]
        tab.append((P.hold.values[i] * np.maximum(after, 0) + P.short.values[i] * np.maximum(-after, 0)).mean(axis=1))
    return tab


def solve_store(P: pd.DataFrame, D: np.ndarray, I0: np.ndarray, budget: float, capacity: float,
                time_limit_s: float = 20.0, integer: bool = True):
    """Exact fast MILP. Expected cost is convex in the pack count n_i, so it equals the max of the chords between
    consecutive integers: z_i >= C_i(k) + (C_i(k+1)-C_i(k)) * (n_i - k).  Identical optimum to solve_store_explicit
    with 40 integer + 40 continuous variables instead of 2*n*K scenario variables."""
    n, K = D.shape
    ub = np.ceil(np.maximum(np.quantile(D, 0.999, axis=1) - I0, 0) / P.pack.values) + 1
    tab = expected_cost_table(P, D, I0, ub)
    s = pywraplp.Solver.CreateSolver("SCIP"); s.SetTimeLimit(int(time_limit_s * 1000))
    nv = [s.IntVar(0, float(ub[i]), f"n{i}") if integer else s.NumVar(0, float(ub[i]), f"n{i}") for i in range(n)]
    zv = [s.NumVar(0, s.infinity(), f"z{i}") for i in range(n)]
    for i in range(n):
        c = tab[i]
        for k in range(len(c) - 1):                       # chord through (k, C(k)) and (k+1, C(k+1))
            slope = float(c[k + 1] - c[k])
            ct = s.Constraint(float(c[k] - slope * k), s.infinity())     # z - slope*n >= C(k) - slope*k
            ct.SetCoefficient(zv[i], 1.0); ct.SetCoefficient(nv[i], -slope)
    obj = s.Objective()
    for i in range(n): obj.SetCoefficient(zv[i], 1.0)
    obj.SetMinimization()
    cb = s.Constraint(0.0, float(budget)); cc = s.Constraint(-s.infinity(), float(capacity - (P.vol.values * I0).sum()))
    for i in range(n):
        cb.SetCoefficient(nv[i], float(P.cost.values[i] * P.pack.values[i]))
        cc.SetCoefficient(nv[i], float(P.vol.values[i] * P.pack.values[i]))
    status = s.Solve(); ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    q = np.array([round(nv[i].solution_value()) * P.pack.values[i] for i in range(n)]) if ok and integer else \
        (np.array([nv[i].solution_value() * P.pack.values[i] for i in range(n)]) if ok else np.zeros(n))
    return q, {"status": {0: "OPTIMAL", 1: "FEASIBLE"}.get(status, str(status)), "obj": s.Objective().Value() if ok else np.nan,
               "budget_used": float((P.cost.values * q).sum()), "cap_used": float((P.vol.values * (I0 + q)).sum()),
               "solve_s": s.WallTime() / 1000}


def store_limits(P, D, I0, budget_factor, cap_factor):
    mean = D.mean(axis=1)
    budget = budget_factor * float((P.cost.values * np.maximum(mean - I0, 0)).sum()) + 1e-6
    capacity = max(cap_factor * float((P.vol.values * mean).sum()), float((P.vol.values * I0).sum()) + 1e-6)
    return budget, capacity


def cost_of(P, q, I0, demand):
    after = I0 + q - demand
    over = np.maximum(after, 0); under = np.maximum(-after, 0)
    return float((P.hold.values * over).sum() + (P.short.values * under).sum()), over, under


def heuristic_order(P, mean, sigma, I0, z, budget, capacity):
    """Order-up-to (mean + z*sigma) rounded UP to packs, then proportionally scaled to meet budget/capacity."""
    need = np.maximum(mean + z * sigma - I0, 0)
    q = np.ceil(need / P.pack.values) * P.pack.values
    f = min(1.0, budget / max((P.cost.values * q).sum(), 1e-9),
            (capacity - (P.vol.values * I0).sum()) / max((P.vol.values * q).sum(), 1e-9))
    return np.floor(q * max(f, 0) / P.pack.values) * P.pack.values


def backtest(qfc: pd.DataFrame, point_fc: pd.DataFrame, S_all: pd.DataFrame, h: pd.DataFrame, price_now: pd.Series,
             rho: float, budget_factor=1.05, cap_factor=1.5, K=200, weeks=4, d_start=C.TEST_START, label="milp",
             sigma_hist: pd.Series | None = None, z=1.28):
    """Weekly re-planning over the test window with inventory carry-over. Returns per store-week-policy rows."""
    P_all = item_params(h, price_now)
    rows = []; sku_rows = []
    state = {pol: pd.Series(0.0, index=P_all.index) for pol in ("milp", "mean", "safety", "oracle")}
    for w in range(weeks):
        d0 = d_start + 7 * w; d1 = d0 + 6
        for store, g in h.groupby("store_id"):
            skus = list(g.sku); P = P_all.loc[skus]
            D = cycle_scenarios(qfc, skus, d0, d1, K, rho, seed=1000 * w + zlib.crc32(store.encode()) % 997)
            act = S_all.loc[d0:d1, skus].sum().values
            mean_pt = point_fc[point_fc.node.isin(skus) & point_fc.d.between(d0, d1)].groupby("node").point.sum().reindex(skus).fillna(0).values
            sig = (sigma_hist.reindex(skus).fillna(0).values if sigma_hist is not None else np.zeros(len(skus)))
            I = {pol: state[pol].loc[skus].values for pol in state}
            budget, cap = store_limits(P, D, I["milp"], budget_factor, cap_factor)
            q_m, info = solve_store(P, D, I["milp"], budget, cap)
            orders = {"milp": q_m,
                      "mean": heuristic_order(P, mean_pt, 0 * sig, I["mean"], 0, budget, cap),
                      "safety": heuristic_order(P, mean_pt, sig, I["safety"], z, budget, cap),
                      "oracle": np.ceil(np.maximum(act - I["oracle"], 0) / P.pack.values) * P.pack.values}
            for pol, q in orders.items():
                c, over, under = cost_of(P, q, I[pol], act)
                rows.append(dict(model=label, policy=pol, week=w + 1, store=store, cost=c, units_demand=act.sum(),
                                 units_short=under.sum(), units_end_inv=over.sum(), order_value=float((P.cost.values * q).sum()),
                                 end_inv_value=float((P.cost.values * over).sum()), stockout_skus=int((under > 1e-9).sum()),
                                 n_skus=len(skus), solve_s=info["solve_s"] if pol == "milp" else 0.0,
                                 budget=budget, budget_used=float((P.cost.values * q).sum())))
                state[pol].loc[skus] = over
    return pd.DataFrame(rows)


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby(["model", "policy"]).agg(cost=("cost", "sum"), demand=("units_demand", "sum"), short=("units_short", "sum"),
                                         end_inv_value=("end_inv_value", "sum"), n=("store", "count"),
                                         stockout_skus=("stockout_skus", "sum"), n_skus=("n_skus", "sum"),
                                         order_value=("order_value", "sum"))
    g["fill_rate"] = 1 - g.short / g.demand
    g["stockout_rate"] = g.stockout_skus / g.n_skus
    g["avg_end_inv_value"] = g.end_inv_value / g.n
    return g.reset_index()
