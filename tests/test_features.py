import numpy as np, pandas as pd
from m5fc.features import sales_features, BOTTOM_FEATS, AGG_FEATS
from m5fc.config import MIN_LAG


def test_no_target_leakage_synthetic():
    """Features of row t must not change when sales at t-27..end are altered (i.e. only info <= t-28 is used)."""
    rng = np.random.default_rng(1)
    S = pd.DataFrame(rng.poisson(3, (500, 3)).astype(float))
    f0 = sales_features(S)
    t = 400
    S2 = S.copy(); S2.iloc[t - MIN_LAG + 1:] += 1000
    f1 = sales_features(S2)
    for k in f0:
        a, b = f0[k].iloc[t].values, f1[k].iloc[t].values
        assert np.allclose(a, b, equal_nan=True), f"feature {k} leaks information newer than t-{MIN_LAG}"


def test_all_lags_at_least_horizon(data):
    bt, at, S, h = data
    for f in BOTTOM_FEATS:
        if f.startswith("lag_"):
            assert int(f.split("_")[1]) >= 28


def test_lag28_matches_target_real(data):
    bt, *_ = data
    g = bt[bt.sku == bt.sku.iloc[0]].sort_values("d")
    assert np.allclose(g.lag_28.values[28:], g.y.values[:-28], equal_nan=True)


def test_feature_columns_exist_and_finite_future_known(data):
    bt, at, *_ = data
    assert set(BOTTOM_FEATS) <= set(bt.columns) and set(AGG_FEATS) <= set(at.columns)
    test = bt[bt.d.between(1914, 1941)]
    for c in ["snap", "wday", "days_to_event", "days_since_event"]:
        assert test[c].notna().all()
