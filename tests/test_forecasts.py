import numpy as np, pandas as pd, pytest
from m5fc.store import load, qcols, FC
from m5fc import config as C

MODELS = ["naive", "snaive", "prophet", "lgbm", "lgbm_mint"]


@pytest.mark.parametrize("m", MODELS)
def test_forecast_shape_finite_nonneg_monotone(data, m):
    _, _, S, h = data
    f = load(m)
    for w, (d0, d1) in {"cal": (C.CAL_START, C.CAL_END), "test": (C.TEST_START, C.TEST_END)}.items():
        g = f[f.window == w]
        if g.empty: continue
        assert g.d.min() == d0 and g.d.max() == d1
        assert g.groupby("node").d.nunique().eq(28).all()
        assert not g.duplicated(["node", "d"]).any()
        assert np.isfinite(g[["point"] + qcols()]).all().all()
        assert (g[qcols()].values >= -1e-9).all() and (g.point >= -1e-9).all()
        assert (np.diff(g[qcols()].values, axis=1) >= -1e-9).all(), "quantile crossing"


def test_models_cover_all_nodes(data):
    _, _, S, h = data
    for m in ["lgbm_mint", "prophet", "snaive"]:
        assert set(load(m, "test").node) == set(S.columns)


def test_lgbm_beats_baselines_and_is_statistically_significant():
    import json
    te = pd.read_csv(C.RES / "metrics_test.csv"); b = te[te.level == "Item-Store"].set_index("model")
    assert b.loc["lgbm_mint", "WAPE"] < b.loc["prophet", "WAPE"] < b.loc["snaive", "WAPE"]
    assert b.loc["lgbm_mint", "WRMSSE"] < b.loc["prophet", "WRMSSE"]
    ci = json.load(open(C.RES / "bootstrap_wape_diff.json"))["WAPE(prophet) - WAPE(lgbm_mint)"]["ci95"]
    assert ci[0] > 0                                            # 95% CI excludes zero


def test_interval_calibration_item_level():
    te = pd.read_csv(C.RES / "metrics_test.csv"); b = te[(te.level == "Item-Store") & (te.model == "lgbm_mint")].iloc[0]
    assert 0.75 <= b.Cov80 <= 0.90 and 0.93 <= b.Cov95 <= 0.995


def test_reconciliation_improves_top_levels():
    te = pd.read_csv(C.RES / "metrics_test.csv").set_index(["model", "level"])
    assert te.loc[("lgbm_mint", "Total"), "WAPE"] < te.loc[("lgbm", "Total"), "WAPE"]
    assert te.loc[("lgbm_mint", "State"), "WAPE"] < te.loc[("lgbm", "State"), "WAPE"]


def test_no_test_data_used_for_training():
    """Models were trained on d <= LAST_TRAIN_D; saved boosters must not have been fit beyond it (structural check)."""
    assert C.LAST_TRAIN_D < C.CAL_START <= C.CAL_END < C.TEST_START <= C.TEST_END
    assert C.CAL_END - C.CAL_START + 1 == C.H == C.TEST_END - C.TEST_START + 1
