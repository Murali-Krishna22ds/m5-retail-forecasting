import json, numpy as np, pandas as pd, pytest
from m5fc import config as C
from m5fc.store import load, qcols, FC, to_wide

KEYS = ["window", "d", "node"]


def _info():
    return json.load(open(C.RES / "ensemble_info.json"))


def test_blend_weight_chosen_on_calibration_window_and_in_range():
    i = _info(); w = i["blend_weight_tft"]
    assert 0.0 <= w <= 1.0
    grid = {float(k): v for k, v in i["blend_grid_cal_score"].items()}
    assert grid[w] == min(grid.values())                       # argmin of the calibration-window objective
    assert grid[0.0] == pytest.approx(2.0, abs=1e-6)           # weight 0 == the LightGBM baseline (score 2 by construction)


def test_ensemble_point_equals_weighted_average_before_cqr_and_reconciliation(data):
    w = _info()["blend_weight_tft"]
    a = load("lgbm").set_index(KEYS).sort_index(); b = load(_info()["tft_used"]).set_index(KEYS).sort_index(); e = load("ens_raw").set_index(KEYS).sort_index()
    idx = b.index[:2000]
    assert np.allclose(e.loc[idx, "point"], (1 - w) * a.loc[idx, "point"] + w * b.loc[idx, "point"])
    agg = [n for n in a.index.get_level_values("node").unique() if "|" not in n][:3]           # aggregate nodes untouched by the blend
    sel = e.index[e.index.get_level_values("node").isin(agg)]
    assert np.allclose(e.loc[sel, "point"], a.loc[sel, "point"])


def test_cqr_changes_only_bottom_intervals_and_never_the_point(data):
    _, _, S, h = data
    for raw, cq in [("lgbm", "lgbm_cqr"), ("ens_raw", "ens")]:
        a = load(raw).set_index(KEYS).sort_index(); b = load(cq).set_index(KEYS).sort_index()
        assert np.allclose(a["point"], b["point"])
        agg = a.index[~a.index.get_level_values("node").isin(set(h.sku))]
        assert np.allclose(a.loc[agg, qcols()], b.loc[agg, qcols()])


def test_cqr_gives_valid_calibration_window_coverage(data):
    """After recalibration the pooled bottom-level coverage on the calibration window must reach the nominal level (conformal guarantee)."""
    _, _, S, h = data
    for m in ["tft_cqr", "ens"]:
        f = load(m, "cal"); f = f[f.node.isin(set(h.sku))]
        y = S.stack().reindex(pd.MultiIndex.from_arrays([f.d, f.node])).values
        assert np.mean((y >= f["q0.1"]) & (y <= f["q0.9"])) >= 0.80 - 0.01
        assert np.mean((y >= f["q0.025"]) & (y <= f["q0.975"])) >= 0.95 - 0.01


def test_no_model_trained_or_selected_on_test_window():
    """Selection window for tuning ends before calibration starts; calibration ends before the test window starts."""
    bp = json.load(open(C.RES / "best_params.json"))
    assert bp["selection_window"][1] < C.CAL_START <= C.CAL_END < C.TEST_START
