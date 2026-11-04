import numpy as np, pytest
from m5fc.store import load, to_wide
from m5fc.reconcile import coherence_error

NAMES = ["lgbm_bu", "lgbm_td", "lgbm_ols", "lgbm_wlss", "lgbm_mint", "ens_mint", "ens_bu", "lgbm_v1_mint"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("window", ["cal", "test"])
def test_reconciled_forecasts_are_coherent(data, name, window):
    _, _, S, h = data
    P = to_wide(load(name, window), "point")[S.columns]
    assert max(coherence_error(P, h).values()) < 1e-8
    assert (P.values >= -1e-9).all()                       # non-negative after clipping


def test_base_forecasts_are_incoherent_so_reconciliation_matters(data):
    _, _, S, h = data
    P = to_wide(load("lgbm", "test"), "point")[S.columns]
    assert max(coherence_error(P, h).values()) > 1.0


def test_store_totals_match_state_totals_explicitly(data):
    _, _, S, h = data
    P = to_wide(load("ens_mint", "test"), "point")
    for state, g in h.drop_duplicates("store_id").groupby("state_id"):
        assert np.allclose(P[state], P[list(g.store_id)].sum(axis=1), atol=1e-8)
