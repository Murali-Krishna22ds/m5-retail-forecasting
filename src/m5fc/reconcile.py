"""Hierarchical forecast reconciliation with Darts (Total > State > Store > Item-Store).
Methods: bottom-up, top-down, MinT (ols / wls_struct / wls_var).
Non-negativity: negative bottom values are clipped to 0 and every upper level is then re-summed from the
bottom, so the final forecasts are *exactly* coherent AND non-negative."""
from __future__ import annotations
import numpy as np, pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import MinTReconciliator, BottomUpReconciliator, TopDownReconciliator
from .config import TEST_START, TEST_END, CAL_START, CAL_END
from .data import aggregate
from .store import load, to_wide, qcols, save


def hierarchy_dict(h: pd.DataFrame) -> dict:
    hd = {r.sku: [r.store_id] for r in h.itertuples()}
    for st, s in h.drop_duplicates("store_id").set_index("store_id").state_id.items(): hd[st] = [s]
    for s in h.state_id.unique(): hd[s] = ["Total"]
    return hd


def to_ts(W: pd.DataFrame, hd: dict) -> TimeSeries:
    W = W.copy(); W.index = pd.date_range("2016-01-01", periods=len(W), freq="D")   # any regular index
    return TimeSeries.from_times_and_values(W.index, W.values.astype("float64"), columns=list(W.columns), hierarchy=hd)


def coherent_from_bottom(Pb: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    return aggregate(Pb.clip(lower=0), h)


def reconcile_point(base: pd.DataFrame, S_all: pd.DataFrame, h: pd.DataFrame, method: str, resid_cal: pd.DataFrame | None = None):
    """base: horizon x nodes (incoherent). Returns coherent horizon x nodes frame."""
    cols = list(S_all.columns); hd = hierarchy_dict(h); base = base[cols]
    ts = to_ts(base, hd)
    if method == "bottom_up":
        rec = BottomUpReconciliator().transform(ts)
    elif method == "top_down":
        hist = to_ts(S_all.loc[: CAL_START - 1], hd)
        t = TopDownReconciliator(); t.fit(hist); rec = t.transform(ts)
    else:
        m = MinTReconciliator(method=method)
        if method in ("wls_var", "mint_cov"):
            m.fit(to_ts(resid_cal[cols], hd))
        else:
            m.fit(to_ts(base, hd))
        rec = m.transform(ts)
    R = pd.DataFrame(rec.values(), index=base.index, columns=cols)
    return coherent_from_bottom(R[list(h.sku)], h)[cols]


def reconcile_model(model: str, S_all, h, method: str, out_name: str):
    """Reconcile a stored model's point forecast (both windows); shift quantiles by the same adjustment."""
    pieces = []
    base_c, base_t = load(model, "cal"), load(model, "test")
    resid = S_all.loc[CAL_START:CAL_END] - to_wide(base_c, "point")[S_all.columns]
    for win, fc in (("cal", base_c), ("test", base_t)):
        P = to_wide(fc, "point")
        R = reconcile_point(P, S_all, h, method, resid)
        adj = (R - P[R.columns]).stack().rename("adj").reset_index(); adj.columns = ["d", "node", "adj"]
        o = fc.merge(adj, on=["d", "node"]); o["point"] = o["point"] + o["adj"]
        has_q = all(c in o.columns for c in qcols())
        if has_q:
            # Shift interval bands only for aggregate nodes (large counts). Bottom series are small integers and their
            # quantile bounds are integer-valued: shifting by a fraction would turn covered y=k into misses (coverage
            # collapsed 82% -> 64% in an earlier run), so bottom intervals keep their calibrated native values.
            agg = ~o["node"].isin(set(h.sku))
            for c in qcols(): o.loc[agg, c] = (o.loc[agg, c] + o.loc[agg, "adj"]).clip(lower=0)
            o[qcols()] = np.sort(o[qcols()].values, axis=1)
        pieces.append(o.drop(columns="adj"))
    save(out_name, pd.concat(pieces, ignore_index=True))


def coherence_error(fc_wide: pd.DataFrame, h: pd.DataFrame) -> dict:
    """max |parent - sum(children)| across the horizon, per level pair."""
    st = h.drop_duplicates("store_id").set_index("store_id").state_id
    store_sum = {s: fc_wide[h[h.store_id == s].sku].sum(axis=1) for s in st.index}
    e_store = max((fc_wide[s] - v).abs().max() for s, v in store_sum.items())
    e_state = max((fc_wide[s] - sum(fc_wide[x] for x in st[st == s].index)).abs().max() for s in st.unique())
    e_total = (fc_wide["Total"] - sum(fc_wide[s] for s in st.unique())).abs().max()
    return {"store = sum(items)": float(e_store), "state = sum(stores)": float(e_state), "total = sum(states)": float(e_total)}
