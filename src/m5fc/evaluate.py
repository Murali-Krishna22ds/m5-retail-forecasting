"""Score any stored model at every hierarchy level on the cal or test window."""
from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .config import CAL_START, TEST_START, H, RES
from .store import load, to_wide, qcols, WINDOWS

LEVELS = ["Total", "State", "Store", "Item-Store"]


def level_of(node: str, bottoms: set) -> str:
    if node == "Total": return "Total"
    if node in ("CA", "TX", "WI"): return "State"
    if node in bottoms: return "Item-Store"
    return "Store"


def revenue_weights(S_all, bt) -> pd.Series:
    last = bt[(bt.d > CAL_START - 1 - 28) & (bt.d <= CAL_START - 1)]
    return (last.assign(r=last.y * last.sell_price.fillna(0)).groupby("sku").r.sum())


def score(model: str, S_all: pd.DataFrame, bottoms: set, window: str = "test", wts: pd.Series | None = None,
          fc: pd.DataFrame | None = None) -> pd.DataFrame:
    fc = load(model, window) if fc is None else fc[fc.window == window]
    P = to_wide(fc, "point")
    d0, d1 = WINDOWS[window]
    Y = S_all.loc[d0:d1, P.columns]
    TR = S_all.loc[: d0 - 1, P.columns]
    ps = M.evaluate_point(Y, P, TR, wts)
    has_q = all(c in fc.columns for c in qcols())
    if has_q:
        Q = {q: to_wide(fc, f"q{q}") for q in (0.025, 0.1, 0.5, 0.9, 0.975)}
        ps["spl"] = [M.scaled_pinball(Y[c].values, {t: Q[t][c].values for t in Q}, M.abs_scale(TR[c].values)) for c in Y.columns]
        ps["cov80"] = [M.coverage(Y[c], Q[0.1][c], Q[0.9][c]) for c in Y.columns]
        ps["cov95"] = [M.coverage(Y[c], Q[0.025][c], Q[0.975][c]) for c in Y.columns]
        ps["width80"] = [(Q[0.9][c] - Q[0.1][c]).mean() for c in Y.columns]
    ps["level"] = [level_of(c, bottoms) for c in ps.index]
    rows = []
    for lv in LEVELS:
        g = ps[ps.level == lv]
        if g.empty: continue
        Yl, Pl = Y[g.index], P[g.index]
        r = {"model": model, "window": window, "level": lv, "n_series": len(g),
             "WAPE": M.wape(Yl.values, Pl.values), "RMSSE": g.rmsse.mean()}
        if lv == "Item-Store" and wts is not None:
            w = g["weight"]; ok = g.rmsse.notna()
            r["WRMSSE"] = float((g.rmsse[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan
        if has_q:
            r.update(SPL=g.spl.mean(), Cov80=g.cov80.mean() if lv == "Item-Store" else M.coverage(Yl.values, to_wide(fc, "q0.1")[g.index].values, to_wide(fc, "q0.9")[g.index].values),
                     Cov95=g.cov95.mean() if lv == "Item-Store" else M.coverage(Yl.values, to_wide(fc, "q0.025")[g.index].values, to_wide(fc, "q0.975")[g.index].values),
                     Width80=g.width80.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def score_all(models, S_all, bottoms, bt, window="test") -> pd.DataFrame:
    w = revenue_weights(S_all, bt)
    return pd.concat([score(m, S_all, bottoms, window, w) for m in models], ignore_index=True)
