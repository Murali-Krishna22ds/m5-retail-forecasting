"""Ensembling + interval recalibration. Every choice (blend weight, conformal adjustment) uses the CALIBRATION window only."""
from __future__ import annotations
import numpy as np, pandas as pd
from . import metrics as M
from .config import QUANTILES, CAL_START, CAL_END
from .store import load, save, to_wide, qcols

KEYS = ["window", "d", "node"]


def average(names: list[str], out: str):
    """Equal-weight average of point and quantiles (quantile averaging = Vincentisation) over models on the same nodes."""
    fs = [load(n).set_index(KEYS).sort_index() for n in names]
    cols = ["point"] + qcols()
    avg = sum(f[cols] for f in fs) / len(fs)
    avg[qcols()] = np.sort(avg[qcols()].values, axis=1)
    save(out, avg.reset_index())


def blend(a: str, b: str, w: float, out: str, bottoms: set):
    """(1-w)*a + w*b on the nodes b covers (bottom series); nodes only in a are kept unchanged. Quantiles averaged then sorted."""
    fa, fb = load(a).set_index(KEYS).sort_index(), load(b).set_index(KEYS).sort_index()
    cols = ["point"] + qcols()
    common = fa.index.intersection(fb.index)
    mix = fa.copy()
    mix.loc[common, cols] = (1 - w) * fa.loc[common, cols].values + w * fb.loc[common, cols].values
    mix[qcols()] = np.sort(mix[qcols()].values, axis=1)
    save(out, mix.reset_index())


def item_objective(fc: pd.DataFrame, S_all: pd.DataFrame, bottoms: list, window="cal"):
    """(WAPE, mean RMSSE) at item-store level on a window."""
    P = to_wide(fc[fc.window == window], "point")[bottoms]
    d0, d1 = (CAL_START, CAL_END) if window == "cal" else (1914, 1941)
    Y = S_all.loc[d0:d1, bottoms]
    r = np.nanmean([M.rmsse(Y[c], P[c], M.rmsse_scale(S_all.loc[: d0 - 1, c].values)) for c in bottoms])
    return M.wape(Y.values, P.values), float(r)


def choose_weight(a: str, b: str, S_all, bottoms: list, grid=np.round(np.arange(0, 1.01, 0.1), 2)):
    """Weight on b minimising WAPE/WAPE_a + RMSSE/RMSSE_a on the calibration window (a = baseline, score 2.0)."""
    import tempfile
    fa, fb = load(a).set_index(KEYS).sort_index(), load(b).set_index(KEYS).sort_index()
    cal_a, cal_b = fa.xs("cal", level="window").reset_index(), fb.xs("cal", level="window").reset_index()
    cal_a["window"] = cal_b["window"] = "cal"
    w0, r0 = item_objective(cal_a, S_all, bottoms)
    res = {}
    for w in grid:
        m = cal_a.set_index(["d", "node"]); bb = cal_b.set_index(["d", "node"]); idx = m.index.intersection(bb.index)
        mix = m.loc[idx, ["point"]].copy(); mix["point"] = (1 - w) * m.loc[idx, "point"] + w * bb.loc[idx, "point"]
        mix = mix.reset_index(); mix["window"] = "cal"
        ww, rr = item_objective(mix, S_all, bottoms); res[float(w)] = float(ww / w0 + rr / r0)
    best = min(res, key=res.get)
    return best, res


def scaled_cqr(name: str, out: str, S_all: pd.DataFrame, bottoms: set):
    """Locally-weighted conformalised quantile regression on the bottom series, calibrated on the cal window (pooled over
    series x days). Score = max(lo - y, y - hi) / (1 + interval width); adjust bounds by +/- quantile * (1 + width).
    Aggregate nodes (only 14 series) are left unchanged."""
    fc = load(name).copy(); info = {}
    cal = fc[(fc.window == "cal") & fc.node.isin(bottoms)].copy()
    cal["y"] = S_all.stack().reindex(pd.MultiIndex.from_arrays([cal.d, cal.node])).values
    isb = fc.node.isin(bottoms)
    for lo, hi, alpha in [(0.1, 0.9, 0.2), (0.025, 0.975, 0.05)]:
        sc = 1.0 + cal[f"q{hi}"] - cal[f"q{lo}"]
        e = (np.maximum(cal[f"q{lo}"] - cal.y, cal.y - cal[f"q{hi}"]) / sc).values
        n = len(e); adj = float(np.quantile(e, min(1.0, (1 - alpha) * (1 + 1 / n)))); info[f"{lo}-{hi}"] = adj
        s_all = 1.0 + fc[f"q{hi}"] - fc[f"q{lo}"]
        fc.loc[isb, f"q{lo}"] = (fc[f"q{lo}"] - adj * s_all)[isb]; fc.loc[isb, f"q{hi}"] = (fc[f"q{hi}"] + adj * s_all)[isb]
    m = fc.loc[isb, "q0.5"]
    fc.loc[isb, "q0.1"] = np.minimum(fc.loc[isb, "q0.1"], m); fc.loc[isb, "q0.9"] = np.maximum(fc.loc[isb, "q0.9"], m)
    fc.loc[isb, "q0.025"] = np.minimum(fc.loc[isb, "q0.025"], fc.loc[isb, "q0.1"]); fc.loc[isb, "q0.975"] = np.maximum(fc.loc[isb, "q0.975"], fc.loc[isb, "q0.9"])
    for c in qcols(): fc.loc[isb, c] = fc.loc[isb, c].clip(lower=0)
    fc[qcols()] = np.sort(fc[qcols()].values, axis=1)
    save(out, fc); return info
