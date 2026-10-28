"""Accuracy metrics. WAPE & RMSSE follow the M5 definitions; pinball = M5 'uncertainty' SPL."""
from __future__ import annotations
import numpy as np, pandas as pd


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9))


def rmsse_scale(train: np.ndarray) -> float:
    """Mean squared one-step naive error on the training history, starting at first sale (M5 rule)."""
    train = np.asarray(train, float)
    nz = np.flatnonzero(train)
    if len(nz) == 0:
        return np.nan
    t = train[nz[0]:]
    if len(t) < 2:
        return np.nan
    return float(np.mean(np.diff(t) ** 2))


def rmsse(y, yhat, scale: float) -> float:
    if not scale or np.isnan(scale) or scale == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2) / scale))


def pinball(y, q, tau):
    d = np.asarray(y, float) - np.asarray(q, float)
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def scaled_pinball(y, qdict: dict[float, np.ndarray], scale_abs: float) -> float:
    """Mean over quantiles of pinball / mean-abs-naive-diff (M5 'SPL'). scale_abs=mean|diff(train)|."""
    if not scale_abs or np.isnan(scale_abs):
        return np.nan
    return float(np.mean([pinball(y, qv, t) for t, qv in qdict.items()]) / scale_abs)


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def winkler(y, lo, hi, alpha):
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    w = hi - lo + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(w.mean())


def abs_scale(train: np.ndarray) -> float:
    train = np.asarray(train, float)
    nz = np.flatnonzero(train)
    if len(nz) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(train[nz[0]:]))))


def evaluate_point(Y: pd.DataFrame, P: pd.DataFrame, TR: pd.DataFrame, weights: pd.Series | None = None) -> pd.DataFrame:
    """Y, P: horizon x series (actual / forecast); TR: train history x series. Returns per-series metrics."""
    rows = []
    for c in Y.columns:
        s = rmsse_scale(TR[c].values)
        rows.append((c, wape(Y[c], P[c]), rmsse(Y[c], P[c], s)))
    out = pd.DataFrame(rows, columns=["series", "wape", "rmsse"]).set_index("series")
    if weights is not None:
        out["weight"] = weights.reindex(out.index).fillna(0)
    return out


def summarise(per_series: pd.DataFrame, Y: pd.DataFrame, P: pd.DataFrame) -> dict:
    w = per_series["weight"] if "weight" in per_series else pd.Series(1.0, index=per_series.index)
    w = w / w.sum()
    ok = per_series["rmsse"].notna()
    return {
        "WAPE (pooled)": wape(Y.values, P.values),
        "WAPE (mean of series)": per_series["wape"].mean(),
        "RMSSE (mean)": per_series["rmsse"].mean(),
        "WRMSSE (revenue-weighted)": float((per_series["rmsse"][ok] * w[ok] / w[ok].sum()).sum()),
    }
