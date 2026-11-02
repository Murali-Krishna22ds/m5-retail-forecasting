"""Baselines: naive, seasonal-naive (+ empirical-residual intervals) and Prophet (with covariates)."""
from __future__ import annotations
import logging, numpy as np, pandas as pd
from .config import QUANTILES, CAL_START, CAL_END, TEST_START, TEST_END, H
from .store import qcols, save, WINDOWS


def _naive(S: pd.DataFrame, origin: int, kind: str, h: int = H) -> pd.DataFrame:
    hist = S.loc[:origin]
    if kind == "naive":
        v = np.tile(hist.iloc[-1].values, (h, 1))
    else:  # seasonal naive, period 7
        last7 = hist.iloc[-7:].values
        v = np.vstack([last7[i % 7] for i in range(h)])
    return pd.DataFrame(v, index=range(origin + 1, origin + 1 + h), columns=S.columns)


def run_naive(S: pd.DataFrame):
    for kind in ("naive", "snaive"):
        rows = []
        # residuals of the previous window give empirical intervals for the next one
        for win, origin in (("cal", CAL_START - 1), ("test", TEST_START - 1)):
            P = _naive(S, origin, kind)
            prev_origin = origin - H
            Pprev = _naive(S, prev_origin, kind)
            R = S.loc[Pprev.index] - Pprev          # actual - forecast on previous window
            out = P.stack().rename("point").reset_index()
            out.columns = ["d", "node", "point"]
            for q in QUANTILES:
                adj = R.quantile(q)                  # per-node residual quantile
                Q = (P + adj).clip(lower=0)
                out[f"q{q}"] = Q.stack().values
            out["window"] = win
            rows.append(out)
        save(kind, pd.concat(rows, ignore_index=True))


def _prophet_one(y: pd.Series, dates: pd.Series, extra: pd.DataFrame, holidays, origin: int, h: int):
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.ERROR); logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    train = pd.DataFrame({"ds": dates.loc[:origin].values, "y": y.loc[:origin].values})
    fut = pd.DataFrame({"ds": dates.loc[origin + 1: origin + h].values})
    m = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                interval_width=0.8, uncertainty_samples=200)
    for c in extra.columns:
        train[c] = extra.loc[:origin, c].values; fut[c] = extra.loc[origin + 1: origin + h, c].values
        m.add_regressor(c)
    m.fit(train)
    np.random.seed(0)                       # Prophet's interval sampling uses the global numpy RNG -> make runs reproducible
    p80 = m.predict(fut)
    m.interval_width = 0.95
    m.uncertainty_samples = 200
    np.random.seed(1)
    p95 = m.predict(fut)
    return pd.DataFrame({"point": p80.yhat.values, "q0.1": p80.yhat_lower.values, "q0.9": p80.yhat_upper.values,
                         "q0.025": p95.yhat_lower.values, "q0.975": p95.yhat_upper.values,
                         "q0.5": p80.yhat.values})


def run_prophet(S_all: pd.DataFrame, df: pd.DataFrame, at: pd.DataFrame, bt: pd.DataFrame, h_frame: pd.DataFrame,
                nodes: list[str] | None = None, history_days: int = 1095, log=print):
    """Prophet on the *test* origin only (compute budget). Regressors: snap + price_ratio (known ahead)."""
    from tqdm import tqdm
    cal = df.drop_duplicates("d").set_index("d").sort_index()
    dates = cal["date"]
    ev = cal[cal.event_name_1 != "none"][["date", "event_name_1"]].rename(columns={"date": "ds", "event_name_1": "holiday"})
    holidays = ev.assign(lower_window=0, upper_window=0)
    origin = TEST_START - 1
    nodes = nodes or list(S_all.columns)
    sku_state = h_frame.set_index("sku")["state_id"].to_dict()
    sku_store = h_frame.set_index("sku")["store_id"].to_dict()
    out = []
    for n in tqdm(nodes, disable=False):
        lo = origin - history_days
        y = S_all[n].loc[lo:]
        if n in sku_state:                                  # bottom series
            b = bt[bt.sku == n].set_index("d").loc[lo: TEST_END]
            extra = pd.DataFrame({"snap": b.snap, "price_ratio": b.price_ratio.fillna(0)})
        else:                                               # aggregate
            a = at[at.node == n].set_index("d").loc[lo: TEST_END]
            extra = pd.DataFrame({"snap": a.snap, "price_ratio": a.price_ratio})
        extra = extra.reindex(range(lo, TEST_END + 1)).ffill().fillna(0)
        r = _prophet_one(y.reindex(range(lo, TEST_END + 1)), dates.loc[lo:TEST_END], extra, holidays, origin, H)
        r["d"] = range(TEST_START, TEST_END + 1); r["node"] = n; r["window"] = "test"
        for c in ["point"] + qcols():
            r[c] = r[c].clip(lower=0)
        r[qcols()] = np.sort(r[qcols()].values, axis=1)      # two separate sampling runs (80%/95%) can cross -> sort
        out.append(r)
    return pd.concat(out, ignore_index=True)
