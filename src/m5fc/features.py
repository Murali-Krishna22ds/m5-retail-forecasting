"""Feature engineering for the direct 28-day-ahead global models.

Leakage rule: every feature derived from SALES is computed on S.shift(MIN_LAG) (MIN_LAG = 28 = H),
so a model trained once can forecast any day of a 28-day horizon from a single origin.
Price / promo / calendar / SNAP features are *known in advance* (retailer plans them) and are not lagged.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .config import MIN_LAG, PROC
from .data import build_subset, hierarchy_frame, wide, aggregate

LAGS = [28, 35, 42, 49, 56, 91, 364]
ROLL = [7, 14, 28, 56, 168]


def sales_features(S: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """S: days x series. Returns dict name -> days x series, all using info <= t-MIN_LAG."""
    f = {}
    L = S.shift(MIN_LAG)
    for k in LAGS:
        f[f"lag_{k}"] = S.shift(k)
    for w in ROLL:
        f[f"rmean_{w}"] = L.rolling(w, min_periods=max(2, w // 4)).mean()
    for w in (7, 28):
        f[f"rstd_{w}"] = L.rolling(w, min_periods=max(2, w // 4)).std()
    f["zero_rate_28"] = (L == 0).astype(float).where(L.notna()).rolling(28, min_periods=7).mean()
    f["dow_mean_4w"] = sum(S.shift(k) for k in range(28, 56, 7)) / 4
    f["dow_mean_8w"] = sum(S.shift(k) for k in range(28, 84, 7)) / 8
    f["trend_ratio"] = f["rmean_7"] / (f["rmean_56"] + 1e-3)
    f["rmax_28"] = L.rolling(28, min_periods=7).max()
    return f


def calendar_frame(df: pd.DataFrame) -> pd.DataFrame:
    cal = df.drop_duplicates("d")[["d", "date", "wday", "month", "event_name_1", "event_type_1"]].sort_values("d")
    cal = cal.set_index("d")
    cal["dom"] = cal.date.dt.day
    cal["woy"] = cal.date.dt.isocalendar().week.astype(int)
    cal["is_event"] = (cal.event_name_1 != "none").astype(int)
    cal["event_type"] = cal.event_type_1.astype("category").cat.codes
    cal["event_code"] = cal.event_name_1.astype("category").cat.codes
    # days to the next / since the last event (known in advance)
    ev = cal.is_event.values
    idx = np.arange(len(ev))
    nxt = np.where(ev == 1, idx, np.nan); nxt = pd.Series(nxt).bfill().values
    prv = np.where(ev == 1, idx, np.nan); prv = pd.Series(prv).ffill().values
    cal["days_to_event"] = np.nan_to_num(np.clip(nxt - idx, 0, 30), nan=30.0)   # no event ahead/behind -> cap at 30
    cal["days_since_event"] = np.nan_to_num(np.clip(idx - prv, 0, 30), nan=30.0)
    return cal[["wday", "month", "dom", "woy", "is_event", "event_type", "event_code",
                "days_to_event", "days_since_event"]]


def build_all(force: bool = False):
    """Returns (bottom_table, agg_table, S_all, meta). Cached as parquet."""
    pb, pa, ps = PROC / "feat_bottom.parquet", PROC / "feat_agg.parquet", PROC / "S_all.parquet"
    df = build_subset()
    h = hierarchy_frame(df)
    S_b = wide(df, "sales").astype("float32")
    S_all = aggregate(S_b, h).astype("float32")
    if pb.exists() and pa.exists() and ps.exists() and not force:
        return pd.read_parquet(pb), pd.read_parquet(pa), pd.read_parquet(ps), h
    cal = calendar_frame(df)

    # ---- price / promo (known in advance) -------------------------------
    P = wide(df, "sell_price").astype("float32")
    pmax = P.rolling(91, min_periods=1).max()
    ratio = (P / pmax)                                   # 1.0 = regular price, <1 = markdown
    chg = P / P.shift(7) - 1
    item_mean = P.T.groupby(h.set_index("sku").item_id).transform("mean").T
    rel_item = P / item_mean
    avail = P.notna().astype("float32")
    snap = wide(df, "snap").astype("float32")

    # ---- bottom table -----------------------------------------------------
    f = sales_features(S_b)
    f.update({"sell_price": P, "price_ratio": ratio, "price_change_wk": chg, "price_rel_item": rel_item,
              "promo": (ratio < 0.97).astype("float32").where(P.notna()), "snap": snap})
    long = {k: v.stack(future_stack=True).rename(k) for k, v in f.items()}
    bt = pd.concat(long.values(), axis=1)
    bt["y"] = S_b.stack(future_stack=True)
    bt = bt.reset_index().rename(columns={"level_0": "d", "level_1": "sku"})
    bt = bt.merge(cal.reset_index(), on="d", how="left").merge(h, on="sku", how="left")
    bt["avail"] = avail.stack(future_stack=True).values
    for c in ["sku", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        bt[c + "_code"] = bt[c].astype("category").cat.codes.astype("int16")
    bt = bt.astype({c: "float32" for c in bt.columns if str(bt[c].dtype) == "float64"})

    # ---- aggregate table (Total, states, stores) -------------------------
    top = [c for c in S_all.columns if c not in set(h.sku)]
    Sa = S_all[top]
    fa = sales_features(Sa)
    members = {"Total": list(h.sku)}
    for s, g in h.groupby("state_id"): members[s] = list(g.sku)
    for s, g in h.groupby("store_id"): members[s] = list(g.sku)
    fa["promo_share"] = pd.DataFrame({n: (ratio[m] < 0.97).mean(axis=1) for n, m in members.items()})
    fa["price_ratio"] = pd.DataFrame({n: ratio[m].mean(axis=1) for n, m in members.items()})
    fa["avail_share"] = pd.DataFrame({n: avail[m].mean(axis=1) for n, m in members.items()})
    fa["snap"] = pd.DataFrame({n: snap[m].mean(axis=1) for n, m in members.items()})
    la = {k: v.stack(future_stack=True).rename(k) for k, v in fa.items()}
    at = pd.concat(la.values(), axis=1)
    at["y"] = Sa.stack(future_stack=True)
    at = at.reset_index().rename(columns={"level_0": "d", "level_1": "node"})
    at = at.merge(cal.reset_index(), on="d", how="left")
    at["node_code"] = at["node"].astype("category").cat.codes.astype("int16")
    at["level"] = np.where(at.node == "Total", 0, np.where(at.node.isin(["CA", "TX", "WI"]), 1, 2)).astype("int8")
    at = at.astype({c: "float32" for c in at.columns if str(at[c].dtype) == "float64"})

    bt.to_parquet(pb); at.to_parquet(pa); S_all.to_parquet(ps)
    return bt, at, S_all, h


BOTTOM_FEATS = (["lag_%d" % k for k in LAGS] + ["rmean_%d" % w for w in ROLL] + ["rstd_7", "rstd_28", "zero_rate_28",
                "dow_mean_4w", "dow_mean_8w", "trend_ratio", "rmax_28", "sell_price", "price_ratio",
                "price_change_wk", "price_rel_item", "promo", "snap", "wday", "month", "dom", "woy", "is_event",
                "event_type", "event_code", "days_to_event", "days_since_event",
                "item_id_code", "dept_id_code", "cat_id_code", "store_id_code", "state_id_code"])
AGG_FEATS = (["lag_%d" % k for k in LAGS] + ["rmean_%d" % w for w in ROLL] + ["rstd_7", "rstd_28", "zero_rate_28",
             "dow_mean_4w", "dow_mean_8w", "trend_ratio", "rmax_28", "promo_share", "price_ratio", "avail_share",
             "snap", "wday", "month", "dom", "woy", "is_event", "event_type", "event_code", "days_to_event",
             "days_since_event", "node_code", "level"])
BOTTOM_CATS = ["item_id_code", "dept_id_code", "cat_id_code", "store_id_code", "state_id_code", "event_type", "event_code"]
AGG_CATS = ["node_code", "event_type", "event_code"]
