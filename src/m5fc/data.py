"""Load M5, pick a stratified subset, return a tidy long frame + hierarchy."""
from __future__ import annotations
import numpy as np, pandas as pd
from .config import RAW, PROC, N_ITEMS, SEED, LAST_TRAIN_D

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def select_items(sales: pd.DataFrame) -> list[str]:
    """Stratified by category; items spread over the volume distribution
    (top-90% down to median volume) so we get fast AND intermittent sellers."""
    rng = np.random.default_rng(SEED)
    dcols = [f"d_{i}" for i in range(LAST_TRAIN_D - 364, LAST_TRAIN_D + 1)]
    vol = sales.groupby(["cat_id", "item_id"])[dcols].sum().sum(axis=1)
    chosen = []
    for cat, n in N_ITEMS.items():
        v = vol.loc[cat].sort_values(ascending=False)
        pool = v.iloc[int(0.02 * len(v)): int(0.5 * len(v))]
        idx = np.linspace(0, len(pool) - 1, n).astype(int)
        chosen += list(pool.index[idx])
    return chosen


def build_subset(force: bool = False) -> pd.DataFrame:
    out = PROC / "subset.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    hdr = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=1)
    dcols = [c for c in hdr.columns if c.startswith("d_")]
    dtypes = {c: "int16" for c in dcols}
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv", dtype=dtypes)
    items = select_items(sales)
    sales = sales[sales.item_id.isin(items)].copy()
    long = sales.melt(id_vars=ID_COLS, value_vars=dcols, var_name="d", value_name="sales")
    long["d"] = long["d"].str[2:].astype("int16")
    cal = pd.read_csv(RAW / "calendar.csv")
    cal["d"] = np.arange(1, len(cal) + 1).astype("int16")  # mirror has no `d` column; row i == d_i
    cal["date"] = pd.to_datetime(cal["date"])
    long = long.merge(cal[["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1",
                           "event_type_1", "snap_CA", "snap_TX", "snap_WI"]], on="d", how="left")
    long["snap"] = np.select([long.state_id == "CA", long.state_id == "TX"],
                             [long.snap_CA, long.snap_TX], long.snap_WI).astype("int8")
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    pr = pd.read_csv(RAW / "sell_prices.csv")
    long = long.merge(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["sku"] = long["item_id"] + "|" + long["store_id"]
    long = long.sort_values(["sku", "d"]).reset_index(drop=True)
    for c in ["event_name_1", "event_type_1"]:
        long[c] = long[c].fillna("none")
    long.to_parquet(out)
    return long


def hierarchy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per bottom series with its ancestors."""
    h = df[["sku", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].drop_duplicates("sku")
    return h.sort_values("sku").reset_index(drop=True)


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """days x sku matrix."""
    return df.pivot(index="d", columns="sku", values=col).sort_index()


def node_names(h: pd.DataFrame):
    """Hierarchy node names: Total > State > Store > sku (bottom)."""
    states = sorted(h.state_id.unique())
    stores = sorted(h.store_id.unique())
    return ["Total"], states, stores, list(h.sku)


def aggregate(bottom: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """bottom: days x sku -> days x [Total, states, stores, skus] (sum)."""
    st = h.set_index("sku")
    out = {"Total": bottom.sum(axis=1)}
    for s, g in st.groupby("state_id"):
        out[s] = bottom[g.index].sum(axis=1)
    for s, g in st.groupby("store_id"):
        out[s] = bottom[g.index].sum(axis=1)
    top = pd.DataFrame(out)
    return pd.concat([top, bottom], axis=1)


def date_index() -> pd.Series:
    """d -> calendar date (d_1 = 2011-01-29)."""
    cal = pd.read_csv(RAW / "calendar.csv", usecols=["date"])
    return pd.Series(pd.to_datetime(cal["date"]).values, index=np.arange(1, len(cal) + 1), name="date")
