"""Check that the processed subset equals the official Kaggle M5 files (d_1..d_1913 are public in sales_train_validation.csv).
Usage: python scripts/verify_against_kaggle.py /path/to/m5-forecasting-accuracy   (folder with calendar.csv, sell_prices.csv, sales_train_validation.csv)
Prints PASS/FAIL per check; exits non-zero on any mismatch."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from m5fc.features import build_all
from m5fc.data import build_subset
K = (sys.argv[1] if len(sys.argv) > 1 else "data/kaggle").rstrip("/") + "/"
bt, at, S, h = build_all(); df = build_subset(); mine = set(h.sku); ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)
hdr = pd.read_csv(K + "sales_train_validation.csv", nrows=1); dc = [c for c in hdr.columns if c.startswith("d_")]
sv = pd.read_csv(K + "sales_train_validation.csv", dtype={c: "int16" for c in dc}); sv["sku"] = sv.item_id + "|" + sv.store_id
sv = sv[sv.sku.isin(mine)].set_index("sku").loc[sorted(mine)]
chk(f"all {len(mine)} series present in Kaggle file", len(sv) == len(mine))
chk(f"sales identical for d_1..d_{len(dc)}", np.array_equal(sv[dc].T.values.astype(float), S.loc[1:len(dc), list(sv.index)].values))
cal = pd.read_csv(K + "calendar.csv"); mc = df.drop_duplicates("d").sort_values("d")
chk("calendar dates identical (d_1..d_1941)", (pd.to_datetime(cal.date.iloc[:1941]).values == mc.date.values).all())
chk("calendar events identical", (cal.event_name_1.fillna("none").iloc[:1941].values == mc.event_name_1.values).all())
pr = pd.read_csv(K + "sell_prices.csv"); pr["sku"] = pr.item_id + "|" + pr.store_id; pr = pr[pr.sku.isin(mine)]
m = df[["sku", "wm_yr_wk", "sell_price"]].drop_duplicates(["sku", "wm_yr_wk"]).dropna()
j = m.merge(pr[["sku", "wm_yr_wk", "sell_price"]], on=["sku", "wm_yr_wk"], how="left", suffixes=("_p", "_k"))
chk(f"all {len(m):,} price rows present and equal", j.sell_price_k.notna().all() and np.allclose(j.sell_price_p, j.sell_price_k))
print(f"NOTE: Kaggle validation file ends at d_{len(dc)}; the test window d1914-d1941 comes from the evaluation release.")
sys.exit(0 if ok else 1)
