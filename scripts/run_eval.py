"""Step 6: score every model (test + calibration windows), bootstrap CIs, category breakdown."""
import sys, json, warnings, os; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from m5fc import metrics as M
from m5fc.config import TEST_START, TEST_END, RES
from m5fc.features import build_all
from m5fc.evaluate import score_all, level_of
from m5fc.store import load, to_wide, save, FC
bt, at, S, h = build_all(); bottoms = set(h.sku)

# TFT covers bottom series only -> also provide a coherent bottom-up version (point forecasts, all levels)
if (FC / "tft.parquet").exists():
    from m5fc.data import aggregate
    t = load("tft"); parts = []
    for w, g in t.groupby("window"):
        P = to_wide(g, "point")[list(h.sku)]
        A = aggregate(P, h).stack().rename("point").reset_index(); A.columns = ["d", "node", "point"]; A["window"] = w
        parts.append(A)
    save("tft_bu", pd.concat(parts, ignore_index=True))

models = ["naive", "snaive", "prophet", "lgbm", "lgbm_bu", "lgbm_td", "lgbm_ols", "lgbm_wlss", "lgbm_mint"]
models += [m for m in ("tft", "tft_bu") if (FC / f"{m}.parquet").exists()]
for w in ("test", "cal"):
    ms = [m for m in models if w in set(load(m)["window"]) ]
    r = score_all(ms, S, bottoms, bt, w); r.to_csv(RES / f"metrics_{w}.csv", index=False)
te = pd.read_csv(RES / "metrics_test.csv")
pd.set_option("display.width", 250)
print(te.drop(columns=["window"]).round(3).to_string())

# ---- paired bootstrap over item-store series (pooled WAPE difference) ----------------------------
Y = S.loc[TEST_START:TEST_END, list(h.sku)]
def abs_err(m): return (Y - to_wide(load(m, "test"), "point")[Y.columns]).abs().sum()
ae = {m: abs_err(m) for m in ["snaive", "prophet", "lgbm", "lgbm_mint"] + (["tft"] if "tft" in models else [])}
tot = Y.sum(); rng = np.random.default_rng(0); idx = np.arange(len(Y.columns)); out = {}
for a, b in [("prophet", "lgbm_mint"), ("snaive", "lgbm_mint"), ("lgbm", "lgbm_mint")] + ([("tft", "lgbm_mint"), ("prophet", "tft")] if "tft" in ae else []):
    d = []
    for _ in range(2000):
        s = rng.choice(idx, len(idx)); d.append((ae[a].values[s].sum() - ae[b].values[s].sum()) / tot.values[s].sum())
    out[f"WAPE({a}) - WAPE({b})"] = {"mean": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
json.dump(out, open(RES / "bootstrap_wape_diff.json", "w"), indent=2); print(json.dumps(out, indent=1))

# ---- breakdown by category & intermittency -----------------------------------------------------
rows = []
adi = {c: (len(Y) / max((S.loc[:TEST_START - 1, c] > 0).sum(), 1)) for c in h.sku}
for m in ["snaive", "prophet", "lgbm_mint"] + (["tft"] if "tft" in models else []):
    P = to_wide(load(m, "test"), "point")[Y.columns]
    for cat, g in h.groupby("cat_id"):
        rows.append(dict(model=m, group=cat, WAPE=M.wape(Y[g.sku].values, P[g.sku].values)))
    hist = S.loc[:TEST_START - 1, list(h.sku)]
    zr = (hist.iloc[-365:] == 0).mean()
    for name, mask in {"smooth (<30% zero days)": zr < 0.3, "intermittent (30-60%)": (zr >= 0.3) & (zr < 0.6), "sparse (>=60%)": zr >= 0.6}.items():
        cols = list(zr[mask].index)
        if cols: rows.append(dict(model=m, group=name, WAPE=M.wape(Y[cols].values, P[cols].values), n_series=len(cols)))
pd.DataFrame(rows).to_csv(RES / "wape_by_group.csv", index=False)
print(pd.DataFrame(rows).pivot(index="group", columns="model", values="WAPE").round(3))
