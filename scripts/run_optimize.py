"""Step 7: OR-Tools inventory optimisation + 4-week rolling backtest on the real test window."""
import sys, json, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd
from m5fc import config as C, optimize as O
from m5fc.features import build_all
from m5fc.store import load, FC
bt, at, S, h = build_all(); skus = list(h.sku)
price = bt[bt.d.between(C.TEST_START, C.TEST_END)].pivot(index="d", columns="sku", values="sell_price")
price_now = price.iloc[0].fillna(price.mean()).fillna(bt.groupby("sku").sell_price.mean())
wk = S.loc[C.TEST_START - 56: C.TEST_START - 1, skus].values.reshape(8, 7, -1).sum(1)
sigma_week = pd.Series(wk.std(0, ddof=1), index=skus)      # weekly demand std over the last 8 weeks
fast = "--quick" in sys.argv

PRIMARY = "Ensemble + MinT" if (FC / "ens_mint.parquet").exists() else "LightGBM + MinT"
SOURCES = {}
if (FC / "ens_mint.parquet").exists(): SOURCES["Ensemble + MinT"] = "ens_mint"
SOURCES.update({"LightGBM + MinT": "lgbm_mint", "LightGBM v1 + MinT": "lgbm_v1_mint", "Prophet": "prophet", "Seasonal naive": "snaive"})
if (FC / "tft_cqr.parquet").exists(): SOURCES["TFT (recalibrated)"] = "tft_cqr"
# calibrate the copula correlation on the calibration window (honest: test is untouched)
cal = load(SOURCES[PRIMARY], "cal")
rho, grid = O.calibrate_rho(cal, S, skus)
print("rho calibration (80% interval coverage of 7-day demand):", {k: round(v, 3) for k, v in grid.items()}, "-> rho =", rho)
out = {"rho": rho, "rho_grid": grid}; allbt = []
for label, m in SOURCES.items():
    t = time.time(); fc = load(m, "test")
    r = O.backtest(fc, fc, S, h, price_now, rho, label=label, sigma_hist=sigma_week, K=100 if fast else 200)
    allbt.append(r); print(label, f"{time.time()-t:.0f}s")
bt_all = pd.concat(allbt); bt_all.to_csv(C.RES / "inventory_backtest_rows.csv", index=False)
summ = O.summarise_backtest(bt_all); summ.to_csv(C.RES / "inventory_summary.csv", index=False)
pd.set_option("display.width", 250); print(summ.round(3).to_string())

# sensitivity: purchasing-budget tightness (primary forecasts)
fc = load(SOURCES[PRIMARY], "test"); sens = []
for bf in [0.85, 0.95, 1.05, 1.25, 10.0]:
    r = O.backtest(fc, fc, S, h, price_now, rho, budget_factor=bf, label=f"budget x{bf}", sigma_hist=sigma_week, K=100)
    s = O.summarise_backtest(r); s["budget_factor"] = bf; sens.append(s)
pd.concat(sens).to_csv(C.RES / "inventory_budget_sensitivity.csv", index=False)
print(pd.concat(sens)[["budget_factor", "policy", "cost", "fill_rate", "stockout_rate"]].round(3).to_string())
out["primary"] = PRIMARY
json.dump(out, open(C.RES / "inventory_info.json", "w"))
