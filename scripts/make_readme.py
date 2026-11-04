"""Builds README.md from result files (numbers.json is produced by scripts/export_numbers.py)."""
import json, sys; sys.path.insert(0, "src")
import pandas as pd
from m5fc.config import RES
N = json.load(open("presentation/numbers.json")); dec = json.load(open(RES / "rolling_origin_decision.json")); best = json.load(open(RES / "best_params.json"))
t = pd.read_csv(RES / "metrics_test.csv"); g = lambda m, l, c: t[(t.model == m) & (t.level == l)][c].values[0]
R = N["rows"]; I = N["inv"]
tab = "\n".join(f"| {n} | " + " | ".join(R[k]) + " |" for n, k in [("Seasonal naive", "snaive"), ("Prophet (+holidays, SNAP, price)", "prophet"), ("LightGBM + MinT (previous version)", "lgbm_v1_mint"), ("**Ensemble + MinT (final)**", "ens_mint"), ("TFT, avg of 2 (item level only)", "tft_avg")])
open("README.md", "w").write(f"""# M5 Retail Demand Forecasting & Replenishment

End-to-end pipeline on the real **M5 (Walmart)** data (official Kaggle files; the whole pipeline is executed by `scripts/run_all.sh`):
features -> baselines (naive, Prophet) -> **LightGBM** + **Temporal Fusion Transformer** (2 models) -> **ensemble** -> **hierarchical reconciliation (Darts)** ->
**calibrated prediction intervals** -> **OR-Tools order optimisation** -> **Plotly Dash** app.

![architecture](docs/architecture.png)

## Headline results (held-out test window d1914-d1941; nothing was selected, tuned or calibrated on it)
| | WAPE Total | WAPE State | WAPE Store | WAPE item-store | RMSSE item-store | 80% / 95% coverage (item-store) |
|---|---|---|---|---|---|---|
{tab}

* **Ensemble + MinT vs Prophet**: WAPE gain {N["boot_prophet"]} (paired bootstrap over series).
* **Ensemble + MinT vs the previous version of this project**: item-level WAPE gain {N["boot_v1"]} ({N["item_gain"]} relative). RMSSE is level. **Top levels are slightly worse**
  (Total {R["ens_mint"][0]} vs {R["lgbm_v1_mint"][0]}, Store {R["ens_mint"][2]} vs {R["lgbm_v1_mint"][2]}) - a real trade-off, not hidden. Details and all other models in [`docs/RESULTS.md`](docs/RESULTS.md).
* **Prediction intervals**: final ensemble coverage {N["cov"]["ens_mint"]} (nominal 0.80 / 0.95) after conformal recalibration on the calibration window; scaled pinball loss {N["spl"]["ens_mint"]} (Prophet {N["spl"]["prophet"]}, seasonal naive {N["spl"]["snaive"]}).
  *Correction of an earlier claim:* raw TFT intervals look badly under-covered ({N["cov"]["tft_avg"]}) only because TFT's lower bounds are strictly positive (e.g. 0.0004) while demand is often exactly 0; recalibration brings TFT to {N["cov"]["tft_cqr"]}.
* **Reconciliation**: all methods give exact coherence (gap ~1e-13, `results/coherence_audit.csv`). Its effect on accuracy is mixed (LightGBM Total WAPE {N["recon"]["lgbm"][0]} -> {N["recon"]["lgbm_mint"][0]}, Item {N["recon"]["lgbm"][3]} -> {N["recon"]["lgbm_mint"][3]}; ensemble Total {N["recon"]["ens"][0]} -> {N["recon"]["ens_mint"][0]}, Item {N["recon"]["ens"][3]} -> {N["recon"]["ens_mint"][3]}). You pay a little accuracy for guaranteed coherence.
* **Inventory (OR-Tools MILP, 4-week rolling backtest on real demand, 10 stores x 40 SKUs)**: realised holding + stockout cost **{I["milp"]}** vs {I["mean"]} ordering the mean forecast ({I["cut_vs_mean"]} lower) and {I["safety"]} for mean + 1.28σ safety stock ({I["cut_vs_safety"]} lower);
  fill rate {I["fill_milp_1dp"]} vs ~{I["fill_mean"]}. With the previous version's forecasts the same optimiser costs {I["v1_milp"]}. Perfect-foresight floor {I["oracle"]}. The MILP carries more stock - it buys service (see budget sensitivity in `results/`).

## How accuracy was (and was not) improved - honestly
1. **Extended features** (cross-series item/store context, recency, EWMs, extra price features; all leak-tested).
2. **Search on a separate selection window** (d1858-d1885) found hyper-parameters that looked better *there*. They **did not generalise**: a rolling-origin check on four earlier clean windows
   (decision rule written before running) scored them {dec["scores_vs_A"]["B_tuned"]:.4f} vs 1.0 for the previous config, and they were worse on the test window too. The rule adopted
   **{best["adopted_config"]}** (extended features, default parameters; score {dec["scores_vs_A"].get("C_ext_default", float("nan")):.4f}).
3. **Ensembling** LightGBM with a 2-model TFT average (TFT weight {N["weight"]}, chosen on the calibration window) gave the real item-level improvement.
4. Conformal recalibration of intervals; deterministic seeding; {N["tests"]} tests.
**Disclosure:** the tuned configuration's test score was viewed once before the rolling-origin check was run (it was worse than v1). The check's decision rule was written before running and uses only four clean pre-selection windows (d1718-d1829), never the test window; the ensemble design and final-model choice were fixed before their test results existed; nothing was changed after seeing the final test results.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
# put the official Kaggle files in data/raw/ (calendar.csv, sell_prices.csv, sales_train_evaluation.csv)  -- or: make download (Nixtla mirror)
make all                                   # = scripts/run_all.sh ; ~100 min on 1 CPU (SKIP_* env vars reuse finished steps)
make app                                   # http://127.0.0.1:8050
```
Deploy: `docker build -t m5 . && docker run -p 8050:8050 m5` or Render via `render.yaml` (serves precomputed forecasts + live OR-Tools re-optimisation).

## Dashboard
**Forecasts** (any of 414 nodes, model overlay, 80/95% bands, per-node WAPE/RMSSE/coverage) · **Accuracy** · **Reconciliation** (gap before/after + audit) ·
**Order recommendations** (live MILP per store/week; sliders for budget, shelf capacity, stockout penalty, on-hand stock; expected vs realised cost and fill rate; backtest vs heuristics).

## Design decisions worth knowing
1. **Direct multi-horizon with lags >= 28** - one model for all 28 days; `test_no_leakage_*` corrupt future data to prove no leakage (also for the cross-series features).
2. **Disjoint windows**: selection d1858-1885 (models <= d1857), calibration d1886-1913 (early stopping, blend weight, MinT weights, conformal, copula rho), test d1914-1941.
3. **Reconciliation**: Darts `MinTReconciliator(wls_var)`; negatives clipped and upper levels re-summed so output is exactly coherent *and* non-negative. Interval bands shift only for aggregate nodes (integer-valued bottom bounds must not move by fractions).
4. **Optimiser**: scenarios from quantile forecasts via a Gaussian copula (rho calibrated on the calibration window); expected cost per SKU is convex in packs ordered, so it is linearised exactly -> small MILP solved by SCIP in ~0.5 s per store, verified equal to the explicit scenario formulation.

## Bugs found by testing (fixed, with regression tests)
NaN calendar features crashed TFT · shifting integer interval bounds collapsed coverage · Prophet quantiles crossed · `hash()`-seeded scenarios and unseeded Prophet sampling made runs non-reproducible ·
Dash crashed for TFT above item level · cross-series features initially produced zeros/explosions at the start of history (caught by range tests) · a flawed monotonicity test (capacity binding).

## Limitations (honest)
* **Subset**: 400 of 30,490 series (1 CPU / 3 GB build machine); items sampled across the volume range, so scores are **not comparable to the Kaggle leaderboard**. One 28-day test window; improvements of ~1% are near the noise level (see bootstrap CIs).
* Models are not refit on the calibration window before testing. TFT is small and briefly trained. Aggregate-level (Total/State/Store) quantile models see only 14 series (Total 95% lower band is too wide); intervals are not jointly coherent.
* **Inventory economics are assumptions** (`docs/DATA.md`); dollar figures illustrate the method, relative comparisons are the result. One backtest, no CI on cost savings.
* Not deployed (no hosting access) and no screen recording: see `docs/DEMO.md`. Individual reflections must be written by each intern (`docs/REFLECTION_TEMPLATE.md`).

## Layout
```
src/m5fc/   data, features, metrics, baselines, lgbm, tft, ensemble, reconcile, evaluate, optimize, store
scripts/    run_all.sh and run_*.py steps, tune_lgbm.py, rolling_origin_lgbm.py, apply_decision.py, make_report.py, make_readme.py, export_numbers.py
app/app.py  Dash app        tests/  {N["tests"]} pytest tests        docs/  architecture, DATA, RESULTS, DEMO, demo.html, reflection template
results/    metrics, forecasts, figures, inventory backtests, tuning / rolling-origin logs
```
""")
print("README written")
