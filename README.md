# M5 Retail Demand Forecasting & Replenishment

End-to-end pipeline on the real **M5 (Walmart)** data: features -> baselines (naive, Prophet) -> **LightGBM** and **Temporal Fusion Transformer** ->
**hierarchical reconciliation (Darts)** -> **prediction intervals** -> **OR-Tools order optimisation** -> **Plotly Dash** app.

![architecture](docs/architecture.png)

## Headline results (held-out test window d1914-d1941, never used for fitting/tuning)
| | WAPE Total | WAPE State | WAPE Store | WAPE item-store | RMSSE item-store | 80% / 95% coverage (item-store) |
|---|---|---|---|---|---|---|
| Seasonal naive | 0.185 | 0.202 | 0.236 | 0.884 | 0.971 | 0.78 / 0.86 |
| Prophet (+holidays, SNAP, price) | 0.143 | 0.140 | 0.205 | 0.812 | 0.812 | 0.76 / 0.89 |
| LightGBM (direct, global) | 0.121 | 0.124 | 0.149 | 0.689 | 0.735 | 0.82 / 0.97 |
| **LightGBM + MinT (Darts)** | **0.099** | **0.119** | 0.155 | 0.696 | 0.736 | 0.82 / 0.97 |
| TFT (PyTorch Forecasting) | - | - | - | **0.658** | 0.777 | 0.47 / 0.56 (under-covers) |

* LightGBM beats Prophet by 0.116 WAPE (paired bootstrap 95% CI [0.076, 0.162]).
* **Reconciliation**: base forecasts were incoherent by up to ~80 units; all 5 methods give exact coherence (gap ~1e-13, `results/coherence_audit.csv`). MinT improves Total WAPE 18% (0.121 -> 0.099) and State 4%, at a small cost at Store (+4%) and Item level (+1%) - a real trade-off, reported not hidden.
* **TFT** has the best item-level WAPE (it forecasts the median, which WAPE rewards on intermittent demand) but worse RMSSE than LightGBM, **badly under-covered intervals (47% vs 80% nominal)** and biased aggregates when summed bottom-up (Total WAPE 0.265). In this compute budget (11 min on 1 CPU, 33.8k parameters) LightGBM is the better production choice.
* **Inventory (OR-Tools MILP, 4-week rolling backtest on real demand, 10 stores x 40 SKUs)**: realised holding + stockout cost **$6.1k vs $10.9k** ordering the mean forecast (-44%) and $10.6k for mean + 1.28σ safety stock (-43%); fill rate 78.8% vs 61.3% / 62.4%. Holds for every forecast source (Prophet $7.1k, seasonal naive $6.9k, TFT scenarios $5.9k). Perfect foresight floor: $0.45k. The MILP carries more inventory ($307 vs $110 per store-week) - it is buying service, and the budget sensitivity (`results/inventory_budget_sensitivity.csv`) shows the trade-off curve.

Full tables, bootstrap CIs, per-segment WAPE (smooth / intermittent / sparse) and all figures: [`docs/RESULTS.md`](docs/RESULTS.md). Data & API docs: [`docs/DATA.md`](docs/DATA.md).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
make download                              # real M5 from the Nixtla GitHub mirror (or put Kaggle files in data/raw/)
make all                                   # data -> features -> baselines -> lgbm -> tft -> reconcile -> eval -> optimize -> report -> test
make app                                   # http://127.0.0.1:8050
```
Runtime on 1 CPU / 3 GB RAM: features 5 s, Prophet 100 s, LightGBM 110 s, TFT ~12 min, inventory backtest ~4 min, tests 9 s.
Deploy: `docker build -t m5 . && docker run -p 8050:8050 m5` or Render via `render.yaml` (serves precomputed forecasts + live OR-Tools re-optimisation).

## Dashboard
Four tabs: **Forecasts** (any of 414 nodes, model overlay, 80/95% bands, per-node WAPE/RMSSE/coverage) · **Accuracy** (metric x level charts and full table) ·
**Reconciliation** (state-minus-sum-of-stores gap before/after + coherence audit) · **Order recommendations** (live MILP per store/week with sliders for purchasing budget, shelf capacity, stockout penalty and on-hand stock; KPIs for expected vs realised cost and fill rate; backtest vs heuristics).

## Design decisions worth knowing
1. **Direct multi-horizon with lags >= 28** - one model for all 28 days, no recursive error, and a unit test (`test_no_target_leakage_synthetic`) perturbs future data to prove no leakage.
2. **Three disjoint windows**: train <= d1885, calibration d1886-1913 (early stopping, MinT weights, copula rho), test d1914-1941.
3. **Intervals**: LightGBM/TFT quantile heads at 2.5/10/50/90/97.5 %. Conformal (CQR) check on the calibration window found adjustment = 0 at the bottom level.
4. **Reconciliation**: Darts `MinTReconciliator(wls_var)` on calibration residuals; negatives clipped and upper levels re-summed so output is exactly coherent *and* non-negative.
5. **Optimiser**: demand scenarios from the quantile forecasts via a Gaussian copula (rho = 0.2 chosen on calibration window so 7-day sums are ~80% covered). Expected cost per SKU is convex in packs ordered, so it is linearised exactly with chord constraints -> MILP with 80 variables per store solved by SCIP in ~0.5 s; verified equal to the explicit scenario formulation (`tests/test_optimize.py`).

## Bugs found by testing (kept in history as commits)
* NaN `days_to_event` at calendar edges crashed TFT. · Shifting integer-valued interval bounds by reconciliation adjustments dropped coverage 82% -> 64% (fixed: shift aggregates only). · Prophet's 80% and 95% bands came from separate sampling runs and crossed (caught by `test_forecast_shape_finite_nonneg_monotone`). · `hash()`-seeded scenarios made backtests non-reproducible (now CRC32; verified bit-identical). · Dash app crashed when TFT was selected above item level.

## Limitations (honest)
* **Subset**: 40 items x 10 stores (400 of 30,490 series) because of the 1-CPU/3 GB build machine; items were sampled across the volume range, not randomly from all of M5, so absolute scores are **not comparable to the Kaggle leaderboard**. Single 28-day test window, no rolling-origin CV.
* Models are not refit on the calibration window before testing (slightly stale; refitting should help).
* TFT is small and briefly trained, and its quantile heads are poorly calibrated; no hyper-parameter search. TFT's *point* WAPE edge is partly a median-vs-mean effect.
* The aggregate-level (Total/State/Store) quantile models see only 14 series; the Total 95% lower band is too wide (see `fig1`). Reconciliation here adjusts point forecasts; intervals are not jointly coherent (probabilistic reconciliation would be the next step - the copula sampler in `optimize.py` is a starting point).
* **Inventory economics are assumptions** (70% cost ratio, 6%/week holding, margin + 10% goodwill, case packs, volumes, budget = 1.05x mean-order cost; `docs/DATA.md`). Dollar figures illustrate the method; relative comparisons are the result. One 4-week backtest on 400 SKUs - no confidence intervals on the cost savings.
* Not deployed: no hosting account was available; Dockerfile/render.yaml are provided but untested on a host. No screen recording: see `docs/DEMO.md`.

## Layout
```
src/m5fc/   data, features, metrics, baselines, lgbm, tft, reconcile, evaluate, optimize, store
scripts/    run_*.py pipeline steps, make_report.py, make_architecture.py
app/app.py  Plotly Dash app           tests/  48 pytest tests           docs/  architecture, DATA, RESULTS, DEMO, reflection
results/    metrics, forecasts, figures, inventory backtests
```
