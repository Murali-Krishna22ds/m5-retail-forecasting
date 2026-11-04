# Data & API documentation

## 1. Source data (M5 Forecasting - Accuracy, Walmart)
**Official Kaggle M5 files**, supplied by the team and read directly from `data/raw/` (`sales_train_evaluation.csv`, `sales_train_validation.csv`,
`calendar.csv`, `sell_prices.csv`, `sample_submission.csv`). The whole pipeline was executed end-to-end on these files (`scripts/run_all.sh`).
Earlier development used the identical Nixtla GitHub mirror; the loader accepts both layouts (the mirror has no `id`/`d` columns).

| file | grain | key columns |
|---|---|---|
| `sales_train_evaluation.csv` | item x store x day (wide: `d_1..d_1941`) | `item_id, dept_id, cat_id, store_id, state_id` |
| `calendar.csv` | day (1969 rows incl. 28-day horizon) | `date, wm_yr_wk, wday, month, year, event_name_1/2, event_type_1/2, snap_CA/TX/WI` |
| `sell_prices.csv` | store x item x walmart-week | `sell_price` (missing => item not on sale that week) |

**Hierarchy** (M5 has 12 aggregation levels; we model the 4 that carry the reconciliation question):

```
Total (1)  >  State (3: CA, TX, WI)  >  Store (10: CA_1-4, TX_1-3, WI_1-3)  >  Item-Store (400 bottom series)
                                                         item -> dept -> category (FOODS / HOUSEHOLD / HOBBIES) are series attributes
```
414 nodes = 1 + 3 + 10 + 400.  Summing matrix `S` is built by Darts from the hierarchy dictionary in `reconcile.hierarchy_dict`.

### Subset used (compute budget: 1 CPU, 3 GB RAM)
Stratified by category (20 FOODS / 12 HOUSEHOLD / 8 HOBBIES items), sampled evenly across the volume distribution between the
98th and 50th percentile of last-365-day volume (so both smooth and intermittent sellers), `seed=42`.
`config.N_ITEMS` changes this; the pipeline is full-M5 compatible (memory/time scale roughly linearly).
All 10 stores/3 states are kept, so the Total > State > Store reconciliation is faithful to M5.

### Time split (no look-ahead anywhere)
| window | days | use |
|---|---|---|
| selection | `d_1858 .. d_1885` (models trained `<= d_1857`) | feature / hyper-parameter search only (`scripts/tune_lgbm.py`) |
| rolling-origin check | 4 windows d1718-1829 + report-only windows | decide which LightGBM config to keep (`scripts/rolling_origin_lgbm.py`) |
| train | `d_1 .. d_1885` | fit all models |
| calibration | `d_1886 .. d_1913` (28 d) | early stopping, MinT residual weights, copula correlation, conformal check |
| **test** | `d_1914 .. d_1941` (28 d, 2016-04-25 .. 2016-05-22) | all reported numbers; never used for fitting or tuning |

### Provenance
The processed subset built from the official files is identical (series, sales, prices, SNAP, events, dates) to the one built earlier from the mirror,
and the full pipeline reproduces bit-for-bit (LightGBM, TFT, reconciliation, MILP backtest; Prophet intervals after seeding).
`python scripts/verify_against_kaggle.py <folder>` re-checks `sales_train_validation.csv` (d_1..d_1913) against the processed subset.

## 2. Processed artefacts (`data/processed/`, git-ignored, rebuilt by `make data features`)
| file | content |
|---|---|
| `subset.parquet` | tidy long table: `sku, d, date, sales, sell_price, snap, events, ids` (776,400 rows) |
| `feat_bottom.parquet` | 776,400 x 59 model table for bottom series |
| `feat_agg.parquet` | 27,174 x 40 model table for Total/State/Store nodes |
| `S_all.parquet` | `1941 x 414` actuals for every hierarchy node |

### Features (all sales-derived features use data `<= t-28`; enforced by `tests/test_features.py`, including the cross-series ones)
`lag_{28,35,42,49,56,91,364}`, `rmean_{7,14,28,56,168}`, `rstd_{7,28}`, `zero_rate_28`, `dow_mean_{4w,8w}`, `trend_ratio`, `rmax_28`;
known-in-advance: `sell_price, price_ratio` (price / 91-day max), `price_change_wk, price_rel_item, promo` (ratio < 0.97), `snap`,
`wday, month, dom, woy, is_event, event_type, event_code, days_to_event, days_since_event`; categorical ids for item/dept/cat/store/state.
Added in the optimised version (+10, adopted by the rolling-origin rule): `ewm_01, ewm_03, days_since_sale`, cross-series context `item_rmean28_xstore, store_rmean28, share_in_store, store_trend`, price context `price_rel_catstore, days_since_price_change, price_vs_min91`.
Rows where the item has no price (not on sale) are excluded from training and forecast as exactly 0 (M5 convention; verified by `test_no_price_means_no_sales_mostly`).

## 3. Forecast store (`results/forecasts/<model>.parquet`)
Long format, one file per model - the single interface between modelling, evaluation, reconciliation, optimiser and app.

| column | meaning |
|---|---|
| `window` | `cal` (d1886-1913) or `test` (d1914-1941) |
| `d`, `node` | day index, hierarchy node (`Total`, `CA`, `CA_1`, `FOODS_3_090\|CA_1`, ...) |
| `point` | point forecast (mean for LGBM/Prophet, median for TFT) |
| `q0.025 q0.1 q0.5 q0.9 q0.975` | quantile forecasts -> 80% = [q0.1, q0.9], 95% = [q0.025, q0.975]; sorted, non-negative |

Model ids: `naive, snaive, prophet, lgbm` (base, incoherent), `lgbm_bu/_td/_ols/_wlss/_mint` (reconciled), `lgbm_v1*` (archived previous version, for before/after), `tft`, `tft2`, `tft_avg`, `tft_cqr`, `tft_bu`, `ens_raw` (blend), `ens` (blend + conformal), `ens_mint` (**final**), `ens_bu`.

## 4. Python API (`src/m5fc`)
| module | main entry points |
|---|---|
| `data` | `build_subset()`, `hierarchy_frame(df)`, `aggregate(bottom_wide, h)` |
| `features` | `build_all()` -> `(bottom_table, agg_table, S_all, hierarchy)`, `sales_features(S)` |
| `metrics` | `wape, rmsse, rmsse_scale, pinball, scaled_pinball, coverage, winkler` |
| `baselines` | `run_naive(S)`, `run_prophet(...)` |
| `lgbm` | `run()` (train point+quantile models using `results/best_params.json`), `cqr()` |
| `ensemble` | `average`, `blend`, `choose_weight` (calibration window), `scaled_cqr` (conformal recalibration) |
| `tft` | `run(bt, epochs, batches_per_epoch)` |
| `reconcile` | `reconcile_model(model, S, h, method, out_name)`, `coherence_error(...)` |
| `evaluate` | `score(model, ...)`, `score_all(models, ...)` |
| `optimize` | `item_params`, `cycle_scenarios`, `calibrate_rho`, `solve_store` (fast exact MILP), `solve_store_explicit`, `backtest`, `summarise_backtest` |

### Command-line pipeline (`Makefile`)
`make all` = `scripts/run_all.sh`: features, baselines, tune, rolling-origin, decision, lgbm, tft x2, ensemble, reconcile, eval, optimize, report, tests (SKIP_* env vars reuse finished steps).

## 5. Inventory-model assumptions (all in `config.py` - **not** M5 data, stated explicitly)
| parameter | value | rationale |
|---|---|---|
| unit cost | 70% of shelf price | typical grocery gross margin ~30% |
| holding + shrink | 6% of unit cost per week | capital + storage + spoilage/markdown; makes critical ratio ~0.9 |
| stockout cost | 1.0 x margin + 10% of price | lost margin + goodwill |
| case pack | FOODS 6, HOUSEHOLD 4, HOBBIES 3 | integer ordering |
| shelf volume / unit | 1.0 / 2.0 / 1.5 | capacity constraint |
| budget | x1.05 of the cost of ordering the mean forecast (UI slider 0.7-1.5) | binding purchasing limit |
| shelf capacity | x1.5 of mean-demand volume | storage limit |
| cycle | 7 days, weekly re-plan with leftover stock carried over | |
Because these are assumptions, absolute dollar figures illustrate the method; **relative** comparisons (MILP vs heuristics, forecast A vs B) are the result.
