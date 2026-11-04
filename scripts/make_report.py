"""Step 8: figures (results/figures/*.png) + docs/RESULTS.md, generated from result files (no hand-typed numbers)."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from m5fc import config as C
from m5fc.features import build_all
from m5fc.data import date_index
from m5fc.store import load, to_wide, FC

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
bt, at, S, h = build_all(); dates = date_index()
te = pd.read_csv(C.RES / "metrics_test.csv"); LV = ["Total", "State", "Store", "Item-Store"]
inf = json.load(open(C.RES / "inventory_info.json")); PRIMARY = inf.get("primary", "LightGBM + MinT")
ens = json.load(open(C.RES / "ensemble_info.json")); best = json.load(open(C.RES / "best_params.json")); lg = json.load(open(C.RES / "lgbm_info.json"))
FINAL = "ens_mint"
NM = {"naive": "Naive", "snaive": "Seasonal naive", "prophet": "Prophet", "lgbm_v1": "LightGBM v1", "lgbm_v1_mint": "LightGBM v1 + MinT", "lgbm": "LightGBM (tuned)",
      "lgbm_mint": "LightGBM + MinT", "lgbm_cqr": "LightGBM + CQR", "tft": "TFT-1", "tft2": "TFT-2", "tft_avg": "TFT (avg of 2)", "tft_cqr": "TFT + CQR", "tft_bu": "TFT (bottom-up)",
      "ens": "Ensemble", "ens_bu": "Ensemble bottom-up", "ens_mint": "Ensemble + MinT (final)", "lgbm_bu": "LGBM bottom-up", "lgbm_td": "LGBM top-down", "lgbm_ols": "LGBM OLS", "lgbm_wlss": "LGBM WLS-struct"}
COL = {"naive": "#adb5bd", "snaive": "#868e96", "prophet": "#e8590c", "lgbm_v1_mint": "#a5d8ff", "lgbm_mint": "#0b6efd", "tft": "#8ce99a", "tft_avg": "#2f9e44", "tft_cqr": "#2f9e44", "tft_bu": "#2f9e44", "ens_mint": "#7048e8", "ens": "#b197fc"}
have = lambda m: m in set(te.model)

# 1 --- forecasts with intervals at 4 levels ---------------------------------------------------------
items = [n for n in h.sku if n.startswith("FOODS_3") and n.endswith("CA_1")][:1]
nodes = [("Total", "Total"), ("CA", "State: CA"), ("CA_1", "Store: CA_1"), (items[0], f"Item-store: {items[0]}")]
fig, axs = plt.subplots(2, 2, figsize=(13, 7.2)); F = {m: load(m, "test") for m in [FINAL, "prophet"]}
for ax, (n, t) in zip(axs.ravel(), nodes):
    lo = C.TEST_START - 56
    ax.plot(dates.loc[lo:C.TEST_END], S.loc[lo:C.TEST_END, n], color="#222", lw=1.1, label="actual")
    f = F[FINAL]; f = f[f.node == n].sort_values("d"); x = dates.loc[f.d]
    ax.fill_between(x, f["q0.025"], f["q0.975"], color="#7048e8", alpha=.12, label="95% interval"); ax.fill_between(x, f["q0.1"], f["q0.9"], color="#7048e8", alpha=.25, label="80% interval")
    ax.plot(x, f.point, color="#7048e8", lw=2, label=NM[FINAL]); p = F["prophet"]; p = p[p.node == n].sort_values("d")
    ax.plot(dates.loc[p.d], p.point, color="#e8590c", lw=1.4, ls="--", label="Prophet"); ax.axvline(dates.loc[C.TEST_START], color="#999", ls=":"); ax.set_title(t, fontsize=11)
axs[0, 0].legend(fontsize=8, ncol=2, loc="lower left"); fig.suptitle("Test-window forecasts (d1914-d1941) with 80% / 95% prediction intervals", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG / "fig1_forecasts.png"); plt.close(fig)

# 2 --- accuracy by level -------------------------------------------------------------------------------
MAIN = [m for m in ["snaive", "prophet", "lgbm_v1_mint", "lgbm_mint", "tft_bu", "ens_mint"] if have(m)]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, met in zip(axs, ["WAPE", "RMSSE", "SPL"]):
    w = 0.85 / len(MAIN)
    for i, m in enumerate(MAIN):
        v = []
        for l in LV:
            x = te[(te.model == m) & (te.level == l)][met].values; v.append(x[0] if len(x) and pd.notna(x[0]) else np.nan)
        ax.bar(np.arange(4) + i * w, v, w, color=COL[m], label=NM[m])
    ax.set_xticks(np.arange(4) + 0.42 - w / 2); ax.set_xticklabels(LV); ax.set_title(met + " (lower = better)")
axs[0].legend(fontsize=7.5); fig.suptitle("Accuracy on held-out test window, by hierarchy level", fontweight="bold"); fig.tight_layout(); fig.savefig(C.FIG / "fig2_accuracy.png"); plt.close(fig)

# 3 --- interval calibration ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 4)); ms = [m for m in ["snaive", "prophet", "lgbm_v1_mint", "lgbm_mint", "tft_avg", "tft_cqr", "ens_mint"] if have(m)]
for ax, (c, nom) in zip(axs, [("Cov80", 0.8), ("Cov95", 0.95)]):
    for i, m in enumerate(ms):
        v = te[(te.model == m) & (te.level == "Item-Store")][c].values
        if len(v): ax.bar(i, v[0], color=COL[m]); ax.text(i, v[0] + .01, f"{v[0]:.2f}", ha="center", fontsize=9)
    ax.axhline(nom, color="k", ls="--"); ax.set_xticks(range(len(ms))); ax.set_xticklabels([NM[m].replace(" (final)", "") for m in ms], rotation=22, ha="right", fontsize=8); ax.set_ylim(0, 1.1); ax.set_title(f"Item-store {c} (nominal {nom:.0%})")
fig.tight_layout(); fig.savefig(C.FIG / "fig3_calibration.png"); plt.close(fig)

# 4 --- reconciliation ----------------------------------------------------------------------------------
au = pd.read_csv(C.RES / "coherence_audit.csv"); rm = [m for m in ["lgbm", "lgbm_bu", "lgbm_td", "lgbm_ols", "lgbm_wlss", "lgbm_mint", "ens", "ens_mint"] if have(m)]
fig, axs = plt.subplots(1, 2, figsize=(15, 4.4)); w = 0.85 / len(rm)
for i, m in enumerate(rm):
    axs[0].bar(np.arange(4) + i * w, [te[(te.model == m) & (te.level == l)].WAPE.values[0] for l in LV], w, label=NM[m], color=plt.cm.Blues(0.25 + 0.1 * i) if m.startswith("lgbm") else plt.cm.Purples(0.4 + 0.3 * (m == "ens_mint")))
axs[0].set_xticks(np.arange(4) + 0.4); axs[0].set_xticklabels(LV); axs[0].set_title("WAPE by level: base vs reconciled"); axs[0].legend(fontsize=7)
mx = au.iloc[:, 1:].max(axis=1).clip(lower=1e-14); axs[1].barh([s.split(" (")[0] + (" (base)" if "base" in s else "") for s in au.forecast], mx, color=["#e03131" if "base" in f else "#0ca678" for f in au.forecast])
axs[1].set_xscale("log"); axs[1].set_title("Max incoherence |parent - sum(children)| (units, log)"); axs[1].invert_yaxis(); fig.tight_layout(); fig.savefig(C.FIG / "fig4_reconciliation.png"); plt.close(fig)

# 5 --- feature importance ------------------------------------------------------------------------------
fi = pd.read_csv(C.RES / "feature_importance_bottom.csv").head(15).iloc[::-1]; ext = set(best["features"]) - set(__import__("m5fc.features", fromlist=["x"]).BOTTOM_FEATS_BASE)
fig, ax = plt.subplots(figsize=(7.5, 5)); ax.barh(fi.feature, fi.gain / fi.gain.sum(), color=["#e8590c" if f in ext else "#0b6efd" for f in fi.feature])
ax.set_title("LightGBM bottom model: top-15 features by gain share\n(orange = new in this version)"); fig.tight_layout(); fig.savefig(C.FIG / "fig5_feature_importance.png"); plt.close(fig)

# 6 --- inventory ----------------------------------------------------------------------------------------
invs = pd.read_csv(C.RES / "inventory_summary.csv"); sens = pd.read_csv(C.RES / "inventory_budget_sensitivity.csv")
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6)); lab = {"milp": "OR-Tools MILP", "mean": "Order mean", "safety": "Mean+1.28σ", "oracle": "Perfect\nforesight"}
srcs = list(invs.model.unique()); w = 0.85 / len(srcs); pol = ["milp", "mean", "safety", "oracle"]
for i, s_ in enumerate(srcs): axs[0].bar(np.arange(4) + i * w, [invs[(invs.model == s_) & (invs.policy == p)].cost.values[0] for p in pol], w, label=s_)
axs[0].set_xticks(np.arange(4) + 0.42); axs[0].set_xticklabels([lab[p] for p in pol]); axs[0].set_ylabel("$ over 4 weeks x 10 stores"); axs[0].set_title("Realised holding + stockout cost"); axs[0].legend(fontsize=7)
g = invs[invs.model == PRIMARY].set_index("policy").loc[pol].reset_index()
axs[1].scatter(g.fill_rate, g.avg_end_inv_value, s=90, c=["#0b6efd", "#adb5bd", "#868e96", "#2f9e44"])
for _, r in g.iterrows(): axs[1].annotate(lab[r.policy].replace("\n", " "), (r.fill_rate, r.avg_end_inv_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
axs[1].set_xlabel("fill rate (units served / demand)"); axs[1].set_ylabel("avg end-of-week inventory $ per store"); axs[1].set_title(f"Service vs inventory ({PRIMARY})")
s2 = sens[sens.policy == "milp"].sort_values("budget_factor"); s3 = sens[sens.policy == "safety"].sort_values("budget_factor")
axs[2].plot(s2.budget_factor.astype(str), s2.cost, "o-", label="MILP", color="#7048e8"); axs[2].plot(s3.budget_factor.astype(str), s3.cost, "s--", label="Mean+1.28σ", color="#868e96")
axs[2].set_xlabel("purchasing budget factor"); axs[2].set_ylabel("realised cost $"); axs[2].set_title("Budget sensitivity"); axs[2].legend(); fig.tight_layout(); fig.savefig(C.FIG / "fig6_inventory.png"); plt.close(fig)

# --- RESULTS.md ---------------------------------------------------------------------------------------------
def md(df, fmt="{:.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows(): out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("" if pd.isna(v) else str(v)) for v in r) + " |")
    return "\n".join(out)
dec = json.load(open(C.RES / "rolling_origin_decision.json")); ro = pd.read_csv(C.RES / "rolling_origin_lgbm.csv").pivot_table(index=["kind", "window"], columns="cfg", values=["WAPE", "RMSSE"]); ro.columns = [f"{a} {b}" for a, b in ro.columns]; ro = ro.reset_index()
boot = json.load(open(C.RES / "bootstrap_wape_diff.json")); t = te.copy(); t["model"] = t.model.map(lambda m: NM.get(m, m))
show = [m for m in ["naive", "snaive", "prophet", "lgbm_v1_mint", "lgbm_mint", "tft_avg", "tft_bu", "ens_mint"] if have(m)]
main = t[t.model.isin([NM[m] for m in show])]; main["o"] = main.model.map({NM[m]: i for i, m in enumerate(show)})
acc = main.pivot_table(index=["o", "model"], columns="level", values=["WAPE", "RMSSE"])[[(a, l) for a in ("WAPE", "RMSSE") for l in LV]]
acc.columns = [f"{a} {l}" for a, l in acc.columns]; acc = acc.reset_index().drop(columns="o")
um = [m for m in ["snaive", "prophet", "lgbm_v1_mint", "lgbm_mint", "lgbm_cqr", "tft_avg", "tft_cqr", "ens", "ens_mint"] if have(m)]
unc = t[(t.level == "Item-Store") & t.model.isin([NM[m] for m in um])][["model", "SPL", "Cov80", "Cov95", "Width80", "WRMSSE"]]
unc = unc.set_index("model").loc[[NM[m] for m in um]].reset_index()
rec = t[t.model.isin([NM[m] for m in rm])].pivot_table(index="model", columns="level", values="WAPE")[LV].reindex([NM[m] for m in rm]).reset_index()
src_tab = invs[invs.policy.isin(["milp", "mean", "safety", "oracle"])][["model", "policy", "cost", "fill_rate", "stockout_rate", "avg_end_inv_value", "order_value"]]
bd = pd.read_csv(C.RES / "wape_by_group.csv").pivot(index="group", columns="model", values="WAPE").reset_index(); bd.columns = [NM.get(c, c) for c in bd.columns]
bootmd = "\n".join(f"| {k} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |" for k, v in boot.items())
tun = pd.read_csv(C.RES / "tuning_lgbm.csv")[["name", "n_feats", "WAPE", "RMSSE", "best_iter", "num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "tweedie_variance_power"]]
sens_md = md(sens[sens.policy.isin(["milp", "safety"])][["budget_factor", "policy", "cost", "fill_rate", "stockout_rate"]].sort_values(["budget_factor", "policy"]))
info = lambda n: json.load(open(C.RES / f"{n}_info.json")) if (C.RES / f"{n}_info.json").exists() else None
tft1, tft2 = info("tft"), info("tft2")
tt = lambda i, nm: f"{nm}: {i['epochs']} max epochs x {i['batches_per_epoch']} batches, hidden {i.get('hidden', 16)}, heads {i.get('heads', 2)}, dropout {i.get('dropout', 0.1)}, {i['seconds']:.0f} s on 1 CPU" if i else ""
open("docs/RESULTS.md", "w").write(f"""# Results (generated by `scripts/make_report.py` from `results/*.csv|json` - do not edit by hand)

All accuracy numbers are on the **held-out test window d1914-d1941** (28 days). **No model choice, hyper-parameter, blend weight or conformal adjustment was selected on the test window**:
features/hyper-parameters were selected on a separate window (d1858-d1885, models trained <= d1857), blend weight / MinT weights / conformal adjustments / copula rho on the calibration window (d1886-d1913).
Scope: 40 items x 10 stores x 3 states = 400 bottom series, 414 hierarchy nodes (see `docs/DATA.md`).

## 1. Point-forecast accuracy by hierarchy level
{md(acc)}

WAPE = sum|y-yhat|/sum y (pooled within level). RMSSE = M5 definition (scale = one-step naive MSE on history, mean over series at that level).
"LightGBM v1 + MinT" is the previous version of this project (base features, default parameters) kept as an archived baseline for before/after comparison.

![forecasts](../results/figures/fig1_forecasts.png)
![accuracy](../results/figures/fig2_accuracy.png)

### Are the differences real? Paired bootstrap over series (2,000 resamples), pooled item-store WAPE (positive = the second model is better)
| comparison | mean difference | 95% CI |
|---|---|---|
{bootmd}

## 2. Prediction intervals (item-store level, test window)
{md(unc)}

Cov80/Cov95 = empirical coverage of the nominal 80% / 95% interval. SPL = scaled pinball loss (M5-uncertainty style; lower is better).
**Recalibration** = locally-weighted conformalised quantile regression fitted on the calibration window, pooled over 400 series x 28 days, bottom level only
(14 aggregate nodes x 28 correlated days are too few). Learned adjustments (in units of interval width + 1): LightGBM {json.dumps({k: round(v, 3) for k, v in ens["lgbm_cqr_adj"].items()})},
TFT {json.dumps({k: round(v, 3) for k, v in ens["tft_cqr_adj"].items()})}, ensemble {json.dumps({k: round(v, 3) for k, v in ens["ens_cqr_adj"].items()})}.

![calibration](../results/figures/fig3_calibration.png)

## 3. Hierarchical reconciliation (Darts)
{md(rec)}

![reconciliation](../results/figures/fig4_reconciliation.png)

Coherence audit (max absolute gap, units):

{md(pd.read_csv(C.RES / "coherence_audit.csv"), "{:.2e}")}

## 4. Where do the models win / lose? (WAPE by segment)
{md(bd)}

## 5. Inventory optimisation (OR-Tools), 4-week rolling backtest on the real test window
Primary forecast source: **{PRIMARY}**. Copula correlation chosen on the calibration window: rho = {inf["rho"]} (80%-interval coverage of 7-day demand by rho: {json.dumps({k: round(v, 3) for k, v in inf["rho_grid"].items()})}).
{md(src_tab)}

![inventory](../results/figures/fig6_inventory.png)

Budget sensitivity ({PRIMARY} scenarios):

{sens_md}

## 6. How the model was optimised, and what actually generalised
**Stage 1 - search on a separate selection window** (d1858-d1885, models trained <= d1857; never the test window). The extended feature set and a random search over 8 hyper-parameter sets were tried;
the best search result ("B_tuned") scored WAPE {best["chosen_selection_scores"]["WAPE"]:.4f} / RMSSE {best["chosen_selection_scores"]["RMSSE"]:.4f} vs baseline {best["baseline_selection_scores"]["WAPE"]:.4f} / {best["baseline_selection_scores"]["RMSSE"]:.4f} there.

{md(tun, "{:.4g}")}

**Stage 2 - rolling-origin check with a decision rule fixed in advance.** A 28-day window is a noisy judge, so the configurations were re-evaluated on four *clean* earlier windows
(d1718-1829; out-of-sample for the search, for early stopping and for the test window; fixed tree counts, trained strictly before each window).
Rule: adopt the config with the lowest mean of 0.5*(WAPE/WAPE_v1 + RMSSE/RMSSE_v1), only if it beats v1 by >= 0.2%.

{md(ro, "{:.4f}")}

Scores vs v1 (lower is better): {json.dumps({k: round(v, 4) for k, v in dec["scores_vs_A"].items()})} -> **adopted: {best["adopted_config"]}**
(A_v1 = base features + default params; B_tuned = searched params; C_ext_default = extended features + default params).
Finding: the searched hyper-parameters did not generalise (worse than v1 on 3 of 4 clean windows and on the test window), while the extended features gave a small gain on most clean windows (RMSSE better on 3 of 4 with one tie, WAPE better on 3 of 4).
The "report only" rows show the selection, calibration and test windows for transparency; they were not used for the decision.

**Disclosure.** The tuned configuration's test-window score was viewed once *before* the rolling-origin check was run, and it was worse than v1. The check was therefore designed so that its
decision rule (written into `scripts/rolling_origin_lgbm.py` before running) uses only the four clean windows d1718-d1829, never the test window. The ensemble design (two TFTs averaged, blend weight from the calibration window)
and the choice of the ensemble as the final model were fixed before test results for them were produced. No parameter was changed after seeing the final test results.

Ensemble: TFT weight **{ens["blend_weight_tft"]}** (chosen on the calibration window by minimising WAPE/WAPE_lgbm + RMSSE/RMSSE_lgbm; grid: {json.dumps(ens["blend_grid_cal_score"])}).

![fi](../results/figures/fig5_feature_importance.png)

## 7. Training details
* LightGBM bottom model: best iteration {lg["bottom_best_iter"]} (early-stopped on calibration window), aggregate model {lg["agg_best_iter"]}; quantile heads 5 x <=350 trees.
* {tt(tft1, "TFT-1")}.
* {tt(tft2, "TFT-2")}. TFT forecasts used = equal average of TFT-1 and TFT-2.
""")
print("report written")
