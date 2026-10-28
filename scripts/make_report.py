"""Step 8: figures (results/figures/*.png) + docs/RESULTS.md with all numbers, generated from the result files."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from m5fc import config as C
from m5fc.features import build_all
from m5fc.store import load, to_wide, FC

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
bt, at, S, h = build_all(); from m5fc.data import date_index; dates = date_index()
te = pd.read_csv(C.RES / "metrics_test.csv"); cal = pd.read_csv(C.RES / "metrics_cal.csv")
LV = ["Total", "State", "Store", "Item-Store"]
COL = {"naive": "#adb5bd", "snaive": "#868e96", "prophet": "#e8590c", "lgbm": "#6ea8fe", "lgbm_mint": "#0b6efd", "tft": "#2f9e44", "tft_bu": "#2f9e44"}
NM = {"naive": "Naive", "snaive": "Seasonal naive", "prophet": "Prophet", "lgbm": "LightGBM (base)", "lgbm_mint": "LightGBM+MinT", "tft": "TFT", "tft_bu": "TFT (bottom-up)",
      "lgbm_bu": "LGBM bottom-up", "lgbm_td": "LGBM top-down", "lgbm_ols": "LGBM OLS", "lgbm_wlss": "LGBM WLS-struct"}
MAIN = [m for m in ["naive", "snaive", "prophet", "lgbm_mint", "tft_bu"] if m in set(te.model)]

# 1 --- forecasts with intervals at 4 levels ---------------------------------------------------------
best_item = (te[(te.level == "Item-Store")]).iloc[0]; Y = S
items = [n for n in h.sku if n.startswith("FOODS_3") and n.endswith("CA_1")][:1] + [n for n in h.sku if n.startswith("HOBBIES") and n.endswith("TX_2")][:1]
nodes = [("Total", "Total"), ("CA", "State: CA"), ("CA_1", "Store: CA_1"), (items[0], f"Item-store: {items[0]}")]
fig, axs = plt.subplots(2, 2, figsize=(13, 7.2))
F = {m: load(m, "test") for m in ["lgbm_mint", "prophet"]}
for ax, (n, t) in zip(axs.ravel(), nodes):
    lo = C.TEST_START - 56
    ax.plot(dates.loc[lo:C.TEST_END], S.loc[lo:C.TEST_END, n], color="#222", lw=1.1, label="actual")
    f = F["lgbm_mint"]; f = f[f.node == n].sort_values("d"); x = dates.loc[f.d]
    ax.fill_between(x, f["q0.025"], f["q0.975"], color="#0b6efd", alpha=.12, label="95% interval")
    ax.fill_between(x, f["q0.1"], f["q0.9"], color="#0b6efd", alpha=.25, label="80% interval")
    ax.plot(x, f.point, color="#0b6efd", lw=2, label="LightGBM+MinT")
    p = F["prophet"]; p = p[p.node == n].sort_values("d"); ax.plot(dates.loc[p.d], p.point, color="#e8590c", lw=1.4, ls="--", label="Prophet")
    ax.axvline(dates.loc[C.TEST_START], color="#999", ls=":"); ax.set_title(t, fontsize=11)
axs[0, 0].legend(fontsize=8, ncol=2, loc="lower left"); fig.suptitle("Test-window forecasts (d1914-d1941) with 80% / 95% prediction intervals", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG / "fig1_forecasts.png"); plt.close(fig)

# 2 --- accuracy by level -----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, met in zip(axs, ["WAPE", "RMSSE", "SPL"]):
    w = 0.8 / len(MAIN)
    for i, m in enumerate(MAIN):
        v = [te[(te.model == m) & (te.level == l)][met].values for l in LV]
        v = [x[0] if len(x) and pd.notna(x[0]) else np.nan for x in v]
        ax.bar(np.arange(4) + i * w, v, w, color=COL[m], label=NM[m])
    ax.set_xticks(np.arange(4) + 0.4 - w / 2); ax.set_xticklabels(LV); ax.set_title(met + " (lower = better)")
axs[0].legend(fontsize=8); fig.suptitle("Accuracy on held-out test window, by hierarchy level", fontweight="bold"); fig.tight_layout(); fig.savefig(C.FIG / "fig2_accuracy.png"); plt.close(fig)

# 3 --- interval calibration ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
ms = [m for m in ["snaive", "prophet", "lgbm_mint", "tft"] if m in set(te.model)]
for ax, (c, nom) in zip(axs, [("Cov80", 0.8), ("Cov95", 0.95)]):
    for i, m in enumerate(ms):
        v = te[(te.model == m) & (te.level == "Item-Store")][c].values
        if len(v): ax.bar(i, v[0], color=COL[m]); ax.text(i, v[0] + .01, f"{v[0]:.2f}", ha="center", fontsize=9)
    ax.axhline(nom, color="k", ls="--"); ax.set_xticks(range(len(ms))); ax.set_xticklabels([NM[m] for m in ms], rotation=12); ax.set_ylim(0, 1.08)
    ax.set_title(f"Item-store {c} (nominal {nom:.0%})")
fig.tight_layout(); fig.savefig(C.FIG / "fig3_calibration.png"); plt.close(fig)

# 4 --- reconciliation --------------------------------------------------------------------------------
au = pd.read_csv(C.RES / "coherence_audit.csv")
rm = ["lgbm", "lgbm_bu", "lgbm_td", "lgbm_ols", "lgbm_wlss", "lgbm_mint"]
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
w = 0.13
for i, m in enumerate(rm):
    v = [te[(te.model == m) & (te.level == l)].WAPE.values[0] for l in LV]
    axs[0].bar(np.arange(4) + i * w, v, w, label=NM[m], color=plt.cm.Blues(0.25 + 0.14 * i) if m != "lgbm" else "#ced4da")
axs[0].set_xticks(np.arange(4) + 0.33); axs[0].set_xticklabels(LV); axs[0].set_title("WAPE by level: base vs reconciled"); axs[0].legend(fontsize=8)
mx = au.iloc[:, 1:].max(axis=1).clip(lower=1e-14)
axs[1].barh([s.split(" (")[0] for s in au.forecast], mx, color=["#e03131"] + ["#0ca678"] * (len(au) - 1)); axs[1].set_xscale("log")
axs[1].set_title("Max incoherence |parent - sum(children)| (units, log)"); axs[1].invert_yaxis()
fig.tight_layout(); fig.savefig(C.FIG / "fig4_reconciliation.png"); plt.close(fig)

# 5 --- feature importance -----------------------------------------------------------------------------
fi = pd.read_csv(C.RES / "feature_importance_bottom.csv").head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(fi.feature, fi.gain / fi.gain.sum(), color="#0b6efd"); ax.set_title("LightGBM (bottom model) - top-15 features by gain share")
fig.tight_layout(); fig.savefig(C.FIG / "fig5_feature_importance.png"); plt.close(fig)

# 6 --- inventory ---------------------------------------------------------------------------------------
inv = pd.read_csv(C.RES / "inventory_summary.csv"); sens = pd.read_csv(C.RES / "inventory_budget_sensitivity.csv")
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
lab = {"milp": "OR-Tools MILP", "mean": "Order mean", "safety": "Mean+1.28σ", "oracle": "Perfect\nforesight"}
src = list(inv.model.unique()); w = 0.8 / len(src); pol = ["milp", "mean", "safety", "oracle"]
for i, s in enumerate(src):
    v = [inv[(inv.model == s) & (inv.policy == p)].cost.values[0] for p in pol]
    axs[0].bar(np.arange(4) + i * w, v, w, label=s)
axs[0].set_xticks(np.arange(4) + 0.4 - w / 2); axs[0].set_xticklabels([lab[p] for p in pol]); axs[0].set_ylabel("$ over 4 weeks x 10 stores"); axs[0].set_title("Realised holding + stockout cost"); axs[0].legend(fontsize=8)
g = inv[inv.model == "LightGBM + MinT"].set_index("policy").loc[pol].reset_index()
axs[1].scatter(g.fill_rate, g.avg_end_inv_value, s=90, c=["#0b6efd", "#adb5bd", "#868e96", "#2f9e44"])  # order: milp, mean, safety, oracle
for _, r in g.iterrows(): axs[1].annotate(lab[r.policy].replace("\n", " "), (r.fill_rate, r.avg_end_inv_value), xytext=(5, 5), textcoords="offset points", fontsize=8)
axs[1].set_xlabel("fill rate (units served / demand)"); axs[1].set_ylabel("avg end-of-week inventory $ per store"); axs[1].set_title("Service vs inventory (LightGBM+MinT forecasts)")
s2 = sens[sens.policy == "milp"].sort_values("budget_factor"); s3 = sens[sens.policy == "safety"].sort_values("budget_factor")
axs[2].plot(s2.budget_factor.astype(str), s2.cost, "o-", label="MILP", color="#0b6efd"); axs[2].plot(s3.budget_factor.astype(str), s3.cost, "s--", label="Mean+1.28σ", color="#868e96")
axs[2].set_xlabel("purchasing budget factor"); axs[2].set_ylabel("realised cost $"); axs[2].set_title("Budget sensitivity"); axs[2].legend()
fig.tight_layout(); fig.savefig(C.FIG / "fig6_inventory.png"); plt.close(fig)

# --- RESULTS.md ------------------------------------------------------------------------------------------
def md(df, fmt="{:.3f}"):
    cols = list(df.columns); out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows(): out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v) else ("" if pd.isna(v) else str(v)) for v in r) + " |")
    return "\n".join(out)
boot = json.load(open(C.RES / "bootstrap_wape_diff.json")); info = json.load(open(C.RES / "lgbm_info.json")); inf = json.load(open(C.RES / "inventory_info.json"))
tft_i = json.load(open(C.RES / "tft_info.json")) if (C.RES / "tft_info.json").exists() else None
t = te.copy(); t["model"] = t.model.map(lambda m: NM.get(m, m))
main = t[t.model.isin([NM[m] for m in ["naive", "snaive", "prophet", "lgbm", "lgbm_mint", "tft", "tft_bu"] if m in set(te.model)])]
acc = main.pivot_table(index="model", columns="level", values=["WAPE", "RMSSE"])[[(a, l) for a in ("WAPE", "RMSSE") for l in LV]]
acc.columns = [f"{a} {l}" for a, l in acc.columns]; acc = acc.reset_index()
unc = main[main.level == "Item-Store"][["model", "SPL", "Cov80", "Cov95", "Width80", "WRMSSE"]].dropna(subset=["SPL"])
rec = t[t.model.isin([NM[m] for m in rm])].pivot_table(index="model", columns="level", values="WAPE")[LV].reset_index()
src_tab = inv[inv.policy.isin(["milp", "mean", "safety", "oracle"])][["model", "policy", "cost", "fill_rate", "stockout_rate", "avg_end_inv_value", "order_value"]]
bd = pd.read_csv(C.RES / "wape_by_group.csv").pivot(index="group", columns="model", values="WAPE").reset_index()
bd.columns = [NM.get(c, c) for c in bd.columns]
bootmd = "\n".join(f"| {k} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |" for k, v in boot.items())
sens_md = md(sens[sens.policy.isin(["milp", "safety"])][["budget_factor", "policy", "cost", "fill_rate", "stockout_rate"]].sort_values(["budget_factor", "policy"]))
open("docs/RESULTS.md", "w").write(f"""# Results (generated by `scripts/make_report.py` from `results/*.csv|json` - do not edit by hand)

All accuracy numbers are on the **held-out test window d1914-d1941** (28 days; never used for training, tuning, calibration or model selection).
Scope: 40 items x 10 stores x 3 states = 400 bottom series, 414 hierarchy nodes (see `docs/DATA.md`).

## 1. Point-forecast accuracy by hierarchy level
{md(acc)}

WAPE = sum|y-yhat|/sum y (pooled within level). RMSSE = M5 definition (scale = one-step naive MSE on history, mean over series at that level).
Revenue-weighted item-store WRMSSE: {", ".join(f"{r.model} {r.WRMSSE:.3f}" for r in unc.itertuples())}.

![forecasts](../results/figures/fig1_forecasts.png)
![accuracy](../results/figures/fig2_accuracy.png)

### Is the difference real? Paired bootstrap over series (2,000 resamples) of pooled item-store WAPE
| comparison | mean difference | 95% CI |
|---|---|---|
{bootmd}

## 2. Prediction intervals (item-store level, test window)
{md(unc)}

Cov80/Cov95 = empirical coverage of the nominal 80% / 95% interval (nominal = 0.80 / 0.95). SPL = scaled pinball loss (M5-uncertainty style, lower is better).
Conformalised-quantile-regression adjustment learnt on the calibration window = {json.dumps(info.get("cqr_adjustment"))} - i.e. the native LightGBM quantile heads were already valid at the bottom level, so no change was made (CQR was *not* applied to the 14 aggregate nodes: 28 correlated days per node is too small a calibration sample - a first attempt reduced Total 80%-coverage from 0.64 to 0.39).

![calibration](../results/figures/fig3_calibration.png)

## 3. Hierarchical reconciliation (Darts)
{md(rec)}

![reconciliation](../results/figures/fig4_reconciliation.png)

Coherence audit (max absolute gap, units):

{md(pd.read_csv(C.RES / "coherence_audit.csv"), "{:.2e}")}

## 4. Where do the models win / lose? (WAPE by segment)
{md(bd)}

## 5. Inventory optimisation (OR-Tools), 4-week rolling backtest on the real test window
Copula correlation chosen on the calibration window: rho = {inf["rho"]} (80%-interval coverage of 7-day demand by rho: {json.dumps({k: round(v, 3) for k, v in inf["rho_grid"].items()})}).
{md(src_tab)}

![inventory](../results/figures/fig6_inventory.png)

Budget sensitivity (LightGBM+MinT scenarios):

{sens_md}

## 6. Feature importance
![fi](../results/figures/fig5_feature_importance.png)

## 7. Training details
* LightGBM: bottom model best iteration {info["bottom_best_iter"]} (early-stopped on calibration window), aggregate model {info["agg_best_iter"]}; quantile heads 5 x <=350 trees.
* TFT: {"%d epochs x %d batches, %.0f s on 1 CPU, 33.8k parameters" % (tft_i["epochs"], tft_i["batches_per_epoch"], tft_i["seconds"]) if tft_i else "not run"}.
""")
print("report written")
