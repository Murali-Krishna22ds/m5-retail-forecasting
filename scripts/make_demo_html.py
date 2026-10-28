"""Self-contained interactive HTML demo (no server needed) built from the real result files."""
import sys, warnings, json; warnings.filterwarnings("ignore"); sys.path.insert(0, "src"); sys.path.insert(0, "app")
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio
import app as A
from m5fc import config as C
D = A.store_data(); S, dates, h = D["S"], D["dates"], D["h"]
W = dict(template="plotly_white", margin=dict(l=45, r=10, t=40, b=40), font=dict(size=12), paper_bgcolor="#fff", plot_bgcolor="#fff")
def band(fig, x, lo, hi, color, name, vis):
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0), fillcolor=color, name=name, hoverinfo="skip", visible=vis)

# 1 forecast explorer ------------------------------------------------------------------------------------
nodes = ["Total", "CA", "TX", "WI", "CA_1", "CA_3", "TX_2", "WI_1"] + [n for n in h.sku if n.endswith("CA_1")][:6] + [n for n in h.sku if n.endswith("WI_3") and n.startswith("HOBBIES")][:2]
fig = go.Figure(); per = 5
for i, n in enumerate(nodes):
    v = i == 0; lo = C.TEST_START - 56
    f = A.get_fc("lgbm_mint", n); p = A.get_fc("prophet", n); x = dates.loc[f.d]
    fig.add_scatter(x=dates.loc[lo:C.TEST_END], y=S.loc[lo:C.TEST_END, n], name="Actual", line=dict(color="#222", width=1.4), visible=v)
    band(fig, x, f["q0.025"].values, f["q0.975"].values, "rgba(11,110,253,0.13)", "95% interval", v)
    band(fig, x, f["q0.1"].values, f["q0.9"].values, "rgba(11,110,253,0.28)", "80% interval", v)
    fig.add_scatter(x=x, y=f.point, name="LightGBM + MinT", line=dict(color="#0b6efd", width=2.4), visible=v)
    fig.add_scatter(x=dates.loc[p.d], y=p.point, name="Prophet", line=dict(color="#e8590c", width=1.6, dash="dash"), visible=v)
btn = [dict(label=n.replace("|", " | "), method="update", args=[{"visible": [j // per == i for j in range(len(nodes) * per)]}, {"title": f"{n} - test window forecast"}]) for i, n in enumerate(nodes)]
fig.update_layout(**W, height=440, title=f"{nodes[0]} - test window forecast", hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                  updatemenus=[dict(buttons=btn, direction="down", x=0, y=1.18, xanchor="left", showactive=True)], yaxis_title="units / day")
fig.add_vline(x=dates.loc[C.TEST_START], line_dash="dot", line_color="#999")
f1 = pio.to_html(fig, include_plotlyjs=True, full_html=False, div_id="p1", config={"displaylogo": False})

# 2 accuracy ----------------------------------------------------------------------------------------------
te = pd.read_csv(C.RES / "metrics_test.csv"); NM = {"naive": "Naive", "snaive": "Seasonal naive", "prophet": "Prophet", "lgbm": "LightGBM", "lgbm_mint": "LightGBM + MinT", "tft": "TFT"}
fig = go.Figure(); cols = {"naive": "#adb5bd", "snaive": "#868e96", "prophet": "#e8590c", "lgbm": "#6ea8fe", "lgbm_mint": "#0b6efd", "tft": "#2f9e44"}
for m in NM:
    g = te[te.model == m].set_index("level").reindex(["Total", "State", "Store", "Item-Store"])
    fig.add_bar(x=g.index, y=g.WAPE, name=NM[m], marker_color=cols[m])
fig.update_layout(**W, height=360, barmode="group", title="WAPE by hierarchy level (lower is better; TFT item-level only)", legend=dict(orientation="h", y=-0.2))
f2 = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="p2", config={"displaylogo": False})
t = te[te.model.isin(NM)].copy(); t["model"] = t.model.map(NM)
it = t[t.level == "Item-Store"][["model", "WAPE", "RMSSE", "SPL", "Cov80", "Cov95", "WRMSSE"]].round(3).fillna("-")
tab = it.to_html(index=False, border=0, classes="tbl")

# 3 reconciliation --------------------------------------------------------------------------------------
f3 = pio.to_html(A.reconciliation_figure("CA").update_layout(**W), include_plotlyjs=False, full_html=False, div_id="p3", config={"displaylogo": False})
aud = pd.read_csv(C.RES / "coherence_audit.csv"); aud.iloc[:, 1:] = aud.iloc[:, 1:].astype(float)
aud_html = aud.to_html(index=False, border=0, classes="tbl", float_format=lambda v: f"{v:.1e}")

# 4 inventory ---------------------------------------------------------------------------------------------
res, k = A.plan_store("CA_1", 1, 1.05, 1.5, 1.0, 0)
f4 = pio.to_html(A.order_figure(res).update_layout(**W), include_plotlyjs=False, full_html=False, div_id="p4", config={"displaylogo": False})
f5 = pio.to_html(A.policy_figure().update_layout(**W), include_plotlyjs=False, full_html=False, div_id="p5", config={"displaylogo": False})
otab = res.sort_values("Order $", ascending=False).head(12).to_html(index=False, border=0, classes="tbl")
inv = pd.read_csv(C.RES / "inventory_summary.csv"); g = inv[inv.model == "LightGBM + MinT"].set_index("policy")
kp = "".join(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>' for l, v in [("MILP cost (4 wk, 10 stores)", f"${g.loc['milp','cost']:,.0f}"), ("Order-the-mean cost", f"${g.loc['mean','cost']:,.0f}"), ("MILP fill rate", f"{g.loc['milp','fill_rate']:.1%}"), ("Mean-order fill rate", f"{g.loc['mean','fill_rate']:.1%}")])
html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>M5 forecasting &amp; replenishment - demo</title><style>
:root{{--bg:#f6f7f9;--fg:#1b1f24;--mut:#5c6670;--card:#fff;--bd:#dfe3e8;--ac:#0b6efd;color-scheme:light dark;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#101317;--fg:#e8eaed;--mut:#9aa4ae;--bd:#2b323a;--ac:#6ea8fe}}}}
:root[data-theme="dark"]{{--bg:#101317;--fg:#e8eaed;--mut:#9aa4ae;--bd:#2b323a;--ac:#6ea8fe}}
html{{scroll-padding-top:env(safe-area-inset-top,0px)}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:14px 14px 40px}} h1{{font-size:22px;margin:6px 0 2px}} h2{{font-size:17px;margin:26px 0 6px}}
p.s{{color:var(--mut);margin:2px 0 10px}} .card{{background:#fff;color:#1b1f24;border:1px solid var(--bd);border-radius:10px;padding:8px;margin:8px 0;overflow-x:auto}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}} .kpi{{background:#fff;color:#1b1f24;border:1px solid var(--bd);border-radius:10px;padding:10px 14px;min-width:150px}}
.kpi b{{display:block;font-size:21px}} .kpi span{{font-size:12px;color:#5c6670}} .tbl{{border-collapse:collapse;font-size:12.5px;width:100%}}
.tbl th{{background:#f1f3f5;text-align:left}} .tbl td,.tbl th{{padding:4px 8px;border-bottom:1px solid #e5e8eb}} .note{{font-size:12.5px;color:var(--mut)}}
</style></head><body><main>
<h1>Retail demand forecasting &amp; replenishment (M5 / Walmart)</h1>
<p class="s">Static, interactive snapshot of the Dash app's results. 40 items x 10 stores x 3 states, held-out test window d1914-d1941 (2016-04-25 to 2016-05-22). Choose a node in the dropdown; hover for values.</p>
<h2>1 &middot; Forecasts with 80% / 95% prediction intervals</h2><div class="card">{f1}</div>
<h2>2 &middot; Accuracy</h2><div class="card">{f2}</div><div class="card">{tab}</div>
<p class="note">Item-store level, 400 series. WAPE/RMSSE lower is better; SPL = scaled pinball loss; Cov = interval coverage (nominal 0.80 / 0.95). TFT intervals under-cover (0.47 / 0.56).</p>
<h2>3 &middot; Hierarchical reconciliation (Darts MinT)</h2><div class="card">{f3}</div><div class="card">{aud_html}</div>
<p class="note">Max |parent - sum of children| in units. Base LightGBM forecasts disagree across levels; every reconciled version is coherent to ~1e-13.</p>
<h2>4 &middot; Order recommendations (OR-Tools MILP)</h2>
<p class="s">Store CA_1, week 1. Budget = 1.05x cost of ordering the mean forecast, shelf capacity 1.5x, case packs. Solver {k['status']} in {k['solve_s']:.2f}s; budget used ${k['budget_used']:.0f} of ${k['budget']:.0f}.</p>
<div class="card">{f4}</div><div class="card">{otab}</div>
<h2>5 &middot; Rolling 4-week backtest on real demand</h2><div class="kpis">{kp}</div><div class="card">{f5}</div>
<p class="note">Cost = holding + stockout under stated assumptions (70% cost ratio, 6%/week holding, margin + 10% goodwill); see repository docs. Live re-optimisation with sliders is in the Dash app (<code>make app</code>).</p>
</main></body></html>"""
html = html.replace("\ufffd", "\\uFFFD")  # plotly.js contains a literal U+FFFD; use the JS escape
open("docs/demo.html", "w").write(html); open("/mnt/user-data/outputs/m5_demo.html", "w").write(html); print(len(html) / 1e6, "MB")
