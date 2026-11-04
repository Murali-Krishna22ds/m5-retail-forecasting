"""Plotly Dash app: forecasts + intervals, accuracy, hierarchical reconciliation, live OR-Tools order recommendations.
Run:  python app/app.py   (http://127.0.0.1:8050)   |  prod: gunicorn app.app:server"""
from __future__ import annotations
import sys, json, pathlib, functools, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback_context, no_update
from m5fc import config as C, optimize as O, metrics as M
from m5fc.features import build_all
from m5fc.store import load, to_wide, FC
from m5fc.reconcile import coherence_error

COL = {"actual": "#222", "ens_mint": "#7048e8", "tft_avg": "#2f9e44", "lgbm_mint": "#0b6efd", "lgbm": "#6ea8fe", "prophet": "#e8590c", "snaive": "#868e96", "naive": "#adb5bd", "tft": "#2f9e44"}
NAMES = {"ens_mint": "Ensemble + MinT", "tft_avg": "TFT (avg of 2)", "lgbm_mint": "LightGBM + MinT", "lgbm": "LightGBM (base)", "tft_bu": "TFT (bottom-up)", "prophet": "Prophet", "snaive": "Seasonal naive", "naive": "Naive", "tft": "TFT"}


@functools.lru_cache(maxsize=1)
def store_data():
    bt, at, S, h = build_all()
    models = [m for m in ["ens_mint", "lgbm_mint", "lgbm", "prophet", "snaive", "naive", "tft_avg", "tft"] if (FC / f"{m}.parquet").exists()]
    F = {m: load(m, "test") for m in models}
    if (FC / "tft_bu.parquet").exists(): F["tft_bu"] = load("tft_bu", "test")      # TFT summed up the hierarchy (point only)
    primary = "ens_mint" if "ens_mint" in models else "lgbm_mint"
    if (FC / "ens.parquet").exists(): F["ens"] = load("ens", "test")                 # unreconciled ensemble (for the coherence plot)
    F_cal = {m: load(m, "cal") for m in models if "cal" in set(load(m)["window"])}
    price = bt[bt.d.between(C.TEST_START, C.TEST_END)].pivot(index="d", columns="sku", values="sell_price")
    price_now = price.iloc[0].fillna(price.mean()).fillna(0.5)
    from m5fc.data import date_index; dates = date_index()
    met = pd.read_csv(C.RES / "metrics_test.csv")
    info = json.load(open(C.RES / "inventory_info.json")) if (C.RES / "inventory_info.json").exists() else {"rho": 0.2}
    return dict(bt=bt, S=S, h=h, models=models, F=F, price_now=price_now, dates=dates, met=met, rho=info["rho"], primary=primary,
                audit=pd.read_csv(C.RES / "coherence_audit.csv") if (C.RES / "coherence_audit.csv").exists() else None)


def get_fc(m: str, node: str) -> pd.DataFrame:
    """Forecast rows for (model, node); TFT only exists at item level -> fall back to its bottom-up sum above that."""
    D = store_data(); f = D["F"][m]; f = f[f.node == node]
    if f.empty and m in ("tft", "tft_avg") and "tft_bu" in D["F"]:
        f = D["F"]["tft_bu"]; f = f[f.node == node]
    return f.sort_values("d")


def nodes_for(level: str):
    D = store_data(); h = D["h"]
    return {"Total": ["Total"], "State": sorted(h.state_id.unique()), "Store": sorted(h.store_id.unique()),
            "Item-Store": list(h.sku)}[level]


# ----------------------------------------------------------------------------------- forecast tab
def forecast_figure(node, models, band, hist_days):
    D = store_data(); S, dates = D["S"], D["dates"]
    lo = C.TEST_START - hist_days
    fig = go.Figure()
    fig.add_scatter(x=dates.loc[lo:C.TEST_END], y=S.loc[lo:C.TEST_END, node], name="Actual", line=dict(color=COL["actual"], width=1.5))
    for m in models:
        f = get_fc(m, node); x = dates.loc[f.d].values
        if f.empty: continue
        has_q = "q0.1" in f and f["q0.1"].notna().all()
        if band != "none" and m == models[0] and has_q:
            a, b = ("q0.1", "q0.9") if band == "80" else ("q0.025", "q0.975")
            fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[f[b], f[a][::-1]], fill="toself", line=dict(width=0),
                            fillcolor="rgba(11,110,253,0.18)", name=f"{band}% interval ({NAMES[m]})", hoverinfo="skip")
        fig.add_scatter(x=x, y=f.point, name=NAMES[m], line=dict(color=COL[m], width=2, dash="solid" if m == models[0] else "dot"))
    fig.add_vline(x=dates.loc[C.TEST_START], line_dash="dash", line_color="#999", annotation_text="forecast origin")
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=40, r=10, t=30, b=30), legend=dict(orientation="h", y=-0.15),
                      yaxis_title="units / day", hovermode="x unified")
    return fig


def node_metrics(node, models):
    D = store_data(); S = D["S"]; Y = S.loc[C.TEST_START:C.TEST_END, node]; sc = M.rmsse_scale(S.loc[: C.TEST_START - 1, node].values)
    rows = []
    for m in models:
        f = get_fc(m, node)
        if f.empty: continue
        r = dict(model=NAMES[m] + (" (bottom-up)" if m in ("tft", "tft_avg") and node not in set(D["h"].sku) else ""), WAPE=round(M.wape(Y.values, f.point.values), 3), RMSSE=round(M.rmsse(Y.values, f.point.values, sc), 3))
        if "q0.1" in f and f["q0.1"].notna().all():
            r["Cov80"] = round(M.coverage(Y.values, f["q0.1"], f["q0.9"]), 2); r["Cov95"] = round(M.coverage(Y.values, f["q0.025"], f["q0.975"]), 2)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- accuracy tab
def accuracy_figure(metric, level):
    met = store_data()["met"]; g = met[(met.level == level) & met.model.isin(["naive", "snaive", "prophet", "lgbm", "lgbm_mint", "ens_mint", "tft_avg", "tft_bu"])]
    g = g.dropna(subset=[metric])
    fig = go.Figure(go.Bar(x=[NAMES.get(m, m) for m in g.model], y=g[metric], marker_color=[COL.get(m.replace("_bu", ""), "#999") for m in g.model],
                           text=g[metric].round(3), textposition="outside"))
    if metric in ("Cov80", "Cov95"):
        fig.add_hline(y=0.8 if metric == "Cov80" else 0.95, line_dash="dash", annotation_text="nominal")
    fig.update_layout(template="plotly_white", height=360, title=f"{metric} - {level} (test window d1914-d1941)", margin=dict(t=50, b=30))
    return fig


# ----------------------------------------------------------------------------------- reconciliation tab
def reconciliation_figure(state):
    D = store_data(); h = D["h"]; stores = sorted(h[h.state_id == state].store_id.unique())
    base_name = "ens" if D["primary"] == "ens_mint" and "ens" in D["F"] else "lgbm"
    base = to_wide(D["F"][base_name].assign(window="test"), "point")
    rec = to_wide(D["F"][D["primary"]], "point"); dates = D["dates"]
    fig = go.Figure()
    for lbl, W in (("Base (unreconciled)", base), (f"Reconciled ({NAMES[D['primary']]})", rec)):
        gap = W[state] - W[stores].sum(axis=1)
        fig.add_bar(x=dates.loc[W.index], y=gap, name=lbl)
    fig.update_layout(template="plotly_white", barmode="group", height=340, title=f"{state}: state forecast minus sum of its stores' forecasts (units)",
                      yaxis_title="incoherence", margin=dict(t=50, b=30))
    return fig


# ----------------------------------------------------------------------------------- inventory tab
def plan_store(store, week, budget_factor, cap_factor, penalty_mult, cover_days, model=None, K=300):
    D = store_data(); h, S = D["h"], D["S"]; model = model or D["primary"]
    g = h[h.store_id == store]; skus = list(g.sku)
    d0 = C.TEST_START + 7 * (week - 1); d1 = d0 + 6
    P = O.item_params(h, D["price_now"], margin_mult=penalty_mult).loc[skus]
    fc = D["F"][model]
    Dsc = O.cycle_scenarios(fc, skus, d0, d1, K, D["rho"], seed=week)
    mean_daily = Dsc.mean(1) / 7
    I0 = np.floor(cover_days * mean_daily)
    budget, cap = O.store_limits(P, Dsc, I0, budget_factor, cap_factor)
    q, info = O.solve_store(P, Dsc, I0, budget, cap)
    exp_cost, _, _ = None, None, None
    short = np.maximum(Dsc - (I0 + q)[:, None], 0)
    res = pd.DataFrame({"SKU": [s.split("|")[0] for s in skus], "Category": g.cat_id.values, "Price": P.price.round(2).values,
                        "Fcst mean": Dsc.mean(1).round(1), "P10": np.quantile(Dsc, .1, 1).round(1), "P90": np.quantile(Dsc, .9, 1).round(1),
                        "On hand": I0, "Order qty": q.astype(int), "Packs": (q / P.pack.values).astype(int),
                        "Order $": (q * P.cost.values).round(2),
                        "P(no stockout)": np.mean(Dsc <= (I0 + q)[:, None], axis=1).round(2)})
    actual = S.loc[d0:d1, skus].sum().values
    kpi = dict(budget=budget, budget_used=info["budget_used"], cap=cap, cap_used=info["cap_used"], status=info["status"], solve_s=info["solve_s"],
               exp_fill=1 - short.sum() / max(Dsc.sum(), 1e-9), exp_cost=info["obj"], actual_units=float(actual.sum()),
               real_fill=1 - np.maximum(actual - I0 - q, 0).sum() / max(actual.sum(), 1e-9),
               real_cost=O.cost_of(P, q, I0, actual)[0])
    return res, kpi


def order_figure(res):
    r = res.sort_values("Fcst mean", ascending=False).head(25)
    fig = go.Figure()
    fig.add_bar(x=r.SKU, y=r["P90"] - r["P10"], base=r["P10"], name="P10-P90 demand", marker_color="rgba(11,110,253,0.25)")
    fig.add_scatter(x=r.SKU, y=r["Fcst mean"], mode="markers", name="Forecast mean", marker=dict(color="#0b6efd", size=8))
    fig.add_scatter(x=r.SKU, y=r["Order qty"] + r["On hand"], mode="markers", name="Stock after order", marker=dict(color="#e8590c", symbol="diamond", size=9))
    fig.update_layout(template="plotly_white", height=380, title="Top-25 SKUs: demand range vs recommended stock", margin=dict(t=50, b=80), xaxis_tickangle=-45, yaxis_title="units / cycle")
    return fig


def policy_figure():
    p = C.RES / "inventory_summary.csv"
    if not p.exists(): return go.Figure()
    s = pd.read_csv(p); s = s[s.model == json.load(open(C.RES / "inventory_info.json")).get("primary", "LightGBM + MinT")]
    lab = {"milp": "OR-Tools MILP", "mean": "Order mean forecast", "safety": "Mean + 1.28 sigma", "oracle": "Perfect foresight"}
    fig = go.Figure(go.Bar(x=[lab[x] for x in s.policy], y=s.cost, text=s.cost.round(0), textposition="outside", marker_color=["#0b6efd", "#adb5bd", "#868e96", "#2f9e44"]))
    fig.update_layout(template="plotly_white", height=330, title="4-week backtest: realised holding + stockout cost ($, 10 stores)", margin=dict(t=50, b=30))
    return fig


# ----------------------------------------------------------------------------------- layout
def kpi(title, value, sub=""):
    return html.Div([html.Div(title, style={"fontSize": 12, "color": "#666"}), html.Div(value, style={"fontSize": 22, "fontWeight": 600}),
                     html.Div(sub, style={"fontSize": 11, "color": "#888"})],
                    style={"padding": "10px 14px", "border": "1px solid #e5e5e5", "borderRadius": 8, "minWidth": 150, "background": "#fff"})


def table(df, id_=None, page=12):
    kw = {"id": id_} if id_ else {}
    return dash_table.DataTable(data=df.to_dict("records"), columns=[{"name": c, "id": c} for c in df.columns], page_size=page, **kw,
                                style_cell={"fontSize": 12, "padding": "4px 8px", "textAlign": "left"},
                                style_header={"fontWeight": 600, "background": "#f5f5f5"}, sort_action="native")


def make_app() -> Dash:
    app = Dash(__name__, title="M5 Forecasting & Replenishment", suppress_callback_exceptions=True)
    D = store_data()
    app.layout = html.Div([
        html.H2("Retail demand forecasting & replenishment (M5 / Walmart)", style={"margin": "8px 0 0"}),
        html.Div("Subset: 40 items x 10 stores x 3 states, test window = last 28 days of M5 (d1914-d1941). "
                 "Forecasts: LightGBM + TFT ensemble (MinT reconciliation), Prophet, naive. Orders: OR-Tools MILP.", style={"color": "#666", "marginBottom": 8}),
        dcc.Tabs(id="tabs", value="fc", children=[
            dcc.Tab(label="Forecasts", value="fc", children=html.Div([
                html.Div([
                    html.Div([html.Label("Level"), dcc.Dropdown(["Total", "State", "Store", "Item-Store"], "Store", id="lvl", clearable=False)], style={"width": 160}),
                    html.Div([html.Label("Node"), dcc.Dropdown(id="node", clearable=False)], style={"width": 260}),
                    html.Div([html.Label("Models (first = shown with interval)"), dcc.Dropdown([{"label": NAMES[m], "value": m} for m in D["models"]], [store_data()["primary"], "prophet"], multi=True, id="models")], style={"width": 380}),
                    html.Div([html.Label("Interval"), dcc.RadioItems([{"label": "80%", "value": "80"}, {"label": "95%", "value": "95"}, {"label": "none", "value": "none"}], "80", id="band", inline=True)]),
                    html.Div([html.Label("History (days)"), dcc.Slider(28, 365, 28, value=112, id="hist", marks={28: "28", 112: "112", 224: "224", 364: "364"})], style={"width": 260}),
                ], style={"display": "flex", "gap": 18, "flexWrap": "wrap", "padding": "10px 0"}),
                dcc.Graph(id="fc-graph"), html.Div(id="fc-table")], style={"padding": 8})),
            dcc.Tab(label="Accuracy", value="acc", children=html.Div([
                html.Div([dcc.Dropdown(["WAPE", "RMSSE", "SPL", "Cov80", "Cov95"], "WAPE", id="acc-metric", clearable=False, style={"width": 160}),
                          dcc.Dropdown(["Total", "State", "Store", "Item-Store"], "Item-Store", id="acc-level", clearable=False, style={"width": 160})], style={"display": "flex", "gap": 12}),
                dcc.Graph(id="acc-graph"),
                html.H4("All metrics (test window)"), table(D["met"].drop(columns=["window"]).round(3), page=14),
                html.Div("WAPE = sum|y-yhat| / sum y. RMSSE = sqrt(MSE / MSE of one-step naive on history). WRMSSE = revenue-weighted RMSSE (item-store). "
                         "SPL = scaled pinball loss over 5 quantiles. Cov80/95 = share of actuals inside the interval.", style={"fontSize": 12, "color": "#666", "marginTop": 6})], style={"padding": 8})),
            dcc.Tab(label="Reconciliation", value="rec", children=html.Div([
                dcc.Dropdown(["CA", "TX", "WI"], "CA", id="rec-state", clearable=False, style={"width": 120}), dcc.Graph(id="rec-graph"),
                html.H4("Coherence audit (max |parent - sum of children|, units)"),
                table(D["audit"].round(10) if D["audit"] is not None else pd.DataFrame(), page=8)], style={"padding": 8})),
            dcc.Tab(label="Order recommendations", value="inv", children=html.Div([
                html.Div([
                    html.Div([html.Label("Store"), dcc.Dropdown(sorted(D["h"].store_id.unique()), "CA_1", id="inv-store", clearable=False)], style={"width": 120}),
                    html.Div([html.Label("Week of test window"), dcc.Dropdown([1, 2, 3, 4], 1, id="inv-week", clearable=False)], style={"width": 120}),
                    html.Div([html.Label("Budget factor (x cost of mean-forecast order)"), dcc.Slider(0.7, 1.5, 0.05, value=1.05, id="inv-budget", marks={0.7: "0.7", 1.05: "1.05", 1.5: "1.5"})], style={"width": 300}),
                    html.Div([html.Label("Shelf capacity factor"), dcc.Slider(0.8, 2.0, 0.1, value=1.5, id="inv-cap", marks={0.8: "0.8", 1.5: "1.5", 2: "2"})], style={"width": 240}),
                    html.Div([html.Label("Stockout penalty (x margin)"), dcc.Slider(0.5, 3, 0.25, value=1.0, id="inv-pen", marks={0.5: "0.5", 1: "1", 3: "3"})], style={"width": 240}),
                    html.Div([html.Label("On-hand (days of cover)"), dcc.Slider(0, 5, 1, value=0, id="inv-cover", marks={0: "0", 2: "2", 5: "5"})], style={"width": 200}),
                    html.Button("Re-optimise", id="inv-go", n_clicks=0, style={"height": 36, "alignSelf": "end"}),
                ], style={"display": "flex", "gap": 18, "flexWrap": "wrap", "padding": "10px 0"}),
                html.Div(id="inv-kpis", style={"display": "flex", "gap": 10, "flexWrap": "wrap", "margin": "6px 0"}),
                dcc.Graph(id="inv-graph"), html.Div(id="inv-table"),
                html.H4("Rolling 4-week backtest vs heuristics"), dcc.Graph(figure=policy_figure())], style={"padding": 8})),
        ]),
    ], style={"fontFamily": "system-ui, sans-serif", "maxWidth": 1280, "margin": "0 auto", "padding": "0 12px"})

    @app.callback(Output("node", "options"), Output("node", "value"), Input("lvl", "value"))
    def _nodes(level):
        n = nodes_for(level); return [{"label": x, "value": x} for x in n], n[0]

    @app.callback(Output("fc-graph", "figure"), Output("fc-table", "children"), Input("node", "value"), Input("models", "value"), Input("band", "value"), Input("hist", "value"))
    def _fc(node, models, band, hist):
        if not node or not models: return go.Figure(), ""
        return forecast_figure(node, models, band, hist), table(node_metrics(node, models), page=6)

    @app.callback(Output("acc-graph", "figure"), Input("acc-metric", "value"), Input("acc-level", "value"))
    def _acc(m, l): return accuracy_figure(m, l)

    @app.callback(Output("rec-graph", "figure"), Input("rec-state", "value"))
    def _rec(s): return reconciliation_figure(s)

    @app.callback(Output("inv-kpis", "children"), Output("inv-graph", "figure"), Output("inv-table", "children"),
                  Input("inv-go", "n_clicks"), State("inv-store", "value"), State("inv-week", "value"), State("inv-budget", "value"),
                  State("inv-cap", "value"), State("inv-pen", "value"), State("inv-cover", "value"))
    def _inv(_, store, week, bf, cf, pen, cover):
        res, k = plan_store(store, week, bf, cf, pen, cover)
        kp = [kpi("Solver", k["status"], f"{k['solve_s']:.2f}s"), kpi("Budget used", f"${k['budget_used']:.0f}", f"of ${k['budget']:.0f}"),
              kpi("Shelf volume", f"{k['cap_used']:.0f}", f"of {k['cap']:.0f}"), kpi("Expected cost", f"${k['exp_cost']:.1f}", "holding + stockout"),
              kpi("Expected fill rate", f"{k['exp_fill']:.1%}"), kpi("Realised fill rate", f"{k['real_fill']:.1%}", f"actual demand {k['actual_units']:.0f} u"),
              kpi("Realised cost", f"${k['real_cost']:.1f}")]
        return kp, order_figure(res), table(res.sort_values("Order $", ascending=False), page=15)

    return app


app = make_app(); server = app.server
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
