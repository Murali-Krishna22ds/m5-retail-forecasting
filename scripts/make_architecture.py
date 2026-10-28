"""Draws docs/architecture.png (pure matplotlib, no external binaries)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 8.4)); ax.set_xlim(0, 150); ax.set_ylim(0, 84); ax.axis("off")
def box(x, y, w, h, title, body="", fc="#eef4ff", ec="#2b5fb8", tc="#0b2a5b"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=tc)
    if body: ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=8.3, color="#222", linespacing=1.35)
def arrow(x1, y1, x2, y2, c="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.5, color=c))

ax.text(75, 81.5, "M5 retail forecasting & replenishment - system architecture", ha="center", fontsize=15, fontweight="bold")
box(2, 56, 26, 20, "1  DATA", "M5 CSVs (Walmart)\nsales 30,490 x 1,941 d\ncalendar / events / SNAP\nsell_prices\n-> stratified subset:\n40 items x 10 stores\n= 400 series, 414 nodes", "#fff4e6", "#e8590c", "#7a2e00")
box(33, 56, 28, 20, "2  FEATURES", "lags 28-364, rolling mean/std\nzero-rate, same-weekday means\nprice ratio, promo flag, price chg\nholidays/events, SNAP, calendar\nall sales features shifted >= 28 d\n(leak-free direct strategy)", "#fff4e6", "#e8590c", "#7a2e00")
box(66, 56, 38, 20, "3  FORECAST MODELS", "Baselines: naive, seasonal naive, Prophet\nLightGBM (global): Tweedie point +\n5 quantile models (2.5/10/50/90/97.5)\nTFT (PyTorch Forecasting): quantile loss,\n56 d encoder, 28 d decoder, static+known inputs", "#eef4ff", "#2b5fb8", "#0b2a5b")
box(109, 56, 39, 20, "4  RECONCILIATION (Darts)", "Total > State > Store > Item-Store\nbottom-up | top-down | OLS |\nWLS-struct | MinT (WLS-var, residuals\nfrom calibration window)\nclip >= 0 & re-sum -> exact coherence", "#e6fcf5", "#0ca678", "#04503b")
box(2, 28, 40, 21, "5  PREDICTION INTERVALS", "native quantile heads (LGBM / TFT)\n80% & 95% central intervals\nconformal check (CQR) on cal window\nmetrics: scaled pinball, coverage,\nWinkler  |  paired bootstrap CIs", "#f3f0ff", "#7048e8", "#2b1a73")
box(47, 28, 48, 21, "6  OR-TOOLS OPTIMISER (per store, weekly)", "scenarios: quantile inverse-CDF + Gaussian copula\n(rho calibrated on cal window)\nMILP (SCIP): min E[holding + stockout]\ns.t. case packs (integer), purchasing budget,\nshelf volume capacity, on-hand carry-over", "#fff0f6", "#d6336c", "#7a1038")
box(100, 28, 48, 21, "7  EVALUATION & BACKTEST", "test = last 28 d (never trained on)\nWAPE, RMSSE, WRMSSE at 4 hierarchy levels\n4-week rolling replenishment backtest:\nMILP vs mean / safety-stock / oracle\npytest: metrics, leakage, coherence, MILP", "#f8f9fa", "#495057", "#212529")
box(2, 3, 146, 19, "8  PLOTLY DASH APP  (app/app.py, gunicorn-ready)", "Forecast explorer (any node, models, 80/95% bands)  |  Accuracy tables & charts  |  Reconciliation audit\n"
    "Order recommendations: live OR-Tools re-optimisation with budget / capacity / stockout-penalty / on-hand sliders  |  policy backtest", "#e7f5ff", "#1c7ed6", "#0b3d6b")
for x in (28, 61, 104): arrow(x + 0.6, 66, x + 5.2, 66)
arrow(85, 56, 22, 49.5); arrow(125, 56, 72, 49.5); arrow(125, 56, 124, 49.5)
arrow(42, 38.5, 46.4, 38.5); arrow(95, 38.5, 99.4, 38.5)
for x in (22, 71, 124): arrow(x, 28, x, 22.6)
plt.savefig("docs/architecture.png", dpi=140, bbox_inches="tight", facecolor="white")
