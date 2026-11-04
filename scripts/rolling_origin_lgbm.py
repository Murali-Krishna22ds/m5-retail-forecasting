"""Rolling-origin check of LightGBM configurations (A = v1: base features + default params; B = tuned; C = extended features + default params).

DECISION RULE (fixed before running, uses ONLY the four clean windows that precede the selection window d1858-1885, so they are out-of-sample for
feature/param selection, for early stopping and for the test window):
    score(cfg) = mean over clean windows of 0.5 * (WAPE_cfg/WAPE_A + RMSSE_cfg/RMSSE_A)       (A = 1.0)
    adopt the config with the lowest score if it beats A by >= 0.2%, otherwise keep A (the simpler v1).
Models are trained strictly on d < window start with a FIXED number of trees (no early stopping). Selection/cal/test windows are reported for transparency only."""
import sys, json, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from m5fc import metrics as M
from m5fc.features import build_all, BOTTOM_FEATS_BASE, BOTTOM_FEATS, BOTTOM_CATS
from m5fc.lgbm import _params
from m5fc.config import RES

bt, at, S, h = build_all(); skus = list(h.sku); best = json.load(open(RES / "best_params.json")); info = json.load(open(RES / "lgbm_info.json"))
CFG = {"A_v1": (BOTTOM_FEATS_BASE, _params("tweedie", tweedie_variance_power=1.15), 513),
       "B_tuned": (best["features"], _params("tweedie", **best["point"]), info["bottom_best_iter"]),
       "C_ext_default": (BOTTOM_FEATS, _params("tweedie", tweedie_variance_power=1.15), 600)}
CLEAN = [(1802, 1829), (1774, 1801), (1746, 1773), (1718, 1745)]
REPORT = [(1858, 1885), (1886, 1913), (1914, 1941)]
rows = []


def run(cfg, w0, w1, tag):
    feats, params, n = CFG[cfg]; cats = [c for c in BOTTOM_CATS if c in feats]; t = time.time()
    tr = bt[(bt.d >= 500) & (bt.d < w0) & (bt.avail == 1) & bt.y.notna()]; va = bt[(bt.d >= w0) & (bt.d <= w1)]
    m = lgb.train(params, lgb.Dataset(tr[feats], tr.y, categorical_feature=cats), n)
    p = pd.Series(0.0, index=va.index); ok = va.avail == 1; p[ok] = np.clip(m.predict(va.loc[ok, feats]), 0, None)
    P = pd.DataFrame({"sku": va.sku.values, "d": va.d.values, "p": p.values}).pivot(index="d", columns="sku", values="p")[skus]; Y = S.loc[w0:w1, skus]
    r = dict(cfg=cfg, window=f"d{w0}-{w1}", kind=tag, WAPE=M.wape(Y.values, P.values),
             RMSSE=float(np.nanmean([M.rmsse(Y[c], P[c], M.rmsse_scale(S.loc[:w0 - 1, c].values)) for c in skus])), secs=round(time.time() - t))
    rows.append(r); pd.DataFrame(rows).to_csv(RES / "rolling_origin_lgbm.csv", index=False); print(r, flush=True)


for w0, w1 in CLEAN:
    for c in CFG: run(c, w0, w1, "clean (decision)")
for w0, w1 in REPORT:
    for c in ("A_v1", "B_tuned"): run(c, w0, w1, "report only")
df = pd.DataFrame(rows); cl = df[df.kind.str.startswith("clean")]; a = cl[cl.cfg == "A_v1"].set_index("window"); score = {}
for c, g in cl.groupby("cfg"):
    g = g.set_index("window"); score[c] = float(np.mean([0.5 * (g.loc[w, "WAPE"] / a.loc[w, "WAPE"] + g.loc[w, "RMSSE"] / a.loc[w, "RMSSE"]) for w in g.index]))
choice = min(score, key=score.get); choice = choice if score[choice] <= 0.998 else "A_v1"
json.dump({"scores_vs_A": score, "chosen": choice, "rule": "lowest mean relative score on 4 clean windows (d1718-1829); must beat A by >=0.2%, else keep A"}, open(RES / "rolling_origin_decision.json", "w"), indent=1)
print("SCORES", score, "-> CHOSEN", choice, flush=True)
