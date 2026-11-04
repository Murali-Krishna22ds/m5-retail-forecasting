"""Select LightGBM features + hyper-parameters on a SEPARATE selection window (train <= d1857, select on d1858-d1885).
Neither the calibration window (d1886-1913) nor the test window (d1914-1941) is touched. Writes results/best_params.json."""
import sys, json, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from m5fc import metrics as M
from m5fc.features import build_all, BOTTOM_FEATS_BASE, BOTTOM_FEATS, BOTTOM_CATS
from m5fc.lgbm import _params
from m5fc.config import RES, SEED

T_END, V0, V1 = 1857, 1858, 1885
bt, at, S, h = build_all(); skus = list(h.sku)
tr = bt[(bt.d >= 500) & (bt.d <= T_END) & (bt.avail == 1) & bt.y.notna()]
va = bt[(bt.d >= V0) & (bt.d <= V1)]; va_fit = va[va.avail == 1]
Y = S.loc[V0:V1, skus]; scale = {c: M.rmsse_scale(S.loc[:T_END, c].values) for c in skus}
LOG = RES / "tuning_lgbm.csv"; rows = []
print(f"train rows {len(tr):,}  selection rows {len(va):,}  (d{V0}-d{V1})", flush=True)


def wide_pred(p_fit):
    p = pd.Series(0.0, index=va.index); p.loc[va_fit.index] = np.clip(p_fit, 0, None)
    return pd.DataFrame({"sku": va.sku.values, "d": va.d.values, "p": p.values}).pivot(index="d", columns="sku", values="p")[skus]


def fit_eval(name, feats, params, rounds=900):
    t = time.time(); cats = [c for c in BOTTOM_CATS if c in feats]
    dtr = lgb.Dataset(tr[feats], tr.y, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(va_fit[feats], va_fit.y, categorical_feature=cats, reference=dtr)
    m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
    P = wide_pred(m.predict(va_fit[feats], num_iteration=m.best_iteration))
    r = dict(name=name, n_feats=len(feats), WAPE=M.wape(Y.values, P.values), RMSSE=float(np.nanmean([M.rmsse(Y[c], P[c], scale[c]) for c in skus])),
             best_iter=m.best_iteration, secs=round(time.time() - t), **{k: params[k] for k in ("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "tweedie_variance_power", "learning_rate")})
    rows.append(r); pd.DataFrame(rows).to_csv(LOG, index=False); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True); return r


base = _params("tweedie", tweedie_variance_power=1.15, learning_rate=0.08)
e0 = fit_eval("base_features/default", BOTTOM_FEATS_BASE, base)
e1 = fit_eval("ext_features/default", BOTTOM_FEATS, base)
rel = lambda r: r["WAPE"] / e0["WAPE"] + r["RMSSE"] / e0["RMSSE"]          # lower is better; baseline = 2.0
feats, fname = (BOTTOM_FEATS, "ext") if rel(e1) < rel(e0) else (BOTTOM_FEATS_BASE, "base")
print("-> feature set:", fname, flush=True)

rng = np.random.default_rng(0); best, best_p = (rel(e1) if fname == "ext" else rel(e0)), base
for i in range(8):
    p = _params("tweedie", learning_rate=0.08, num_leaves=int(rng.choice([31, 63, 127, 255])), min_data_in_leaf=int(rng.choice([30, 60, 120, 250])),
                feature_fraction=float(rng.choice([0.5, 0.7, 0.9])), lambda_l2=float(rng.choice([0, 1, 10])), tweedie_variance_power=float(rng.choice([1.05, 1.15, 1.3, 1.5])))
    r = fit_eval(f"random_{i}", feats, p)
    if rel(r) < best: best, best_p = rel(r), p
print("-> best point params:", {k: best_p[k] for k in ("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "tweedie_variance_power")}, "score", round(best, 4), flush=True)

# quantile heads: old settings vs point-tuned settings, judged by pinball on the selection window (q0.5 and q0.9)
def qfit(alpha, params, nround=350):
    cats = [c for c in BOTTOM_CATS if c in feats]; dtr = lgb.Dataset(tr[feats], tr.y, categorical_feature=cats)
    m = lgb.train(dict(params, objective="quantile", alpha=alpha, learning_rate=0.05), dtr, nround)
    P = wide_pred(m.predict(va_fit[feats])); return M.pinball(Y.values, P.values, alpha)
old_q = _params("quantile", num_leaves=31)
new_q = dict(best_p, num_leaves=min(best_p["num_leaves"], 63)); new_q.pop("tweedie_variance_power", None)
qres = {}
for a in (0.5, 0.9):
    qres[a] = (qfit(a, old_q), qfit(a, new_q)); print(f"quantile {a}: pinball old={qres[a][0]:.4f} tuned={qres[a][1]:.4f}", flush=True)
use_new_q = all(n < o for o, n in qres.values())
point = {k: best_p[k] for k in ("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "tweedie_variance_power", "bagging_fraction", "max_bin")}
quant = {k: new_q[k] for k in ("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "bagging_fraction", "max_bin")} if use_new_q else \
        {k: old_q[k] for k in ("num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "bagging_fraction", "max_bin")}
base_row = {"WAPE": e0["WAPE"], "RMSSE": e0["RMSSE"]}; best_row = min([r for r in rows if r["name"] != "base_features/default"], key=rel)
json.dump({"feature_set": fname, "features": feats, "point": point, "quantile": quant, "quantile_tuned_used": bool(use_new_q),
           "selection_window": [V0, V1], "baseline_selection_scores": base_row, "chosen_selection_scores": {"WAPE": best_row["WAPE"], "RMSSE": best_row["RMSSE"]}},
          open(RES / "best_params.json", "w"), indent=1)
print("DONE", flush=True)
