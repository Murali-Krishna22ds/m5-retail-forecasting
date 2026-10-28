"""Global LightGBM models: point (Tweedie) + quantile (pinball) for bottom series, and a second
model for the aggregate nodes (Total/State/Store). Direct strategy (lags >= 28) -> no recursion error.
Conformalised quantile regression (CQR) recalibrates intervals on the 28-day calibration window."""
from __future__ import annotations
import json, numpy as np, pandas as pd, lightgbm as lgb
from .config import LAST_TRAIN_D, CAL_START, CAL_END, TEST_START, TEST_END, QUANTILES, RES, SEED
from .features import BOTTOM_FEATS, AGG_FEATS, BOTTOM_CATS, AGG_CATS
from .store import save, qcols

MODELS = RES / "models"; MODELS.mkdir(exist_ok=True, parents=True)
TRAIN_FROM_BOTTOM, TRAIN_FROM_AGG = 500, 400


def _params(obj, **kw):
    p = dict(objective=obj, learning_rate=0.05, num_leaves=63, min_data_in_leaf=60, feature_fraction=0.7,
             bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbose=-1, seed=SEED, num_threads=1)
    p.update(kw); return p


def _split(t, feats, cats, d0):
    tr = t[(t.d >= d0) & (t.d <= LAST_TRAIN_D) & t.y.notna()]
    ca = t[(t.d >= CAL_START) & (t.d <= CAL_END)]
    te = t[(t.d >= TEST_START) & (t.d <= TEST_END)]
    return tr, ca, te


def fit_predict(t: pd.DataFrame, feats, cats, d0, id_col, tag, bottom: bool, log=print):
    tr, ca, te = _split(t, feats, cats, d0)
    if bottom:                      # do not train on days when the item was not on sale (no price)
        tr = tr[tr.avail == 1]
    dtr = lgb.Dataset(tr[feats], tr.y, categorical_feature=cats, free_raw_data=False)
    ca_fit = ca[ca.avail == 1] if bottom else ca
    dca = lgb.Dataset(ca_fit[feats], ca_fit.y, categorical_feature=cats, reference=dtr)
    obj = "tweedie" if bottom else "regression"
    extra = dict(tweedie_variance_power=1.15) if bottom else {}
    log(f"[{tag}] train rows={len(tr):,} cal rows={len(ca_fit):,}")
    pm = lgb.train(_params(obj, **extra), dtr, 1500, valid_sets=[dca], callbacks=[lgb.early_stopping(60, verbose=False)])
    best = pm.best_iteration; log(f"[{tag}] point best_iter={best}")
    pm.save_model(str(MODELS / f"{tag}_point.txt"))
    preds = {"point": {w: np.clip(pm.predict(x[feats], num_iteration=best), 0, None) for w, x in (("cal", ca), ("test", te))}}
    imp = pd.DataFrame({"feature": feats, "gain": pm.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(RES / f"feature_importance_{tag}.csv", index=False)
    for q in QUANTILES:
        qm = lgb.train(_params("quantile", alpha=q, num_leaves=31), dtr, min(max(best, 150), 350))
        qm.save_model(str(MODELS / f"{tag}_q{q}.txt"))
        preds[f"q{q}"] = {w: np.clip(qm.predict(x[feats]), 0, None) for w, x in (("cal", ca), ("test", te))}
        log(f"[{tag}] quantile {q} done")
    out = []
    for w, x in (("cal", ca), ("test", te)):
        o = x[["d", id_col]].rename(columns={id_col: "node"}).copy(); o["window"] = w
        o["point"] = preds["point"][w]
        arr = np.sort(np.column_stack([preds[f"q{q}"][w] for q in QUANTILES]), axis=1)   # fix quantile crossing
        for i, q in enumerate(QUANTILES): o[f"q{q}"] = arr[:, i]
        if bottom:
            na = x.avail.values != 1
            for c in ["point"] + qcols(): o.loc[na, c] = 0.0                              # not on shelf -> 0
        out.append(o)
    return pd.concat(out, ignore_index=True), best


ADJ: dict = {}


def cqr(fc: pd.DataFrame, S_all: pd.DataFrame, group_of: dict | None = None) -> pd.DataFrame:
    """Conformalised quantile regression (Romano et al. 2019): widen/shrink each central interval by the
    (1-alpha)(1+1/n)-quantile of conformity scores on the calibration window. Grouped by hierarchy level
    for aggregates (pooled over bottom series)."""
    fc = fc.copy()
    cal = fc[fc.window == "cal"].copy()
    cal["y"] = [S_all.at[d, n] for d, n in zip(cal.d, cal.node)]
    cal["grp"] = cal.node.map(group_of) if group_of else "all"
    cal = cal[cal.grp.str.startswith("bottom")] if group_of else cal    # conformalise only where we can pool (400 series x 28 d)
    pairs = [(0.1, 0.9, 0.2), (0.025, 0.975, 0.05)]
    for lo, hi, alpha in pairs:
        adj = {}
        for g, x in cal.groupby("grp"):
            e = np.maximum(x[f"q{lo}"] - x.y, x.y - x[f"q{hi}"]).values
            n = len(e); lev = min(1.0, (1 - alpha) * (1 + 1 / n))
            adj[g] = np.quantile(e, lev)
        a = (fc.node.map(group_of) if group_of else pd.Series("all", index=fc.index)).map(adj).fillna(0.0).values
        ADJ[(lo, hi)] = adj
        fc[f"q{lo}"] = np.clip(fc[f"q{lo}"] - a, 0, None); fc[f"q{hi}"] = fc[f"q{hi}"] + a
    # keep the median as is, re-sort for monotonicity
    arr = np.sort(fc[qcols()].values, axis=1); fc[qcols()] = arr
    return fc


def run(log=print):
    from .features import build_all
    bt, at, S_all, h = build_all()
    fb, ib = fit_predict(bt, BOTTOM_FEATS, BOTTOM_CATS, TRAIN_FROM_BOTTOM, "sku", "bottom", True, log)
    fa, ia = fit_predict(at, AGG_FEATS, AGG_CATS, TRAIN_FROM_AGG, "node", "agg", False, log)
    base = pd.concat([fa, fb], ignore_index=True)
    save("lgbm", base)
    lvl = {n: ("agg_" + n if n in set(fa.node) else "bottom") for n in base.node.unique()}
    save("lgbm_cqr", cqr(base, S_all, lvl))
    json.dump({"bottom_best_iter": int(ib), "agg_best_iter": int(ia), "cqr_adjustment": {f"{k[0]}-{k[1]}": {g: float(v) for g, v in d.items()} for k, d in ADJ.items()}}, open(RES / "lgbm_info.json", "w"))
    return base
