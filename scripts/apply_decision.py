"""Applies results/rolling_origin_decision.json to results/best_params.json (which scripts/run_lgbm.py reads).
A_v1: base features + default params | B_tuned: keep tuned params | C_ext_default: extended features + default params."""
import sys, json; sys.path.insert(0, "src")
from m5fc.config import RES
from m5fc.features import BOTTOM_FEATS_BASE, BOTTOM_FEATS
dec = json.load(open(RES / "rolling_origin_decision.json")); tuned = json.load(open(RES / "best_params_tuned_B.json"))
DEF_P = dict(num_leaves=63, min_data_in_leaf=60, feature_fraction=0.7, lambda_l2=1.0, tweedie_variance_power=1.15, bagging_fraction=0.7, max_bin=127)
DEF_Q = dict(num_leaves=31, min_data_in_leaf=60, feature_fraction=0.7, lambda_l2=1.0, bagging_fraction=0.7, max_bin=127)
c = dec["chosen"]
out = dict(tuned)
if c == "A_v1":   out.update(feature_set="base", features=BOTTOM_FEATS_BASE, point=DEF_P, quantile=DEF_Q)
elif c == "C_ext_default": out.update(feature_set="ext", features=BOTTOM_FEATS, point=DEF_P, quantile=DEF_Q)
out["quantile_tuned_used"] = False if c != "B_tuned" else tuned["quantile_tuned_used"]
out["adopted_config"] = c; out["decision_scores_vs_A"] = dec["scores_vs_A"]
json.dump(out, open(RES / "best_params.json", "w"), indent=1); print("adopted:", c, "| features:", out["feature_set"], len(out["features"]), "| point:", out["point"])
