"""Step 5: hierarchical reconciliation (Darts) of the LightGBM base forecasts + coherence audit."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
from m5fc.features import build_all
from m5fc.reconcile import reconcile_model, coherence_error
from m5fc.store import load, to_wide
bt, at, S, h = build_all()
METHODS = {"bottom_up": "lgbm_bu", "top_down": "lgbm_td", "ols": "lgbm_ols", "wls_struct": "lgbm_wlss", "wls_var": "lgbm_mint"}
aud = []
base = to_wide(load("lgbm", "test"), "point")[S.columns]
aud.append({"forecast": "lgbm (base, unreconciled)", **coherence_error(base, h)})
for m, name in METHODS.items():
    reconcile_model("lgbm", S, h, m, name)
    aud.append({"forecast": f"{name} ({m})", **coherence_error(to_wide(load(name, "test"), "point")[S.columns], h)})
pd.DataFrame(aud).to_csv("results/coherence_audit.csv", index=False)
print(pd.DataFrame(aud).to_string())
