"""Step 5: hierarchical reconciliation (Darts) of the LightGBM and ensemble base forecasts + coherence audit."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import pandas as pd
from m5fc.features import build_all
from m5fc.reconcile import reconcile_model, coherence_error
from m5fc.store import load, to_wide, FC
bt, at, S, h = build_all()
JOBS = [("lgbm", m, n) for m, n in {"bottom_up": "lgbm_bu", "top_down": "lgbm_td", "ols": "lgbm_ols", "wls_struct": "lgbm_wlss", "wls_var": "lgbm_mint"}.items()]
if (FC / "ens.parquet").exists(): JOBS += [("ens", "wls_var", "ens_mint"), ("ens", "bottom_up", "ens_bu")]
if (FC / "lgbm_v1.parquet").exists(): JOBS += [("lgbm_v1", "wls_var", "lgbm_v1_mint")]
aud = []
for base_name in ["lgbm", "ens"]:
    if (FC / f"{base_name}.parquet").exists():
        aud.append({"forecast": f"{base_name} (base, unreconciled)", **coherence_error(to_wide(load(base_name, "test"), "point")[S.columns], h)})
for base_name, m, name in JOBS:
    reconcile_model(base_name, S, h, m, name)
    aud.append({"forecast": f"{name} ({m})", **coherence_error(to_wide(load(name, "test"), "point")[S.columns], h)})
pd.DataFrame(aud).to_csv("results/coherence_audit.csv", index=False); print(pd.DataFrame(aud).to_string())
