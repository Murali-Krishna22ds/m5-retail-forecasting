"""Step: TFT averaging, LightGBM+TFT blend (weight chosen on the calibration window), conformal recalibration of intervals."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import numpy as np
from m5fc import config as C
from m5fc.features import build_all
from m5fc.store import FC
from m5fc.ensemble import average, blend, choose_weight, scaled_cqr
bt, at, S, h = build_all(); bottoms = list(h.sku); bset = set(bottoms); info = {}
tft = "tft"
if (FC / "tft2.parquet").exists():
    average(["tft", "tft2"], "tft_avg"); tft = "tft_avg"; info["tft_members"] = ["tft", "tft2"]
info["tft_used"] = tft
info["tft_cqr_adj"] = scaled_cqr(tft, "tft_cqr", S, bset)
info["lgbm_cqr_adj"] = scaled_cqr("lgbm", "lgbm_cqr", S, bset)
w, grid = choose_weight("lgbm", tft, S, bottoms)
info["blend_weight_tft"] = w; info["blend_grid_cal_score"] = {str(k): round(v, 4) for k, v in grid.items()}
blend("lgbm", tft, w, "ens_raw", bset)
info["ens_cqr_adj"] = scaled_cqr("ens_raw", "ens", S, bset)
json.dump(info, open(C.RES / "ensemble_info.json", "w"), indent=1); print(json.dumps(info, indent=1))
