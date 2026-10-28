"""Step 3: LightGBM point + quantile models -> results/forecasts/lgbm*.parquet"""
import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
from m5fc.lgbm import run
t = time.time(); run(); print(f"lgbm done {time.time()-t:.0f}s")
