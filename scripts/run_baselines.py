"""Step 2: naive + seasonal-naive + Prophet baselines -> results/forecasts/*.parquet"""
import sys, warnings, time; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from m5fc.data import build_subset
from m5fc.features import build_all
from m5fc.baselines import run_naive, run_prophet
from m5fc.store import save
df = build_subset(); bt, at, S, h = build_all()
run_naive(S)
if "--no-prophet" not in sys.argv:
    t = time.time(); out = run_prophet(S, df, at, bt, h)
    save("prophet", out); print(f"prophet done in {time.time()-t:.0f}s")
