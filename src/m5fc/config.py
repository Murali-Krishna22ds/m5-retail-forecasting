"""Central configuration. Change here, everything downstream follows."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"
for p in (PROC, RES, FIG):
    p.mkdir(parents=True, exist_ok=True)

# ---- M5 timeline (d_1 = 2011-01-29) -------------------------------------
LAST_TRAIN_D = 1885          # models are fit on d <= 1885
CAL_START, CAL_END = 1886, 1913   # calibration / validation window (28 d)
TEST_START, TEST_END = 1914, 1941  # held-out test window (28 d)
H = 28

# ---- subset (full M5 = 30,490 items; compute budget here = 1 CPU / 3 GB) --
N_ITEMS = {"FOODS": 20, "HOUSEHOLD": 12, "HOBBIES": 8}
SEED = 42
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]   # -> 80% and 95% central intervals
MIN_LAG = 28                 # direct multi-horizon: every lag feature >= H

# ---- inventory economics (assumptions, all editable; see docs/DATA.md) ---
COST_FRAC = 0.70             # unit cost = 70% of shelf price
HOLD_RATE_WEEK = 0.06        # weekly holding+shrink as fraction of unit cost
STOCKOUT_MARGIN_MULT = 1.0   # lost margin
GOODWILL_FRAC = 0.10         # extra penalty as fraction of price per unit short
CYCLE_DAYS = 7               # replenishment cycle
