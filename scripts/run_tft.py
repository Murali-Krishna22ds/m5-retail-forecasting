"""Step 4: Temporal Fusion Transformer -> results/forecasts/tft.parquet"""
import sys, time, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import argparse
from m5fc.features import build_all
from m5fc.tft import run
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batches", type=int, default=300)
a = ap.parse_args()
bt, at, S, h = build_all()
t = time.time(); fc, hist = run(bt, a.epochs, a.batches); print(f"tft done {time.time()-t:.0f}s")
json.dump({"seconds": time.time() - t, "metrics": hist, "epochs": a.epochs, "batches_per_epoch": a.batches}, open("results/tft_info.json", "w"))
