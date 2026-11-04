"""Step 4: Temporal Fusion Transformer -> results/forecasts/tft.parquet"""
import sys, time, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "src")
import argparse
from m5fc.features import build_all
from m5fc.tft import run
ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batches", type=int, default=300)
ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--heads", type=int, default=2); ap.add_argument("--hidden-cont", type=int, default=8)
ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--lr", type=float, default=0.01); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--name", default="tft")
a = ap.parse_args()
bt, at, S, h = build_all()
t = time.time(); fc, hist = run(bt, a.epochs, a.batches, hidden=a.hidden, heads=a.heads, hidden_cont=a.hidden_cont, dropout=a.dropout, lr=a.lr, seed=a.seed, name=a.name); print(f"tft done {time.time()-t:.0f}s")
json.dump({"seconds": time.time() - t, "metrics": hist, "epochs": a.epochs, "batches_per_epoch": a.batches, "hidden": a.hidden, "heads": a.heads, "dropout": a.dropout, "lr": a.lr, "seed": a.seed}, open(f"results/{a.name}_info.json", "w"))
