"""Writes presentation/numbers.json (pre-formatted strings) from result files, so the deck never contains hand-typed numbers."""
import json, subprocess, sys; sys.path.insert(0, "src")
import pandas as pd
from m5fc.config import RES
t = pd.read_csv(RES / "metrics_test.csv"); inv = pd.read_csv(RES / "inventory_summary.csv"); boot = json.load(open(RES / "bootstrap_wape_diff.json"))
ens = json.load(open(RES / "ensemble_info.json")); prim = json.load(open(RES / "inventory_info.json"))["primary"]
g = lambda m, l, c: t[(t.model == m) & (t.level == l)][c].values[0]
f3 = lambda v: f"{v:.3f}"
row = lambda m: [f3(g(m, l, "WAPE")) for l in ("Total", "State", "Store", "Item-Store")] + [f3(g(m, "Item-Store", "RMSSE")), f"{g(m, 'Item-Store', 'Cov80'):.2f} / {g(m, 'Item-Store', 'Cov95'):.2f}"]
N = {"rows": {m: row(m) for m in ["snaive", "prophet", "lgbm_v1_mint", "ens_mint"]}}
N["rows"]["tft_avg"] = ["-", "-", "-", f3(g("tft_avg", "Item-Store", "WAPE")), f3(g("tft_avg", "Item-Store", "RMSSE")), f"{g('tft_avg', 'Item-Store', 'Cov80'):.2f} / {g('tft_avg', 'Item-Store', 'Cov95'):.2f} raw"]
ci = lambda k: f"{boot[k]['mean']:+.3f}, 95% CI [{boot[k]['ci95'][0]:+.3f}, {boot[k]['ci95'][1]:+.3f}]"
N["boot_prophet"] = ci("WAPE(prophet) - WAPE(ens_mint)"); N["boot_v1"] = ci("WAPE(lgbm_v1_mint) - WAPE(ens_mint)")
N["spl"] = {m: f3(g(m, "Item-Store", "SPL")) for m in ["snaive", "prophet", "lgbm_v1_mint", "tft_avg", "tft_cqr", "ens_mint"]}
N["cov"] = {m: f"{g(m, 'Item-Store', 'Cov80'):.2f} / {g(m, 'Item-Store', 'Cov95'):.2f}" for m in ["tft_avg", "tft_cqr", "ens_mint", "lgbm_v1_mint"]}
N["recon"] = {m: [f3(g(m, l, "WAPE")) for l in ("Total", "State", "Store", "Item-Store")] for m in ["lgbm", "lgbm_mint", "ens", "ens_mint"]}
p = inv[inv.model == prim].set_index("policy"); v1 = inv[inv.model == "LightGBM v1 + MinT"].set_index("policy")
k = lambda v: f"${v / 1000:.1f}k"
N["inv"] = {"milp": k(p.loc["milp", "cost"]), "mean": k(p.loc["mean", "cost"]), "safety": k(p.loc["safety", "cost"]), "oracle": f"${p.loc['oracle', 'cost'] / 1000:.2f}k",
            "fill_milp": f"{p.loc['milp', 'fill_rate']:.0%}", "fill_mean": f"{p.loc['mean', 'fill_rate']:.0%}", "cut_vs_mean": f"{1 - p.loc['milp', 'cost'] / p.loc['mean', 'cost']:.0%}",
            "cut_vs_safety": f"{1 - p.loc['milp', 'cost'] / p.loc['safety', 'cost']:.0%}", "v1_milp": k(v1.loc["milp", "cost"]), "v1_fill": f"{v1.loc['milp', 'fill_rate']:.1%}", "fill_milp_1dp": f"{p.loc['milp', 'fill_rate']:.1%}"}
N["weight"] = f"{ens['blend_weight_tft']:.1f}"
N["tests"] = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], capture_output=True, text=True).stdout.strip().splitlines()[-1].split()[0]
N["item_gain"] = f"{(1 - g('ens_mint', 'Item-Store', 'WAPE') / g('lgbm_v1_mint', 'Item-Store', 'WAPE')):.1%}"
json.dump(N, open("presentation/numbers.json", "w"), indent=1); print(json.dumps(N, indent=1)[:1800])
