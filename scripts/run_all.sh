#!/bin/sh
# Full pipeline on the files in data/raw. Logs to results/pipeline.log, writes results/pipeline.done when finished.
cd "$(dirname "$0")/.."; L=results/pipeline.log; : > $L
step() { echo "[$(date +%H:%M:%S)] START $1" >> $L; shift; "$@" >> results/step.log 2>&1 || { echo "FAILED: $1" >> $L; exit 1; }; echo "[$(date +%H:%M:%S)] done" >> $L; }
step features python -c "import sys; sys.path.insert(0,'src'); from m5fc.features import build_all; build_all(force=True)"
step baselines python scripts/run_baselines.py
step lgbm python scripts/run_lgbm.py
step tft python scripts/run_tft.py --epochs 10 --batches 400
step reconcile python scripts/run_reconcile.py
step eval python scripts/run_eval.py
step optimize python scripts/run_optimize.py
step report sh -c "python scripts/make_architecture.py && python scripts/make_report.py && python scripts/make_demo_html.py"
step tests python -m pytest -q tests
echo ALL_DONE >> $L; touch results/pipeline.done
