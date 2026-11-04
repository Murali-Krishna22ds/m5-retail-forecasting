#!/bin/sh
# Full pipeline on the files in data/raw. Logs to results/pipeline.log, creates results/pipeline.done at the end.
# Env: SKIP_BASELINES / SKIP_TUNE / SKIP_ROLLING / SKIP_TFT1 / SKIP_TFT2 = 1 reuse those (deterministic) outputs. Total from scratch: ~100 min on 1 CPU.
cd "$(dirname "$0")/.."; L=results/pipeline.log; : > $L; rm -f results/pipeline.done
step() { n=$1; shift; echo "[$(date +%H:%M:%S)] START $n" >> $L; "$@" >> results/step.log 2>&1 || { echo "FAILED: $n" >> $L; exit 1; }; echo "[$(date +%H:%M:%S)] done $n" >> $L; }
step features python -c "import sys; sys.path.insert(0,'src'); from m5fc.features import build_all; build_all(force=True)"
[ -z "$SKIP_BASELINES" ] && step baselines python scripts/run_baselines.py
[ -z "$SKIP_TUNE" ] && step tune python scripts/tune_lgbm.py
[ -z "$SKIP_ROLLING" ] && step rolling python scripts/rolling_origin_lgbm.py
step decision python scripts/apply_decision.py
step lgbm python scripts/run_lgbm.py
[ -z "$SKIP_TFT1" ] && step tft1 python scripts/run_tft.py --epochs 10 --batches 400
[ -z "$SKIP_TFT2" ] && step tft2 python scripts/run_tft.py --epochs 10 --batches 400 --hidden 24 --heads 4 --hidden-cont 12 --dropout 0.15 --lr 0.008 --seed 7 --name tft2
step ensemble python scripts/run_ensemble.py
step reconcile python scripts/run_reconcile.py
step eval python scripts/run_eval.py
step optimize python scripts/run_optimize.py
step report sh -c "python scripts/make_architecture.py && python scripts/make_report.py && python scripts/make_demo_html.py"
step tests python -m pytest -q tests
echo ALL_DONE >> $L; touch results/pipeline.done
