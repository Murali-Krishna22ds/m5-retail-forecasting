PY ?= python
.PHONY: all data features baselines lgbm tft reconcile eval optimize report test app clean
all: data features baselines lgbm tft reconcile eval optimize report test

download:        ## fetch M5 (Nixtla GitHub mirror, ~50 MB zip; or place Kaggle files in data/raw/)
	mkdir -p data/raw && curl -L -o data/raw/m5.zip https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip && unzip -o data/raw/m5.zip -d data/raw
data:            ; $(PY) -c "import sys; sys.path.insert(0,'src'); from m5fc.data import build_subset; build_subset(force=True)"
features:        ; $(PY) -c "import sys; sys.path.insert(0,'src'); from m5fc.features import build_all; build_all(force=True)"
baselines:       ; $(PY) scripts/run_baselines.py
lgbm:            ; $(PY) scripts/run_lgbm.py
tft:             ; $(PY) scripts/run_tft.py --epochs 10 --batches 400
reconcile:       ; $(PY) scripts/run_reconcile.py
eval:            ; $(PY) scripts/run_eval.py
optimize:        ; $(PY) scripts/run_optimize.py
report:          ; $(PY) scripts/make_architecture.py && $(PY) scripts/make_report.py
test:            ; $(PY) -m pytest -q tests
app:             ; $(PY) app/app.py
