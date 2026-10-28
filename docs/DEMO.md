# Demo

1. **Interactive snapshot (no install):** `docs/demo.html` (published copy: https://claude.ai/artifact/LY5o3MSNJ97M1E4yH6xqSA) - node dropdown over the hierarchy with 80/95% bands, accuracy, reconciliation, order recommendations, backtest. Built from the real result files by `scripts/make_demo_html.py`.
2. **Full app:** `make app` -> http://127.0.0.1:8050. Live OR-Tools re-optimisation (budget / capacity / stockout-penalty / on-hand sliders) only works here. Callbacks were exercised over HTTP and by `tests/test_app.py`.
3. **Screen recording (to do by a human):** not produced - the build environment had no browser/screen capture. Suggested 90-second script:
   Forecasts tab (Total -> CA -> CA_1 -> one item; toggle 95% band, add TFT) -> Accuracy tab (WAPE by level, Cov80) -> Reconciliation tab (CA gap before/after) ->
   Order tab (CA_1 week 1; drag budget factor 1.05 -> 0.7, watch orders and fill rate change; raise stockout penalty; set 3 days on-hand).
4. **Hosting:** `docker build -t m5 . && docker run -p 8050:8050 m5`, or Render via `render.yaml`. Not deployed from the build environment.
