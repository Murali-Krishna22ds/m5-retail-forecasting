const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");
const R = "/home/claude/m5-retail-forecasting/"; const N = require("./numbers.json");
const THEME = { name: "M5 Forecasting", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1B1F24", lt1: "FFFFFF", dk2: "0B2A5B", lt2: "F1F5FB", accent1: "0B6EFD", accent2: "E8590C", accent3: "2F9E44", accent4: "7048E8", accent5: "868E96", accent6: "D6336C", hlink: "0B6EFD", folHlink: "7048E8" } };
const pres = new pptxgen(); pres.layout = "LAYOUT_16x9"; pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
pres.title = "M5 retail forecasting & replenishment"; pres.author = "Project team";
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "TITLE_SLIDE", background: { color: THEME.colors.dk2 }, objects: [
  { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.5, w: 8.6, h: 1.4, fontSize: 36, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
  { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.1, w: 8.6, h: 1.2, fontSize: 16, color: C.background2, valign: "top", margin: 0 }, text: "" } } ] });
pres.defineSlideMaster({ title: "CONTENT", background: { color: THEME.colors.lt1 }, slideNumber: { x: 9.2, y: 5.25, fontSize: 10, color: THEME.colors.accent5 },
  objects: [ { rect: { x: 0, y: 0, w: 0.18, h: 5.625, fill: { color: THEME.colors.accent1 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9.0, h: 0.7, fontSize: 24, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } } ] });

const sections = ["Problem", "Results", "Wrap-up"]; sections.forEach(s => pres.addSection({ title: s }));
function content(title, sec, notes) { const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec }); s.addText(title, { placeholder: "title" }); if (notes) s.addNotes(notes); return s; }
function txt(s, runs, o) { s.addText(runs, Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, margin: 0, valign: "top", paraSpaceAfter: 6 }, o)); }
function bullets(arr) { return arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1 } })); }
function pic(s, f, x, y, w, h) { s.addImage({ path: R + f, x, y, w, h, altText: f }); }
function stat(s, x, y, big, small, col) { s.addShape(pres.ShapeType.roundRect, { x, y, w: 2.1, h: 1.05, fill: { color: THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, rectRadius: 0.08, objectName: "stat card" });
  s.addText(big, { x, y: y + 0.08, w: 2.1, h: 0.55, align: "center", fontSize: 26, bold: true, color: col, isTextBox: true, margin: 0 });
  s.addText(small, { x: x + 0.1, y: y + 0.62, w: 1.9, h: 0.4, align: "center", fontSize: 10.5, color: C.text1, isTextBox: true, margin: 0 }); }

// 1 title
let s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: "Problem" });
s.addText("From forecasts to orders: demand forecasting and replenishment on Walmart M5", { placeholder: "title" });
s.addText("LightGBM + TFT ensemble with calibrated prediction intervals, Darts reconciliation, OR-Tools ordering and a Dash app", { placeholder: "body" });
s.addNotes("Hook: a forecast is only useful if it turns into an order. We built the whole chain on the official Kaggle M5 files and tested every link on a held-out 28 days.");

// 2 problem
s = content("The problem: forecast every shelf, then decide what to order", "Problem", "Explain the hierarchy: 400 item-store series roll up to 10 stores, 3 states, total. Planners at each level must see the same numbers. The real objective is cost: stockouts vs leftovers under budget and shelf limits. Scope honestly: 40 items x 10 stores because of the 1-CPU build machine.");
txt(s, bullets(["Official Kaggle M5 data: sales, prices, promotions, holidays, SNAP days (2011-2016)", "Scope: 40 items x 10 stores x 3 states = 400 series, 414 hierarchy nodes", "Hard part 1: demand is intermittent (many zero days); a point forecast hides the risk", "Hard part 2: store forecasts must add up to state and total forecasts", "Hard part 3: the output that matters is an order quantity, not a forecast", "Held-out test: last 28 days (d1914-d1941), never used for fitting, selecting or tuning"]), { x: 0.5, y: 1.2, w: 5.4, h: 3.9, fontSize: 15 });
stat(s, 6.4, 1.3, "400", "bottom series, 414 nodes", THEME.colors.accent1); stat(s, 6.4, 2.55, "28 d", "held-out horizon", THEME.colors.accent2); stat(s, 6.4, 3.8, "1 CPU", "whole project fit on it", THEME.colors.accent3);

// 3 design
s = content("Design: leak-free pipeline with three separate data windows", "Problem", "Walk left to right. Every sales-derived feature is lagged at least 28 days so one model forecasts all 28 days, proven by a unit test that corrupts future data. Windows: model and feature selection on d1858-1885, calibration on d1886-1913, test on d1914-1941. Because one 28-day window is a noisy judge, a rolling-origin check on four earlier windows decided which LightGBM configuration to keep, with the rule written before running it.");
pic(s, "docs/architecture.png", 0.5, 1.05, 6.7, 3.75);
txt(s, bullets(["Direct 28-day forecasts, lags >= 28", "Select: d1858-1885", "Calibrate: d1886-1913", "Test: d1914-1941", "Rolling-origin decision rule", N.tests + " tests"]), { x: 7.4, y: 1.3, w: 2.2, h: 3.4, fontSize: 13 });

// 4 accuracy table
s = content("The ensemble beats Prophet and last version at item level", "Results", "Read by row. Prophet already uses holidays, SNAP and price, a fair baseline. The final model is a LightGBM and TFT blend with TFT weight " + N.weight + ", chosen on the calibration window, then reconciled with MinT. Against the previous version the item-level gain is small but statistically supported; at Total, State and Store the previous version is slightly better. Be upfront about that trade-off.");
const H = (t) => ({ text: t, options: { bold: true, color: THEME.colors.lt1, fill: { color: THEME.colors.dk2 }, align: "center" } });
const row = (n, a, bold) => [{ text: n, options: { bold, align: "left" } }, ...a.map(v => ({ text: v, options: { bold, align: "center" } }))];
s.addTable([[H("Model"), H("WAPE Total"), H("WAPE State"), H("WAPE Store"), H("WAPE Item"), H("RMSSE Item"), H("Cov80 / Cov95")],
  row("Seasonal naive", N.rows.snaive), row("Prophet", N.rows.prophet), row("LightGBM + MinT (previous)", N.rows.lgbm_v1_mint), row("Ensemble + MinT (final)", N.rows.ens_mint, true), row("TFT (avg of 2)", N.rows.tft_avg)],
  { x: 0.5, y: 1.2, w: 9.0, colW: [2.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.7], fontSize: 12, color: THEME.colors.dk1, border: { type: "solid", pt: 0.5, color: "CED4DA" }, rowH: 0.42, valign: "middle" });
txt(s, bullets(["WAPE gain vs Prophet: " + N.boot_prophet, "WAPE gain vs previous version (item level): " + N.boot_v1 + "; Total, State, Store are slightly worse", "TFT: best raw item WAPE (median forecast) but worse RMSSE; its 'raw' coverage is an artifact, see slide 6"]), { x: 0.5, y: 3.9, w: 9.0, h: 1.3, fontSize: 12.5 });

// 5 forecasts figure
s = content("Forecasts come with 80% and 95% intervals at every level", "Results", "Point at the dotted line: the forecast origin. Everything to the right is the test window. Bands widen at item level where demand is count data. Be honest about the weak spot: aggregate-level intervals come from a model that sees only 14 series, so the Total 95 percent lower band is too wide.");
pic(s, "results/figures/fig1_forecasts.png", 1.0, 1.0, 7.4, 4.1);
txt(s, [{ text: "Item-store coverage:\n80% interval " + N.cov.ens_mint.split(" / ")[0] + "\n95% interval " + N.cov.ens_mint.split(" / ")[1], options: {} }], { x: 8.5, y: 1.6, w: 1.4, h: 1.6, fontSize: 11 });

// 6 calibration
s = content("Intervals: a counting artifact, found and fixed", "Results", "Important correction story. Raw TFT looked badly under-covered, about 48 percent for an 80 percent interval. The cause: TFT outputs are strictly positive, so a lower bound like 0.0004 misses a true value of exactly 0. Demand is a count with many zeros. A tiny conformal adjustment, fitted on the calibration window, fixes coverage to nominal. LightGBM was already valid. The final ensemble is recalibrated the same way and lands on nominal coverage on the test window. SPL is scaled pinball loss, lower is better; LightGBM previous version is slightly sharper.");
pic(s, "results/figures/fig3_calibration.png", 0.5, 1.15, 6.0, 1.85);
txt(s, bullets(["Raw TFT coverage " + N.cov.tft_avg + ": lower bounds never exactly 0", "Conformal recalibration (calibration window): " + N.cov.tft_cqr, "Final ensemble: " + N.cov.ens_mint, "Scaled pinball loss: final " + N.spl.ens_mint + ", previous LightGBM " + N.spl.lgbm_v1_mint + ", Prophet " + N.spl.prophet]), { x: 6.7, y: 1.2, w: 2.9, h: 3.6, fontSize: 12.5 });

// 7 reconciliation
s = content("Reconciliation guarantees coherence; its accuracy effect is mixed", "Results", "Base forecasts disagreed across levels by tens of units; after Darts MinT the gap is about 1e-13, verified by tests on every window. Be balanced about accuracy: for LightGBM, MinT helps Total and State but costs a little at Store and Item. For the ensemble it helps Total only. Coherence is the benefit you are paying for, not accuracy.");
pic(s, "results/figures/fig4_reconciliation.png", 0.5, 1.1, 6.6, 1.94);
txt(s, bullets(["Gap between levels: tens of units before, ~1e-13 after", "LightGBM Total WAPE " + N.recon.lgbm[0] + " to " + N.recon.lgbm_mint[0] + "; Item " + N.recon.lgbm[3] + " to " + N.recon.lgbm_mint[3], "Ensemble Total " + N.recon.ens[0] + " to " + N.recon.ens_mint[0] + "; Item " + N.recon.ens[3] + " to " + N.recon.ens_mint[3], "Top-down is clearly worst"]), { x: 7.3, y: 1.2, w: 2.3, h: 3.6, fontSize: 12 });
txt(s, [{ text: "Store totals equal state totals exactly: verified on every window.", options: {} }], { x: 0.5, y: 3.4, w: 6.6, h: 0.6, fontSize: 13 });

// 8 inventory
s = content("OR-Tools cuts realised cost by " + N.inv.cut_vs_mean + " on a real 4-week backtest", "Results", "The business result. Quantile forecasts become demand scenarios, then a mixed-integer program per store minimises expected holding plus stockout cost with case packs, a purchasing budget and shelf capacity. Weekly re-planning with leftover stock carried over. Cost parameters are stated assumptions, so the relative comparison is the result. The optimiser holds more stock: it buys service. The previous version's forecasts give " + N.inv.v1_milp + " with the same optimiser, so better forecasts translate into a small saving.");
stat(s, 0.5, 1.1, N.inv.milp, "OR-Tools MILP cost", THEME.colors.accent1); stat(s, 2.75, 1.1, N.inv.mean, "order the mean forecast", THEME.colors.accent5); stat(s, 5.0, 1.1, N.inv.fill_milp, "MILP fill rate (vs " + N.inv.fill_mean + ")", THEME.colors.accent3); stat(s, 7.25, 1.1, N.inv.oracle, "perfect-foresight floor", THEME.colors.accent2);
pic(s, "results/figures/fig6_inventory.png", 0.5, 2.35, 9.0, 2.48);
txt(s, [{ text: "10 stores x 40 SKUs x 4 weeks. Cost = holding + stockout under stated assumptions. Same ranking for every forecast source.", options: {} }], { x: 0.5, y: 4.9, w: 9.0, h: 0.4, fontSize: 11, color: THEME.colors.accent5 });

// 9 lessons
s = content("What we learned about improving accuracy honestly", "Wrap-up", "The key story. We searched features and hyper-parameters on a separate window and the searched parameters looked better there, but they did not generalise: on four clean earlier windows and on the test window they lost to the defaults. A rule written in advance picked extended features with default parameters. The ensemble gave the real item-level gain. We also found that the apparent TFT miscalibration was an artifact, plus several bugs caught by tests. Disclose: we had looked at the tuned configuration test score once before running the rolling-origin check, which is why its decision rule used only clean earlier windows and was written in advance.");
txt(s, bullets(["Searched hyper-parameters won the selection window but lost on 4 clean windows and on test: overfit to one 28-day window", "A decision rule fixed in advance on rolling-origin windows chose extended features + default parameters (+0.3%)", "Ensembling LightGBM + TFT gave the real item-level gain (" + N.item_gain + " WAPE vs previous version)", "TFT 'under-coverage' was a zero-bound artifact, fixed by conformal recalibration", "Bugs caught by tests: interval shifts, crossing quantiles, non-deterministic seeding, app crash"]), { x: 0.5, y: 1.15, w: 5.7, h: 3.9, fontSize: 13 });
stat(s, 6.7, 1.3, N.tests, "tests passing", THEME.colors.accent3); stat(s, 6.7, 2.6, "4", "clean windows decided the config, not test", THEME.colors.accent2);
txt(s, [{ text: "Honest limits: 400 of 30,490 series, one 28-day test window, assumed inventory costs.", options: {} }], { x: 6.5, y: 3.9, w: 3.1, h: 1.0, fontSize: 12, color: THEME.colors.accent5 });

// 10 next steps
s = content("Next steps", "Wrap-up", "Close with what you would do with more time: scale to full M5, rolling-origin evaluation of the whole stack, probabilistic reconciliation, and deploy with real cost data. Mention the demo link and the repo.");
txt(s, bullets(["Scale to all 30,490 series and evaluate the whole stack with rolling-origin cross-validation, not one window", "Refit on the calibration window before testing; train TFT longer with a count-aware (zero-inflated) output", "Probabilistic reconciliation so intervals, not just points, are coherent across levels", "Replace assumed costs with real margins, lead times and supplier constraints; add multi-period lookahead", "Deploy the Dash app (Dockerfile and render.yaml are ready) and monitor forecast coverage in production"]), { x: 0.5, y: 1.2, w: 9.0, h: 3.0, fontSize: 16 });
txt(s, [{ text: "Repo: README, docs/DATA.md, docs/RESULTS.md  |  Dash app: python app/app.py  |  Interactive demo page included", options: {} }], { x: 0.5, y: 4.6, w: 9.0, h: 0.4, fontSize: 12, color: THEME.colors.accent5 });

pres.writeFile({ fileName: R + "presentation/M5_final_presentation.pptx" }).then(f => applyTheme(f, THEME)).then(() => console.log("ok"));
