const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");
const R = "/home/claude/m5-retail-forecasting/";
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
s.addText("Hierarchical LightGBM / TFT forecasts with prediction intervals, Darts reconciliation, OR-Tools ordering and a Dash app", { placeholder: "body" });
s.addNotes("Hook: a forecast is only useful if it turns into an order. We built the whole chain on real Walmart M5 data and tested every link on a held-out 28 days.");

// 2 problem
s = content("The problem: forecast every shelf, then decide what to order", "Problem", "Explain the hierarchy: 400 item-store series roll up to 10 stores, 3 states, total. Planners at each level must see the same numbers. Then explain that the real business objective is cost: stockouts vs leftovers under budget and shelf limits. Mention scope honestly: 40 items x 10 stores because of the 1-CPU build machine.");
txt(s, bullets(["Real M5 data: sales, prices, promotions, holidays, SNAP days (2011-2016)", "Scope: 40 items x 10 stores x 3 states = 400 series, 414 hierarchy nodes", "Hard part 1: demand is intermittent (many zero days); a point forecast hides the risk", "Hard part 2: store forecasts must add up to state and total forecasts", "Hard part 3: the output that matters is an order quantity, not a forecast", "Held-out test: last 28 days (d1914-d1941), never used for fitting or tuning"]), { x: 0.5, y: 1.2, w: 5.4, h: 3.9, fontSize: 15 });
stat(s, 6.4, 1.3, "400", "bottom series, 414 nodes", THEME.colors.accent1); stat(s, 6.4, 2.55, "28 d", "held-out horizon", THEME.colors.accent2); stat(s, 6.4, 3.8, "1 CPU", "whole project fit on it", THEME.colors.accent3);

// 3 architecture + design
s = content("Design: leak-free pipeline from data to dashboard", "Problem", "Walk left to right. Key design decision: every sales-derived feature is lagged at least 28 days, so one model forecasts all 28 days and there is no recursive error. We prove no leakage with a unit test that changes future data and checks features do not move. Three windows: train, calibration, test.");
pic(s, "docs/architecture.png", 0.5, 1.05, 6.7, 3.75);
txt(s, bullets(["Direct 28-day forecasts, lags >= 28", "Train <= d1885", "Calibrate d1886-1913", "Test d1914-1941", "48 pytest tests"]), { x: 7.4, y: 1.3, w: 2.2, h: 3.2, fontSize: 13 });

// 4 accuracy table
s = content("LightGBM beats Prophet by a wide, statistically solid margin", "Results", "Read the table by row. Prophet already uses holidays, SNAP and price, so it is a fair baseline. LightGBM wins at every level. The paired bootstrap over series gives a 95 percent interval of 0.076 to 0.162 for the WAPE gap, so it is not noise. TFT has the best item-level WAPE but I will show on the next slides why I would not ship it.");
const H = (t) => ({ text: t, options: { bold: true, color: THEME.colors.lt1, fill: { color: THEME.colors.dk2 }, align: "center" } });
const row = (n, a, bold) => [{ text: n, options: { bold, align: "left" } }, ...a.map(v => ({ text: v, options: { bold, align: "center" } }))];
s.addTable([[H("Model"), H("WAPE Total"), H("WAPE State"), H("WAPE Store"), H("WAPE Item"), H("RMSSE Item"), H("Cov80 / Cov95")],
  row("Seasonal naive", ["0.185", "0.202", "0.236", "0.884", "0.971", "0.78 / 0.86"]), row("Prophet", ["0.143", "0.140", "0.205", "0.812", "0.812", "0.76 / 0.89"]),
  row("LightGBM", ["0.121", "0.124", "0.149", "0.689", "0.735", "0.82 / 0.97"]), row("LightGBM + MinT", ["0.099", "0.119", "0.155", "0.696", "0.736", "0.82 / 0.97"], true),
  row("TFT", ["-", "-", "-", "0.658", "0.777", "0.47 / 0.56"])], { x: 0.5, y: 1.2, w: 9.0, colW: [2.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.5], fontSize: 12, color: THEME.colors.dk1, border: { type: "solid", pt: 0.5, color: "CED4DA" }, rowH: 0.42, valign: "middle" });
txt(s, bullets(["WAPE gap Prophet minus LightGBM+MinT: +0.116, 95% CI [0.076, 0.162]", "TFT: best item WAPE (it forecasts the median) but worse RMSSE and badly under-covered intervals"]), { x: 0.5, y: 4.0, w: 9.0, h: 1.0, fontSize: 13 });

// 5 forecasts figure
s = content("Forecasts come with 80% and 95% intervals at every level", "Results", "Point at the dotted line: that is the forecast origin. Everything to the right is the test window. Note how the bands widen at item level where demand is count data. Admit the weak spot: the Total 95 percent lower band is too wide because the aggregate quantile model sees only 14 series.");
pic(s, "results/figures/fig1_forecasts.png", 1.0, 1.0, 7.4, 4.1);
txt(s, [{ text: "Item-store coverage:\n80% interval 0.82\n95% interval 0.97", options: { breakLine: false } }], { x: 8.5, y: 1.6, w: 1.4, h: 1.6, fontSize: 11 });

// 6 calibration
s = content("Intervals: LightGBM is calibrated, TFT is not", "Results", "Coverage should hit the dashed line. LightGBM is on target. TFT only covers 47 percent for a nominal 80 percent. Conformal recalibration on LightGBM found an adjustment of exactly zero, meaning it was already valid. A first conformal attempt on single aggregate nodes made coverage worse because 28 correlated days is too small a sample, so we restricted it.");
pic(s, "results/figures/fig3_calibration.png", 0.5, 1.15, 6.0, 2.18);
txt(s, bullets(["Native quantile heads: 2.5 / 10 / 50 / 90 / 97.5%", "Conformal (CQR) check: adjustment = 0, already valid", "TFT under-covers: 47% vs 80% nominal", "Why we ship LightGBM: better RMSSE, calibrated, coherent aggregates"]), { x: 6.7, y: 1.2, w: 2.9, h: 3.6, fontSize: 13 });
txt(s, [{ text: "Scaled pinball loss, item level: LightGBM 0.210, TFT 0.231, Prophet 0.254, seasonal naive 0.317", options: {} }], { x: 0.5, y: 3.6, w: 6.0, h: 0.8, fontSize: 13, color: THEME.colors.accent5 });

// 7 reconciliation
s = content("Reconciliation makes every level add up, and helps at the top", "Results", "Base forecasts disagreed by up to 80 units between a state and the sum of its stores. After Darts MinT the gap is 1e-13. Be balanced: Total WAPE improves 18 percent, state 4 percent, but store gets 4 percent worse and item 1 percent worse. Top-down is clearly worst. We clip negatives and re-sum so the result is both coherent and non-negative.");
pic(s, "results/figures/fig4_reconciliation.png", 0.5, 1.1, 6.6, 1.98);
txt(s, bullets(["Base gap up to ~80 units; reconciled ~1e-13", "MinT: Total WAPE 0.121 to 0.099 (-18%)", "State 0.124 to 0.119; Store 0.149 to 0.155 (worse); Item +1%", "5 methods compared; top-down worst"]), { x: 7.3, y: 1.2, w: 2.3, h: 3.6, fontSize: 12.5 });
txt(s, [{ text: "Store totals equal state totals exactly: verified by tests on every window.", options: {} }], { x: 0.5, y: 3.4, w: 6.6, h: 0.6, fontSize: 13 });

// 8 inventory
s = content("OR-Tools cuts realised cost by 44% on a real 4-week backtest", "Results", "This is the business result. We turn quantile forecasts into demand scenarios, then solve a mixed-integer program per store: minimise expected holding plus stockout cost with case packs, a purchasing budget and shelf capacity. Backtest re-plans weekly with leftover stock carried forward. Be clear that the cost parameters are stated assumptions, so the relative comparison is the result. The optimiser holds more stock: it is buying service.");
stat(s, 0.5, 1.1, "$6.1k", "OR-Tools MILP cost", THEME.colors.accent1); stat(s, 2.75, 1.1, "$10.9k", "order the mean forecast", THEME.colors.accent5); stat(s, 5.0, 1.1, "79%", "MILP fill rate (vs 61%)", THEME.colors.accent3); stat(s, 7.25, 1.1, "$0.45k", "perfect-foresight floor", THEME.colors.accent2);
pic(s, "results/figures/fig6_inventory.png", 0.5, 2.35, 9.0, 2.48);
txt(s, [{ text: "10 stores x 40 SKUs x 4 weeks. Cost = holding + stockout under stated assumptions. Same ranking for Prophet, seasonal-naive and TFT scenarios.", options: {} }], { x: 0.5, y: 4.9, w: 9.0, h: 0.4, fontSize: 11, color: THEME.colors.accent5 });

// 9 challenges
s = content("What went wrong, and how testing caught it", "Wrap-up", "Tell the story of the bugs, it shows rigour. Calendar NaNs crashed TFT. Shifting integer interval bounds collapsed coverage from 82 to 64 percent: a fraction of a unit turns covered into missed for counts. Prophet quantiles crossed, caught by a test. Non-deterministic hash seeding made backtests irreproducible. And the Dash app crashed when TFT was selected above item level.");
txt(s, bullets(["Interval shift broke coverage 82% to 64%: integer-valued bounds moved by fractions; now shift aggregates only", "Prophet's 80% and 95% bands crossed: found by monotonicity test, now sorted", "Backtest differed run to run: Python hash() seeding; now CRC32, verified bit-identical", "Calendar NaNs crashed TFT; the app crashed for TFT above item level: both fixed with regression tests", "Conformal calibration on single nodes made coverage worse: restricted to pooled bottom series"]), { x: 0.5, y: 1.15, w: 5.6, h: 3.9, fontSize: 13.5 });
stat(s, 6.7, 1.3, "48", "tests passing", THEME.colors.accent3); stat(s, 6.7, 2.6, "5", "bugs found by testing", THEME.colors.accent2);
txt(s, [{ text: "Honest limits: 400 of 30,490 series, one 28-day window, assumed inventory costs.", options: {} }], { x: 6.5, y: 3.9, w: 3.1, h: 1.0, fontSize: 12, color: THEME.colors.accent5 });

// 10 next steps
s = content("Next steps", "Wrap-up", "Close with what you would do with more time: scale to full M5, rolling-origin evaluation, probabilistic reconciliation, better TFT, and deploy with real cost data. Mention the demo link and the repo.");
txt(s, bullets(["Scale to all 30,490 series and run rolling-origin cross-validation instead of one window", "Refit on the calibration window before testing; tune LightGBM and train TFT longer with calibrated quantiles", "Probabilistic reconciliation so intervals, not just points, are coherent across levels", "Replace assumed costs with real margins, lead times and supplier constraints; add multi-period lookahead", "Deploy the Dash app (Dockerfile and render.yaml are ready) and monitor forecast coverage in production"]), { x: 0.5, y: 1.2, w: 9.0, h: 3.0, fontSize: 16 });
txt(s, [{ text: "Repo: README, docs/DATA.md, docs/RESULTS.md  |  Dash app: python app/app.py  |  Interactive demo page included", options: {} }], { x: 0.5, y: 4.6, w: 9.0, h: 0.4, fontSize: 12, color: THEME.colors.accent5 });

pres.writeFile({ fileName: R + "presentation/M5_final_presentation.pptx" }).then(f => applyTheme(f, THEME)).then(() => console.log("ok"));
