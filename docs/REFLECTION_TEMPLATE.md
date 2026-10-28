# Individual reflection (each intern writes their own - 1 page)

> This is a template. It must be written by you, in your words, about what *you* did. Do not submit it unedited.

**What I owned** (name specific modules/decisions; e.g. features + leakage test / LightGBM + intervals / TFT / reconciliation / OR-Tools model / Dash app / tests / presentation):

**What I learned** (technical, 3-4 sentences). Facts from this project you can speak to from experience:
- Lagging every sales feature by >= the horizon gives a direct multi-horizon model; a perturbation test proves no leakage.
- WAPE rewards medians on intermittent demand, RMSSE penalises large misses - TFT won one and lost the other.
- Coverage of integer-valued intervals is brittle: shifting a bound by 0.2 turned 82% coverage into 64%.
- Reconciliation can help the top of the hierarchy while slightly hurting the middle - a trade-off, not a free lunch.
- Expected cost is convex in the pack count, so a scenario newsvendor MILP linearises exactly and solves 40x faster than the naive formulation.
- Non-deterministic seeding (`hash()`) silently broke reproducibility.

**A mistake I made and how I found it:**

**What I would do differently / next:**

**Where I needed help and how I got unstuck:**
