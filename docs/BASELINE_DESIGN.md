# Baseline design (v1, 2026-11-07)

Nothing here has been run yet. This fixes what we'll measure *before* seeing any results, so the comparison stays honest.

## What the data showed first (`src/explore.py`, `results/country_summary.csv`)
- UCI Online Retail II: a UK **online gift wholesaler**, Dec 2009 – Dec 2011 (25 months), about 1.04M sales lines after removing cancellations, returns and non-product codes.
- UK: 36,184 orders (about 1,447 a month). **Germany: 753 orders (about 30 a month)**, 2,168 products, 2,166 of them also sold in the UK. Japan: 33 orders in total, too few to use.
- Germany weekly units: median about 1,800, with only 3 zero weeks across 106 weeks, but lumpy. Individual wholesale orders cause spikes (Apr and Jul 2010, May 2011).
- Both markets peak in autumn: Oct–Nov hold about 25% of a year's units in each. Germany's monthly units run at roughly 12–45 per 1,000 UK units.

**Design consequence:** with about 30 orders a month, Germany has almost no signal per product per day. So the forecast target is **weekly units**, at two levels:
1. **Total Germany weekly units** (the headline: does transfer help at all?).
2. **The top 20 Germany products** (does it help per product, where data is thinnest?). These hold about 14% of Germany's units.

## The test setup
- **Test window:** the last 13 weeks (about Sep 9 – Dec 9, 2011), which includes the autumn peak. Forecast 4 weeks ahead, rolling forward one week at a time.
- **Target history given to each model:** the most recent **1, 3, 6 or 12 months** of German data before each forecast date. This is the "how little data" dial.
- **Source history:** the full UK history up to the forecast date. A model never sees any data after the date it is forecasting from.
- **Metric:** WAPE = sum of |actual − forecast| / sum of actual, over the test window. Lower is better. Also report per-product WAPE for the top 20.

## Models
### Target-only baselines (Germany data only)
- **B0 Naive:** next week = last week.
- **B1 Moving average:** next week = mean of the last 4 weeks.
- **B2 LightGBM (Germany only):** features are recent lags (1–4 weeks), rolling means, week of year and a Christmas-season flag.
- *Not possible with 1–6 months of history:* a seasonal naive (same week last year) needs a full year of the target's own data. Seeing last Christmas is exactly what a data-poor market lacks, and it's why transfer might help.

### Transfer models (Germany plus UK history)
- **T1 Scaled UK:** Germany forecast = UK's recent trend for the same product or total, times the Germany/UK ratio measured in the available German months. It borrows the UK's shape and Germany's level.
- **T2 Pooled LightGBM (global model):** one LightGBM trained on UK and German weekly series together, with a country feature and the same lag and calendar features. This is the "global model" idea behind Amazon's DeepAR, in its simplest form.
- **T1b Top-down with UK weekly shape (Duy's idea, 2026-11-07):** forecast Germany's **monthly** total (smoother, so more reliable), then split it into weeks using the **UK's within-month weekly profile**. The UK's data is good at timing; Germany's own data is good enough for the level once it's averaged over a month. Assumes Germany's timing within a month follows the UK's. That's fair for UK → Germany (shared Christmas), but it would need a Japan calendar for US → Japan.
- Later: **T3 Chronos zero-shot** (no training on this data), then **T4 embedding twins** (Duy's idea: match products by description to borrow history).

## What would count as a result
- Plot **WAPE vs. months of German history** (1, 3, 6, 12) for each model.
- Transfer "works" where T1 or T2 beats the best target-only baseline. It "fails" where the gap closes or reverses.
- **Expected, not yet shown:** transfer helps most at 1–3 months, especially across the autumn peak, and the advantage shrinks by 12 months.

## Known limitations
- **Shared calendar:** the UK and Germany both peak for Christmas, so this tests data scarcity, not holiday shift. US → Japan would also need calendar adjustment (see `PLAN.md`).
- **Wholesale, not retail:** customers are businesses ordering in bulk, so a few large orders cause demand spikes. Consider capping extreme orders, and report results with and without them.
- **Small test window:** 13 weeks is one season, so treat differences of a few WAPE points as noise unless they hold across several rolling start dates.
