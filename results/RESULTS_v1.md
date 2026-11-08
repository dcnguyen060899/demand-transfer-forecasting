# Results v1: Germany total weekly units (2026-11-07)

Script: `src/baselines.py`. Raw predictions: `baseline_predictions.csv`. Table: `baseline_wape.csv`.
Test: 13 weeks, 2011-09-05 to 2011-11-28, horizons 1–4 weeks, pooled.

## WAPE (%), lower is better, by months of German history

| Model | 1 mo | 3 mo | 6 mo | 12 mo | Bias at 3 mo |
|---|---|---|---|---|---|
| B0 naive (DE only) | 70.9 | 70.9 | 70.9 | 70.9 | −2% |
| **B1 4-week average (DE only)** | **52.3** | **52.3** | **52.3** | 52.3 | −6% |
| B2 LightGBM (DE only) | can't train | 71.0 | 70.5 | 58.6 | −2% |
| T1 UK × recent ratio | 61.1 | 61.1 | 61.1 | 61.1 | +20% |
| T1b top-down, UK shape (Duy) | 61.1 | 60.1 | 60.4 | 57.3 | +16% |
| **T2 LightGBM pooled UK+DE** | 72.5 | 63.5 | 60.2 | **52.2** | +23% |

B0, B1 and T1 use only the last 1–4 weeks, so their scores don't change with k.

## What it says
1. **The simple 4-week average beat every transfer model at 1–6 months.** The prediction "transfer helps most when data is scarce" was **not supported** here.
2. **The transfer models over-forecast by about 15–40%.** They copied the UK's autumn rise, but Germany's autumn 2011 rose less than the UK's (Germany ran about 35 units per 1,000 UK units in Oct but about 20 in Nov).
3. **Transfer catches up as German data grows.** Pooled LightGBM goes from 72.5 to 52.2, tying the 4-week average at 12 months, and its bias falls from +42% to +1%. With more German rows, it learns Germany's own level instead of the UK's.
4. **Germany-only LightGBM can't train on 1 month** and is poor on 3–6 months. Too few rows for machine learning.
5. **Noise is large.** At 3 months, the bootstrap 90% intervals overlap for every model (roughly 40–97%). With 13 test weeks and lumpy wholesale orders, none of these differences is conclusive yet.

## Next steps
- Fix the over-forecast: estimate the Germany/UK ratio more robustly, or shrink the UK's seasonal lift toward Germany's own level.
- Repeat with more test windows (for example, rolling through 2011) so the noise estimate shrinks.
- Run the top 20 products, where Germany's data is thinnest.
- Cap single huge orders and report with and without the cap.

## Round 2: borrow the size of the rise from peer markets (Duy's idea), `src/peer_lift.py`

Same test. Both lift models use one formula, so only the source changes: Germany's average weekly level × the source's lift from the window weeks to the target week, one year earlier (smoothed over 5 weeks).

| Model | 1 mo | 3 mo | 6 mo | 12 mo | Bias at 3 mo |
|---|---|---|---|---|---|
| B1 4-week average (DE only) | 52.3 | 52.3 | 52.3 | 52.3 | −6% |
| T1c UK lift | 51.3 | 53.3 | 53.3 | 52.3 | +15% |
| T3 peer lift (FR, IE, NL, ES, BE) | **50.6** | **51.3** | **48.6** | **47.0** | **−38%** |

Autumn lift (Sep–Nov vs Jun–Aug weekly average): 2010: UK 1.78, peers 0.88, Germany 1.43. 2011 (the test): UK 1.68, **Germany 1.59**.

**Reading it honestly**
- T3 has the lowest WAPE at every history length, but it **under-forecasts by 14–38%**. Peers barely rose in autumn 2010 (0.88), so T3 predicts almost no rise. It wins by being cautious, not by being right.
- **Germany's real 2011 rise (1.59) was close to the UK's (1.68)**, not the peers' (0.88). The size of the UK's rise was roughly right. The round 1 over-forecasts came from how T1/T1b estimated Germany's level (ratio and growth terms), not mainly from the UK's lift.
- Why a cautious model can score well: Germany's weeks are lumpy, so a forecast that stays low misses the spikes but stays close to the many quiet weeks. **WAPE alone can reward under-forecasting**, so bias must be reported next to it.
- Bootstrap intervals still overlap (3 months: B1 39–67, T1c 39–74, T3 39–61). No winner is conclusive.

**Lesson for the write-up:** "similar size" doesn't mean "similar seasonality". The peer markets were a worse guide to Germany's rise than the UK, even though Germany is closer to them in size. A good next model would combine the sources (for example, average the UK and peer lifts, weighted by how well each predicted Germany in past seasons) and judge models on WAPE and bias together.

## Round 3: T4 signal extraction (Duy's derivation, `docs/DERIVATION_T4.md`), `src/t4_signal.py`

Germany's level is filtered with a local level Kalman filter, starting from a borrowed prior (the average peer market's weekly units), then multiplied by the UK's seasonal timing. The noise ratio q and the relative noise are fitted on the peer markets, using data before each origin only.

| Model | 1 mo | 3 mo | 6 mo | 12 mo | Bias at 3 mo |
|---|---|---|---|---|---|
| B1 4-week average | 52.3 | 52.3 | 52.3 | 52.3 | −6% |
| T1b top-down (Duy) | 61.1 | 60.1 | 60.4 | 57.3 | +16% |
| T4-level (filtered level, no seasonality) | 50.0 | 48.9 | 48.6 | 47.1 | −27% |
| **T4 filtered level × UK timing** | **47.3** | **44.9** | **44.2** | **43.6** | **−10%** |

Bootstrap 90% interval at 3 months: T4 33–58 vs B1 39–67. Better on average, still overlapping.

**What the filter learned:** maximum likelihood on the peers put q at the bottom of the grid (q ≈ 1e-7 on a wider grid): week-to-week movement in these small markets is almost all one-off noise, so the level should barely react to any single week. Final Kalman gains were 0.13 (1 month), 0.06 (3), 0.03 (6) and 0.02 (12). In practice, T4's level is a long, heavily smoothed average of Germany's window, anchored to the borrowed prior.

**Is it a lucky prior?** Sensitivity (`src/t4_sensitivity.py`, `t4_sensitivity.csv`): with the prior halved or doubled, T4 stays at 44–48 WAPE for 3–12 months. Only at 1 month, where the prior dominates, does a doubled prior hurt (62.0, bias +43%). With q fixed at 0.01 instead of the MLE, results are similar (45–48); at q = 0.1 (less smoothing) they worsen to about 50–51. So the gain comes mainly from **heavy smoothing of Germany's level plus the UK's timing**, not from the particular prior, except when Germany has only one month of data.

**Lesson:** the macro framing (permanent vs. transitory) didn't just describe the problem; fitting it told us *how much* to smooth, and the answer was "a lot". That fixed the main weakness found in Round 1 (estimating Germany's level), and it is the first model that is both the most accurate and close to unbiased. Still one season and 13 test weeks: the next step is a wider rolling test before calling it a win.

## Round 4: wider test, all of 2011 (`src/rolling_2011.py`)

48 target weeks (2011-01-03 to 2011-11-28), horizons 1–4, same history cuts. LightGBM models left out (slow, not competitive in Round 1).

| Model | 1 mo | 3 mo | 6 mo | 12 mo | Bias at 3 mo |
|---|---|---|---|---|---|
| B0 naive | 65.9 | 65.9 | 65.9 | 65.9 | −5% |
| B1 4-week average | 51.4 | 51.4 | 51.4 | 51.4 | −5% |
| T1b top-down (Duy) | 56.4 | 52.2 | 52.5 | 51.9 | +4% |
| T1c UK lift | 52.0 | **48.5** | 50.2 | 48.8 | +4% |
| T3 peer lift | 69.4 | 76.6 | 68.7 | 68.3 | +12% |
| T4-level (flat) | **47.1** | 49.6 | 50.8 | **47.0** | −7% |
| T4 level × UK timing | 47.6 | 50.0 | 52.7 | 48.8 | −1% |

WAPE by quarter at 3 months of history: T4 is best in Q2 (42.1) and **Q4 (39.9 vs B1 51.3)**, but worst in Q1 (71.0 vs B1 42.9), the post-Christmas drop.

Paired bootstrap, WAPE(T4) − WAPE(B1), resampling weeks: 1 mo −3.6 points (90% interval −8.5 to +1.6; T4 wins 88% of resamples), 3 mo −1.4 (−8.7 to +5.9; 62%), 6 mo +1.3 (−6.9 to +9.7; 40%), 12 mo −2.7 (−10.5 to +5.4; 73%).

**Reading it honestly**
- **T4's Round 3 lead mostly doesn't hold across the year.** Over 2011 it beats the 4-week average by about 1–4 points at 1, 3 and 12 months, with near-zero bias, but every interval includes zero. It is a modest, unconfirmed improvement, not a clear win.
- **T4 is strongest in the holiday quarter** (Q4 39.9 vs 51.3) and weakest right after Christmas (Q1 71.0). Its heavily smoothed level is slow to react when demand drops after the peak. That is the cost of trusting a slow-moving level: the same smoothing that ignores one-off spikes also lags real turns.
- **The peer-lift model (T3) collapses over the full year** (68–77). Its Round 2 win was specific to one season, confirming the Round 2 caution.
- **Simple UK lift (T1c) and the flat filtered level are competitive** across the year (about 47–50). No single model dominates every quarter.

**Lesson:** a 13-week test can crown the wrong winner. Across a full year, the honest result is that transfer and signal extraction give small, uncertain gains over a simple average, larger in the holiday quarter, and they hurt after turning points. Next: let the filter react faster at known turning points (for example, a seasonal level or a higher q in January), and test product-level series, where Germany's data is thinner.

## Round 5: T5, nudge harder after Christmas (Duy's fix, `src/t5_turning.py`)

Duy's diagnosis of Round 4: the smoothed level is slow to notice the post-Christmas drop, just as it is slow to notice the autumn rise. Fix: in a known turning window (Dec 15 – Jan 31), multiply the level's movement variance σ_u² by M, so the Kalman gain rises there and the filter follows the new weeks; everywhere else σ_u² is unchanged, so one-off spikes are still ignored. The window comes from the calendar, which is known in advance, so it uses no test data. M = 100 was set as the main value before the run (it moves q from about 1e-7 toward 1e-5, still heavy smoothing); M = 10 and 1000 are sensitivity checks. Same 48-week test as Round 4.

| Model | 1 mo | 3 mo | 6 mo | 12 mo | Bias at 3 mo | Bias at 12 mo |
|---|---|---|---|---|---|---|
| B1 4-week average | 51.4 | 51.4 | 51.4 | 51.4 | −5% | −5% |
| T4 (M = 1) | 47.6 | 50.0 | 52.7 | 48.8 | −1% | +3% |
| T5 M = 10 | 47.6 | 49.8 | 52.2 | 48.4 | −1% | +2% |
| **T5 M = 100 (main)** | **47.6** | **48.2** | **49.3** | **46.6** | **−4%** | **−5%** |
| T5 M = 1000 | 48.1 | 46.1 | 45.9 | 45.6 | −9% | −13% |

WAPE by quarter at 3 months of history: Q1 drops from 71.0 (T4) to 61.7 (M = 100) and 50.8 (M = 1000), still worse than B1's 42.9. Q1 bias confirms the diagnosis: T4 over-forecast January–March by 51% (its level stayed at the December high), T5 M = 100 by 36%, M = 1000 by 10%. Q2–Q4 are unchanged (42.1, 53.0, 39.9), as intended: the fix only acts in the turning window.

Paired bootstrap, WAPE(T5 M = 100) − WAPE(B1): 1 mo −3.7 (90% interval −8.7 to +1.5; T5 wins 88%), 3 mo −3.2 (−9.8 to +3.1; 79%), 6 mo −2.1 (−9.5 to +5.2; 68%), 12 mo −5.0 (−11.9 to +2.2; 87%).

**Reading it honestly**
- **The fix works where it should and nowhere else.** It cuts the Q1 error by about 9 points and leaves the other quarters exactly as they were. T5 now beats B1 by 2–5 points at every history length, including 6 months, where T4 had lost.
- **Still not significant.** Every interval includes zero, though T5 wins in 68–88% of resamples, more often than T4 did (40–88%).
- **Bigger M is not simply better.** M = 1000 has the lowest WAPE (about 46), but it under-forecasts by 9–13%. With 6–12 months of history, January stays inside Germany's window all year; a hard nudge locks the level to the post-Christmas low, and the heavy smoothing afterwards is too slow to lift it back up. That is the same trap as T3 in Round 2, where WAPE rewarded a cautious model, so M = 100 stays the main result.
- **Q1 is still T5's worst quarter.** Even nudging harder, the level starts January at the December peak and needs several weeks to fall. A forecaster that knows the drop is coming (a seasonal factor for January) would help more than a faster reaction to it.

**Lesson:** a model's known weakness can be fixed by telling it *when* to expect real change, using the calendar rather than the test data. The cost is a trade-off between reacting fast and staying unbiased, so WAPE and bias must be judged together when choosing how hard to nudge.
