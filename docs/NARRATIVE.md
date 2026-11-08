# Can a Data-Rich Market Teach a Data-Poor One? Transfer Forecasting for New-Market Demand

**Author:** Duy Nguyen · **Date:** November 8, 2026 · **Status:** working draft, results v2 (exploratory, five rounds)

*Written in the Amazon narrative style: prose first, data in tables, open questions stated plainly. Terms are explained the first time they appear, so a reader new to forecasting can follow. PDF version: `docs/latex/narrative.pdf`.*

## 1. Purpose

When a retailer opens in a new country, it must plan inventory before that country has much sales history. The obvious idea is to borrow the history of a market it already knows well, for example using US demand to plan Japan. This document tests that idea on public data and asks two questions: does borrowing a data-rich market’s history improve forecasts in a data-poor market, and **when does that transfer fail?** The aim is not to ship a model. It is to learn, with honest measurement, where borrowing helps and where it misleads, so the next experiment is better aimed.

**The answer so far, in five rounds.** Borrowing the UK’s history directly did not beat a simple four-week average (Round 1). The error came from estimating Germany’s *size*, not its seasonal *timing* (Round 2). A model from macroeconomics, which separates Germany’s lasting level from one-off orders, gave the best result on the first test (Round 3). On a full year, its lead shrank to a few points and it failed right after Christmas (Round 4). Telling it to react faster at that known turning point fixed most of that failure (Round 5). The final model beats the simple average by 2–5 points at every history length and is close to unbiased, but the gain is not yet statistically confirmed.

## 2. Background

**Demand** here means units sold per week. Strictly, sales are only a stand-in for demand: when a product is out of stock, customers who wanted it can’t buy it, so recorded sales undercount true demand (*censored demand*). This dataset has no stockout records, so the study uses sales and says so.

Forecasting is framed as **supervised regression**. The label is the number of units sold in a future week. The inputs are facts known before that week, such as last week’s units, the average of the last four weeks and the week of the year. Turning a time series into a table of “what happened 1, 2, 3 weeks ago” is called building **lag features**. The one rule that keeps this honest is no **leakage**: an input may never use information from on or after the week being predicted.

Amazon has published several forecasting models: DeepAR (an LSTM-based recurrent network trained as one *global model* across many products), MQ-RNN and MQ-CNN (which forecast many weeks ahead at once and output ranges such as P50 and P90), MQ-Transformer, and Chronos (a pretrained time-series foundation model). This study deliberately starts below all of them, because a complex model is only worth its cost if it beats simple ones.

## 3. Tenets

1.  **Beat a baseline before claiming anything.** A *baseline* is the simplest reasonable forecast, such as “next week equals last week”. A model that can’t beat it isn’t helping, however clever it is.

2.  **Decide the test before seeing results.** The setup, metric and models were written down before any model ran.

3.  **Report error and direction together.** A model can score well by always guessing low, so accuracy is never reported without bias.

4.  **Treat small differences as noise until shown otherwise.** Every comparison carries an uncertainty range.

## 4. The stand-in problem and the data

Public data doesn’t contain the same products sold in the US and Japan, so the study uses a stand-in with the same shape: **UCI Online Retail II**, the transactions of a UK online gift wholesaler from December 2009 to December 2011. After removing cancellations, returns and non-product codes, it has about 1.04 million sales lines.

| Market | Orders (Dec 2009–Dec 2011) | Orders per month | Role |
|---|---|---|---|
| United Kingdom | 36,184 | ~1,450 | Source (data-rich) |
| Germany | 753 | ~30 | Target (data-poor) |
| France, Ireland, Netherlands, Spain, Belgium | 581 down to 143 each | 6–24 | Peer markets |
| Japan | 33 in total | — | Too few to use |

Germany sold 2,168 products, and 2,166 of them were also sold in the UK, so there is plenty to borrow. Germany’s weekly units are very lumpy: in 2011 they ranged from 305 to 7,015, because a single wholesale order can triple a week’s total. The UK, with 250 to 600 orders a week, is smooth.

**A known limitation:** the UK and Germany share Christmas, so this setup tests *data scarcity* more than *calendar differences*. US to Japan would add a harder problem, since peaks such as Golden Week, Obon and New Year fall in different weeks.

## 5. How the test works

The question is posed from early September 2011, before the holiday season. Each model forecasts Germany’s weekly units for the 13 weeks from September 5 to November 28, 2011, from 1 to 4 weeks ahead. To mimic a new market, each model is shown only the most recent **1, 3, 6 or 12 months** of German history. UK and peer history is never cut. With 1 to 6 months, Germany’s own data contains no Christmas; with 12 months, it contains one.

Accuracy is measured by **WAPE** (weighted absolute percentage error): total absolute error divided by total actual units. Lower is better, and it stays stable when some weeks have very few sales. **Bias** is the total forecast minus total actual, as a percentage: positive means over-forecasting. Uncertainty comes from a **bootstrap**, which resamples the 13 test weeks many times to see how much the score could move by chance.

Rounds 1–3 use those 13 autumn weeks. Rounds 4–5 repeat the same test for **every week of 2011** (48 target weeks, January 3 to November 28), so each model faces quiet months, the autumn rise and the drop after Christmas. Models are then compared with a **paired bootstrap**, which resamples the same weeks for both models each time, so the comparison isn’t blurred by which weeks happened to be drawn.

| Model | Uses | Idea |
|---|---|---|
| B0 Naive | Germany only | Next week = last week |
| B1 Moving average | Germany only | Next week = average of the last 4 weeks |
| B2 LightGBM | Germany only | Gradient-boosted decision trees on lag and calendar features |
| T1 Scaled UK | Germany + UK | UK’s pattern, scaled by the recent Germany/UK ratio |
| T1b Top-down (author’s idea) | Germany + UK | Germany’s level from its whole window, timing from the UK |
| T2 Pooled LightGBM | Germany + UK | One model trained on both countries, with country as a feature |
| T1c UK lift | Germany + UK | Germany’s level × the UK’s seasonal rise last year |
| T3 Peer lift (author’s idea) | Germany + peers | Germany’s level × the peer markets’ seasonal rise last year |
| T4 Signal extraction (author’s model) | Germany + peers + UK | Germany’s level filtered from one-off orders (Kalman filter), × the UK’s timing |
| T5 Turning point (author’s fix) | Germany + peers + UK | T4, but reacting faster from mid-December through January |

## 6. Results

### Round 1: does borrowing the UK’s history help?

WAPE (%), lower is better, by months of German history:

| Model                     |        1 mo |     3 mo |     6 mo |    12 mo | Bias at 3 mo |
|:--------------------------|------------:|---------:|---------:|---------:|-------------:|
| B0 Naive                  |        70.9 |     70.9 |     70.9 |     70.9 |     −2% |
| **B1 4-week average**     |    **52.3** | **52.3** | **52.3** |     52.3 |     −6% |
| B2 LightGBM, Germany only | can’t train |     71.0 |     70.5 |     58.6 |     −2% |
| T1 Scaled UK              |        61.1 |     61.1 |     61.1 |     61.1 |    +20% |
| T1b Top-down              |        61.1 |     60.1 |     60.4 |     57.3 |    +16% |
| **T2 Pooled LightGBM**    |        72.5 |     63.5 |     60.2 | **52.2** |    +23% |

The prediction going in was that borrowing would help most when Germany had little data. **It did not.** The simple four-week average beat every borrowing model at 1, 3 and 6 months. The borrowing models over-forecast by 15–40%. The pooled model improved steadily as German data grew, from 72.5 to 52.2, and tied the simple average at 12 months, with its bias falling from +42% to +1%. With enough German rows, it learned Germany’s own level instead of copying the UK’s. The Germany-only LightGBM couldn’t train on one month and was weak at three to six, a reminder that machine learning needs data.

### Round 2: borrow only the size of the rise, from peers instead of the UK

The over-forecasts raised a question: should the size of Germany’s autumn rise come from smaller peer markets rather than the large UK? T1c and T3 use the same formula, so only the source of the rise changes.

| Model             |     1 mo |     3 mo |     6 mo |    12 mo |       Bias at 3 mo |
|:------------------|---------:|---------:|---------:|---------:|-------------------:|
| B1 4-week average |     52.3 |     52.3 |     52.3 |     52.3 |           −6% |
| T1c UK lift       |     51.3 |     53.3 |     53.3 |     52.3 |          +15% |
| T3 Peer lift      | **50.6** | **51.3** | **48.6** | **47.0** | **-38\%** |

Autumn rise, measured as September–November weekly sales divided by June–August:

|                |           2010 |    2011 (the test) |
|:---------------|---------------:|-------------------:|
| UK             | 1.78× |     1.68× |
| Peers combined | 0.88× |                  — |
| Germany        | 1.43× | **1.59×** |

T3 has the lowest WAPE at every history length, but **for the wrong reason**. The peers barely rose in autumn 2010 (they dipped to 0.88×), so T3 predicted almost no rise and under-forecast by 14–38%. It scored well because Germany’s weeks are lumpy: a forecast that stays low misses the spikes but sits close to the many quiet weeks. Germany’s real rise (1.59×) was in fact close to the UK’s (1.68×). That reverses the explanation offered after Round 1: the UK’s rise was roughly the right size, and the earlier over-forecasts came from how those models estimated Germany’s level. The bootstrap 90% ranges still overlapped for every model (at 3 months: B1 39–67%, T1c 39–74%, T3 39–61%), so none of these differences is settled.

### Round 3: separate Germany’s lasting level from one-off orders

Round 2 located the problem in Germany’s level, so Round 3 asks how to estimate that level from lumpy weeks. The author’s idea came from macroeconomics. In the permanent-income model of consumption, a household’s income mixes a lasting part with one-off windfalls, and it should spend based on the lasting part. Germany’s weekly sales have the same structure: a slowly moving level, plus one-off wholesale orders. The author wrote this as a **local level model**:
``` math
d_t = L_t + e_t, \qquad L_t = L_{t-1} + u_t,
```
where $`d_t`$ is the week’s units, $`L_t`$ the lasting level, $`e_t`$ a one-off shock with variance $`\sigma_e^2`$, and $`u_t`$ a lasting change with variance $`\sigma_u^2`$. Each week, the estimate of the level moves by a fraction $`k`$ of the surprise:
``` math
\hat L_t = \hat L_{t-1} + k_t\,(d_t - \hat L_{t-1}).
```
If weeks are mostly one-off noise ($`\sigma_e^2 \gg \sigma_u^2`$), $`k`$ is near 0 and a big week is ignored; if they are mostly lasting change, $`k`$ is near 1. The author’s first formula, $`k = \sigma_u^2/(\sigma_u^2 + \sigma_e^2)`$, is a special case of the exact answer, the **Kalman filter**, which also tracks how unsure the current estimate is ($`P`$):
``` math
P_{t|t-1} = P_{t-1} + \sigma_u^2, \qquad
k_t = \frac{P_{t|t-1}}{P_{t|t-1} + \sigma_e^2}, \qquad
P_t = (1 - k_t)\,P_{t|t-1}.
```

Borrowing enters in two places. A new market has too few weeks to learn its own noise ratio $`q = \sigma_u^2/\sigma_e^2`$, so $`q`$ is fitted by maximum likelihood on the five peer markets, using only data before each forecast date. The filter also starts from a borrowed guess, the average peer market’s weekly units, with wide uncertainty, so Germany’s own data takes over as it arrives. The forecast, T4, is the filtered level × the UK’s seasonal timing from the year before.

| Model (13 autumn weeks) | 1 mo | 3 mo | 6 mo | 12 mo | Bias at 3 mo |
|:---|---:|---:|---:|---:|---:|
| B1 4-week average | 52.3 | 52.3 | 52.3 | 52.3 | −6% |
| T1b Top-down | 61.1 | 60.1 | 60.4 | 57.3 | +16% |
| T4 level only (no seasonality) | 50.0 | 48.9 | 48.6 | 47.1 | −27% |
| **T4 level × UK timing** | **47.3** | **44.9** | **44.2** | **43.6** | −10% |

Fitting the model gave a clear answer about how much to smooth: **a lot**. The fitted $`q`$ was about 10^−7, meaning week-to-week movement in these small markets is almost all one-off noise. By the end of Germany’s window the weight on each new week was only 0.13 (1 month of history), 0.06 (3), 0.03 (6) and 0.02 (12). T4 was the first model that was both the most accurate and close to unbiased. It did not depend on a lucky starting guess: with the borrowed prior halved or doubled, WAPE stayed at 44–48 for 3 to 12 months of history. Only with one month, where the prior dominates, did a doubled prior hurt (62.0). The bootstrap range at 3 months (33–58) still overlapped B1’s (39–67).

### Round 4: the full-year test

Thirteen weeks in one season is a small test, so Round 4 rolls the forecast date through all of 2011. The LightGBM models are left out because they were slow and not competitive in Round 1.

| Model (48 weeks of 2011)      |     1 mo |     3 mo | 6 mo |    12 mo | Bias at 3 mo |
|:------------------------------|---------:|---------:|-----:|---------:|-------------:|
| B0 Naive                      |     65.9 |     65.9 | 65.9 |     65.9 |     −5% |
| B1 4-week average             |     51.4 |     51.4 | 51.4 |     51.4 |     −5% |
| T1b Top-down                  |     56.4 |     52.2 | 52.5 |     51.9 |     +4% |
| T1c UK lift                   |     52.0 | **48.5** | 50.2 |     48.8 |     +4% |
| T3 Peer lift                  |     69.4 |     76.6 | 68.7 |     68.3 |    +12% |
| T4 level only                 | **47.1** |     49.6 | 50.8 | **47.0** |     −7% |
| T4 level × UK timing |     47.6 |     50.0 | 52.7 |     48.8 |     −1% |

**The Round 3 lead mostly did not hold.** Over the full year, T4 beat the four-week average by 1–4 points at 1, 3 and 12 months, with almost no bias, and lost by 1 point at 6 months. In the paired bootstrap every interval included zero. The peer-lift model (T3), which had looked best in Round 2, collapsed to 68–77: its win had been specific to one season.

Splitting the year by quarter showed where T4 fails (table in Round 5, 3 months of history). It was best in the holiday quarter (39.9 against B1’s 51.3) and worst right after Christmas (71.0 against 42.9), when it over-forecast January to March by 51%. The author’s diagnosis: the heavily smoothed level is slow to notice the post-Christmas drop, just as it is slow to notice the autumn rise. The smoothing that correctly ignores one-off orders also ignores real turns.

### Round 5: react faster at a known turning point

The author’s fix: at times when real change is known to happen, let the filter move faster, and keep it slow everywhere else. T5 multiplies the variance of lasting change by a factor $`M`$ in a turning window, December 15 to January 31:
``` math
\sigma_{u,t}^2 = \begin{cases} M\,\sigma_u^2 & \text{if week } t \text{ is in the turning window,}\\
\sigma_u^2 & \text{otherwise.}\end{cases}
```
A larger $`\sigma_u^2`$ raises $`P`$, and so raises the weight $`k`$ on new weeks, only in that window. The window comes from the calendar, which is known in advance, so it uses no test data. *M* = 100 was set as the main value before the run (it moves $`q`$ from about 10^−7 toward 10^−5, still heavy smoothing); *M* = 10 and *M* = 1000 are checks.

| Model (48 weeks of 2011) | 1 mo | 3 mo | 6 mo | 12 mo | Bias 3 mo | Bias 12 mo |
|:---|---:|---:|---:|---:|---:|---:|
| B1 4-week average | 51.4 | 51.4 | 51.4 | 51.4 | −5% | −5% |
| T4 (*M* = 1) | 47.6 | 50.0 | 52.7 | 48.8 | −1% | +3% |
| T5, *M* = 10 | 47.6 | 49.8 | 52.2 | 48.4 | −1% | +2% |
| **T5, *M* = 100 (main)** | **47.6** | **48.2** | **49.3** | **46.6** | −4% | −5% |
| T5, *M* = 1000 | 48.1 | 46.1 | 45.9 | 45.6 | −9% | −13% |

| WAPE by quarter, 3 months of history |       Q1 |   Q2 |   Q3 |   Q4 | Bias in Q1 |
|:-------------------------------------|---------:|-----:|-----:|-----:|-----------:|
| B1 4-week average                    | **42.9** | 57.5 | 52.0 | 51.3 |   +2% |
| T4                                   |     71.0 | 42.1 | 53.0 | 39.9 |  +51% |
| T5, *M* = 100                      |     61.7 | 42.1 | 53.0 | 39.9 |  +36% |
| T5, *M* = 1000                     |     50.8 | 42.0 | 53.0 | 39.9 |  +10% |

**The fix works where it should and nowhere else.** T5 (*M* = 100) cuts the first-quarter error from 71.0 to 61.7 and leaves the other quarters unchanged. It now beats the four-week average at every history length, by 2–5 points, including 6 months, where T4 had lost. The first quarter is still its weakest: even reacting faster, the level starts January at the December peak and needs several weeks to fall.

**Reacting harder is not simply better.** *M* = 1000 has the lowest WAPE, about 46, but under-forecasts by 9–13%. With 6–12 months of history, January stays inside Germany’s window all year. A hard nudge locks the level to the post-Christmas low, and the heavy smoothing afterwards is too slow to lift it back. This is the Round 2 trap again: WAPE rewards a cautious forecast. So *M* = 100 stays the main result.

### How sure are we?

More than after Round 2, but not enough to claim a win. The paired bootstrap of WAPE(T5) − WAPE(B1), over the 48 weeks of 2011, gives:

| German history | Mean difference (points) | 90% interval | Share of resamples T5 wins |
|:---|---:|---:|---:|
| 1 month | −3.7 | −8.7 to +1.5 | 88% |
| 3 months | −3.2 | −9.8 to +3.1 | 79% |
| 6 months | −2.1 | −9.5 to +5.2 | 68% |
| 12 months | −5.0 | −11.9 to +2.2 | 87% |

T5 wins in most resamples at every history length, but every interval still includes zero. One year of one lumpy series cannot confirm a gain of a few points.

## 7. What we learned

1.  **Simple is hard to beat when the target data is thin and noisy.** After five rounds, the best model beats a four-week average by only 2–5 points.

2.  **Borrowing a pattern also borrows a level.** Most of the transfer error came from estimating Germany’s size, not its seasonal shape. The models that worked take only the *timing* from the UK and estimate the level from Germany.

3.  **“Similar size” isn’t “similar seasonality”.** Small peer markets looked like better comparisons for Germany, but the large UK predicted Germany’s rise better. The peer markets were more useful for something else: learning how noisy a small market is.

4.  **Structure from economics told us how much to trust new data.** The permanent-versus-transitory model, fitted on the peers, said “smooth a lot”. It also showed where smoothing fails: at real turning points. Telling the model *when* real change is expected, from the calendar, fixed most of that.

5.  **A short test can crown the wrong winner.** T3 won 13 autumn weeks and collapsed over the year; T4’s 8-point lead shrank to about 2.

6.  **A good score can hide a bad forecast.** Twice, WAPE favored a model that under-forecast: T3 by up to 38% in Round 2, and T5 with *M* = 1000 by 13% in Round 5. At a retailer, that means stockouts, so bias belongs next to accuracy.

7.  **These echo a lesson from the author’s PSB 2027 paper,** where gains seen in training did not carry over to held-out data: a method has to be judged where it will be used, not where it looks best.

## 8. Next steps

1.  Expect the January drop instead of only reacting to it: add a seasonal factor for the level, so the forecast falls on schedule after Christmas.

2.  Forecast the top 20 German products, where data is thinnest and borrowing should matter most.

3.  Combine sources, weighting the UK and peer rises by how well each predicted Germany in past seasons.

4.  Cap or flag single very large orders, and report results with and without them.

5.  Use T5’s filtered level as an input to the LightGBM models, combining the economic structure with machine learning.

6.  Test neural models against T5. Amazon’s Deep State Space Models (Rangapuram et al., 2018) are the neural version of this filter: a recurrent network sets the filter’s parameters for each series, which is a natural next step from T5.

7.  Add a pair with a different holiday calendar to test the calendar-shift problem directly.

## 9. FAQ

**Why not just use a neural network?** On 106 weeks of one noisy series, simple rules and small statistical models are strong, cheap and easy to inspect. A neural model has to earn its place by beating them, which is next step 6.

**What does a macroeconomics model have to do with machine learning?** Both learn parameters from data. The local level model has two (the noise variances), learned by maximum likelihood, the same principle behind training a neural network on a loss. Its advantage here is structure: with very little German data, a model that knows “demand is a level plus noise” needs far fewer examples than one that has to discover that.

**Why weekly and not monthly?** Monthly totals are smoother and easier to forecast, but operations such as staffing and replenishment run week by week. The top-down model (T1b) tries to get both: a monthly level and a weekly split.

**Why is the error so high?** Germany averages about 30 orders a month, and one wholesale order can triple a week. No model can foresee which week a large customer orders, so the goal is to be less wrong than the alternatives, not to reach zero.

**Does this say US to Japan would fail?** No. It says that in this data, borrowing the UK’s numbers directly helped less than expected, and borrowing worked best for timing and for learning how noisy a new market is. US to Japan adds calendar differences that this pair doesn’t test; T5’s turning window would have to move to Japan’s own turning points, such as New Year and Golden Week.

**Isn’t the turning window a form of tuning on the test?** The window’s dates come from the calendar, which a forecaster knows in advance, and *M* = 100 was fixed before the run. *M* = 1000 scored better on WAPE, and it was not adopted. A stricter check would choose $`M`$ on 2010 and test it on 2011.

**Open decision.** T5 with *M* = 1000 has about 2 points lower WAPE than *M* = 100 but is 9–13% low; *M* = 100 is less than 5% low. Which should a retailer ship? This document leaves that open: it depends on the relative cost of a stockout and of excess inventory.

## Appendix: reproducibility

Code and outputs are in the project folder: `src/explore.py` (data cleaning and country summary), `src/baselines.py` (Round 1), `src/peer_lift.py` (Round 2), `src/t4_signal.py` and `src/t4_sensitivity.py` (Round 3), `src/rolling_2011.py` (Round 4), `src/t5_turning.py` (Round 5), and `results/` (predictions, tables and `RESULTS_v1.md`). The full derivation is in `docs/DERIVATION_T4.md`. Data: UCI Machine Learning Repository, Online Retail II (dataset 502), downloaded November 7, 2026. The research questions, the top-down and peer-lift ideas, the local level model and its first gain formula, the Round 4 diagnosis and the turning-point fix, and the interpretation are the author’s; the code was built with Claude Code.
