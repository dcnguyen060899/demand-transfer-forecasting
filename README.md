# Demand-transfer forecasting

**Can a data-rich market's history improve demand forecasts in a data-poor market, and when does that transfer fail?**

New markets and new products have almost no sales history, so forecasters borrow history from markets they know well (the cold-start problem). This study tests that idea on public data, using the UK (about 1,450 orders a month) as the data-rich source and Germany (about 30 orders a month) as the data-poor target, in the UCI Online Retail II dataset. It is a stand-in for a harder case such as US → Japan; because the UK and Germany share Christmas, it tests data scarcity rather than calendar shifts.

The full write-up is [`docs/latex/narrative.pdf`](docs/latex/narrative.pdf) (Markdown version: [`docs/NARRATIVE.md`](docs/NARRATIVE.md)); every round's numbers are in [`results/RESULTS_v1.md`](results/RESULTS_v1.md).

## Result so far (five rounds)

| Round | Question | Answer |
|---|---|---|
| 1 | Does copying the UK's history help? | No. A 4-week moving average beat every borrowing model; borrowing over-forecast by 15-40%. |
| 2 | Size or timing? | The UK predicted Germany's seasonal *timing* well; the error was in Germany's *level*. Small peer markets looked best on 13 weeks but under-forecast by up to 38%. |
| 3 | Separate the lasting level from one-off orders | A local level (Kalman filter) model, adapted from the permanent-vs-transitory income idea in macroeconomics, with its noise ratio learned on peer markets, was the most accurate model on the 13-week test. |
| 4 | Does it hold over a full year? | Partly. Over all 48 weeks of 2011 its lead shrank to 1-4 WAPE points, and it failed right after Christmas. |
| 5 | React faster at known turning points | Raising the level's variance from Dec 15 to Jan 31 cut the post-Christmas error by 9 points; the model now beats the moving average by 2-5 points at every history length, though not yet with statistical confidence. |

WAPE (%) over the 48 weeks of 2011, lower is better, by months of German history available:

| Model | 1 mo | 3 mo | 6 mo | 12 mo |
|---|---|---|---|---|
| 4-week moving average | 51.4 | 51.4 | 51.4 | 51.4 |
| T4 local level × UK timing | 47.6 | 50.0 | 52.7 | 48.8 |
| **T5 turning-point filter** | **47.6** | **48.2** | **49.3** | **46.6** |

Lessons: a short test can crown the wrong model, and WAPE alone can reward a forecast that runs low, so bias is reported next to it throughout.

## Reproduce

1. Download Online Retail II from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (dataset 502, CC BY 4.0) and save it as `data/raw/online_retail_II.xlsx`.
2. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Run from the project root, in order:

| Script | Round |
|---|---|
| `src/explore.py` | cleans the data, writes `data/processed/` and the country summary |
| `src/baselines.py` | Round 1 (about 4 minutes; LightGBM) |
| `src/peer_lift.py` | Round 2 |
| `src/t4_signal.py`, `src/t4_sensitivity.py` | Round 3 |
| `src/rolling_2011.py` | Round 4 |
| `src/t5_turning.py` | Round 5 |

Each script is run as `.venv/bin/python -I src/<script>.py` and writes its predictions to `results/`.

## Docs

- [`docs/BASELINE_DESIGN.md`](docs/BASELINE_DESIGN.md): setup, models and metric, fixed before any results
- [`docs/DERIVATION_T4.md`](docs/DERIVATION_T4.md): the local level model, the Kalman filter, and the turning-point fix

## Credits

Research question, model ideas (top-down forecast, peer lift, the local level model and its first gain formula, the turning-point fix) and interpretation: Duy Nguyen. Code built with Claude Code. Data: Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository, CC BY 4.0.
