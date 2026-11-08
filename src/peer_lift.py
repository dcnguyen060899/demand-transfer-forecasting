"""T3: borrow the size of the seasonal rise from peer markets instead of the UK (Duy's idea, 2026-11-07).

Run from the project root:  .venv/bin/python -I src/peer_lift.py

Both lift models use the same formula; only the source of the lift changes:
    forecast = Germany's average weekly units over its window
               x (source units around the target week LAST year / source units over the window's weeks LAST year)
The ratio is the "lift": how much bigger the target weeks were than the window weeks, one year earlier.
Last year's values are smoothed over 5 weeks so single big orders don't dominate.
- T1c: lift from the UK (same formula as T3, so the comparison isolates the source).
- T3:  lift from peers = France + EIRE + Netherlands + Spain + Belgium added together (Germany and UK excluded).
Same test weeks, horizons and German history cuts as src/baselines.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script's folder from the path
from baselines import HORIZONS, KS, TEST, de_window, wape, b1_moving_average

ROOT = Path(__file__).resolve().parents[1]
PEERS = ["France", "EIRE", "Netherlands", "Spain", "Belgium"]


def weekly_by_country():
    df = pd.read_parquet(ROOT / "data/processed/transactions.parquet")
    wk = (df.set_index("date").groupby("Country")["Quantity"]
            .resample("W-MON", label="left", closed="left").sum().unstack(0).fillna(0))
    return wk["Germany"], wk["United Kingdom"], wk[PEERS].sum(axis=1)


def smooth_last_year(series, week):
    centre = week - pd.Timedelta(weeks=52)
    window = series[(series.index >= centre - pd.Timedelta(weeks=2)) & (series.index <= centre + pd.Timedelta(weeks=2))]
    return window.mean()


def lift_forecast(de_hist, source, target):
    level = de_hist.mean()
    # Weeks before the data starts (Dec 2009) have no last-year value; average over the weeks that do.
    base = np.nanmean([smooth_last_year(source, w) for w in de_hist.index])
    return level * smooth_last_year(source, target) / base


def main():
    de, uk, peers = weekly_by_country()
    rows = []
    for k in KS:
        for target in TEST:
            for h in HORIZONS:
                origin = target - pd.Timedelta(weeks=h - 1)
                hist = de_window(de, origin, k)
                preds = {"B1 4-week average (DE only)": b1_moving_average(hist),
                         "T1c UK lift": lift_forecast(hist, uk, target),
                         "T3 peer lift (FR, IE, NL, ES, BE)": lift_forecast(hist, peers, target)}
                for name, p in preds.items():
                    rows.append(dict(k=k, model=name, target=target, h=h, actual=de[target], pred=p))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results/peer_lift_predictions.csv", index=False)
    order = list(dict.fromkeys(r.model))
    w = r.groupby(["model", "k"]).apply(lambda g: wape(g.actual, g.pred), include_groups=False).unstack().reindex(order)
    b = r.groupby(["model", "k"]).apply(lambda g: (g.pred.sum() - g.actual.sum()) / g.actual.sum(),
                                        include_groups=False).unstack().reindex(order)
    w.columns = b.columns = [f"{k} mo" for k in w.columns]
    print("WAPE %\n", (w * 100).round(1).to_string(), "\n\nBias % (+ = over-forecast)\n", (b * 100).round(1).to_string())

    # How big was each source's autumn lift last year? (Sep-Nov 2010 vs Jun-Aug 2010, weekly averages)
    def lift(s):
        return s["2010-09-01":"2010-11-30"].mean() / s["2010-06-01":"2010-08-31"].mean()
    print("\nAutumn lift in 2010 (Sep-Nov vs Jun-Aug):",
          {"UK": round(lift(uk), 2), "peers": round(lift(peers), 2), "Germany": round(lift(de), 2)})
    print("Autumn lift in 2011 (what we are forecasting):",
          {"UK": round(uk["2011-09-01":"2011-11-30"].mean() / uk["2011-06-01":"2011-08-31"].mean(), 2),
           "Germany": round(de["2011-09-01":"2011-11-30"].mean() / de["2011-06-01":"2011-08-31"].mean(), 2)})

    rng = np.random.default_rng(0)
    print("\nBootstrap 90% interval for WAPE at k = 3 months")
    for name in order:
        g = r[(r.k == 3) & (r.model == name)]
        weeks = g.target.unique()
        vals = [wape(*pd.concat([g[g.target == x] for x in rng.choice(weeks, len(weeks))])[["actual", "pred"]].T.values)
                for _ in range(1000)]
        print(f"  {name:36s} {np.percentile(vals, 5) * 100:5.1f} - {np.percentile(vals, 95) * 100:5.1f}")


if __name__ == "__main__":
    main()
