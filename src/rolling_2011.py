"""Wider test: roll the forecast origin through all of 2011 (target weeks 2011-01-03 .. 2011-11-28, 48 weeks).

Run from the project root:  .venv/bin/python -I src/rolling_2011.py

Same horizons (1-4 weeks) and German history cuts (1, 3, 6, 12 months) as the earlier rounds. LightGBM models are
left out here because they are slow and were not competitive in Round 1.
Reports WAPE and bias overall and by quarter, plus a paired bootstrap of the WAPE difference T4 - B1 (resampling
weeks, so both models are scored on the same weeks each time).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HORIZONS, KS, de_window, wape, b0_naive, b1_moving_average, t1b_top_down  # noqa: E402
from peer_lift import weekly_by_country, smooth_last_year, lift_forecast, PEERS  # noqa: E402
import t4_signal as t4  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGETS = pd.date_range("2011-01-03", "2011-11-28", freq="7D")


def main():
    de, uk, peer_sum = weekly_by_country()
    df = pd.read_parquet(ROOT / "data/processed/transactions.parquet")
    peers = (df.set_index("date").groupby("Country")["Quantity"].resample("W-MON", label="left", closed="left")
               .sum().unstack(0).fillna(0))[PEERS]
    fits, rows = {}, []
    for k in KS:
        for target in TARGETS:
            for h in HORIZONS:
                origin = target - pd.Timedelta(weeks=h - 1)
                hist = de_window(de, origin, k)
                if len(hist) < 2:
                    continue
                if origin not in fits:
                    past = peers[peers.index < origin]
                    fits[origin] = t4.fit_q([(past[c] / past[c].mean()).values for c in PEERS])
                q, cv_e = fits[origin]
                before = peers[peers.index < hist.index[0]].iloc[-8:]
                l0 = before.mean(axis=1).mean() if len(before) else peers[peers.index < origin].iloc[:8].mean(axis=1).mean()
                s2 = (cv_e * l0) ** 2
                level, *_ = t4.filter_level(hist.values, s2, q * s2, l0, l0 ** 2)
                base = np.nanmean([smooth_last_year(uk, w) for w in hist.index[-4:]])
                preds = {
                    "B0 naive": b0_naive(hist),
                    "B1 4-week average": b1_moving_average(hist),
                    "T1b top-down (Duy)": t1b_top_down(de_hist=hist, uk=uk, origin=origin, target=target),
                    "T1c UK lift": lift_forecast(hist, uk, target),
                    "T3 peer lift": lift_forecast(hist, peer_sum, target),
                    "T4-level (flat)": level,
                    "T4 level x UK timing": level * smooth_last_year(uk, target) / base,
                }
                for name, p in preds.items():
                    rows.append(dict(k=k, model=name, target=target, h=h, actual=de[target], pred=p))
    r = pd.DataFrame(rows).dropna(subset=["pred"])
    r.to_csv(ROOT / "results/rolling_2011_predictions.csv", index=False)
    order = list(dict.fromkeys(r.model))

    def table(x, fn):
        t = x.groupby(["model", "k"]).apply(fn, include_groups=False).unstack().reindex(order)
        t.columns = [f"{c} mo" for c in t.columns]
        return t
    wap = lambda g: wape(g.actual, g.pred)
    bias = lambda g: (g.pred.sum() - g.actual.sum()) / g.actual.sum()
    print(f"Target weeks: {r.target.min().date()} .. {r.target.max().date()} ({r.target.nunique()} weeks), horizons 1-4\n")
    print("WAPE %, all of 2011\n", (table(r, wap) * 100).round(1).to_string())
    print("\nBias %, all of 2011\n", (table(r, bias) * 100).round(1).to_string())
    r["quarter"] = "Q" + r.target.dt.quarter.astype(str)
    q = r[r.k == 3].groupby(["model", "quarter"]).apply(wap, include_groups=False).unstack().reindex(order)
    print("\nWAPE % by quarter, 3 months of history\n", (q * 100).round(1).to_string())

    rng = np.random.default_rng(0)
    print("\nPaired bootstrap: WAPE(T4) - WAPE(B1), resampling weeks (negative = T4 better)")
    for k in KS:
        a = r[(r.k == k) & (r.model == "T4 level x UK timing")].set_index(["target", "h"]).sort_index()
        b = r[(r.k == k) & (r.model == "B1 4-week average")].set_index(["target", "h"]).sort_index()
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        weeks = a.index.get_level_values(0).unique()
        diffs = []
        for _ in range(2000):
            pick = rng.choice(weeks, len(weeks))
            aa = pd.concat([a.loc[w] for w in pick]); bb = pd.concat([b.loc[w] for w in pick])
            diffs.append(wape(aa.actual, aa.pred) - wape(bb.actual, bb.pred))
        diffs = np.array(diffs) * 100
        print(f"  {k:2d} mo: mean {diffs.mean():5.1f} points, 90% interval {np.percentile(diffs, 5):5.1f} to "
              f"{np.percentile(diffs, 95):5.1f}, share of resamples where T4 wins {np.mean(diffs < 0):.0%}")


if __name__ == "__main__":
    main()
