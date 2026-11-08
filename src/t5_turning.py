"""T5: T4 that nudges harder after Christmas (Duy's fix, 2026-11-08).

Run from the project root:  .venv/bin/python -I src/t5_turning.py

T4's heavily smoothed level is slow to follow the real post-Christmas drop. T5 raises the level's movement variance
(σ_u²) by a factor M for weeks in a known turning-point window, Dec 15 - Jan 31, so the filter trusts new data more
there and stays slow everywhere else. In Kalman terms, the change at a turning point is treated as permanent.

M = 100 is the main setting, chosen from reasoning (move q from ~1e-7 toward ~1e-5, still small), not tuned on the
test; M = 10 and M = 1000 are reported as sensitivity. Same full-year test as src/rolling_2011.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HORIZONS, KS, de_window, wape, b1_moving_average  # noqa: E402
from peer_lift import weekly_by_country, smooth_last_year, PEERS  # noqa: E402
import t4_signal as t4  # noqa: E402
from rolling_2011 import TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MS = [1, 10, 100, 1000]  # M = 1 is T4 itself


def turning(week):
    return (week.month == 12 and week.day >= 15) or week.month == 1


def filter_level_tv(weeks, y, sigma_e2, sigma_u2, l0, p0, m):
    level, p = l0, p0
    for w, obs in zip(weeks, y):
        p_pred = p + sigma_u2 * (m if turning(w) else 1)
        k = p_pred / (p_pred + sigma_e2)
        level += k * (obs - level)
        p = (1 - k) * p_pred
    return level


def main():
    de, uk, _ = weekly_by_country()
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
                shape = smooth_last_year(uk, target) / np.nanmean([smooth_last_year(uk, w) for w in hist.index[-4:]])
                rows.append(dict(k=k, model="B1 4-week average", target=target, h=h, actual=de[target],
                                 pred=b1_moving_average(hist)))
                for m in MS:
                    level = filter_level_tv(hist.index, hist.values, s2, q * s2, l0, l0 ** 2, m)
                    name = "T4 (M=1)" if m == 1 else f"T5 M={m}"
                    rows.append(dict(k=k, model=name, target=target, h=h, actual=de[target], pred=level * shape))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results/t5_predictions.csv", index=False)
    order = list(dict.fromkeys(r.model))
    wap = lambda g: wape(g.actual, g.pred)
    bias = lambda g: (g.pred.sum() - g.actual.sum()) / g.actual.sum()

    def table(x, fn):
        t = x.groupby(["model", "k"]).apply(fn, include_groups=False).unstack().reindex(order)
        t.columns = [f"{c} mo" for c in t.columns]
        return t
    print("WAPE %, all of 2011\n", (table(r, wap) * 100).round(1).to_string())
    print("\nBias %, all of 2011\n", (table(r, bias) * 100).round(1).to_string())
    r["quarter"] = "Q" + r.target.dt.quarter.astype(str)
    q3 = r[r.k == 3].groupby(["model", "quarter"]).apply(wap, include_groups=False).unstack().reindex(order)
    print("\nWAPE % by quarter, 3 months of history\n", (q3 * 100).round(1).to_string())

    rng = np.random.default_rng(0)
    print("\nPaired bootstrap: WAPE(T5 M=100) - WAPE(B1), resampling weeks (negative = T5 better)")
    for k in KS:
        a = r[(r.k == k) & (r.model == "T5 M=100")].set_index(["target", "h"]).sort_index()
        b = r[(r.k == k) & (r.model == "B1 4-week average")].set_index(["target", "h"]).sort_index()
        weeks = a.index.get_level_values(0).unique()
        d = []
        for _ in range(2000):
            pick = rng.choice(weeks, len(weeks))
            aa = pd.concat([a.loc[w] for w in pick]); bb = pd.concat([b.loc[w] for w in pick])
            d.append(wape(aa.actual, aa.pred) - wape(bb.actual, bb.pred))
        d = np.array(d) * 100
        print(f"  {k:2d} mo: mean {d.mean():5.1f}, 90% interval {np.percentile(d, 5):5.1f} to {np.percentile(d, 95):5.1f},"
              f" T5 wins {np.mean(d < 0):.0%}")


if __name__ == "__main__":
    main()
