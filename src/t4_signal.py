"""T4: signal extraction (local level Kalman filter) for Germany's level, plus the UK's seasonal timing.
Derivation: docs/DERIVATION_T4.md (model and first formula by Duy).

Run from the project root:  .venv/bin/python -I src/t4_signal.py

Per forecast origin (only data before it is used):
1. Noise-to-signal ratio q = σ_u²/σ_e² and relative noise cv_e are fitted by maximum likelihood on the peer
   markets' weekly series (each scaled by its own mean), because a new market has too little data of its own.
2. Germany's level is filtered through its available window, starting from a borrowed prior:
   L0 = average peer market's weekly units over the 8 weeks before the window, uncertainty P0 = L0² (CV 100%).
   σ_e = cv_e × L0, σ_u² = q × σ_e².
3. Forecast = filtered level × UK seasonal shape from the window's last 4 weeks to the target week, one year
   earlier (5-week smoothing).
Models reported: B1 (4-week average), T1b (Duy's top-down), T4-level (filtered level, flat, no seasonality),
T4 (filtered level × UK timing).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HORIZONS, KS, TEST, de_window, wape, b1_moving_average, t1b_top_down  # noqa: E402
from peer_lift import weekly_by_country, smooth_last_year, PEERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QS = np.logspace(-4, 1, 60)


def filter_level(y, sigma_e2, sigma_u2, l0, p0):
    """Kalman filter for the local level model. Returns final level, gains, innovations and their variances."""
    level, p = l0, p0
    gains, v, f = [], [], []
    for obs in y:
        p_pred = p + sigma_u2
        f_t = p_pred + sigma_e2
        k = p_pred / f_t
        v_t = obs - level
        level = level + k * v_t
        p = (1 - k) * p_pred
        gains.append(k); v.append(v_t); f.append(f_t)
    return level, np.array(gains), np.array(v), np.array(f)


def fit_q(series_list):
    """Concentrated maximum likelihood for q on several mean-scaled series. Returns q and relative σ_e."""
    best = None
    for q in QS:
        v2f, logf, n = 0.0, 0.0, 0
        for y in series_list:
            if len(y) < 8:
                continue
            _, _, v, f = filter_level(y[1:], 1.0, q, y[0], 1e6)  # near-diffuse start at the first observation
            v, f = v[1:], f[1:]
            v2f += np.sum(v ** 2 / f); logf += np.sum(np.log(f)); n += len(v)
        s2 = v2f / n
        ll = -0.5 * (logf + n * np.log(s2) + n)
        if best is None or ll > best[0]:
            best = (ll, q, np.sqrt(s2))
    return best[1], best[2]


def main():
    de, uk, _ = weekly_by_country()
    df = pd.read_parquet(ROOT / "data/processed/transactions.parquet")
    wk = (df.set_index("date").groupby("Country")["Quantity"]
            .resample("W-MON", label="left", closed="left").sum().unstack(0).fillna(0))
    peers = wk[PEERS]
    fits = {}
    rows, gains_log = [], []
    for k in KS:
        for target in TEST:
            for h in HORIZONS:
                origin = target - pd.Timedelta(weeks=h - 1)
                hist = de_window(de, origin, k)
                if origin not in fits:
                    past = peers[peers.index < origin]
                    fits[origin] = fit_q([(past[c] / past[c].mean()).values for c in PEERS])
                q, cv_e = fits[origin]
                before = peers[peers.index < hist.index[0]].iloc[-8:]
                l0 = before.mean(axis=1).mean()  # average single peer market's weekly units
                sigma_e2 = (cv_e * l0) ** 2
                level, gains, _, _ = filter_level(hist.values, sigma_e2, q * sigma_e2, l0, l0 ** 2)
                last4 = hist.index[-4:]
                base = np.nanmean([smooth_last_year(uk, w) for w in last4])
                shape = smooth_last_year(uk, target) / base
                preds = {"B1 4-week average": b1_moving_average(hist),
                         "T1b top-down (Duy)": t1b_top_down(de_hist=hist, uk=uk, origin=origin, target=target),
                         "T4-level (filtered level, flat)": level,
                         "T4 filtered level x UK timing": level * shape}
                for name, p in preds.items():
                    rows.append(dict(k=k, model=name, target=target, h=h, actual=de[target], pred=p))
                if h == 1:
                    gains_log.append(dict(k=k, origin=origin, q=q, cv_e=cv_e, prior=l0, final_gain=gains[-1],
                                          weeks=len(hist), level=level))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results/t4_predictions.csv", index=False)
    g = pd.DataFrame(gains_log)
    g.to_csv(ROOT / "results/t4_filter_log.csv", index=False)
    order = list(dict.fromkeys(r.model))
    w = r.groupby(["model", "k"]).apply(lambda x: wape(x.actual, x.pred), include_groups=False).unstack().reindex(order)
    b = r.groupby(["model", "k"]).apply(lambda x: (x.pred.sum() - x.actual.sum()) / x.actual.sum(),
                                        include_groups=False).unstack().reindex(order)
    w.columns = b.columns = [f"{k} mo" for k in w.columns]
    print("WAPE %\n", (w * 100).round(1).to_string(), "\n\nBias %\n", (b * 100).round(1).to_string())
    print("\nFitted on peers (median over origins): q =", round(g.q.median(), 4), " cv_e =", round(g.cv_e.median(), 2),
          " steady-state gain k* =", round(float(np.median((-g.q + np.sqrt(g.q ** 2 + 4 * g.q)) / 2)), 3))
    print("Borrowed prior (avg peer weekly units, median):", int(g.prior.median()),
          " vs Germany's actual weekly median 2011:", int(de["2011"].median()))
    print("Final Kalman gain by months of history (median):",
          g.groupby("k").final_gain.median().round(3).to_dict())

    rng = np.random.default_rng(0)
    print("\nBootstrap 90% interval for WAPE at k = 3 months")
    for name in order:
        x = r[(r.k == 3) & (r.model == name)]
        weeks = x.target.unique()
        vals = [wape(*pd.concat([x[x.target == t] for t in rng.choice(weeks, len(weeks))])[["actual", "pred"]].T.values)
                for _ in range(1000)]
        print(f"  {name:34s} {np.percentile(vals, 5) * 100:5.1f} - {np.percentile(vals, 95) * 100:5.1f}")


if __name__ == "__main__":
    main()
