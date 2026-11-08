"""Run the v1 baselines on Germany's total weekly units (see docs/BASELINE_DESIGN.md).

Run from the project root:  .venv/bin/python -I src/baselines.py

Setup
- Series: weekly units (weeks start Monday), Germany (target) and UK (source).
- Test weeks: 2011-09-05 .. 2011-11-28 (13 full weeks; the last data week, Dec 5-9, is partial and excluded).
- Each test week is forecast 1, 2, 3 and 4 weeks ahead, from origins before it. A model only sees data from
  weeks before its origin.
- German history is cut to the last k months before each origin (k = 1, 3, 6, 12); UK history is never cut.
- Metric: WAPE = sum|actual - forecast| / sum(actual), pooled over test weeks and horizons.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
WEEKS_PER_MONTH = 52 / 12
TEST = pd.date_range("2011-09-05", "2011-11-28", freq="7D")
HORIZONS = [1, 2, 3, 4]
KS = [1, 3, 6, 12]
LAGS = [1, 2, 3, 4]


def weekly():
    w = pd.read_csv(ROOT / "data/processed/weekly_germany_uk.csv", parse_dates=["week_start"]).set_index("week_start")
    return w["units"].astype(float), w["uk_units"].astype(float)


def wape(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return np.abs(actual - pred).sum() / actual.sum()


def de_window(de, origin, k):
    start = origin - pd.Timedelta(weeks=round(k * WEEKS_PER_MONTH))
    return de[(de.index >= start) & (de.index < origin)]


def uk_forecast(uk, target):
    """UK forecast for a target week from data before the origin: same week last year x recent growth."""
    return uk.get(target - pd.Timedelta(weeks=52), np.nan)


def uk_growth(uk, origin):
    recent = uk[uk.index < origin].iloc[-4:]
    last_year = uk[(uk.index >= origin - pd.Timedelta(weeks=56)) & (uk.index < origin - pd.Timedelta(weeks=52))]
    return recent.sum() / last_year.sum()


# ---------------------------------------------------------------- models
def b0_naive(de_hist, **_):
    return de_hist.iloc[-1]


def b1_moving_average(de_hist, **_):
    return de_hist.iloc[-4:].mean()


def t1_ratio_recent(de_hist, uk, origin, target, **_):
    """Germany = UK forecast x Germany/UK ratio over the last 4 weeks only (short, noisy level)."""
    weeks = de_hist.index[-4:]
    r = de_hist.loc[weeks].sum() / uk.loc[weeks].sum()
    return r * uk_forecast(uk, target) * uk_growth(uk, origin)


def t1b_top_down(de_hist, uk, origin, target, **_):
    """Duy's idea: Germany's level from its whole window (aggregated, smoother) x the UK's weekly shape."""
    r = de_hist.sum() / uk.loc[de_hist.index].sum()
    return r * uk_forecast(uk, target) * uk_growth(uk, origin)


def make_rows(series, scale, origins, horizon_target, is_de):
    """Feature rows: lags of the scaled series before each origin; target h weeks later."""
    rows = []
    for o in origins:
        past = series[series.index < o]
        if len(past) < max(LAGS):
            continue
        for h in HORIZONS:
            tgt = o + pd.Timedelta(weeks=h - 1)
            if horizon_target is not None and tgt not in horizon_target:
                continue
            feat = {f"lag{l}": past.iloc[-l] / scale for l in LAGS}
            feat.update(roll4=past.iloc[-4:].mean() / scale, h=h, woy=tgt.isocalendar().week, is_de=is_de)
            feat["y"] = series.get(tgt, np.nan) / scale
            feat["target"] = tgt
            rows.append(feat)
    return pd.DataFrame(rows)


PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=3, verbose=-1)


def fit_predict(train, test_row):
    feats = [c for c in train.columns if c not in ("y", "target")]
    m = lgb.LGBMRegressor(**PARAMS).fit(train[feats], train["y"])
    return m.predict(test_row[feats])[0]


def lgbm(de_hist, uk, origin, target, pooled, **_):
    scale_de = de_hist.mean()
    h = (target - origin).days // 7 + 1
    # German training rows: origins and targets strictly inside the German window.
    de_origins = de_hist.index[max(LAGS):]
    tr = make_rows(de_hist, scale_de, de_origins, set(de_hist.index), is_de=1)
    if pooled:
        uk_hist = uk[uk.index < origin]
        uk_scale = uk_hist.iloc[-13:].mean()
        tr = pd.concat([tr, make_rows(uk_hist, uk_scale, uk_hist.index[max(LAGS):], set(uk_hist.index), is_de=0)])
    tr = tr.dropna()
    if len(tr) < 5:
        return np.nan  # not enough rows to train
    row = make_rows(de_hist, scale_de, [origin], None, is_de=1)
    row = row[row["h"] == h]
    if row.empty:
        return np.nan
    return fit_predict(tr, row) * scale_de


MODELS = {
    "B0 naive (DE only)": b0_naive,
    "B1 4-week average (DE only)": b1_moving_average,
    "B2 LightGBM (DE only)": lambda **kw: lgbm(pooled=False, **kw),
    "T1 UK x recent ratio": t1_ratio_recent,
    "T1b top-down, UK shape (Duy)": t1b_top_down,
    "T2 LightGBM pooled UK+DE": lambda **kw: lgbm(pooled=True, **kw),
}


def main():
    de, uk = weekly()
    records = []
    for k in KS:
        for target in TEST:
            for h in HORIZONS:
                origin = target - pd.Timedelta(weeks=h - 1)
                hist = de_window(de, origin, k)
                for name, fn in MODELS.items():
                    pred = fn(de_hist=hist, uk=uk, origin=origin, target=target)
                    records.append(dict(k=k, model=name, target=target, h=h, actual=de[target], pred=pred))
    r = pd.DataFrame(records)
    r.to_csv(ROOT / "results/baseline_predictions.csv", index=False)

    def score(g):
        ok = g.dropna(subset=["pred"])
        return pd.Series({"WAPE": wape(ok.actual, ok.pred) if len(ok) else np.nan,
                          "coverage": len(ok) / len(g),
                          "bias": (ok.pred.sum() - ok.actual.sum()) / ok.actual.sum() if len(ok) else np.nan})

    s = r.groupby(["model", "k"]).apply(score, include_groups=False).reset_index()
    table = s.pivot(index="model", columns="k", values="WAPE").reindex(MODELS)
    cov = s.pivot(index="model", columns="k", values="coverage").reindex(MODELS)
    bias = s.pivot(index="model", columns="k", values="bias").reindex(MODELS)
    table.columns = [f"{k} mo" for k in table.columns]
    pd.set_option("display.width", 140)
    print("WAPE on Germany total weekly units (lower is better), test weeks 2011-09-05..2011-11-28, horizons 1-4\n")
    print((table * 100).round(1).to_string())
    print("\nBias (+ = over-forecast), %\n")
    print((bias * 100).round(1).set_axis(table.columns, axis=1).to_string())
    print("\nCoverage (share of forecasts a model could make)\n")
    print(cov.round(2).set_axis(table.columns, axis=1).to_string())
    table.to_csv(ROOT / "results/baseline_wape.csv")

    # Noise check: WAPE spread across the 13 test weeks, by bootstrap over weeks.
    rng = np.random.default_rng(0)
    print("\nBootstrap 90% interval for WAPE at k = 3 months (resampling test weeks)\n")
    for name in MODELS:
        g = r[(r.k == 3) & (r.model == name)].dropna(subset=["pred"])
        if g.empty:
            continue
        weeks = g.target.unique()
        vals = []
        for _ in range(2000):
            pick = rng.choice(weeks, len(weeks))
            gg = pd.concat([g[g.target == w] for w in pick])
            vals.append(wape(gg.actual, gg.pred))
        print(f"  {name:32s} {np.percentile(vals, 5) * 100:5.1f} - {np.percentile(vals, 95) * 100:5.1f}")


if __name__ == "__main__":
    main()
