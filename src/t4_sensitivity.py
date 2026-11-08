"""Sensitivity checks for T4: is the gain from the method or from a lucky borrowed prior?
Run from the project root:  .venv/bin/python -I src/t4_sensitivity.py
Varies (a) the prior level (x0.5, x1, x2) and (b) whether q comes from the peers' MLE (wider grid) or is fixed larger.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t4_signal as t4
from baselines import HORIZONS, KS, TEST, de_window, wape
from peer_lift import weekly_by_country, smooth_last_year, PEERS

ROOT = Path(__file__).resolve().parents[1]
t4.QS = np.logspace(-7, 1, 80)  # wider grid: the first run hit the lower edge (1e-4)
de, uk, _ = weekly_by_country()
df = pd.read_parquet(ROOT / "data/processed/transactions.parquet")
wk = (df.set_index("date").groupby("Country")["Quantity"].resample("W-MON", label="left", closed="left").sum()
        .unstack(0).fillna(0))
peers = wk[PEERS]
fits = {}
out = []
for qmode in ["MLE", "q=0.01", "q=0.1"]:
    for mult in [0.5, 1.0, 2.0]:
        for k in KS:
            a, p_ = [], []
            for target in TEST:
                for h in HORIZONS:
                    origin = target - pd.Timedelta(weeks=h - 1)
                    hist = de_window(de, origin, k)
                    if origin not in fits:
                        past = peers[peers.index < origin]
                        fits[origin] = t4.fit_q([(past[c] / past[c].mean()).values for c in PEERS])
                    q, cv_e = fits[origin]
                    if qmode != "MLE":
                        q = float(qmode.split("=")[1])
                    l0 = peers[peers.index < hist.index[0]].iloc[-8:].mean(axis=1).mean() * mult
                    s2 = (cv_e * l0) ** 2
                    level, *_ = t4.filter_level(hist.values, s2, q * s2, l0, l0 ** 2)
                    base = np.nanmean([smooth_last_year(uk, w) for w in hist.index[-4:]])
                    a.append(de[target]); p_.append(level * smooth_last_year(uk, target) / base)
            a, p_ = np.array(a), np.array(p_)
            out.append(dict(q=qmode, prior=f"x{mult}", k=k, wape=round(wape(a, p_) * 100, 1),
                            bias=round((p_.sum() - a.sum()) / a.sum() * 100, 1)))
r = pd.DataFrame(out)
print("median MLE q with wider grid:", f"{np.median([v[0] for v in fits.values()]):.2e}")
print(r.pivot_table(index=["q", "prior"], columns="k", values="wape").to_string())
print()
print(r.pivot_table(index=["q", "prior"], columns="k", values="bias").to_string())
r.to_csv(ROOT / "results/t4_sensitivity.csv", index=False)
