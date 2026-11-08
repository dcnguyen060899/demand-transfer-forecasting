# T4: signal extraction for how much to trust the target market (derivation, 2026-11-08)

Idea (Duy): Germany's weekly sales mix a slowly moving level with one-off wholesale orders, like permanent vs. transitory income in the ECON 305 consumption model. Weight new German data by how much of it is signal.

## Model (Duy)
    d_t = L_t + e_t          e_t ~ (0, σ_e²)   transitory: one-off orders (the "windfall")
    L_t = L_{t-1} + u_t      u_t ~ (0, σ_u²)   permanent: lasting changes in the level
e and u are uncorrelated with each other and over time. This is the local level model ("random walk plus noise").

## Update rule
    L̂_t = L̂_{t-1} + k_t (d_t − L̂_{t-1})
- If σ_e² ≫ σ_u² (mostly noise), k → 0: a big week is probably one order, so keep the old estimate. (Duy)
- If σ_e² ≪ σ_u² (mostly signal), k → 1: follow each new week.

First approximation (Duy): k = σ_u² / (σ_u² + σ_e²).

Exact version (Kalman filter): the weight also counts how unsure we already are about the level.
    P_t|t−1 = P_{t−1} + σ_u²                (uncertainty before seeing week t)
    k_t     = P_t|t−1 / (P_t|t−1 + σ_e²)
    P_t     = (1 − k_t) P_t|t−1             (uncertainty after seeing week t)
Duy's formula is the special case P_{t−1} = 0, as if the old estimate were known exactly. In steady state, with q = σ_u²/σ_e²:
    k* = (−q + sqrt(q² + 4q)) / 2
which is larger than Duy's approximation, because real uncertainty about the level makes new data more useful.

## Where borrowing enters
Start the filter from a borrowed guess instead of Germany's first noisy weeks:
- L̂_0 = Germany/UK (or peer) ratio × the source market's level, with prior uncertainty P_0.
- Large P_0: little trust in the borrowed guess, so German data takes over quickly.
- Small P_0: the borrowed guess dominates until German data accumulates.
So the borrowing weight is not fixed: it falls automatically as German weeks arrive, which is the "help shrinks with more data" pattern, derived rather than assumed.

## Forecast
T4 forecast for a future week = filtered German level L̂ × the source's seasonal shape (the timing from T1b), estimated with data before the origin only. σ_e² and σ_u² are estimated from history before each origin (for example by maximum likelihood on the local level model, or from the source market when German history is too short).

## Test
Same as rounds 1–2: 13 test weeks, horizons 1–4, German history cut to 1, 3, 6 or 12 months; report WAPE, bias and bootstrap ranges against B1 and T1b.

## T5: nudge harder at known turning points (Duy's fix, Round 5)
Diagnosis (Duy, after Round 4): the smoothed level is slow to notice the post-Christmas drop, just as it is slow to notice the autumn rise.
Fix: let the movement variance depend on the calendar,
    σ_u,t² = M σ_u²   for weeks from Dec 15 to Jan 31 (the turning window)
    σ_u,t² = σ_u²     otherwise
A larger σ_u,t² raises P_t|t−1 and so the gain k_t, only inside the window: the filter treats change there as lasting and follows the new weeks, then returns to heavy smoothing. M = 100 is the main setting (fixed before the run); 10 and 1000 are checks. Results: `results/RESULTS_v1.md`, Round 5.
