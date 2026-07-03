# ITERATION_LOG — refinement loop (Step 3)

Baseline (v1, 2026-07-02): OOS folds 3–8 (2024-01→2026-06), threshold 0.55:
portfolio Sharpe 0.44 (block-bootstrap 95% CI −0.61…+1.57), total +26.9%,
max DD −46% — vs buy-and-hold −2.1%, max DD −73%. OOS AUC 0.554.
Skill is asymmetric: bearish buckets (p<0.45) hit 55–61%; 0.45–0.65 ≈ coin-flip.

## Queued experiments
1. **Threshold 0.50** — sensitivity sweep already shows Sharpe 0.62 vs 0.44 at
   0.55; verify it isn't a costs/turnover artifact before adopting.
2. **Volatility-standardized target** (direction beyond ±0.1σ; Gemini round-2
   suggestion) — stops memecoin noise dominating the loss; changes signal
   semantics, so bucket stats must be rebuilt.
3. Regime-conditional evaluation of cross-sectional ranks (Gemini's
   non-stationarity warning): check rank-feature importance stability across
   folds before trusting rotation signals.

## Log

**2026-07-02 · Experiment 1: threshold 0.50 vs 0.55 (OOS, per fold).**
0.50 wins in 5/6 OOS folds (e.g. fold 3: Sharpe 0.64 vs 0.10; fold 8: +0.20
vs −2.11); portfolio Sharpe 0.62 vs 0.44. Costs already included, so not a
turnover artifact. **Verdict: robust, but post-hoc** — 0.55 was
pre-registered, so 0.55 remains the headline; both are displayed in the
threshold table. Revisit as the pre-registered default only in a future
version validated on data unseen today.

**2026-07-02 · Experiment 2: volatility-standardized target
(fwd_ret > 0.1·σ7d, Gemini round-2 suggestion).** Full walk-forward retrain,
frozen params. OOS AUC 0.553 (vs 0.554 baseline — no discrimination gain);
portfolio backtest at θ=0.55: Sharpe 0.04, total −4.7% (vs 0.44, +26.9%).
**Verdict: rejected.** Simple direction target stays.

**2026-07-02 · Experiment 3: cross-sectional momentum-rank stability
(Gemini's non-stationarity warning).** Per-fold Spearman(cs_rank_r7, y)
flips sign across folds (−0.035…+0.041). Confirms the rank features are
regime-unstable as standalone signals (consistent with their absence from
the permutation-importance top 15). **Verdict: keep as model inputs (trees
can condition them on regime gauges), never build stance logic on them.**

**2026-07-02 · Recommendation layer added** (crypto/recommend.py + dashboard
tab): stance (LONG / AVOID-EXIT / NO SIGNAL from bucket skill structure),
per-asset signed feature attributions (LightGBM pred_contrib), inverse-vol
conviction sizing capped 20%/asset against a user-set budget, success stated
as bucket OOS hit rate with Wilson CI. Buy/sell-command framing and success
promises remain prohibited per hard rules.
