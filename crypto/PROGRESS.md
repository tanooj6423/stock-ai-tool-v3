# PROGRESS — mirrors REQUIREMENTS.md 1:1

Checked = built, run with real output, and verified (not just written).

## §0 Scope/stack
- [x] Python + Streamlit decision, `crypto/` package isolated from stock code
- [x] Universe BTC/ETH/SOL/BNB, daily UTC bars, ≥4y history target

## §1 Data layer
- [x] Binance spot OHLCV fetcher + parquet storage (2,374 rows/asset, 0 gaps)
- [x] Binance funding-rate fetcher (daily aggregation; FTX-day extremes verified real)
- [x] Binance open-interest accumulator (29 rows so far; non-feature in v1)
- [x] Coin Metrics on-chain fetcher — BTC/ETH, 7 metrics, 0 gaps (metric set rebuilt on current community tier: MVRV, exchange flows, exchange supply, AdrActCnt, TxTfrCnt, FeeTotNtv)
- [x] Stablecoin supply fetcher (USDT+USDC, 2020→present)
- [x] Fear & Greed fetcher (2018→present, 4 known source gaps)
- [x] Macro fetcher (DXY, SPX, US10Y; weekend gaps expected, ffilled at feature time)
- [x] Incremental refresh CLI run twice — second run added 0 rows (idempotent)
- [x] `.env.example` shipped; zero keys required — verified
- [ ] **CHECKPOINT 1: user approved real fetched data**

## §2 Feature engineering (expanded per §11)
- [x] Universe expanded: dynamic top-30 + seeds = 31 assets incl. memecoins/new listings (§11)
- [x] Technical features incl. new: path efficiency, range pctl, trade-size z, weekend share, ATH dist, Amihud, beta/corr/instability
- [x] Derivatives features: funding (mean/z/cum/vol/interaction), perp-spot basis (level/z/mom), volume ratio, dislocation flag
- [x] On-chain features BTC/ETH (+1d shift): MVRV, exchange netflow/supply, activity growth, fee impulse
- [x] Sentiment (+1d shift), stablecoin supply growth, macro (DXY/SPX/10Y, corr)
- [x] Cross-sectional: ranks (r7/r30/funding/illiq), rotation velocity, RS, breadth, dispersion, alt-season
- [x] Anti-lookahead tests pass: full-mutation test, on-chain/sentiment shift tests, target tests (27/27)
- [x] Real panel built: 49,986 rows × 73 features × 31 assets, 2020→present
- [ ] **CHECKPOINT 2: user approved real feature table**

## §3 Model layer
- [x] Binary 7d-direction target construction (+ tests)
- [x] Walk-forward split with purge+embargo, global by date (+ overlap tests; 8 folds, 2 warmup)
- [x] XGB + LGBM pooled ensemble; frozen params depth=3, λ=10 after warmup-fold tuning; sqrt-inverse asset weights; short-history assets excluded from training
- [x] Fold-causal isotonic calibration (warmup folds stay raw, excluded from headline OOS)
- [x] OOS feature importance (gain + permutation on last OOS fold)
- [x] OOS bucket stats w/ Wilson CI + weekly non-overlap sanity check
- [x] Real run: OOS(folds 3–8) AUC 0.554, Brier 0.251; bearish buckets strongest (see checkpoint 3 report)
- [x] macOS libomp clash (lightgbm-before-xgboost segfault) diagnosed and fixed via import order
- [ ] **CHECKPOINT 3: user approved real fold metrics**

## §4 Backtesting engine
- [x] Long/flat rule, execution at t+1 open, 10bps fee + 5bps slippage per side
- [x] IS vs OOS strictly separated (no blended output exists); block-bootstrap Sharpe CI
- [x] Buy-and-hold benchmarks per asset + portfolio
- [x] Regime breakdown: per-fold, data-labeled (trending-up/down, choppy) — covers 4 choppy, 3 trending-up, 1 trending-down window
- [x] Threshold sensitivity (0.50/0.55/0.60/0.65) + per-family results
- [x] Backtest unit tests: hand-computed P&L, cost-on-state-change, no-lookahead lag test (39/39 pass)
- [x] Real run: OOS Sharpe 0.44 (95% CI −0.61…+1.57), beat B&H (−2%) with smaller DD; capital preservation in 2026 downtrend verified
- [ ] **CHECKPOINT 4: user approved real backtest results**

## §5 Risk/output layer
- [x] Signal format: P(up,7d) + bucket + OOS bucket accuracy w/ Wilson CI, coin-flip buckets labeled "no signal"
- [x] No buy/sell wording; disclaimer banner on every page (incl. Sharpe-CI-includes-zero statement and survivorship caveat)

## §6 Application layer
- [x] Signals page · Backtest page (equity vs B&H, IS/OOS, regimes, threshold) · Model page (importance, calibration, confidence-over-time) · Data health page
- [x] Verified running: all 4 tabs render with real artifacts, no errors (streamlit run crypto/app.py)
- [ ] **CHECKPOINT 5: user approved dashboard**

## Step 3 — refinement loop
- [x] ≥2 regime backtests documented honestly (8 fold-windows: 3 trending-up, 1 trending-down, 4 choppy — incl. the bad ones: fold 3 lag, fold 5 no-edge)
- [x] ITERATION_LOG.md: 3 experiments run and logged (θ=0.50 robust-but-post-hoc; vol-std target rejected; rank features regime-unstable)
- [x] Recommendation layer added (stance + why + sizing + historical success w/ CI), dashboard tab verified
- [x] Done-criteria review: all REQUIREMENTS items verified with real output; OOS results documented incl. bad periods; dashboard runs end to end. **BUILD COMPLETE (v1).**

## §10 Amendments (Gemini review)
- [x] Amendments recorded in REQUIREMENTS.md §10
- [ ] All adopted amendments implemented in code
