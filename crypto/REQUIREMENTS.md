# REQUIREMENTS — Crypto Market Analysis System v1

**Status:** DRAFT — awaiting user approval before any code is written.
**Owner:** Claude (architect/builder). Gemini subagent used only for research/second-opinion tasks.
**Constraints:** free data sources only · fast timeline · ~300k token budget · no live trading.

---

## 0. Scope and stack decision

- **Language/stack:** Python 3.11 end-to-end. Dashboard in **Streamlit** (the host repo is already a Streamlit app with xgboost/lightgbm installed; a React+FastAPI split buys nothing for a solo decision-support tool and would blow the timeline/budget).
- **Location:** everything under `crypto/` in this repo, with a standalone entrypoint `crypto/app.py` (`streamlit run crypto/app.py`). The existing NSE/US stock code is not touched.
- **Asset universe (v1):** BTC, ETH, SOL, BNB. Majors only — deepest liquidity, best free-data coverage. Long-tail alts are a non-goal.
- **Bar frequency:** **daily candles** (UTC close). Rationale: the only free on-chain source (Coin Metrics Community) is daily; mixing intraday price with daily on-chain invites alignment bugs, which are this project's #1 stated risk.
- **History:** ≥ 4 years per asset (covers the 2021 top, 2022 bear, 2023 recovery, 2024–25 trend — multiple regimes for Step 3).

## 1. Data layer

| Source | What | Endpoint | Auth | Refresh |
|---|---|---|---|---|
| Binance public REST | Spot OHLCV (daily) | `/api/v3/klines` | none | daily |
| Binance Futures public REST | Funding rate history, open interest history | `/fapi/v1/fundingRate`, `/futures/data/openInterestHist` | none | daily |
| Coin Metrics Community API | On-chain: active addresses, transfer count/value, fees, NVT, realized cap (as available per asset) | `api.coinmetrics.io/v4` community tier | none | daily |
| Alternative.me | Fear & Greed index | `/fng/` | none | daily |
| yfinance (already a dep) | Macro: DXY, S&P 500, gold, US 10Y yield | library | none | daily |

- **Zero API keys required for the core build.** A `.env.example` is still shipped for optional extras (e.g. FRED); no secrets ever committed.
- **Storage:** Parquet, one file per source per asset under `crypto/data/raw/`, plus one canonical merged table `crypto/data/features.parquet`. No database in v1 — pandas + parquet is sufficient at this scale (~6k rows/asset).
- **Timestamps:** everything UTC; a row's timestamp = bar **close** time. On-chain and sentiment data are shifted **+1 day** before merging (they describe day *t* but are only knowable at *t+1*). This shift is enforced in code and covered by tests.
- **Refresh model:** a CLI command `python -m crypto.pipeline refresh` (idempotent, incremental — fetches only missing dates); the dashboard calls the same code with a 6-hour cache. No background daemon in v1.
- **Failure handling:** each source fetch is independent; a source being down degrades (features become NaN and are imputed/flagged) rather than blocking the pipeline.

## 2. Feature engineering (~45 features per asset-day)

All features use only data ≤ *t* (post-shift). Computed in `crypto/features.py`, one tested function per family.

- **Technical (price/volume):** log returns 1/7/30d; realized vol 7/30d; RSI-14; MACD histogram; Bollinger %B; ATR%; distance from 50/200-day SMA; drawdown from 90d high; volume z-score (30d).
- **Derivatives:** funding rate level, 3d mean, 30d z-score; open interest 1d/7d % change; OI-to-volume ratio.
- **On-chain (BTC/ETH strongest; per-asset availability handled gracefully):** active-address 7d growth and 90d z-score; transfer value 7d growth; fee 7d growth; NVT ratio and its 90d z-score.
- **Sentiment:** Fear & Greed level; 7d change; 30d z-score.
- **Macro/cross-asset:** DXY 7d return; SPX 7d return; gold 7d return; rolling 30d BTC–SPX and BTC–DXY correlation; asset return minus BTC return 7d (relative strength); day-of-week.
- **Hygiene:** winsorize at 1st/99th percentile computed on *training folds only*; NaN policy explicit per feature; no target leakage into any scaler/winsorizer.

## 3. Model layer

- **Target — binary direction classification:** `y = 1 if forward 7-day log return > 0 else 0`.
  - **Why classification over return regression:** (a) the required output is a probabilistic signal with confidence — calibrated class probabilities give that directly; (b) crypto daily returns are heavy-tailed and noise-dominated, so regression optimizes toward outliers and its point estimates are hard to present honestly; (c) classification quality (calibration, bucket accuracy) is directly auditable in the dashboard.
  - **Why 7-day horizon:** matches daily bars + weekly-scale on-chain/sentiment signal; 1-day direction on daily crypto bars is close to coin-flip and would encourage overfitting. Single horizon in v1 — multiple horizons are scope creep.
- **Models:** XGBoost + LightGBM classifiers, final signal = mean of the two predicted probabilities, then **isotonic calibration** fit on out-of-fold validation predictions only.
- **Pooling:** one pooled model across the 4 assets with asset identity as a categorical feature (4× the training data; per-asset models can be an iteration-loop experiment if pooled underperforms).
- **Validation — walk-forward, never random k-fold:** expanding-window with ~8 folds; each fold trains on all data up to a cutoff, then a **7-day purge gap** (so no training label overlaps the test window) plus a 7-day embargo, then tests on the next ~90 days. Per-fold metrics reported; aggregate OOS = concatenated fold predictions.
- **Hyperparameters:** small fixed grid (depth, learning rate, min-child-weight, subsample, L1/L2) selected using **only the first 2 folds**, then frozen — no tuning on later (reported) folds.
- **Explainability:** gain importance + permutation importance computed on OOS data only.

## 4. Backtesting engine

- **Strategy under test (deliberately simple):** per asset — long when calibrated P(up) > 0.55, flat otherwise. Long/flat only in v1 (no shorting; funding/borrow modeling for shorts is its own project). Rebalance check daily; a "trade" occurs only when the state flips.
- **Costs:** 10 bps taker fee + 5 bps slippage per side (Binance spot majors; conservative), applied on every state change.
- **Sizing:** fixed 100% of a notional unit per asset, no leverage, no compound cross-asset portfolio math in v1 (per-asset equity curves + a naive equal-weight combined curve).
- **Metrics (all reported IS and OOS separately, never blended):** annualized Sharpe, max drawdown, win rate (per trade), CAGR, total return, exposure %, number of trades, and **buy-and-hold benchmark** for each asset alongside.
- **Regime analysis (Step 3 requirement):** metrics additionally broken out over at least one trending window and one choppy/ranging window, chosen from the actual data (candidates: 2022 bear, 2023 H1 recovery, 2024 chop periods) and labeled explicitly in the report.
- **Honesty rules:** OOS = walk-forward test folds only; the engine refuses to report a "combined" Sharpe that mixes IS and OOS.

## 5. Risk / output layer

- Every displayed signal is: **asset · P(7-day return > 0) · confidence bucket · historical OOS accuracy of that bucket with a Wilson 95% interval · the backtest stats of the strategy that uses this signal**. Example rendering: *"BTC: P(up, 7d) = 0.61 — signals in the 0.60–0.65 bucket were correct 57% of the time OOS (95% CI 51–63%, n=142)."*
- **No buy/sell/hold wording anywhere.** No profit language. A fixed disclaimer banner on every page: research/decision-support tool, not financial advice, past performance ≠ future results.
- A "model confidence over time" chart (predicted probability + rolling OOS accuracy) so degradation is visible.
- If OOS performance is not meaningfully better than chance for a bucket, the UI must say so rather than hide it.

## 6. Application layer (Streamlit, `crypto/app.py`)

1. **Signals** — current per-asset signal cards (format above) + latest data timestamp + data-staleness warnings.
2. **Backtest** — equity curves vs buy-and-hold, IS/OOS metric tables, regime breakout table.
3. **Model** — feature importances (gain + permutation), calibration curve, confidence-over-time chart.
4. **Data health** — last refresh per source, row counts, gap report.

## 7. Testing (pytest, `crypto/tests/`)

Focused where silent bugs live, per the brief:
- **Pipeline:** schema/dtype checks, UTC enforcement, no duplicate timestamps, gap detection, incremental-refresh idempotency.
- **Anti-lookahead:** (a) on-chain/sentiment +1-day shift verified; (b) mutation test — corrupt all data *after* date *t* and assert features at *t* are bit-identical; (c) target at *t* uses only *t+1…t+7*.
- **Walk-forward:** purge/embargo gap verified — no training sample's label window overlaps any test window.
- **Backtest:** costs applied exactly on state changes; a known toy price series produces hand-computed P&L.

## 8. Explicit non-goals (v1)

- ❌ **No automated trade execution** — no order code, no trade-capable API keys, nothing that touches an exchange account. (Live execution, if ever, is a separate later phase with its own guardrails — flagged, not built.)
- ❌ No paid data sources.
- ❌ No intraday/HFT signals; no horizons other than 7d.
- ❌ No assets beyond BTC/ETH/SOL/BNB; no portfolio optimization; no shorting in the backtest.
- ❌ No background schedulers/daemons; refresh is on-demand.
- ❌ No LLM/news-scraping sentiment in v1 (Fear & Greed only).

## 9. Build order and checkpoints (per Step 2)

1. Data pipeline (+ tests) → show real fetched data.
2. Feature engineering (+ tests) → show real feature table.
3. Model training + walk-forward validation → show real fold metrics.
4. Backtesting engine → show real IS/OOS/regime results.
5. Dashboard → show it running end to end.

Each step ends with real output and a user checkpoint. `crypto/PROGRESS.md` mirrors this document 1:1; `crypto/ITERATION_LOG.md` starts in the refinement loop. Done = every item here verified with real output, OOS results (including bad periods) documented honestly, dashboard running.

---

## 10. Amendments after independent review (Gemini second opinion, 2026-07-02)

Adopted:
- **+ Taker buy/sell volume ratio** feature (already in Binance klines payload).
- **+ Stablecoin supply 7d growth** (USDT + USDC, Coin Metrics Community `SplyCur`) as a market-level liquidity feature.
- **− Day-of-week, Bollinger %B, gold returns** removed (weak/redundant).
- **Backtest executes at t+1 open**, not day-t close (removes zero-latency execution bias).
- **Fold-causal calibration:** the isotonic calibrator used for fold *k* fits only on out-of-fold predictions from folds < *k*.
- **Overlapping-label inference:** daily samples of a 7-day target share 6/7 of their future window; headline backtest Sharpe is therefore reported with a **block-bootstrap 95% CI** (block ≥ 7 days), and classification metrics are sanity-checked on non-overlapping weekly subsamples.

Discovered during the data-pipeline build (2026-07-02):
- Coin Metrics Community no longer serves NVT / realized cap / adjusted transfer value. It does serve **MVRV (CapMVRVCur), exchange in/out flows (FlowInExUSD/FlowOutExUSD), exchange supply (SplyExUSD), transfer count, native fees** — the on-chain feature family is rebuilt on these (a net upgrade: MVRV and exchange flows are stronger documented signals than NVT).
- Community on-chain coverage is **BTC/ETH only** (SOL: none; BNB: ended 2019). SOL/BNB rows carry NaN asset-level on-chain features; BTC's on-chain state additionally feeds all assets as market-level features.

Noted as known limitations (free-data constraints, documented not hidden):
- Coin Metrics Community serves *current-state* history; on-chain metrics may have been revised since original publication (backfill bias risk).
- Binance retains only ~30 days of open-interest history → **OI excluded from v1 features**; the pipeline accumulates OI daily so it becomes usable in a future version.

---

## 11. Universe expansion + feature enlargement (user directive, 2026-07-02)

User directive: go beyond majors — "find and exploit any possible opportunity available, no matter how new or unorthodox." Supersedes §0's 4-asset universe. Non-goals that still stand: no execution, free data only, honest stats.

- **Universe:** dynamic top-30 Binance USDT spot pairs by 24h quote volume + seed majors (31 assets at first snapshot: majors, memecoins, new listings, tokenized stocks). Stablecoin/wrapped/leveraged bases excluded. Sub-cent perps resolved via the `1000`-prefix contract convention.
- **Feature set grown to 73 numeric + asset/family categoricals** (registry: `FEATURE_COLS` in features.py), adding — from Gemini round 2 and my own design — perp-spot basis (level/z/momentum), funding×trend interaction, cumulative 30d funding, funding vol, funding-basis dislocation, perp/spot volume ratio, cross-sectional ranks (momentum/funding/illiquidity) + rotation velocity, alt-season gauge, breadth, dispersion, Amihud illiquidity, avg-trade-size z, trade-count-vs-volume spread, taker-buy divergence, path efficiency, range-compression percentile, weekend volume share, ATH distance, beta/corr to BTC + beta instability, BTC/ETH fee-congestion impulse.
- **Training rules for heterogeneous history** (per review): asset-family categorical (major/alt/meme) instead of per-ticker overfit; min ~400d history to enter training (short-history assets are scored, flagged, never trained on); sqrt-inverse asset sample weights; purge/embargo applied globally by date across the pooled panel.
- **Survivorship-by-liquidity bias (top risk, cannot be fully fixed free):** today's top-30 membership backfills coins that got liquid *because they mooned*; collapsed former-leaders (LUNA-class) are absent. Mitigations: backtest results reported per family (majors nearly bias-free), documented prominently in the dashboard; point-in-time universe reconstruction from Binance Vision archives flagged as future work.
- Logged for the refinement loop: volatility-standardized target variant (direction beyond 0.1σ) as an experiment; regime non-stationarity of cross-sectional ranks probed by the regime-breakdown backtests.
