
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import ta
 
from data import (
    get_fundamentals, get_relative_strength, get_support_resistance,
)
from model import get_risk_metrics, predict_holding_days
from model_us import (
    train_model_us, get_signal_us, prepare_us_frame,
    get_spy_df, get_vix_df, get_vix3m_df,
    TB_TP_ATR, TB_SL_ATR, TB_HORIZON,
)
from sentiment import get_news_sentiment
from screener_engine import (
    score_ml_signal, score_volume, score_market_regime,
    score_relative_strength, score_fundamentals,
    score_sentiment, score_fibonacci, score_institutional,
    get_risk_level,
)
from universe_us import (
    get_us_universe, get_us_sector,
    US_SECTOR_INDICES, get_sp500_tickers_live,
)
 
# ── Constants ────────────────────────────────────────────────
MIN_DAILY_TURNOVER_USD = 5_000_000
MIN_PRICE_USD = 2.0
MIN_AVG_VOLUME_US = 500_000
MAX_PICKS_PER_SECTOR = 2
VIX_FEAR_THRESHOLD = 25.0
VIX_TERM_FEAR = 1.0          # VIX/VIX3M > 1 → backwardation → fear
MIN_EV = 0.15                # min expected value per $1 risked
MIN_RR1 = 1.5                # min reward:risk on target 1
MIN_ADX = 15.0               # min trend strength
MIN_MODEL_QUALITY = 0.55     # walk-forward edge gate (0.5 = no edge)
MIN_CONFIDENCE = 0.62
EARNINGS_SKIP_DAYS = 7       # skip if earnings within N days
KELLY_FRACTION = 0.25        # quarter-Kelly
MAX_POSITION_PCT = 0.20      # max 20% of capital in one name
 
 
# ── Market context ───────────────────────────────────────────
 
@st.cache_data(ttl=3600)
def get_spy_data():
    return get_spy_df()
 
 
@st.cache_data(ttl=3600)
def get_us_market_regime(spy_df=None) -> str:
    try:
        if spy_df is None:
            spy_df = get_spy_data()
        if spy_df is None or spy_df.empty:
            return "unknown"
        from data import add_indicators
        spy = add_indicators(spy_df)
        if spy is None or spy.empty:
            return "unknown"
        price = float(spy["Close"].iloc[-1])
        sma50 = float(spy["SMA_50"].iloc[-1])
        sma200 = float(spy["SMA_200"].iloc[-1])
        rsi = float(spy["RSI"].iloc[-1])
        macd = float(spy["MACD"].iloc[-1])
        bull_signals = sum([
            price > sma50, price > sma200, sma50 > sma200,
            rsi > 50, macd > 0,
        ])
        if bull_signals >= 4:
            return "bull"
        elif bull_signals <= 1:
            return "bear"
        return "sideways"
    except Exception:
        return "unknown"
 
 
@st.cache_data(ttl=3600)
def get_vix_state() -> dict:
    """VIX level + term structure. Backwardation (VIX > VIX3M) is a
    stronger near-term fear signal than the absolute level."""
    out = {"vix": None, "term": None, "high_fear": False, "label": "Unknown"}
    try:
        vix_df = get_vix_df()
        if vix_df is None or vix_df.empty:
            return out
        vix = float(vix_df["Close"].iloc[-1])
        term = None
        vix3m_df = get_vix3m_df()
        if vix3m_df is not None and not vix3m_df.empty:
            term = round(vix / float(vix3m_df["Close"].iloc[-1]), 3)
        high_fear = vix > VIX_FEAR_THRESHOLD or (
            term is not None and term > VIX_TERM_FEAR
        )
        if vix < 15:
            label = "Calm"
        elif vix < 20:
            label = "Low"
        elif vix < 25:
            label = "Moderate"
        elif vix < 35:
            label = "Elevated"
        else:
            label = "Extreme Fear"
        if term is not None and term > VIX_TERM_FEAR and vix < 25:
            label += " (term inverted!)"
        return {"vix": round(vix, 1), "term": term,
                "high_fear": high_fear, "label": label}
    except Exception:
        return out
 
 
@st.cache_data(ttl=3600)
def get_breadth_state() -> dict:
    """Equal-weight (RSP) vs cap-weight (SPY) 20d momentum.
    Negative spread = narrow, top-heavy rally → less reliable signals."""
    try:
        rsp = yf.Ticker("RSP").history(period="3mo")
        spy = yf.Ticker("SPY").history(period="3mo")
        if rsp is None or spy is None or len(rsp) < 21 or len(spy) < 21:
            return {"spread": None, "narrow": False}
        r = float(rsp["Close"].iloc[-1] / rsp["Close"].iloc[-21] - 1)
        s = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-21] - 1)
        spread = round((r - s) * 100, 2)
        return {"spread": spread, "narrow": spread < -2.0}
    except Exception:
        return {"spread": None, "narrow": False}
 
 
@st.cache_data(ttl=21600)
def get_us_earnings_risk(ticker: str) -> dict:
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            return {"risk_level": "none", "days": None, "message": ""}
        date_col = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                date_col = pd.Series(ed if isinstance(ed, list) else [ed])
        elif hasattr(cal, "empty") and not cal.empty:
            if "Earnings Date" in getattr(cal, "columns", []):
                date_col = cal["Earnings Date"].dropna()
            elif "Earnings Date" in str(cal.index):
                date_col = pd.Series(cal.loc["Earnings Date"])
        if date_col is None or len(date_col) == 0:
            return {"risk_level": "none", "days": None, "message": ""}
        next_earnings = pd.to_datetime(date_col.iloc[0])
        today = pd.Timestamp.now(tz=getattr(next_earnings, "tzinfo", None))
        days = (next_earnings - today).days
        if days < 0 or days > 30:
            return {"risk_level": "none", "days": days, "message": ""}
        if days <= EARNINGS_SKIP_DAYS:
            return {"risk_level": "high", "days": days,
                    "message": f"⚠️ Earnings in {days}d — inside holding window"}
        elif days <= 14:
            return {"risk_level": "medium", "days": days,
                    "message": f"Earnings in {days}d — exit before report"}
        return {"risk_level": "low", "days": days,
                "message": f"Earnings in {days}d"}
    except Exception:
        return {"risk_level": "none", "days": None, "message": ""}
 
 
@st.cache_data(ttl=3600)
def get_us_sector_momentum(sector_ticker: str) -> float | None:
    try:
        df = yf.Ticker(sector_ticker).history(period="3mo")
        if df is None or df.empty or len(df) < 20:
            return None
        return round(
            (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-20]) - 1) * 100, 2
        )
    except Exception:
        return None
 
 
def score_us_sector_momentum(sector: str) -> int:
    etf = US_SECTOR_INDICES.get(sector)
    if not etf:
        return 4
    momentum = get_us_sector_momentum(etf)
    if momentum is None:
        return 4
    if momentum >= 5:
        return 8
    elif momentum >= 2:
        return 6
    elif momentum >= 0:
        return 4
    return 1
 
 
# ── Trade construction ───────────────────────────────────────
 
def calculate_us_entry_targets(df, regime) -> dict:
    """Targets aligned with the geometry the model was trained on
    (TP 2.2x ATR / SL 1.1x ATR), refined by support/resistance."""
    price = float(df["Close"].iloc[-1])
    atr = float(df["ATR"].iloc[-1])
    sl_mult = TB_SL_ATR * (1.25 if regime == "bear" else 1.0)
    entry = price
    atr_stop = entry - sl_mult * atr
    support_levels, resistance_levels = get_support_resistance(df)
    # Use nearest support below entry if it tightens the stop sensibly
    support_stop = None
    if support_levels:
        s0 = support_levels[0]
        if entry - s0 >= 0.5 * atr and s0 > atr_stop:
            support_stop = s0 * 0.995  # just below support
    stop_loss = support_stop if support_stop else atr_stop
    risk = max(entry - stop_loss, 0.5 * atr)
 
    t1_atr = entry + TB_TP_ATR * atr
    valid_t1 = [r for r in resistance_levels if r > entry + MIN_RR1 * risk]
    target1 = min(valid_t1[0], t1_atr) if valid_t1 else t1_atr
    target2 = entry + 3.5 * atr
    valid_t2 = [r for r in resistance_levels if r > target1 * 1.01]
    if valid_t2:
        target2 = max(target2, valid_t2[0])
 
    rr1 = round((target1 - entry) / risk, 2)
    rr2 = round((target2 - entry) / risk, 2)
    return {
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "rr1": rr1,
        "rr2": rr2,
        "risk_amount": round(risk, 2),
        "atr": round(atr, 2),
    }
 
 
def kelly_position_size(capital, risk_pct, entry, stop_loss,
                        buy_prob, rr1, high_fear=False):
    """Quarter-Kelly sizing capped by the user's fixed risk % rule.
    Returns (shares, cost, effective_risk_pct, kelly_f)."""
    try:
        per_share_risk = abs(entry - stop_loss)
        if per_share_risk <= 0:
            return 0, 0.0, 0.0, 0.0
        # Kelly fraction for a p / RR bet
        f_full = buy_prob - (1 - buy_prob) / max(rr1, 0.1)
        f = max(0.0, f_full) * KELLY_FRACTION
        eff_risk_pct = min(risk_pct / 100.0, f)
        if high_fear:
            eff_risk_pct *= 0.5
        risk_amount = capital * eff_risk_pct
        shares = int(risk_amount / per_share_risk)
        # Cap total exposure
        max_shares = int(capital * MAX_POSITION_PCT / entry)
        shares = min(shares, max_shares)
        return shares, round(shares * entry, 2), round(eff_risk_pct * 100, 2), round(f_full, 3)
    except Exception:
        return 0, 0.0, 0.0, 0.0
 
 
def build_trade_plan(targets, holding_days, ticker) -> list:
    """Concrete execution playbook for the pick."""
    e, sl = targets["entry"], targets["stop_loss"]
    t1, t2, atr = targets["target1"], targets["target2"], targets["atr"]
    return [
        f"ENTRY: Buy near ${e:,.2f} (limit order; avoid first 30 min of session).",
        f"STOP: Hard stop at ${sl:,.2f} ({(e - sl) / e:.1%} below entry). No exceptions.",
        f"T1 (${t1:,.2f}): Sell 50%. Immediately move stop on the rest to breakeven (${e:,.2f}).",
        f"T2 (${t2:,.2f}): Sell remainder, OR trail stop at 2.0x ATR (${2 * atr:,.2f}) below each new high.",
        f"TIME STOP: If neither T1 nor stop is hit in {holding_days} trading days, exit — thesis stale.",
        f"NEVER: add to a losing position in {ticker}, or remove the stop after entry.",
    ]
 
 
def check_us_liquidity(df) -> bool:
    try:
        avg_volume = df["Volume"].tail(20).mean()
        avg_price = df["Close"].tail(20).mean()
        return avg_volume * avg_price >= MIN_DAILY_TURNOVER_USD
    except Exception:
        return True
 
 
@st.cache_data(ttl=21600)
def pre_filter_us_universe(tickers: tuple,
                           min_price: float = MIN_PRICE_USD,
                           min_avg_volume: int = MIN_AVG_VOLUME_US) -> list:
    qualified = []
    batch_size = 50
    tickers = list(tickers)
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(
                batch, period="10d", progress=False,
                group_by="ticker", auto_adjust=True, threads=True,
            )
            if raw is None or raw.empty:
                continue
            for ticker in batch:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if ticker not in raw.columns.get_level_values(0):
                            continue
                        df_t = raw[ticker].dropna()
                    else:
                        df_t = raw.dropna()
                    if df_t.empty or len(df_t) < 3:
                        continue
                    price = float(df_t["Close"].iloc[-1])
                    vol = float(df_t["Volume"].mean())
                    if price >= min_price and vol >= min_avg_volume:
                        qualified.append(ticker)
                except Exception:
                    continue
        except Exception:
            for ticker in batch:
                try:
                    df_t = yf.Ticker(ticker).history(period="10d")
                    if df_t is None or df_t.empty:
                        continue
                    if (float(df_t["Close"].iloc[-1]) >= min_price and
                            float(df_t["Volume"].mean()) >= min_avg_volume):
                        qualified.append(ticker)
                except Exception:
                    continue
    return qualified
 
 
def get_us_signal_breakdown(df, signal, regime, rs,
                            sentiment, sentiment_trend) -> list:
    try:
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns else None
        rows = [
            {"Factor": "RSI", "Reading": f"{rsi:.1f}",
             "Status": ("Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"),
             "Signal": "🟢" if 40 <= rsi <= 65 else "🔴" if rsi > 70 else "🟡"},
            {"Factor": "MACD", "Reading": f"{macd:.2f}",
             "Status": ("Bullish crossover" if macd > 0 and macd_hist > 0 else "Bearish / weak"),
             "Signal": "🟢" if macd > 0 and macd_hist > 0 else "🔴" if macd < 0 else "🟡"},
            {"Factor": "Trend (SMA)",
             "Reading": "Above all SMAs" if price > sma200 else "Below SMA200",
             "Status": ("Uptrend confirmed" if price > sma50 and price > sma200
                        else "Above SMA50 only" if price > sma50 else "Below key averages"),
             "Signal": ("🟢" if price > sma50 and price > sma200
                        else "🔴" if price < sma50 else "🟡")},
            {"Factor": "Volume", "Reading": f"{vol_ratio:.1f}x avg",
             "Status": "Confirming move" if vol_ratio >= 1.2 else "Below average",
             "Signal": "🟢" if vol_ratio >= 1.4 else "🟡" if vol_ratio >= 1.0 else "🔴"},
            {"Factor": "Market regime", "Reading": regime.upper(),
             "Status": ("Favourable" if regime == "bull"
                        else "Headwind" if regime == "bear" else "Neutral"),
             "Signal": "🟢" if regime == "bull" else "🔴" if regime == "bear" else "🟡"},
            {"Factor": "Rel. strength",
             "Reading": f"{rs:+.1f}% vs SPY" if rs is not None else "N/A",
             "Status": ("Outperforming" if rs is not None and rs >= 0 else "Underperforming"),
             "Signal": "🟢" if rs and rs >= 3 else "🔴" if rs and rs < -3 else "🟡"},
            {"Factor": "Sentiment", "Reading": sentiment.capitalize(),
             "Status": (f"Positive & {sentiment_trend}" if sentiment == "positive"
                        else f"Negative / {sentiment_trend}"),
             "Signal": ("🟢" if sentiment == "positive"
                        else "🔴" if sentiment == "negative" else "🟡")},
        ]
        if adx is not None:
            rows.append({
                "Factor": "ADX (trend strength)", "Reading": f"{adx:.0f}",
                "Status": ("Strong trend" if adx >= 25
                           else "Developing" if adx >= 15 else "No trend"),
                "Signal": "🟢" if adx >= 25 else "🟡" if adx >= 15 else "🔴",
            })
        return rows
    except Exception:
        return []
 
 
def _score_multiframe_us(df) -> int:
    score = 0
    try:
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        current = close.iloc[-1]
        conditions = [
            current > sma20.iloc[-1],
            current > sma50.iloc[-1],
            current > sma200.iloc[-1],
            sma50.iloc[-1] > sma200.iloc[-1],
            current > close.tail(5).mean(),
        ]
        met = sum(conditions)
        score = {5: 15, 4: 12, 3: 8, 2: 4}.get(met, 0)
    except Exception:
        pass
    return score
 
 
def _get_us_key_drivers(df, regime, rs, sentiment,
                        confidence, ev, quality) -> list:
    drivers = []
    try:
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])
        if ev >= 0.4:
            drivers.append(
                f"Expected value +${ev:.2f} per $1 risked (calibrated p={confidence:.0%})"
            )
        if quality >= 0.65:
            drivers.append(
                f"Model shows real out-of-sample edge on this name (quality {quality:.2f})"
            )
        if macd > 0 and macd_hist > 0:
            drivers.append("MACD bullish crossover with positive histogram")
        if vol_ratio >= 1.4:
            drivers.append(f"Volume {vol_ratio:.1f}x average — institutional confirmation")
        if price > sma50 and price > sma200:
            drivers.append("Price above SMA50 and SMA200 — structural uptrend")
        if rs and rs >= 5:
            drivers.append(f"Outperforming S&P 500 by {rs:.1f}% — relative strength leader")
        if sentiment == "positive":
            drivers.append("News sentiment positive — news flow aligned with signal")
        if 40 <= rsi <= 55:
            drivers.append(f"RSI at {rsi:.0f} — not overbought, room to run")
        if regime == "bull":
            drivers.append("Bull regime — macro tailwind")
    except Exception:
        pass
    return drivers[:4]
 
 
# ── Main scan ────────────────────────────────────────────────
 
@st.cache_data(ttl=21600)
def run_us_scan(tickers_tuple: tuple,
                capital: float = 10000.0,
                risk_pct: float = 1.5,
                progress_callback=None,
                run_prefilter: bool = True) -> tuple:
    """Full US scan. Returns (picks_list, regime_str). Backward compatible
    with v1 output keys; adds ev, quality, kelly fields, trade_plan, adx."""
    tickers = list(tickers_tuple)
 
    spy_df = get_spy_data()
    regime = get_us_market_regime(spy_df)
    vix_state = get_vix_state()
    high_fear = vix_state["high_fear"]
    breadth = get_breadth_state()
 
    if regime == "bear":
        min_threshold = 55
    elif regime == "unknown":
        min_threshold = 50
    else:
        min_threshold = 45
    if breadth["narrow"]:
        min_threshold += 5  # narrow rally → demand more proof
    min_rs_bear = 3.0
 
    if run_prefilter and len(tickers) > 150:
        if progress_callback:
            progress_callback(0, len(tickers), "PRE-FILTERING")
        tickers = pre_filter_us_universe(tuple(tickers))
        if progress_callback:
            progress_callback(0, len(tickers),
                              f"Pre-filter complete — {len(tickers)} liquid stocks")
 
    results = []
    sector_counts = {}
    total = len(tickers)
 
    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        try:
            # Single enriched frame — same features the model trained on
            df = prepare_us_frame(ticker)
            if df is None or len(df) < 100:
                continue
            if not check_us_liquidity(df):
                continue
 
            sector = get_us_sector(ticker)
            if sector_counts.get(sector, 0) >= MAX_PICKS_PER_SECTOR:
                continue
 
            # Trend strength gate — no trendless chop
            adx = float(df["ADX"].iloc[-1]) if "ADX" in df.columns else 20.0
            if adx < MIN_ADX:
                continue
 
            rs = get_relative_strength(df, spy_df) if spy_df is not None else None
            if regime == "bear" and (rs is None or rs < min_rs_bear):
                continue
 
            # In bear regime require price above SMA200 unless RS leader
            current_price = float(df["Close"].iloc[-1])
            sma200 = float(df["SMA_200"].iloc[-1])
            if regime == "bear" and current_price < sma200 and (rs or 0) < 8:
                continue
 
            earnings = get_us_earnings_risk(ticker)
            if earnings["risk_level"] == "high":
                continue
 
            news_data = get_news_sentiment(ticker)
            sentiment = news_data["sentiment"]
            sent_conf = news_data["confidence"]
            sentiment_score = news_data["sentiment_score"]
            sentiment_trend = news_data["trend"]
            if news_data["risk_flags"]:
                continue
 
            # ML model — calibrated ensemble + quality gate
            bundle, features, quality = train_model_us(ticker)
            if bundle is None or quality < MIN_MODEL_QUALITY:
                continue
            signal, confidence, buy_prob, sell_prob = get_signal_us(
                bundle, df, features
            )
            if signal != "BUY" or confidence < MIN_CONFIDENCE:
                continue
 
            rsi = float(df["RSI"].iloc[-1])
            if rsi > 70 or rsi < 28:
                continue
 
            high_52w = float(df["High"].tail(252).max())
            pct_from_high = (high_52w - current_price) / high_52w * 100
            if regime == "bear" and pct_from_high < 3.0:
                continue
 
            # Trade geometry + EV filter
            targets = calculate_us_entry_targets(df, regime)
            if targets["rr1"] < MIN_RR1:
                continue
            ev = round(buy_prob * targets["rr1"] - (1 - buy_prob), 3)
            if ev < MIN_EV:
                continue
 
            fundamentals = get_fundamentals(ticker)
 
            # ── Composite 10-layer score (kept for UI) ──
            s1 = score_ml_signal(signal, confidence, buy_prob)
            s2 = _score_multiframe_us(df)
            s3 = score_volume(df)
            s4 = score_market_regime(regime)
            s5 = score_relative_strength(rs)
            s6 = score_us_sector_momentum(sector)
            s7 = score_fundamentals(fundamentals)
            s8 = score_sentiment(sentiment, sent_conf, sentiment_trend, sentiment_score)
            s9 = score_fibonacci(df)
            s10 = score_institutional(ticker, df)
            total_score = int(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10)
            if total_score < min_threshold:
                continue
 
            # EV-blended ranking score
            rank_score = round(total_score + 40 * max(ev, 0), 2)
 
            # Kelly-capped position sizing
            shares, cost, eff_risk_pct, kelly_f = kelly_position_size(
                capital, risk_pct, targets["entry"], targets["stop_loss"],
                buy_prob, targets["rr1"], high_fear=high_fear,
            )
            if shares == 0:
                continue
 
            atr_pct = float(df["ATR"].iloc[-1]) / current_price
            vol_ratio = float(df["Volume_ratio"].iloc[-1])
            holding_days = min(
                predict_holding_days(df, signal, confidence, regime, rs),
                TB_HORIZON,
            )
            risk_metrics = get_risk_metrics(df)
            risk_level = get_risk_level(
                total_score, regime, rsi, vol_ratio,
                earnings["risk_level"], confidence,
                targets["entry"], targets["stop_loss"], atr_pct,
            )
            signal_breakdown = get_us_signal_breakdown(
                df, signal, regime, rs, sentiment, sentiment_trend
            )
            key_drivers = _get_us_key_drivers(
                df, regime, rs, sentiment, buy_prob, ev, quality
            )
            trade_plan = build_trade_plan(targets, holding_days, ticker)
 
            capital_note = ""
            if high_fear:
                tnote = (f", VIX term {vix_state['term']}"
                         if vix_state.get("term") else "")
                capital_note = (
                    f"VIX {vix_state['vix']}{tnote} — fear regime, position halved."
                )
            if breadth["narrow"]:
                capital_note += " Narrow rally (RSP lagging SPY) — extra caution."
 
            max_loss = round(
                abs(targets["entry"] - targets["stop_loss"]) * shares, 2
            )
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
 
            results.append({
                "ticker": ticker,
                "sector": sector,
                "score": total_score,
                "rank_score": rank_score,
                "ev": ev,
                "quality": quality,
                "signal": signal,
                "confidence": round(confidence, 4),
                "buy_prob": round(buy_prob, 4),
                "accuracy": quality,  # backward-compat key
                "price": round(current_price, 4),
                "rsi": round(rsi, 1),
                "adx": round(adx, 1),
                "macd": round(float(df["MACD"].iloc[-1]), 4),
                "entry": targets["entry"],
                "stop_loss": targets["stop_loss"],
                "target1": targets["target1"],
                "target2": targets["target2"],
                "rr1": targets["rr1"],
                "rr2": targets["rr2"],
                "risk_amount": targets["risk_amount"],
                "shares": shares,
                "position_cost": cost,
                "effective_risk_pct": eff_risk_pct,
                "kelly_fraction": kelly_f,
                "capital_note": capital_note.strip(),
                "max_loss": max_loss,
                "holding_days": holding_days,
                "trade_plan": trade_plan,
                "risk_level": risk_level,
                "signal_breakdown": signal_breakdown,
                "key_drivers": key_drivers,
                "earnings_message": earnings["message"],
                "earnings_days": earnings["days"],
                "earnings_risk": earnings["risk_level"],
                "high_fear": high_fear,
                "vix": vix_state["vix"],
                "vix_term": vix_state.get("term"),
                "breadth_spread": breadth["spread"],
                "sentiment": sentiment,
                "sent_conf": round(sent_conf, 4),
                "sentiment_score": sentiment_score,
                "sentiment_trend": sentiment_trend,
                "relative_strength": rs,
                "market_regime": regime,
                "pct_from_52w_high": round(pct_from_high, 1),
                "sharpe": risk_metrics["Sharpe Ratio"],
                "max_drawdown": risk_metrics["Max Drawdown"],
                "score_breakdown": {
                    "ML Signal": f"{s1}/20",
                    "Multi-TF": f"{s2}/15",
                    "Volume": f"{s3}/10",
                    "Regime": f"{s4}/10",
                    "Rel. Strength": f"{s5}/10",
                    "Sector": f"{s6}/8",
                    "Fundamentals": f"{s7}/8",
                    "Sentiment": f"{s8}/8",
                    "Fibonacci": f"{s9}/5",
                    "Institutional": f"{s10}/10",
                    "Expected Value": f"+{ev:.2f}/$1",
                    "Model Quality": f"{quality:.2f}",
                },
                "currency": "USD",
            })
        except Exception:
            continue
 
    # Rank by EV-blended score, not raw score
    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results[:10], regime