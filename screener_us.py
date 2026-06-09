"""
screener_us.py — US stock screener for Equitex Intelligence
Adapts the India screener for US markets.

Key differences vs India (screener_engine.py):
  - SPY as market benchmark (not Nifty)
  - ^VIX as fear gauge (not India VIX); VIX > 25 = high fear
  - No FII/DII data → position size reduction triggered by VIX > 25
  - No NSE earnings calendar → yfinance earnings dates
  - No NSE bhavcopy delivery data
  - Liquidity threshold: $5M daily turnover (not ₹50 crore)
  - USD formatting in all outputs
  - US sector ETFs for sector momentum
  - Market hours: 9:30–16:00 ET
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

from data import (
    get_stock_data, add_indicators, get_fundamentals,
    get_support_resistance, get_relative_strength,
    get_fibonacci_levels,
)
from model import (
    train_model_fast, get_signal,
    get_risk_metrics, predict_holding_days,
)
from sentiment import get_news_sentiment
from screener_engine import (
    score_ml_signal, score_volume, score_market_regime,
    score_relative_strength, score_fundamentals,
    score_sentiment, score_fibonacci, score_institutional,
    get_risk_level, calculate_entry_targets,
    calculate_position_size,
)
from universe_us import (
    get_us_universe, get_us_sector,
    US_SECTOR_INDICES, get_sp500_tickers_live,
)

# ── Constants ────────────────────────────────────────────────
MIN_DAILY_TURNOVER_USD = 5_000_000   # $5M daily turnover minimum
MIN_PRICE_USD = 2.0                  # Skip penny stocks
MIN_AVG_VOLUME_US = 500_000          # 500K shares/day minimum
MAX_PICKS_PER_SECTOR = 2
VIX_FEAR_THRESHOLD = 25.0           # VIX > 25 → halve position sizes


# ── Data helpers ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_spy_data():
    """SPY as US market benchmark (equivalent of Nifty 50)."""
    for ticker in ["SPY", "^GSPC", "IVV"]:
        try:
            df = yf.Ticker(ticker).history(period="5y")
            if df is not None and not df.empty and len(df) > 50:
                df.dropna(inplace=True)
                return df
        except Exception:
            continue
    return None


@st.cache_data(ttl=3600)
def get_us_vix():
    """CBOE VIX — US market fear gauge."""
    try:
        df = yf.Ticker("^VIX").history(period="2y")
        if df is None or df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_us_market_regime(spy_df=None) -> str:
    """
    Determines US market regime using SPY.
    Returns 'bull', 'bear', 'sideways', or 'unknown'.
    """
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
            price > sma50,
            price > sma200,
            sma50 > sma200,
            rsi > 50,
            macd > 0,
        ])
        if bull_signals >= 4:
            return "bull"
        elif bull_signals <= 1:
            return "bear"
        else:
            return "sideways"
    except Exception:
        return "unknown"


@st.cache_data(ttl=3600)
def get_vix_state() -> dict:
    """
    Returns VIX level and whether we're in high-fear regime.
    Equivalent of FII sentiment in India screener.
    """
    try:
        vix_df = get_us_vix()
        if vix_df is None or vix_df.empty:
            return {"vix": None, "high_fear": False, "label": "Unknown"}
        current_vix = float(vix_df["Close"].iloc[-1])
        high_fear = current_vix > VIX_FEAR_THRESHOLD
        if current_vix < 15:
            label = "Calm"
        elif current_vix < 20:
            label = "Low"
        elif current_vix < 25:
            label = "Moderate"
        elif current_vix < 35:
            label = "Elevated"
        else:
            label = "Extreme Fear"
        return {
            "vix": round(current_vix, 1),
            "high_fear": high_fear,
            "label": label,
        }
    except Exception:
        return {"vix": None, "high_fear": False, "label": "Unknown"}


@st.cache_data(ttl=21600)
def get_us_earnings_risk(ticker: str) -> dict:
    """
    Check upcoming earnings using yfinance calendar.
    Returns risk level + days to earnings.
    Equivalent of get_earnings_status() for India.
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return {"risk_level": "none", "days": None, "message": ""}
        # Calendar columns vary — try common formats
        if "Earnings Date" in cal.columns:
            date_col = cal["Earnings Date"].dropna()
        elif hasattr(cal, "index") and "Earnings Date" in str(cal.index):
            date_col = pd.Series(cal.loc["Earnings Date"])
        else:
            return {"risk_level": "none", "days": None, "message": ""}
        if len(date_col) == 0:
            return {"risk_level": "none", "days": None, "message": ""}
        next_earnings = pd.to_datetime(date_col.iloc[0])
        today = pd.Timestamp.now(tz=next_earnings.tzinfo)
        days = (next_earnings - today).days
        if days < 0 or days > 30:
            return {"risk_level": "none", "days": days, "message": ""}
        if days <= 3:
            return {
                "risk_level": "high",
                "days": days,
                "message": f"⚠️ Earnings in {days}d — HIGH risk",
            }
        elif days <= 7:
            return {
                "risk_level": "medium",
                "days": days,
                "message": f"Earnings in {days}d",
            }
        else:
            return {
                "risk_level": "low",
                "days": days,
                "message": f"Earnings in {days}d",
            }
    except Exception:
        return {"risk_level": "none", "days": None, "message": ""}


@st.cache_data(ttl=3600)
def get_us_sector_momentum(sector_ticker: str) -> float | None:
    """1-month momentum for a US sector ETF."""
    try:
        df = yf.Ticker(sector_ticker).history(period="3mo")
        if df is None or df.empty or len(df) < 20:
            return None
        return round(
            (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-20]) - 1) * 100,
            2
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
    else:
        return 1


def get_us_signal_breakdown(df, signal, regime, rs,
                             sentiment, sentiment_trend) -> list:
    """Signal breakdown table with US-appropriate labels (vs SPY not Nifty)."""
    try:
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma20 = float(df["SMA_20"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        return [
            {
                "Factor": "RSI",
                "Reading": f"{rsi:.1f}",
                "Status": (
                    "Overbought" if rsi > 70
                    else "Oversold" if rsi < 30
                    else "Neutral"
                ),
                "Signal": (
                    "🟢" if 40 <= rsi <= 65
                    else "🔴" if rsi > 70
                    else "🟡"
                ),
            },
            {
                "Factor": "MACD",
                "Reading": f"{macd:.2f}",
                "Status": (
                    "Bullish crossover"
                    if macd > 0 and macd_hist > 0
                    else "Bearish / weak"
                ),
                "Signal": (
                    "🟢" if macd > 0 and macd_hist > 0
                    else "🔴" if macd < 0
                    else "🟡"
                ),
            },
            {
                "Factor": "Trend (SMA)",
                "Reading": (
                    "Above all SMAs" if price > sma200
                    else "Below SMA200"
                ),
                "Status": (
                    "Uptrend confirmed"
                    if price > sma50 and price > sma200
                    else "Above SMA50 only"
                    if price > sma50
                    else "Below key averages"
                ),
                "Signal": (
                    "🟢" if price > sma50 and price > sma200
                    else "🔴" if price < sma50
                    else "🟡"
                ),
            },
            {
                "Factor": "Volume",
                "Reading": f"{vol_ratio:.1f}x avg",
                "Status": (
                    "Confirming move" if vol_ratio >= 1.2
                    else "Below average"
                ),
                "Signal": (
                    "🟢" if vol_ratio >= 1.4
                    else "🟡" if vol_ratio >= 1.0
                    else "🔴"
                ),
            },
            {
                "Factor": "Market regime",
                "Reading": regime.upper(),
                "Status": (
                    "Favourable" if regime == "bull"
                    else "Headwind" if regime == "bear"
                    else "Neutral"
                ),
                "Signal": (
                    "🟢" if regime == "bull"
                    else "🔴" if regime == "bear"
                    else "🟡"
                ),
            },
            {
                "Factor": "Rel. strength",
                "Reading": (
                    f"{rs:+.1f}% vs SPY" if rs is not None
                    else "N/A"
                ),
                "Status": (
                    "Outperforming" if rs is not None and rs >= 0
                    else "Underperforming"
                ),
                "Signal": (
                    "🟢" if rs and rs >= 3
                    else "🔴" if rs and rs < -3
                    else "🟡"
                ),
            },
            {
                "Factor": "Sentiment",
                "Reading": sentiment.capitalize(),
                "Status": (
                    f"Positive & {sentiment_trend}"
                    if sentiment == "positive"
                    else f"Negative / {sentiment_trend}"
                ),
                "Signal": (
                    "🟢" if sentiment == "positive"
                    else "🔴" if sentiment == "negative"
                    else "🟡"
                ),
            },
        ]
    except Exception:
        return []


def check_us_liquidity(df) -> bool:
    """Check if US stock has ≥ $5M daily dollar volume."""
    try:
        avg_volume = df["Volume"].tail(20).mean()
        avg_price = df["Close"].tail(20).mean()
        daily_dollar_vol = avg_volume * avg_price
        return daily_dollar_vol >= MIN_DAILY_TURNOVER_USD
    except Exception:
        return True


@st.cache_data(ttl=21600)
def pre_filter_us_universe(tickers: tuple,
                            min_price: float = MIN_PRICE_USD,
                            min_avg_volume: int = MIN_AVG_VOLUME_US) -> list:
    """
    Phase 1: Fast batch pre-filter for US universe.
    Same architecture as India pre_filter_universe().
    Typically ~60-90s for 800 stocks.
    """
    qualified = []
    batch_size = 50
    tickers = list(tickers)
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            if len(batch) == 1:
                raw = yf.download(
                    batch[0], period="10d",
                    progress=False, auto_adjust=True
                )
                if raw is None or raw.empty:
                    continue
                price = float(raw["Close"].iloc[-1])
                vol = float(raw["Volume"].mean())
                if price >= min_price and vol >= min_avg_volume:
                    qualified.append(batch[0])
                continue
            raw = yf.download(
                batch, period="10d", progress=False,
                group_by="ticker", auto_adjust=True, threads=True
            )
            if raw is None or raw.empty:
                continue
            for ticker in batch:
                try:
                    if ticker in raw.columns.get_level_values(0):
                        df_t = raw[ticker].dropna()
                    else:
                        continue
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
                    price = float(df_t["Close"].iloc[-1])
                    vol = float(df_t["Volume"].mean())
                    if price >= min_price and vol >= min_avg_volume:
                        qualified.append(ticker)
                except Exception:
                    continue
    return qualified


@st.cache_data(ttl=21600)
def run_us_scan(tickers_tuple: tuple,
                capital: float = 10000.0,
                risk_pct: float = 1.5,
                progress_callback=None,
                run_prefilter: bool = True) -> tuple:
    """
    Full US stock scan. Returns (picks_list, regime_str).

    picks_list: up to 10 dicts, each containing full trade info in USD.
    regime_str: 'bull' | 'bear' | 'sideways' | 'unknown'
    """
    tickers = list(tickers_tuple)

    # Market context
    spy_df = get_spy_data()
    regime = get_us_market_regime(spy_df)
    vix_state = get_vix_state()
    high_fear = vix_state["high_fear"]

    # Score thresholds (same logic as India)
    if regime == "bear":
        min_threshold = 55
    elif regime == "unknown":
        min_threshold = 50
    else:
        min_threshold = 45

    min_rs_bear = 3.0  # Must outperform SPY by 3% in bear

    # Phase 1 — pre-filter
    if run_prefilter and len(tickers) > 150:
        if progress_callback:
            progress_callback(0, len(tickers), "PRE-FILTERING")
        tickers = pre_filter_us_universe(tuple(tickers))
        if progress_callback:
            progress_callback(
                0, len(tickers),
                f"Pre-filter complete — {len(tickers)} liquid stocks"
            )

    results = []
    sector_counts = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        try:
            df = get_stock_data(ticker, period="2y")
            if df is None or len(df) < 60:
                continue
            df = add_indicators(df)
            if df is None or len(df) < 50:
                continue

            # Liquidity check
            if not check_us_liquidity(df):
                continue

            # Sector cap
            sector = get_us_sector(ticker)
            if sector_counts.get(sector, 0) >= MAX_PICKS_PER_SECTOR:
                continue

            # Relative strength vs SPY
            rs = (
                get_relative_strength(df, spy_df)
                if spy_df is not None else None
            )

            # Bear regime: require +3% RS vs SPY
            if regime == "bear" and (rs is None or rs < min_rs_bear):
                continue

            # Earnings risk
            earnings = get_us_earnings_risk(ticker)
            if earnings["risk_level"] == "high":
                continue

            # News sentiment & risk flags
            news_data = get_news_sentiment(ticker)
            sentiment = news_data["sentiment"]
            sent_conf = news_data["confidence"]
            sentiment_score = news_data["sentiment_score"]
            sentiment_trend = news_data["trend"]
            risk_flags = news_data["risk_flags"]
            if risk_flags:
                continue

            # ML model
            model, scaler, features, accuracy = train_model_fast(ticker)
            if model is None:
                continue
            signal, confidence, buy_prob, sell_prob = get_signal(
                model, scaler, df, features
            )
            if signal != "BUY" or confidence < 0.60:
                continue

            # RSI filter
            rsi = float(df["RSI"].iloc[-1])
            if rsi > 70 or rsi < 28:
                continue

            # Not within 3% of 52W high in bear
            high_52w = float(df["High"].tail(252).max())
            current_price = float(df["Close"].iloc[-1])
            pct_from_high = (high_52w - current_price) / high_52w * 100
            if regime == "bear" and pct_from_high < 3.0:
                continue

            # Fundamentals
            fundamentals = get_fundamentals(ticker)

            # ── Composite 10-layer score ──────────────────────────────
            s1 = score_ml_signal(signal, confidence, buy_prob)
            s2 = _score_multiframe_us(df)
            s3 = score_volume(df)
            s4 = score_market_regime(regime)
            s5 = score_relative_strength(rs)
            s6 = score_us_sector_momentum(sector)
            s7 = score_fundamentals(fundamentals)
            s8 = score_sentiment(
                sentiment, sent_conf, sentiment_trend, sentiment_score
            )
            s9 = score_fibonacci(df)
            s10 = score_institutional(ticker, df)
            total_score = int(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10)

            if total_score < min_threshold:
                continue

            # Position sizing (USD-based, same 1.5% risk rule)
            targets = calculate_entry_targets(df, signal, regime)
            # In high-fear (VIX > 25): halve position like FII-bearish in India
            shares, cost = calculate_position_size(
                capital, risk_pct,
                targets["entry"], targets["stop_loss"],
                fii_bearish=high_fear,
            )
            if shares == 0:
                continue

            atr_pct = float(df["ATR"].iloc[-1]) / float(df["Close"].iloc[-1])
            vol_ratio = float(df["Volume_ratio"].iloc[-1])
            holding_days = predict_holding_days(
                df, signal, confidence, regime, rs
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
                df, signal, regime, rs, sentiment, confidence
            )

            capital_note = ""
            if high_fear:
                capital_note = f"VIX {vix_state['vix']} — position halved (fear mode)."

            max_loss = round(targets["risk_amount"] * shares, 2)

            sector_counts[sector] = sector_counts.get(sector, 0) + 1

            results.append({
                "ticker": ticker,
                "sector": sector,
                "score": total_score,
                "signal": signal,
                "confidence": round(confidence, 4),
                "buy_prob": round(buy_prob, 4),
                "accuracy": round(accuracy, 4),
                "price": round(current_price, 4),
                "rsi": round(rsi, 1),
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
                "capital_note": capital_note,
                "max_loss": max_loss,
                "holding_days": holding_days,
                "risk_level": risk_level,
                "signal_breakdown": signal_breakdown,
                "key_drivers": key_drivers,
                "earnings_message": earnings["message"],
                "earnings_days": earnings["days"],
                "earnings_risk": earnings["risk_level"],
                "high_fear": high_fear,
                "vix": vix_state["vix"],
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
                },
                "currency": "USD",
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10], regime


# ── Private helpers ───────────────────────────────────────────

def _score_multiframe_us(df) -> int:
    """Multi-timeframe score — identical logic to India screener."""
    score = 0
    try:
        close = df["Close"]
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        current = close.iloc[-1]
        above_sma20 = current > sma20.iloc[-1]
        above_sma50 = current > sma50.iloc[-1]
        above_sma200 = current > sma200.iloc[-1]
        golden_cross = sma50.iloc[-1] > sma200.iloc[-1]
        weekly_sma = close.tail(5).mean()
        weekly_trend = close.iloc[-1] > weekly_sma
        conditions = [above_sma20, above_sma50, above_sma200, golden_cross, weekly_trend]
        met = sum(conditions)
        if met == 5:
            score = 15
        elif met == 4:
            score = 12
        elif met == 3:
            score = 8
        elif met == 2:
            score = 4
        else:
            score = 0
    except Exception:
        pass
    return score


def _get_us_key_drivers(df, signal, regime, rs,
                         sentiment, confidence) -> list:
    """Key driver bullets for US picks."""
    drivers = []
    try:
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        if confidence >= 0.80:
            drivers.append(
                f"High-conviction signal at {confidence:.0%} model confidence"
            )
        if macd > 0 and macd_hist > 0:
            drivers.append("MACD bullish crossover with positive histogram")
        if vol_ratio >= 1.4:
            drivers.append(
                f"Volume {vol_ratio:.1f}x above average — institutional confirmation"
            )
        if price > sma50 and price > sma200:
            drivers.append("Price above SMA50 and SMA200 — strong structural uptrend")
        if rs and rs >= 5:
            drivers.append(
                f"Outperforming S&P 500 by {rs:.1f}% — relative strength leader"
            )
        if sentiment == "positive":
            drivers.append("News sentiment positive — news flow aligned with signal")
        if 40 <= rsi <= 55:
            drivers.append(f"RSI at {rsi:.0f} — not overbought, room to run")
        if regime == "bull":
            drivers.append("Bull market regime — macro tailwind supporting longs")
    except Exception:
        pass
    return drivers[:3]