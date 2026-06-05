import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from data import (get_stock_data, add_indicators,
                  get_fundamentals, get_support_resistance,
                  get_nifty_correlation, get_relative_strength,
                  get_market_regime, get_sector_momentum,
                  get_fibonacci_levels, get_nifty_data)
from model import (train_model_fast, get_signal,
                   get_risk_metrics, predict_holding_days)
from sentiment import get_news_sentiment
from universe import get_sector, SECTOR_INDICES, is_commodity
from earnings import (get_earnings_status,
                      get_nse_earnings_calendar,
                      get_dividend_risk,
                      get_dividend_exdates,
                      get_nse_fii_dii_flow)

# Minimum daily turnover in crores
MIN_DAILY_TURNOVER_CR = 50
# Max picks per sector
MAX_PICKS_PER_SECTOR = 2

def score_ml_signal(signal, confidence, buy_prob):
    if signal == "BUY":
        base = round(confidence * 20)
        bonus = round(max(0, (buy_prob - 0.5) * 10))
        return int(min(20, base + bonus))
    return 0

def score_multiframe(df_daily):
    score = 0
    try:
        if df_daily is None:
            return 0
        close = df_daily["Close"]
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
        conditions = [
            above_sma20, above_sma50,
            above_sma200, golden_cross, weekly_trend
        ]
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

def score_volume(df):
    try:
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        vol_5d = df["Volume"].tail(5).mean()
        vol_20d = df["Volume"].tail(20).mean()
        vol_trend = vol_5d > vol_20d
        if vol_ratio >= 1.8:
            base = 10
        elif vol_ratio >= 1.4:
            base = 8
        elif vol_ratio >= 1.1:
            base = 6
        elif vol_ratio >= 0.8:
            base = 4
        else:
            base = 2
        if vol_trend:
            base = min(10, base + 1)
        return base
    except Exception:
        return 3

def score_market_regime(regime):
    if regime == "bull":
        return 10
    elif regime == "sideways":
        return 5
    elif regime == "unknown":
        return 5
    return 3

def score_relative_strength(rs):
    try:
        if rs is None:
            return 5
        if rs >= 10:
            return 10
        elif rs >= 5:
            return 8
        elif rs >= 0:
            return 6
        elif rs >= -5:
            return 3
        else:
            return 0
    except Exception:
        return 5

def score_sector_momentum(sector, sector_indices):
    try:
        if sector not in sector_indices:
            return 4
        momentum = get_sector_momentum(
            sector_indices[sector]
        )
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
    except Exception:
        return 4

def score_fundamentals(fundamentals):
    score = 0
    try:
        pe = fundamentals.get("PE Ratio", "N/A")
        if pe != "N/A" and isinstance(pe, (int, float)):
            if 5 < pe < 25:
                score += 3
            elif 25 <= pe < 50:
                score += 2
            elif pe <= 5:
                score += 1
        rev_growth = fundamentals.get(
            "Revenue Growth", "N/A"
        )
        if rev_growth != "N/A" and isinstance(
            rev_growth, (int, float)
        ):
            if rev_growth > 0.15:
                score += 3
            elif rev_growth > 0.05:
                score += 2
            elif rev_growth > 0:
                score += 1
        de = fundamentals.get("Debt to Equity", "N/A")
        if de != "N/A" and isinstance(de, (int, float)):
            if de < 0.5:
                score += 2
            elif de < 1.5:
                score += 1
    except Exception:
        pass
    return min(8, score)

def score_sentiment(sentiment, confidence, trend,
                    sentiment_score):
    try:
        if sentiment == "positive":
            base = min(6, int(confidence * 8))
        elif sentiment == "neutral":
            base = 3
        else:
            base = 0
        if trend == "improving":
            base = min(8, base + 2)
        elif trend == "deteriorating":
            base = max(0, base - 2)
        if sentiment_score > 0.3:
            base = min(8, base + 1)
        return base
    except Exception:
        return 3

def score_fibonacci(df):
    try:
        from data import get_fibonacci_levels
        levels = get_fibonacci_levels(df)
        current = float(df["Close"].iloc[-1])
        for name, level in levels.items():
            if abs(current - level) / current < 0.02:
                return 5
        return 2
    except Exception:
        return 2

def score_institutional(ticker, df):
    try:
        vol_5d = df["Volume"].tail(5).mean()
        vol_20d = df["Volume"].tail(20).mean()
        vol_trend = vol_5d > vol_20d
        price_5d = float(df["Close"].iloc[-1])
        price_10d = float(
            df["Close"].iloc[-10]
            if len(df) > 10 else df["Close"].iloc[0]
        )
        price_trend = price_5d > price_10d
        rsi = float(df["RSI"].iloc[-1])
        rsi_healthy = 40 < rsi < 65
        conditions = [vol_trend, price_trend, rsi_healthy]
        met = sum(conditions)
        if met == 3:
            return 10
        elif met == 2:
            return 7
        elif met == 1:
            return 4
        return 2
    except Exception:
        return 2

def get_risk_level(score, regime, rsi, vol_ratio,
                   earnings_risk, confidence,
                   entry, stop_loss, atr_pct):
    risk_score = 0
    if entry > 0 and stop_loss > 0:
        stop_distance = abs(entry - stop_loss) / entry
        if stop_distance > 0.05:
            risk_score += 4
        elif stop_distance > 0.035:
            risk_score += 3
        elif stop_distance > 0.02:
            risk_score += 2
        elif stop_distance > 0.01:
            risk_score += 1
    if regime == "bear":
        risk_score += 2
    elif regime == "sideways":
        risk_score += 1
    if rsi > 65:
        risk_score += 2
    elif rsi > 58:
        risk_score += 1
    if atr_pct > 0.03:
        risk_score += 2
    elif atr_pct > 0.02:
        risk_score += 1
    if earnings_risk == "medium":
        risk_score += 2
    if confidence < 0.65:
        risk_score += 2
    elif confidence < 0.75:
        risk_score += 1
    if vol_ratio < 0.8:
        risk_score += 1
    if risk_score >= 7:
        return "HIGH"
    elif risk_score >= 4:
        return "MEDIUM"
    else:
        return "LOW"

def get_signal_breakdown(df, signal, regime, rs,
                         sentiment, sentiment_trend):
    try:
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma20 = float(df["SMA_20"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        breakdown = [
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
                )
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
                )
            },
            {
                "Factor": "Trend (SMA)",
                "Reading": (
                    "Above all SMAs"
                    if price > sma200
                    else "Below SMA200"
                ),
                "Status": (
                    "Uptrend confirmed"
                    if price > sma50
                    else "Below key average"
                ),
                "Signal": (
                    "🟢"
                    if price > sma50 and price > sma200
                    else "🔴" if price < sma50
                    else "🟡"
                )
            },
            {
                "Factor": "Volume",
                "Reading": f"{vol_ratio:.1f}x avg",
                "Status": (
                    "Confirming move"
                    if vol_ratio >= 1.2
                    else "Below average"
                ),
                "Signal": (
                    "🟢" if vol_ratio >= 1.4
                    else "🟡" if vol_ratio >= 1.0
                    else "🔴"
                )
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
                )
            },
            {
                "Factor": "Rel. strength",
                "Reading": (
                    f"{rs:+.1f}% vs Nifty"
                    if rs else "N/A"
                ),
                "Status": (
                    "Outperforming"
                    if rs is not None and rs >= 0
                    else "Underperforming"
                ),
                "Signal": (
                    "🟢" if rs and rs >= 3
                    else "🔴" if rs and rs < -3
                    else "🟡"
                )
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
                )
            },
        ]
        return breakdown
    except Exception:
        return []

def get_key_drivers(df, signal, regime, rs,
                    sentiment, confidence):
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
                f"High-conviction signal at "
                f"{confidence:.0%} model confidence"
            )
        if macd > 0 and macd_hist > 0:
            drivers.append(
                "MACD bullish crossover with "
                "positive histogram"
            )
        if vol_ratio >= 1.4:
            drivers.append(
                f"Volume {vol_ratio:.1f}x above average "
                f"— institutional confirmation"
            )
        if price > sma50 and price > sma200:
            drivers.append(
                "Price above SMA50 and SMA200 "
                "— strong structural uptrend"
            )
        if rs and rs >= 5:
            drivers.append(
                f"Outperforming Nifty 50 by "
                f"{rs:.1f}% — relative strength leader"
            )
        if sentiment == "positive":
            drivers.append(
                "News sentiment positive — "
                "news flow aligned with signal"
            )
        if 40 <= rsi <= 55:
            drivers.append(
                f"RSI at {rsi:.0f} — not overbought, "
                f"room to run"
            )
        if regime == "bull":
            drivers.append(
                "Bull market regime — "
                "macro tailwind supporting longs"
            )
    except Exception:
        pass
    return drivers[:3]

def calculate_entry_targets(df, signal, regime):
    try:
        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])
        atr_pct = atr / current_price

        # Wider stops in bear/high volatility
        if regime == "bear":
            atr_multiplier = 2.5
        elif atr_pct > 0.025:
            atr_multiplier = 3.0
        else:
            atr_multiplier = 2.0

        support_levels, resistance_levels = (
            get_support_resistance(df)
        )

        if signal == "BUY":
            entry = current_price
            support_stop = (
                support_levels[0]
                if support_levels and
                (entry - support_levels[0]) >
                atr * 0.5
                else None
            )
            atr_stop = entry - atr_multiplier * atr
            stop_loss = (
                max(support_stop, atr_stop)
                if support_stop
                else atr_stop
            )
            risk = max(entry - stop_loss, atr)
            valid_t1 = [
                r for r in resistance_levels
                if r > entry + risk
            ]
            valid_t2 = [
                r for r in resistance_levels
                if r > entry + risk * 2
            ]
            target1 = (
                valid_t1[0] if valid_t1
                else entry + 2 * risk
            )
            target2 = (
                valid_t2[0] if valid_t2
                else entry + 3 * risk
            )
            rr1 = round((target1 - entry) / risk, 2)
            rr2 = round((target2 - entry) / risk, 2)
        else:
            entry = current_price
            stop_loss = entry + atr_multiplier * atr
            risk = max(stop_loss - entry, atr)
            valid_t1 = [
                s for s in support_levels
                if s < entry - risk
            ]
            valid_t2 = [
                s for s in support_levels
                if s < entry - risk * 2
            ]
            target1 = (
                valid_t1[0] if valid_t1
                else entry - 2 * risk
            )
            target2 = (
                valid_t2[0] if valid_t2
                else entry - 3 * risk
            )
            rr1 = round((entry - target1) / risk, 2)
            rr2 = round((entry - target2) / risk, 2)

        return {
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "target1": round(target1, 2),
            "target2": round(target2, 2),
            "rr1": max(rr1, 1.0),
            "rr2": max(rr2, 2.0),
            "risk_amount": round(risk, 2)
        }
    except Exception:
        price = float(df["Close"].iloc[-1])
        atr = (
            float(df["ATR"].iloc[-1])
            if "ATR" in df.columns
            else price * 0.02
        )
        mult = 2.5 if regime == "bear" else 2.0
        return {
            "entry": round(price, 2),
            "stop_loss": round(price - mult * atr, 2),
            "target1": round(price + 2 * atr, 2),
            "target2": round(price + 3 * atr, 2),
            "rr1": 2.0,
            "rr2": 3.0,
            "risk_amount": round(mult * atr, 2)
        }

def calculate_position_size(capital, risk_pct,
                             entry, stop_loss,
                             fii_bearish=False):
    try:
        risk_amount = capital * (risk_pct / 100)
        # Halve position in bear FII environment
        if fii_bearish:
            risk_amount *= 0.5
        per_share_risk = abs(entry - stop_loss)
        if per_share_risk == 0:
            return 0, 0
        shares = int(risk_amount / per_share_risk)
        total_cost = shares * entry
        return shares, round(total_cost, 2)
    except Exception:
        return 0, 0

def check_liquidity(df, min_turnover_cr=50):
    """
    Check if stock has sufficient daily turnover.
    Minimum ₹50 crore daily to avoid slippage.
    """
    try:
        avg_volume = df["Volume"].tail(20).mean()
        avg_price = df["Close"].tail(20).mean()
        turnover_cr = (avg_volume * avg_price) / 1e7
        return turnover_cr >= min_turnover_cr
    except Exception:
        return True

def check_price_position(df):
    """
    Check how far stock is from 52-week high.
    Avoid stocks within 2% of 52W high in bear market
    as institutional resistance is strongest there.
    """
    try:
        high_52w = df["High"].tail(252).max()
        current = float(df["Close"].iloc[-1])
        pct_from_high = (
            (high_52w - current) / high_52w * 100
        )
        return pct_from_high
    except Exception:
        return 10.0
@st.cache_data(ttl=21600)
def pre_filter_universe(tickers,
                        min_price=10.0,
                        min_avg_volume=200000,
                        min_turnover_cr=3.0):
    """
    Phase 1: Fast batch pre-filter using 10-day data.
    Runs in ~60-90 seconds for 2000 stocks.
    Reduces universe to liquid, priced stocks only.
    """
    qualified = []
    batch_size = 50
    tickers = list(tickers)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
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
                turnover = price * vol / 1e7
                if (price >= min_price and
                        vol >= min_avg_volume and
                        turnover >= min_turnover_cr):
                    qualified.append(batch[0])
                continue

            raw = yf.download(
                batch, period="10d",
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                threads=True
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
                    turnover = price * vol / 1e7
                    if (price >= min_price and
                            vol >= min_avg_volume and
                            turnover >= min_turnover_cr):
                        qualified.append(ticker)
                except Exception:
                    continue
        except Exception:
            for ticker in batch:
                try:
                    df_t = yf.Ticker(
                        ticker
                    ).history(period="10d")
                    if df_t is None or df_t.empty:
                        continue
                    price = float(
                        df_t["Close"].iloc[-1]
                    )
                    vol = float(
                        df_t["Volume"].mean()
                    )
                    turnover = price * vol / 1e7
                    if (price >= min_price and
                            vol >= min_avg_volume and
                            turnover >= min_turnover_cr):
                        qualified.append(ticker)
                except Exception:
                    continue

    return qualified

def run_full_scan(tickers, capital=50000,
                  risk_pct=1.5,
                  progress_callback=None,
                  run_prefilter=True):
    nifty_df = get_nifty_data()
    regime = "unknown"
    if nifty_df is not None and not nifty_df.empty:
        regime = get_market_regime(nifty_df)

    earnings_map = get_nse_earnings_calendar()
    dividend_map = get_dividend_exdates()
    fii_data = get_nse_fii_dii_flow()
    fii_bearish = (
        fii_data is not None and
        fii_data.get("sentiment") == "bearish"
    )
    fii_consecutive_selling = (
        fii_data.get("consecutive_selling_days", 0)
        if fii_data else 0
    )

    min_rs_bear = 3.0

    if regime == "bear":
        min_threshold = 55
    elif regime == "unknown":
        min_threshold = 50
    else:
        min_threshold = 45

    tickers = list(tickers)

    # Phase 1 — pre-filter for large universes
    if run_prefilter and len(tickers) > 150:
        if progress_callback:
            progress_callback(
                0, len(tickers),
                "PRE-FILTERING"
            )
        tickers = pre_filter_universe(
            tuple(tickers)
        )
        if progress_callback:
            progress_callback(
                0, len(tickers),
                f"Pre-filter complete — "
                f"{len(tickers)} liquid stocks"
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

            if not check_liquidity(
                df, MIN_DAILY_TURNOVER_CR
            ):
                continue

            fundamentals = get_fundamentals(ticker)
            sector = get_sector(ticker)

            if sector_counts.get(
                sector, 0
            ) >= MAX_PICKS_PER_SECTOR:
                continue

            correlation = (
                get_nifty_correlation(df, nifty_df)
                if nifty_df is not None else None
            )
            rs = (
                get_relative_strength(df, nifty_df)
                if nifty_df is not None else None
            )

            if regime == "bear" and (
                rs is None or rs < min_rs_bear
            ):
                continue

            div_risk = get_dividend_risk(
                ticker, dividend_map
            )
            if div_risk["flag"]:
                continue

            news_data = get_news_sentiment(ticker)
            sentiment = news_data["sentiment"]
            sent_conf = news_data["confidence"]
            sentiment_score = news_data[
                "sentiment_score"
            ]
            sentiment_trend = news_data["trend"]
            risk_flags = news_data["risk_flags"]

            if risk_flags:
                continue

            earnings_status = get_earnings_status(
                ticker, earnings_map
            )
            if earnings_status["risk_level"] == "high":
                continue

            model, scaler, features, accuracy = (
                train_model_fast(ticker)
            )
            if model is None:
                continue

            signal, confidence, buy_prob, sell_prob = (
                get_signal(model, scaler, df, features)
            )

            if signal != "BUY" or confidence < 0.60:
                continue

            rsi = float(df["RSI"].iloc[-1])
            if rsi > 70 or rsi < 28:
                continue

            pct_from_high = check_price_position(df)
            if regime == "bear" and pct_from_high < 3.0:
                continue

            s1 = score_ml_signal(
                signal, confidence, buy_prob
            )
            s2 = score_multiframe(df)
            s3 = score_volume(df)
            s4 = score_market_regime(regime)
            s5 = score_relative_strength(rs)
            s6 = score_sector_momentum(
                sector, SECTOR_INDICES
            )
            s7 = score_fundamentals(fundamentals)
            s8 = score_sentiment(
                sentiment, sent_conf,
                sentiment_trend, sentiment_score
            )
            s9 = score_fibonacci(df)
            s10 = score_institutional(ticker, df)

            total_score = int(
                s1 + s2 + s3 + s4 + s5 +
                s6 + s7 + s8 + s9 + s10
            )

            if total_score < min_threshold:
                continue

            targets = calculate_entry_targets(
                df, signal, regime
            )
            shares, cost = calculate_position_size(
                capital, risk_pct,
                targets["entry"], targets["stop_loss"],
                fii_bearish=fii_bearish
            )
            risk_metrics = get_risk_metrics(df)

            atr_pct = (
                float(df["ATR"].iloc[-1]) /
                float(df["Close"].iloc[-1])
            )
            vol_ratio = float(
                df["Volume_ratio"].iloc[-1]
            )

            holding_days = predict_holding_days(
                df, signal, confidence, regime, rs
            )

            risk_level = get_risk_level(
                total_score, regime, rsi, vol_ratio,
                earnings_status["risk_level"],
                confidence,
                targets["entry"],
                targets["stop_loss"],
                atr_pct
            )

            signal_breakdown = get_signal_breakdown(
                df, signal, regime, rs,
                sentiment, sentiment_trend
            )

            key_drivers = get_key_drivers(
                df, signal, regime, rs,
                sentiment, confidence
            )

            capital_note = ""
            if shares == 0:
                min_capital = (
                    targets["risk_amount"] *
                    (100 / risk_pct)
                )
                capital_note = (
                    f"Min capital needed: "
                    f"₹{min_capital:,.0f}"
                )
            if fii_bearish:
                capital_note = (
                    "FII selling — position halved. "
                    + capital_note
                )

            max_loss = round(
                targets["risk_amount"] * shares
                if shares > 0
                else targets["risk_amount"], 2
            )

            sector_counts[sector] = (
                sector_counts.get(sector, 0) + 1
            )

            results.append({
                "ticker": ticker,
                "sector": sector,
                "score": total_score,
                "signal": signal,
                "confidence": round(confidence, 4),
                "buy_prob": round(buy_prob, 4),
                "accuracy": round(accuracy, 4),
                "price": round(
                    float(df["Close"].iloc[-1]), 2
                ),
                "rsi": round(float(rsi), 1),
                "macd": round(
                    float(df["MACD"].iloc[-1]), 2
                ),
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
                "div_message": div_risk["message"],
                "div_days": div_risk["days_to_ex"],
                "earnings_message": (
                    earnings_status["message"]
                ),
                "earnings_days": (
                    earnings_status["days_to_earnings"]
                ),
                "earnings_risk": (
                    earnings_status["risk_level"]
                ),
                "fii_bearish": fii_bearish,
                "fii_selling_days": (
                    fii_consecutive_selling
                ),
                "sentiment": sentiment,
                "sent_conf": round(sent_conf, 4),
                "sentiment_score": sentiment_score,
                "sentiment_trend": sentiment_trend,
                "correlation": correlation,
                "relative_strength": rs,
                "market_regime": regime,
                "pct_from_52w_high": round(
                    pct_from_high, 1
                ),
                "sharpe": risk_metrics["Sharpe Ratio"],
                "max_drawdown": (
                    risk_metrics["Max Drawdown"]
                ),
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
                    "Institutional": f"{s10}/10"
                }
            })
        except Exception:
            continue

    results.sort(
        key=lambda x: x["score"], reverse=True
    )
    return results[:10], regime