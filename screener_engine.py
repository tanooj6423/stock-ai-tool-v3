import pandas as pd
import numpy as np
import streamlit as st
from data import (get_stock_data, add_indicators, get_weekly_data,
                  get_fundamentals, get_support_resistance,
                  get_nifty_correlation, get_relative_strength,
                  get_market_regime, get_sector_momentum,
                  get_fibonacci_levels, get_nifty_data)
from model import train_model_fast, get_signal, get_risk_metrics
from sentiment import get_news_sentiment
from universe import get_sector, SECTOR_INDICES, is_commodity

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
        weekly_close = close.iloc[-1]
        weekly_sma = close.tail(5).mean()
        weekly_trend = weekly_close > weekly_sma

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
        momentum = get_sector_momentum(sector_indices[sector])
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

        rev_growth = fundamentals.get("Revenue Growth", "N/A")
        if rev_growth != "N/A" and isinstance(rev_growth, (int, float)):
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

def score_sentiment(sentiment, confidence, trend, sentiment_score):
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

        price_5d = df["Close"].iloc[-1]
        price_10d = df["Close"].iloc[-10] if len(df) > 10 else df["Close"].iloc[0]
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

def calculate_entry_targets(df, signal):
    try:
        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])
        support_levels, resistance_levels = get_support_resistance(df)

        if signal == "BUY":
            entry = current_price
            stop_loss = (
                support_levels[0]
                if support_levels and
                (entry - support_levels[0]) > atr * 0.5
                else entry - 2 * atr
            )
            risk = max(entry - stop_loss, atr)

            valid_t1 = [r for r in resistance_levels if r > entry + risk]
            valid_t2 = [r for r in resistance_levels if r > entry + risk * 2]

            target1 = valid_t1[0] if valid_t1 else entry + 2 * risk
            target2 = valid_t2[0] if valid_t2 else entry + 3 * risk

            rr1 = round((target1 - entry) / risk, 2)
            rr2 = round((target2 - entry) / risk, 2)
        else:
            entry = current_price
            stop_loss = (
                resistance_levels[0]
                if resistance_levels and
                (resistance_levels[0] - entry) > atr * 0.5
                else entry + 2 * atr
            )
            risk = max(stop_loss - entry, atr)

            valid_t1 = [s for s in support_levels if s < entry - risk]
            valid_t2 = [s for s in support_levels if s < entry - risk * 2]

            target1 = valid_t1[0] if valid_t1 else entry - 2 * risk
            target2 = valid_t2[0] if valid_t2 else entry - 3 * risk

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
        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else price * 0.02
        return {
            "entry": round(price, 2),
            "stop_loss": round(price - 2 * atr, 2),
            "target1": round(price + 2 * atr, 2),
            "target2": round(price + 3 * atr, 2),
            "rr1": 2.0,
            "rr2": 3.0,
            "risk_amount": round(2 * atr, 2)
        }

def calculate_position_size(capital, risk_pct, entry, stop_loss):
    try:
        risk_amount = capital * (risk_pct / 100)
        per_share_risk = abs(entry - stop_loss)
        if per_share_risk == 0:
            return 0, 0
        shares = int(risk_amount / per_share_risk)
        total_cost = shares * entry
        return shares, round(total_cost, 2)
    except Exception:
        return 0, 0

def run_full_scan(tickers, capital=50000, risk_pct=1.5,
                  progress_callback=None):
    nifty_df = get_nifty_data()
    regime = "unknown"
    if nifty_df is not None and not nifty_df.empty:
        regime = get_market_regime(nifty_df)

    results = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i, total, ticker)
        try:
            df = get_stock_data(ticker, period="1y")
            if df is None or len(df) < 60:
                continue
            df = add_indicators(df)
            if df is None or len(df) < 50:
                continue

            fundamentals = get_fundamentals(ticker)
            sector = get_sector(ticker)
            correlation = (
                get_nifty_correlation(df, nifty_df)
                if nifty_df is not None else None
            )
            rs = (
                get_relative_strength(df, nifty_df)
                if nifty_df is not None else None
            )

            news_data = get_news_sentiment(ticker)
            sentiment = news_data["sentiment"]
            sent_conf = news_data["confidence"]
            sentiment_score = news_data["sentiment_score"]
            sentiment_trend = news_data["trend"]
            risk_flags = news_data["risk_flags"]

            if risk_flags:
                continue

            model, scaler, features, accuracy = train_model_fast(ticker)
            if model is None:
                continue

            signal, confidence, buy_prob, sell_prob = get_signal(
                model, scaler, df, features
            )

            if signal != "BUY" or confidence < 0.60:
                continue

            rsi = float(df["RSI"].iloc[-1])
            if rsi > 72 or rsi < 28:
                continue

            s1 = score_ml_signal(signal, confidence, buy_prob)
            s2 = score_multiframe(df)
            s3 = score_volume(df)
            s4 = score_market_regime(regime)
            s5 = score_relative_strength(rs)
            s6 = score_sector_momentum(sector, SECTOR_INDICES)
            s7 = score_fundamentals(fundamentals)
            s8 = score_sentiment(sentiment, sent_conf,
                                 sentiment_trend, sentiment_score)
            s9 = score_fibonacci(df)
            s10 = score_institutional(ticker, df)

            total_score = int(s1 + s2 + s3 + s4 + s5 +
                              s6 + s7 + s8 + s9 + s10)

            min_threshold = 45 if regime in ["bear", "unknown"] else 55
            if total_score < min_threshold:
                continue

            targets = calculate_entry_targets(df, signal)
            shares, cost = calculate_position_size(
                capital, risk_pct,
                targets["entry"], targets["stop_loss"]
            )
            risk_metrics = get_risk_metrics(df)

            capital_note = ""
            if shares == 0:
                min_capital = targets["risk_amount"] * (100 / risk_pct)
                capital_note = f"Min capital needed: ₹{min_capital:,.0f}"

            results.append({
                "ticker": ticker,
                "sector": sector,
                "score": total_score,
                "signal": signal,
                "confidence": round(confidence, 4),
                "buy_prob": round(buy_prob, 4),
                "accuracy": round(accuracy, 4),
                "price": round(float(df["Close"].iloc[-1]), 2),
                "rsi": round(float(rsi), 1),
                "macd": round(float(df["MACD"].iloc[-1]), 2),
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
                "sentiment": sentiment,
                "sent_conf": round(sent_conf, 4),
                "sentiment_score": sentiment_score,
                "sentiment_trend": sentiment_trend,
                "correlation": correlation,
                "relative_strength": rs,
                "market_regime": regime,
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
                    "Institutional": f"{s10}/10"
                }
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10], regime