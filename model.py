import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
from data import get_stock_data, add_indicators

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "BB_upper", "BB_lower", "BB_width", "ATR",
    "Volume_ratio", "ROC", "Stoch", "Williams_R", "CCI",
    "Return", "SMA_cross", "Price_to_SMA20",
    "Price_to_SMA200", "Volatility"
]

def build_features(df):
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    available = [f for f in FEATURE_COLS if f in df.columns]
    if not available:
        return None, None, None
    return df[available], df["Target"], available

def get_risk_metrics(df):
    try:
        returns = df["Close"].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        rolling_max = df["Close"].cummax()
        drawdown = (df["Close"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        volatility = returns.std() * np.sqrt(252)
        win_rate = (returns > 0).sum() / len(returns)
        return {
            "Sharpe Ratio": round(float(sharpe), 2),
            "Max Drawdown": f"{float(max_drawdown):.1%}",
            "Annual Volatility": f"{float(volatility):.1%}",
            "Win Rate": f"{float(win_rate):.1%}"
        }
    except Exception:
        return {
            "Sharpe Ratio": 0,
            "Max Drawdown": "N/A",
            "Annual Volatility": "N/A",
            "Win Rate": "N/A"
        }

def get_signal(model, scaler, df, feature_cols):
    if model is None or scaler is None or feature_cols is None:
        return "NEUTRAL", 0.5, 0.5, 0.5
    try:
        available = [f for f in feature_cols if f in df.columns]
        if not available:
            return "NEUTRAL", 0.5, 0.5, 0.5
        latest = df[available].dropna().iloc[-1:]
        if latest.empty:
            return "NEUTRAL", 0.5, 0.5, 0.5
        latest_scaled = scaler.transform(latest)
        pred = model.predict(latest_scaled)[0]
        proba = model.predict_proba(latest_scaled)[0]
        confidence = float(proba[pred])
        buy_prob = float(proba[1])
        sell_prob = float(proba[0])
        signal = "BUY" if pred == 1 else "SELL"
        return signal, confidence, buy_prob, sell_prob
    except Exception:
        return "NEUTRAL", 0.5, 0.5, 0.5

@st.cache_resource
def train_model(ticker):
    try:
        df = get_stock_data(ticker, period="2y")
        if df is None or len(df) < 100:
            return None, None, None, 0.5
        df = add_indicators(df)
        if df is None or len(df) < 100:
            return None, None, None, 0.5
        result = build_features(df)
        if result[0] is None:
            return None, None, None, 0.5
        X, y, features = result
        if len(X) < 100:
            return None, None, None, 0.5

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="logloss", verbosity=0
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=6,
            random_state=42, n_jobs=-1
        )

        if LIGHTGBM_AVAILABLE:
            try:
                lgbm = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, random_state=42,
                    verbose=-1
                )
                estimators = [("xgb", xgb), ("rf", rf), ("lgbm", lgbm)]
            except Exception:
                estimators = [("xgb", xgb), ("rf", rf)]
        else:
            estimators = [("xgb", xgb), ("rf", rf)]

        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        ensemble.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, ensemble.predict(X_test_s))
        return ensemble, scaler, features, round(float(acc), 3)
    except Exception:
        return None, None, None, 0.5

@st.cache_resource
def train_model_fast(ticker):
    try:
        df = get_stock_data(ticker, period="1y")
        if df is None or len(df) < 60:
            return None, None, None, 0.5
        df = add_indicators(df)
        if df is None or len(df) < 50:
            return None, None, None, 0.5
        result = build_features(df)
        if result[0] is None:
            return None, None, None, 0.5
        X, y, features = result
        if len(X) < 50:
            return None, None, None, 0.5

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        xgb = XGBClassifier(
            n_estimators=100, max_depth=3,
            learning_rate=0.1, random_state=42,
            eval_metric="logloss", verbosity=0
        )
        xgb.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, xgb.predict(X_test_s))
        return xgb, scaler, features, round(float(acc), 3)
    except Exception:
        return None, None, None, 0.5
    
def predict_holding_days(df, signal, confidence, regime, rs):
    try:
        score = 0
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma20 = float(df["SMA_20"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])
        volatility = float(df["Volatility"].iloc[-1]) if "Volatility" in df.columns else 0.02

        # Factor 1 - RSI position (higher RSI = less room = shorter hold)
        if 40 <= rsi <= 55:
            score += 3  # early in move
        elif 55 < rsi <= 65:
            score += 2  # mid move
        elif rsi > 65:
            score += 1  # late in move, exit sooner
        else:
            score += 2  # oversold bounce

        # Factor 2 - MACD momentum
        if macd > 0 and macd_hist > 0:
            score += 3  # strong momentum
        elif macd > 0 or macd_hist > 0:
            score += 2  # moderate
        else:
            score += 1  # weak

        # Factor 3 - Trend alignment
        above_sma20 = price > sma20
        above_sma50 = price > sma50
        above_sma200 = price > sma200
        trend_count = sum([above_sma20, above_sma50, above_sma200])
        score += trend_count  # 0-3 points

        # Factor 4 - Volume confirmation
        if vol_ratio >= 1.5:
            score += 2
        elif vol_ratio >= 1.1:
            score += 1

        # Factor 5 - Market regime
        if regime == "bull":
            score += 3
        elif regime == "sideways":
            score += 2
        elif regime == "unknown":
            score += 1
        else:  # bear
            score += 0

        # Factor 6 - Relative strength
        if rs is not None:
            if rs >= 5:
                score += 2
            elif rs >= 0:
                score += 1

        # Factor 7 - Volatility (high volatility = shorter hold)
        daily_vol = volatility
        if daily_vol < 0.015:
            score += 2  # low vol, can hold longer
        elif daily_vol < 0.025:
            score += 1
        else:
            score += 0  # high vol, exit faster

        # Factor 8 - Model confidence
        if confidence >= 0.80:
            score += 2
        elif confidence >= 0.70:
            score += 1

        # Map score to holding days
        # Max possible score = 3+3+3+2+3+2+2+2 = 20
        if score >= 16:
            return 12
        elif score >= 13:
            return 10
        elif score >= 10:
            return 8
        elif score >= 7:
            return 6
        elif score >= 4:
            return 4
        else:
            return 3

    except Exception:
        return 5  # default fallback