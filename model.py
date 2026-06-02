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
    
def predict_holding_days(df, signal, confidence,
                         regime, rs):
    try:
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_hist = float(df["MACD_hist"].iloc[-1])
        vol_ratio = float(df["Volume_ratio"].iloc[-1])
        price = float(df["Close"].iloc[-1])
        sma20 = float(df["SMA_20"].iloc[-1])
        sma50 = float(df["SMA_50"].iloc[-1])
        sma200 = float(df["SMA_200"].iloc[-1])
        atr = float(df["ATR"].iloc[-1])
        atr_pct = atr / price

        volatility = (
            float(df["Volatility"].iloc[-1])
            if "Volatility" in df.columns
            else atr_pct
        )

        # Bear market — always shorter hold
        if regime == "bear":
            if rsi > 62:
                return 3
            elif rsi > 52:
                return 4
            else:
                return 5

        # RSI position — most important factor
        # Early in move = longer hold
        # Late in move = shorter hold
        if rsi < 40:
            rsi_days = 10
        elif rsi < 48:
            rsi_days = 9
        elif rsi < 55:
            rsi_days = 8
        elif rsi < 62:
            rsi_days = 6
        elif rsi < 68:
            rsi_days = 4
        else:
            rsi_days = 3

        # ATR — high volatility = shorter hold
        # stock reaches target faster
        if atr_pct > 0.03:
            atr_adjustment = -2
        elif atr_pct > 0.02:
            atr_adjustment = -1
        elif atr_pct < 0.01:
            atr_adjustment = 2
        else:
            atr_adjustment = 0

        # Trend alignment — strong trend = longer hold
        above_sma20 = price > sma20
        above_sma50 = price > sma50
        above_sma200 = price > sma200
        trend_count = sum([
            above_sma20, above_sma50, above_sma200
        ])
        if trend_count == 3:
            trend_adjustment = 1
        elif trend_count == 2:
            trend_adjustment = 0
        else:
            trend_adjustment = -1

        # MACD momentum
        if macd > 0 and macd_hist > 0:
            macd_adjustment = 1
        elif macd < 0:
            macd_adjustment = -1
        else:
            macd_adjustment = 0

        # Volume confirmation
        if vol_ratio >= 1.5:
            vol_adjustment = 1
        elif vol_ratio < 0.8:
            vol_adjustment = -1
        else:
            vol_adjustment = 0

        # Regime adjustment
        if regime == "bull":
            regime_adjustment = 2
        elif regime == "sideways":
            regime_adjustment = 0
        else:
            regime_adjustment = -1

        # Relative strength
        if rs and rs >= 5:
            rs_adjustment = 1
        elif rs and rs < -5:
            rs_adjustment = -1
        else:
            rs_adjustment = 0

        # Confidence adjustment
        if confidence >= 0.85:
            conf_adjustment = 1
        elif confidence < 0.65:
            conf_adjustment = -1
        else:
            conf_adjustment = 0

        total_days = (
            rsi_days +
            atr_adjustment +
            trend_adjustment +
            macd_adjustment +
            vol_adjustment +
            regime_adjustment +
            rs_adjustment +
            conf_adjustment
        )

        # Clamp to 3-14 days
        return max(3, min(14, total_days))

    except Exception:
        return 6