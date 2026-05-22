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