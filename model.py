import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                              precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import streamlit as st
from data import (get_stock_data, add_indicators,
                  get_nifty_data, add_market_features,
                  get_india_vix, get_nse_delivery_data)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# Base features — available from add_indicators alone
BASE_FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "BB_upper", "BB_lower", "BB_width", "ATR",
    "Volume_ratio", "Volume_surge_3d",
    "ROC", "Stoch", "Williams_R", "CCI",
    "Return", "Return_5d", "Return_20d",
    "SMA_cross", "Price_to_SMA20", "Price_to_SMA200",
    "Volatility", "NATR", "Gap_pct",
    "Price_pos_52w", "Bullish_div",
    "ATR_pct_rank", "Expiry_week"
]

# Advanced features — require external market data
ADVANCED_FEATURE_COLS = [
    "Beta_20d", "Rel_return_5d",
    "Nifty_momentum", "VIX_ratio"
]

def create_triple_barrier_target(
    df, horizon=7, pt_atr_mult=2.0,
    sl_atr_mult=2.5, min_gain=0.025
):
    """
    Triple Barrier Method.
    Label 1: Take Profit hit first within horizon days.
    Label 0: Stop Loss hit first OR time barrier.
    Aligns model objective with actual swing trade
    mechanics (entry, stop, target).
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = df["ATR"].values

    for t in range(n - horizon):
        entry = close[t]
        tp = max(
            entry + pt_atr_mult * atr[t],
            entry * (1 + min_gain)
        )
        sl = entry - sl_atr_mult * atr[t]

        path_high = high[t+1:t+1+horizon]
        path_low = low[t+1:t+1+horizon]

        tp_hits = np.where(path_high >= tp)[0]
        sl_hits = np.where(path_low <= sl)[0]

        first_tp = (
            tp_hits[0] if len(tp_hits) > 0
            else horizon + 1
        )
        first_sl = (
            sl_hits[0] if len(sl_hits) > 0
            else horizon + 1
        )

        if first_tp < first_sl:
            labels[t] = 1

    return labels

def build_features(df, advanced=False):
    """
    Build feature matrix X and target y.
    Returns X, y, feature_list or None, None, None.
    """
    df = df.copy()
    df["Target"] = create_triple_barrier_target(df)
    df.dropna(inplace=True)

    cols = BASE_FEATURE_COLS.copy()
    if advanced:
        cols += [
            c for c in ADVANCED_FEATURE_COLS
            if c in df.columns
        ]

    available = [c for c in cols if c in df.columns]
    if not available or len(df) < 60:
        return None, None, None

    X = df[available].copy()
    y = df["Target"].copy()

    # Drop last horizon rows — no valid label
    X = X.iloc[:-7]
    y = y.iloc[:-7]

    if len(X) < 60:
        return None, None, None

    return X, y, available

def purged_split(X, y, test_frac=0.25,
                 embargo_days=5):
    """
    Walk-forward split with embargo gap.
    Prevents lookahead leakage when target
    uses future price data.
    """
    n = len(X)
    split = int(n * (1 - test_frac))
    split_emb = min(split + embargo_days, n)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]
    X_test = X.iloc[split_emb:]
    y_test = y.iloc[split_emb:]

    return X_train, X_test, y_train, y_test

def get_lgbm_model(scale_pos_weight=1.0):
    if not LGBM_AVAILABLE:
        return None
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        num_leaves=31,
        min_data_in_leaf=15,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )

def get_xgb_model(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )

def get_risk_metrics(df):
    try:
        returns = df["Close"].pct_change().dropna()
        sharpe = (
            (returns.mean() / returns.std()) *
            np.sqrt(252)
        )
        rolling_max = df["Close"].cummax()
        drawdown = (
            (df["Close"] - rolling_max) / rolling_max
        )
        max_drawdown = drawdown.min()
        volatility = returns.std() * np.sqrt(252)
        win_rate = (returns > 0).sum() / len(returns)
        return {
            "Sharpe Ratio": round(float(sharpe), 2),
            "Max Drawdown": (
                f"{float(max_drawdown):.1%}"
            ),
            "Annual Volatility": (
                f"{float(volatility):.1%}"
            ),
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
    if (model is None or scaler is None or
            feature_cols is None):
        return "NEUTRAL", 0.5, 0.5, 0.5
    try:
        available = [
            f for f in feature_cols
            if f in df.columns
        ]
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
def train_model_fast(ticker):
    """
    Fast scan model: LightGBM only.
    Uses 2y data with Triple Barrier target.
    Optimised for speed (<30s) across 100 stocks.
    """
    try:
        df = get_stock_data(ticker, period="2y")
        if df is None or len(df) < 100:
            return None, None, None, 0.5
        df = add_indicators(df)
        if df is None or len(df) < 80:
            return None, None, None, 0.5

        X, y, features = build_features(
            df, advanced=False
        )
        if X is None or len(X) < 60:
            return None, None, None, 0.5

        X_train, X_test, y_train, y_test = (
            purged_split(X, y)
        )

        if len(X_train) < 40 or len(X_test) < 10:
            return None, None, None, 0.5

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / max(pos, 1)

        if LGBM_AVAILABLE:
            model = get_lgbm_model(spw)
        else:
            model = get_xgb_model(spw)

        model.fit(X_train_s, y_train)
        acc = accuracy_score(
            y_test, model.predict(X_test_s)
        )
        return model, scaler, features, round(
            float(acc), 3
        )
    except Exception:
        return None, None, None, 0.5

@st.cache_resource
def train_model(ticker):
    """
    Full analysis model: LightGBM + XGBoost ensemble
    with probability calibration.
    Uses 5y data, advanced features, purged CV.
    """
    try:
        df = get_stock_data(ticker, period="5y")
        if df is None or len(df) < 200:
            return None, None, None, 0.5
        df = add_indicators(df)
        if df is None or len(df) < 150:
            return None, None, None, 0.5

        # Add market features
        nifty_df = get_nifty_data()
        vix_df = get_india_vix()
        df = add_market_features(df, nifty_df, vix_df)

        # Add delivery volume if available
        delivery_pct = get_nse_delivery_data(ticker)
        if delivery_pct is not None:
            df["Delivery_pct"] = delivery_pct
            # Delivery breakout flag
            delivery_series = pd.Series(
                [delivery_pct] * len(df),
                index=df.index
            )
            df["Delivery_breakout"] = (
                delivery_series > 0.5
            ).astype(int)

        X, y, features = build_features(
            df, advanced=True
        )
        if X is None or len(X) < 100:
            return None, None, None, 0.5

        X_train, X_test, y_train, y_test = (
            purged_split(X, y, test_frac=0.25,
                         embargo_days=7)
        )

        if len(X_train) < 80 or len(X_test) < 20:
            return None, None, None, 0.5

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / max(pos, 1)

        # Build ensemble
        estimators = []

        if LGBM_AVAILABLE:
            lgbm = get_lgbm_model(spw)
            lgbm.fit(X_train_s, y_train)
            cal_lgbm = CalibratedClassifierCV(
                lgbm, cv="prefit", method="isotonic"
            )
            cal_lgbm.fit(X_test_s, y_test)
            estimators.append(("lgbm", cal_lgbm))

        xgb = get_xgb_model(spw)
        xgb.fit(X_train_s, y_train)
        cal_xgb = CalibratedClassifierCV(
            xgb, cv="prefit", method="isotonic"
        )
        cal_xgb.fit(X_test_s, y_test)
        estimators.append(("xgb", cal_xgb))

        if len(estimators) == 1:
            final_model = estimators[0][1]
        else:
            final_model = VotingClassifier(
                estimators=estimators,
                voting="soft",
                weights=[0.6, 0.4]
                if LGBM_AVAILABLE else [1]
            )
            final_model.fit(X_train_s, y_train)

        acc = accuracy_score(
            y_test,
            final_model.predict(X_test_s)
        )
        prec = precision_score(
            y_test,
            final_model.predict(X_test_s),
            zero_division=0
        )

        # Return accuracy weighted by precision
        combined = round(
            float((acc + prec) / 2), 3
        )
        return (
            final_model, scaler, features, combined
        )
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

        if regime == "bear":
            if rsi > 62:
                return 3
            elif rsi > 52:
                return 4
            else:
                return 5

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

        if atr_pct > 0.03:
            atr_adj = -2
        elif atr_pct > 0.02:
            atr_adj = -1
        elif atr_pct < 0.01:
            atr_adj = 2
        else:
            atr_adj = 0

        trend_count = sum([
            price > sma20,
            price > sma50,
            price > sma200
        ])
        trend_adj = 1 if trend_count == 3 else (
            0 if trend_count == 2 else -1
        )

        macd_adj = (
            1 if macd > 0 and macd_hist > 0
            else -1 if macd < 0
            else 0
        )

        vol_adj = (
            1 if vol_ratio >= 1.5
            else -1 if vol_ratio < 0.8
            else 0
        )

        regime_adj = (
            2 if regime == "bull"
            else 0 if regime == "sideways"
            else -1
        )

        rs_adj = (
            1 if rs and rs >= 5
            else -1 if rs and rs < -5
            else 0
        )

        conf_adj = (
            1 if confidence >= 0.85
            else -1 if confidence < 0.65
            else 0
        )

        total = (
            rsi_days + atr_adj + trend_adj +
            macd_adj + vol_adj + regime_adj +
            rs_adj + conf_adj
        )
        return max(3, min(14, total))

    except Exception:
        return 6