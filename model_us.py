import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import ta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
 
from data import get_stock_data, add_indicators
 
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False
 
# ── Label parameters (aligned with how trades are actually exited) ──
TB_HORIZON = 10          # max holding days
TB_TP_ATR = 2.2          # take-profit distance in ATRs
TB_SL_ATR = 1.1          # stop distance in ATRs  → 2:1 reward/risk
TB_MIN_GAIN = 0.03       # TP must clear 3% (covers slippage + costs)
 
CONFIDENCE_THRESHOLD = 0.62   # high-confidence cut used for quality metric
 
US_FEATURES = [
    # momentum / oscillators
    "RSI", "RSI_2", "MACD", "MACD_signal", "MACD_hist",
    "Stoch", "Williams_R", "CCI", "ROC", "MFI", "ADX",
    # trend
    "SMA_cross", "Price_to_SMA20", "Price_to_SMA200",
    "Dist_20d_high", "Mom_12_1", "Price_pos_52w",
    # volatility
    "BB_width", "NATR", "Volatility", "Down_vol", "ATR_pct_rank",
    # volume / flow
    "Volume_ratio", "Volume_surge_3d", "OBV_flow", "Dollar_vol_z",
    # price action
    "Return", "Return_5d", "Return_20d", "Gap_pct",
    "Close_loc", "Up_streak_3", "Bullish_div",
    # market context
    "Beta_60d", "Rel_ret_20d", "SPY_mom_20",
    "VIX_level", "VIX_term",
]
 
 
# ── Market context data ─────────────────────────────────────────────
 
@st.cache_data(ttl=3600)
def _get_history(ticker: str, period: str = "5y"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None
 
 
def get_spy_df():
    for t in ["SPY", "^GSPC", "IVV"]:
        df = _get_history(t)
        if df is not None and len(df) > 250:
            return df
    return None
 
 
def get_vix_df():
    return _get_history("^VIX", "5y")
 
 
def get_vix3m_df():
    return _get_history("^VIX3M", "5y")
 
 
# ── Feature engineering ─────────────────────────────────────────────
 
def add_us_features(df, spy_df=None, vix_df=None, vix3m_df=None):
    """Adds US-specific features on top of add_indicators output."""
    if df is None or len(df) < 60:
        return None
    df = df.copy()
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
 
    # Short-term mean reversion (Connors RSI-2)
    try:
        df["RSI_2"] = ta.momentum.RSIIndicator(close, window=2).rsi()
    except Exception:
        df["RSI_2"] = 50.0
 
    # Trend strength
    try:
        df["ADX"] = ta.trend.ADXIndicator(high, low, close).adx()
    except Exception:
        df["ADX"] = 20.0
 
    # Money Flow Index (volume-weighted momentum)
    try:
        df["MFI"] = ta.volume.MFIIndicator(high, low, close, vol).money_flow_index()
    except Exception:
        df["MFI"] = 50.0
 
    # OBV flow: 10d OBV change normalised by 10d total volume
    try:
        obv = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        df["OBV_flow"] = (obv - obv.shift(10)) / (vol.rolling(10).sum() + 1e-9)
    except Exception:
        df["OBV_flow"] = 0.0
 
    # Classic 12-1 month momentum (skip most recent month)
    df["Mom_12_1"] = close.pct_change(252) - close.pct_change(21)
 
    # Distance from 20-day high (breakout proximity)
    df["Dist_20d_high"] = close / (high.rolling(20).max() + 1e-9) - 1
 
    # Downside deviation (20d)
    ret = close.pct_change()
    df["Down_vol"] = ret.clip(upper=0).rolling(20).std()
 
    # Dollar-volume z-score (institutional activity)
    dv = close * vol
    df["Dollar_vol_z"] = (
        (dv.rolling(5).mean() - dv.rolling(60).mean())
        / (dv.rolling(60).std() + 1e-9)
    )
 
    # Close location within day's range
    df["Close_loc"] = (close - low) / (high - low + 1e-9)
 
    # Up-day streak over last 3 days
    df["Up_streak_3"] = (ret > 0).rolling(3).sum()
 
    # Market context — SPY
    if spy_df is not None:
        spy_ret = spy_df["Close"].pct_change()
        aligned = pd.concat([ret, spy_ret], axis=1).dropna()
        aligned.columns = ["s", "m"]
        cov = aligned["s"].rolling(60).cov(aligned["m"])
        var = aligned["m"].rolling(60).var()
        df["Beta_60d"] = (cov / (var + 1e-10)).reindex(df.index).fillna(1.0)
        spy_20 = spy_df["Close"].pct_change(20).reindex(df.index).ffill()
        df["Rel_ret_20d"] = (close.pct_change(20) - spy_20).fillna(0)
        df["SPY_mom_20"] = spy_20.fillna(0)
    else:
        df["Beta_60d"], df["Rel_ret_20d"], df["SPY_mom_20"] = 1.0, 0.0, 0.0
 
    # Market context — VIX level + term structure
    if vix_df is not None:
        vix = vix_df["Close"].reindex(df.index).ffill()
        df["VIX_level"] = (vix / 20.0).fillna(1.0)  # normalised around 20
        if vix3m_df is not None:
            vix3m = vix3m_df["Close"].reindex(df.index).ffill()
            df["VIX_term"] = (vix / (vix3m + 1e-9)).fillna(0.92)
        else:
            df["VIX_term"] = 0.92
    else:
        df["VIX_level"], df["VIX_term"] = 1.0, 0.92
 
    return df
 
 
@st.cache_data(ttl=3600)
def prepare_us_frame(ticker: str, period: str = "3y"):
    """Full enriched frame: OHLCV + indicators + US features. Single source
    of truth used by both training and the screener (no train/serve skew)."""
    df = get_stock_data(ticker, period=period)
    if df is None or len(df) < 120:
        return None
    df = add_indicators(df)
    if df is None or len(df) < 100:
        return None
    df = add_us_features(df, get_spy_df(), get_vix_df(), get_vix3m_df())
    return df
 
 
# ── Labels ───────────────────────────────────────────────────────────
 
def create_tb_target_us(df, horizon=TB_HORIZON, tp_atr=TB_TP_ATR,
                        sl_atr=TB_SL_ATR, min_gain=TB_MIN_GAIN):
    """Triple barrier: 1 if take-profit hit before stop within horizon."""
    n = len(df)
    labels = np.zeros(n, dtype=int)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    atr = df["ATR"].values
    for t in range(n - horizon):
        entry = close[t]
        tp = max(entry + tp_atr * atr[t], entry * (1 + min_gain))
        sl = entry - sl_atr * atr[t]
        ph = high[t + 1:t + 1 + horizon]
        pl = low[t + 1:t + 1 + horizon]
        tp_hits = np.where(ph >= tp)[0]
        sl_hits = np.where(pl <= sl)[0]
        first_tp = tp_hits[0] if len(tp_hits) else horizon + 1
        first_sl = sl_hits[0] if len(sl_hits) else horizon + 1
        if first_tp < first_sl:
            labels[t] = 1
    return labels
 
 
def build_us_features(df):
    df = df.copy()
    df["Target"] = create_tb_target_us(df)
    cols = [c for c in US_FEATURES if c in df.columns]
    if not cols:
        return None, None, None
    sub = df[cols + ["Target"]].dropna()
    # drop last horizon rows — labels there are unresolved
    sub = sub.iloc[:-TB_HORIZON]
    if len(sub) < 150:
        return None, None, None
    return sub[cols], sub["Target"], cols
 
 
# ── Models ───────────────────────────────────────────────────────────
 
def _lgbm(spw):
    return lgb.LGBMClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        num_leaves=15, min_data_in_leaf=20,
        feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=5,
        reg_lambda=1.0, scale_pos_weight=spw,
        n_jobs=-1, verbose=-1, random_state=42,
    )
 
 
def _xgb(spw):
    return XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6,
        reg_lambda=1.0, min_child_weight=5,
        scale_pos_weight=spw, eval_metric="logloss",
        verbosity=0, random_state=42, n_jobs=-1,
    )
 
 
def _time_decay_weights(n, half_life=252):
    age = np.arange(n)[::-1]  # most recent = 0
    return np.power(0.5, age / half_life)
 
 
def _walk_forward_quality(X, y, n_folds=3, embargo=10):
    """Expanding-window walk-forward. Quality = precision of picks with
    calibrated p > threshold, averaged over folds, relative to base rate.
    Returns value in [0, 1]; 0.5 ≈ no edge over base rate."""
    n = len(X)
    precisions, base_rates = [], []
    fold_ends = np.linspace(0.55, 0.85, n_folds)
    for fe in fold_ends:
        tr_end = int(n * fe)
        te_start = min(tr_end + embargo, n)
        te_end = min(te_start + int(n * 0.15), n)
        if te_end - te_start < 20 or tr_end < 100:
            continue
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = X.iloc[te_start:te_end], y.iloc[te_start:te_end]
        if ytr.sum() < 10 or ytr.nunique() < 2:
            continue
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        mdl = _lgbm(spw) if LGBM_AVAILABLE else _xgb(spw)
        w = _time_decay_weights(len(ytr))
        try:
            mdl.fit(Xtr, ytr, sample_weight=w)
            proba = mdl.predict_proba(Xte)[:, 1]
        except Exception:
            continue
        picks = proba >= np.quantile(proba, 0.8)  # top-quintile signals
        if picks.sum() < 5:
            continue
        precisions.append(float(yte[picks].mean()))
        base_rates.append(float(yte.mean()))
    if not precisions:
        return 0.5
    lift = np.mean(precisions) - np.mean(base_rates)
    # map: no lift → 0.5, +20pp precision lift → ~0.9
    return float(np.clip(0.5 + 2.0 * lift, 0.0, 1.0))
 
 
@st.cache_resource
def train_model_us(ticker: str):
    """
    Returns (bundle, features, quality).
    bundle: list of calibrated classifiers (probabilities averaged).
    quality: walk-forward precision lift score in [0,1] (0.5 = no edge).
    """
    try:
        df = prepare_us_frame(ticker)
        if df is None:
            return None, None, 0.5
        X, y, features = build_us_features(df)
        if X is None or y.sum() < 15 or y.nunique() < 2:
            return None, None, 0.5
 
        quality = _walk_forward_quality(X, y)
 
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        w = _time_decay_weights(len(y))
        tscv = TimeSeriesSplit(n_splits=3)
 
        bundle = []
        base_models = []
        if LGBM_AVAILABLE:
            base_models.append(_lgbm(spw))
        base_models.append(_xgb(spw))
        for bm in base_models:
            try:
                cal = CalibratedClassifierCV(bm, method="sigmoid", cv=tscv)
                cal.fit(X, y, sample_weight=w)
                bundle.append(cal)
            except Exception:
                continue
        if not bundle:
            return None, None, 0.5
        return bundle, features, round(quality, 3)
    except Exception:
        return None, None, 0.5
 
 
def get_signal_us(bundle, df, features):
    """Returns (signal, confidence, buy_prob, sell_prob) from latest bar."""
    if bundle is None or features is None:
        return "NEUTRAL", 0.5, 0.5, 0.5
    try:
        avail = [f for f in features if f in df.columns]
        latest = df[avail].dropna().iloc[-1:]
        if latest.empty or len(avail) != len(features):
            return "NEUTRAL", 0.5, 0.5, 0.5
        probs = [m.predict_proba(latest)[0][1] for m in bundle]
        buy_prob = float(np.mean(probs))
        signal = "BUY" if buy_prob >= 0.5 else "SELL"
        confidence = buy_prob if signal == "BUY" else 1 - buy_prob
        return signal, confidence, buy_prob, 1 - buy_prob
    except Exception:
        return "NEUTRAL", 0.5, 0.5, 0.5