import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
import requests
from datetime import datetime, timedelta, date
from universe import SECTOR_INDICES

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="2y"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df is None or df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_weekly_data(ticker):
    try:
        df = yf.Ticker(ticker).history(
            period="2y", interval="1wk"
        )
        if df is None or df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_nifty_data():
    for ticker in ["^NSEI", "NIFTYBEES.NS", "HDFCBANK.NS"]:
        try:
            df = yf.Ticker(ticker).history(period="5y")
            if df is not None and not df.empty and len(df) > 50:
                df.dropna(inplace=True)
                return df
        except Exception:
            continue
    return None

@st.cache_data(ttl=3600)
def get_india_vix():
    try:
        df = yf.Ticker("^INDIAVIX").history(period="2y")
        if df is None or df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600)
def get_sector_data(sector_ticker):
    try:
        df = yf.Ticker(sector_ticker).history(period="3mo")
        if df is None or df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except Exception:
        return None

@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            "PE Ratio": info.get("trailingPE", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52W Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Beta": info.get("beta", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Revenue Growth": info.get("revenueGrowth", "N/A"),
            "Debt to Equity": info.get("debtToEquity", "N/A"),
            "ROE": info.get("returnOnEquity", "N/A"),
            "Promoter Holding": info.get(
                "heldPercentInsiders", "N/A"
            ),
        }
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def get_nse_delivery_data(ticker):
    """
    Fetch delivery % from NSE bhavcopy.
    Returns recent daily delivery percentage (0-1).
    """
    try:
        symbol = ticker.replace(".NS", "").upper()
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,application/csv",
            "Referer": "https://www.nseindia.com"
        }
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers=headers, timeout=8
        )
        delivery_pcts = []
        today = date.today()
        for days_back in range(1, 8):
            dt = today - timedelta(days=days_back)
            if dt.weekday() >= 5:
                continue
            date_str = dt.strftime("%d%m%Y")
            url = (
                f"https://nsearchives.nseindia.com"
                f"/products/content/"
                f"sec_bhavdata_full_{date_str}.csv"
            )
            try:
                resp = session.get(
                    url, headers=headers,
                    timeout=8
                )
                if resp.status_code != 200:
                    continue
                import io
                df_bhav = pd.read_csv(
                    io.StringIO(resp.text)
                )
                df_bhav.columns = (
                    df_bhav.columns.str.strip()
                )
                row = df_bhav[
                    df_bhav["SYMBOL"].str.strip()
                    == symbol
                ]
                if row.empty:
                    continue
                cols = df_bhav.columns.tolist()
                deliv_col = next(
                    (c for c in cols
                     if "DELIV" in c.upper()
                     and "QTY" in c.upper()), None
                )
                traded_col = next(
                    (c for c in cols
                     if "TTL" in c.upper()
                     or "TRDQTY" in c.upper()
                     or "TRADED" in c.upper()), None
                )
                if deliv_col and traded_col:
                    d = float(
                        str(row[deliv_col].iloc[0]
                            ).replace(",", "")
                    )
                    t = float(
                        str(row[traded_col].iloc[0]
                            ).replace(",", "")
                    )
                    if t > 0:
                        delivery_pcts.append(d / t)
                if len(delivery_pcts) >= 3:
                    break
            except Exception:
                continue
        if delivery_pcts:
            return round(
                sum(delivery_pcts) / len(delivery_pcts),
                4
            )
        return None
    except Exception:
        return None

def add_indicators(df):
    if df is None or len(df) < 50:
        return None
    df = df.copy()
    try:
        # Core momentum
        df["RSI"] = ta.momentum.RSIIndicator(
            df["Close"]
        ).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
        df["Stoch"] = ta.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"]
        ).stoch()
        df["Williams_R"] = (
            ta.momentum.WilliamsRIndicator(
                df["High"], df["Low"], df["Close"]
            ).williams_r()
        )
        df["CCI"] = ta.trend.CCIIndicator(
            df["High"], df["Low"], df["Close"]
        ).cci()
        df["ROC"] = ta.momentum.ROCIndicator(
            df["Close"]
        ).roc()

        # Trend
        df["SMA_20"] = ta.trend.SMAIndicator(
            df["Close"], window=20
        ).sma_indicator()
        df["SMA_50"] = ta.trend.SMAIndicator(
            df["Close"], window=50
        ).sma_indicator()
        df["SMA_200"] = ta.trend.SMAIndicator(
            df["Close"], window=200
        ).sma_indicator()
        df["EMA_12"] = ta.trend.EMAIndicator(
            df["Close"], window=12
        ).ema_indicator()
        df["EMA_26"] = ta.trend.EMAIndicator(
            df["Close"], window=26
        ).ema_indicator()

        # Volatility
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()
        df["BB_width"] = bb.bollinger_wband()
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range()

        # Volume
        df["Volume_SMA"] = df["Volume"].rolling(
            window=20
        ).mean()
        df["Volume_ratio"] = (
            df["Volume"] / df["Volume_SMA"]
        )
        df["Volume_surge_3d"] = (
            df["Volume"].rolling(3).mean() /
            df["Volume"].rolling(20).mean()
        )

        # Derived price features
        df["Return"] = df["Close"].pct_change()
        df["Return_5d"] = df["Close"].pct_change(5)
        df["Return_20d"] = df["Close"].pct_change(20)
        df["SMA_cross"] = (
            (df["SMA_20"] - df["SMA_50"]) /
            df["SMA_50"]
        )
        df["Price_to_SMA20"] = (
            (df["Close"] - df["SMA_20"]) /
            df["SMA_20"]
        )
        df["Price_to_SMA200"] = (
            (df["Close"] - df["SMA_200"]) /
            df["SMA_200"]
        )
        df["Volatility"] = df["Return"].rolling(
            window=20
        ).std()

        # NEW: Normalised ATR
        df["NATR"] = (df["ATR"] / df["Close"]) * 100

        # NEW: Gap percentage (overnight gap)
        df["Gap_pct"] = (
            (df["Open"] - df["Close"].shift(1)) /
            df["Close"].shift(1)
        )

        # NEW: Price position in 52W range (0=low, 1=high)
        high_52w = df["High"].rolling(252, min_periods=50).max()
        low_52w = df["Low"].rolling(252, min_periods=50).min()
        df["Price_pos_52w"] = (
            (df["Close"] - low_52w) /
            (high_52w - low_52w + 1e-9)
        )

        # NEW: Bullish divergence
        # Price lower low but RSI higher low over 5 days
        df["Bullish_div"] = (
            (df["Close"] < df["Close"].shift(5)) &
            (df["RSI"] > df["RSI"].shift(5))
        ).astype(int)

        # NEW: ATR percentile (ATR vs its own 252-day history)
        df["ATR_pct_rank"] = df["ATR"].rolling(
            252, min_periods=50
        ).rank(pct=True)

        # NEW: F&O expiry week flag (last Thursday of month)
        try:
            idx = pd.DatetimeIndex(df.index)
            # Week 4 = days 22-31 of month
            df["Expiry_week"] = (
                idx.day >= 22
            ).astype(int)
        except Exception:
            df["Expiry_week"] = 0

        df.dropna(inplace=True)
        return df
    except Exception:
        return None

def add_market_features(df, nifty_df=None,
                        vix_df=None):
    """
    Add features requiring external market data.
    Call after add_indicators.
    """
    if df is None:
        return df
    df = df.copy()
    try:
        # Rolling beta vs Nifty (20-day)
        if nifty_df is not None:
            stock_ret = df["Close"].pct_change()
            nifty_ret = nifty_df["Close"].pct_change()
            aligned = pd.concat(
                [stock_ret, nifty_ret], axis=1
            ).dropna()
            aligned.columns = ["stock", "nifty"]
            window = 20
            cov = aligned["stock"].rolling(
                window
            ).cov(aligned["nifty"])
            var = aligned["nifty"].rolling(
                window
            ).var()
            beta = (cov / (var + 1e-10)).reindex(
                df.index
            )
            df["Beta_20d"] = beta.fillna(1.0)

            # Relative return vs Nifty (5-day)
            nifty_ret_5d = nifty_df[
                "Close"
            ].pct_change(5).reindex(df.index)
            stock_ret_5d = df["Close"].pct_change(5)
            df["Rel_return_5d"] = (
                stock_ret_5d - nifty_ret_5d
            ).fillna(0)

            # Nifty 20-day momentum
            nifty_mom = nifty_df[
                "Close"
            ].pct_change(20).reindex(df.index)
            df["Nifty_momentum"] = nifty_mom.fillna(0)

        # India VIX ratio
        if vix_df is not None and len(vix_df) >= 200:
            vix_sma200 = vix_df[
                "Close"
            ].rolling(200).mean()
            vix_ratio = (
                vix_df["Close"] / (vix_sma200 + 1e-9)
            ).reindex(df.index)
            df["VIX_ratio"] = vix_ratio.fillna(1.0)

    except Exception:
        pass
    return df

def get_support_resistance(df, window=20):
    try:
        highs = df["High"].rolling(
            window=window, center=True
        ).max()
        lows = df["Low"].rolling(
            window=window, center=True
        ).min()
        resistance = df["High"][
            df["High"] == highs
        ].dropna()
        support = df["Low"][
            df["Low"] == lows
        ].dropna()
        current_price = df["Close"].iloc[-1]
        resistance_levels = sorted(
            [float(r) for r in resistance.values
             if float(r) > current_price]
        )[:3]
        support_levels = sorted(
            [float(s) for s in support.values
             if float(s) < current_price],
            reverse=True
        )[:3]
        return support_levels, resistance_levels
    except Exception:
        return [], []

def get_nifty_correlation(stock_df, nifty_df):
    try:
        if stock_df is None or nifty_df is None:
            return None
        stock_ret = stock_df[
            "Close"
        ].pct_change().dropna()
        nifty_ret = nifty_df[
            "Close"
        ].pct_change().dropna()
        aligned = pd.concat(
            [stock_ret, nifty_ret], axis=1
        ).dropna()
        aligned.columns = ["stock", "nifty"]
        if len(aligned) < 10:
            return None
        return round(
            float(aligned["stock"].corr(
                aligned["nifty"]
            )), 3
        )
    except Exception:
        return None

def get_relative_strength(stock_df, nifty_df,
                          period=20):
    try:
        if stock_df is None or nifty_df is None:
            return None
        if (len(stock_df) < period or
                len(nifty_df) < period):
            return None
        stock_return = (
            float(stock_df["Close"].iloc[-1]) /
            float(stock_df["Close"].iloc[-period]) - 1
        ) * 100
        nifty_return = (
            float(nifty_df["Close"].iloc[-1]) /
            float(nifty_df["Close"].iloc[-period]) - 1
        ) * 100
        return round(stock_return - nifty_return, 2)
    except Exception:
        return None

def get_market_regime(nifty_df):
    try:
        if nifty_df is None or nifty_df.empty:
            return "unknown"
        nifty = add_indicators(nifty_df)
        if nifty is None or nifty.empty:
            return "unknown"
        price = float(nifty["Close"].iloc[-1])
        sma50 = float(nifty["SMA_50"].iloc[-1])
        sma200 = float(nifty["SMA_200"].iloc[-1])
        rsi = float(nifty["RSI"].iloc[-1])
        macd = float(nifty["MACD"].iloc[-1])
        bull_signals = sum([
            price > sma50,
            price > sma200,
            sma50 > sma200,
            rsi > 50,
            macd > 0
        ])
        if bull_signals >= 4:
            return "bull"
        elif bull_signals <= 1:
            return "bear"
        else:
            return "sideways"
    except Exception:
        return "unknown"

def get_sector_momentum(sector_ticker):
    try:
        df = get_sector_data(sector_ticker)
        if df is None or df.empty:
            return None
        month_return = (
            float(df["Close"].iloc[-1]) /
            float(df["Close"].iloc[-20]) - 1
        ) * 100
        return round(month_return, 2)
    except Exception:
        return None

def get_fibonacci_levels(df):
    try:
        high = float(df["High"].tail(60).max())
        low = float(df["Low"].tail(60).min())
        diff = high - low
        return {
            "0%": round(low, 2),
            "23.6%": round(low + 0.236 * diff, 2),
            "38.2%": round(low + 0.382 * diff, 2),
            "50%": round(low + 0.5 * diff, 2),
            "61.8%": round(low + 0.618 * diff, 2),
            "100%": round(high, 2)
        }
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def validate_ticker(ticker):
    try:
        ticker = ticker.upper().strip()
        if (not ticker.endswith(".NS") and
                not ticker.endswith("=F")):
            ticker = ticker + ".NS"
        df = yf.Ticker(ticker).history(period="5d")
        return ticker if not df.empty else None
    except Exception:
        return None

def get_india_vix_ratio():
    try:
        vix_df = get_india_vix()
        if vix_df is None or len(vix_df) < 200:
            return 1.0
        current = float(vix_df["Close"].iloc[-1])
        sma200 = float(
            vix_df["Close"].rolling(200).mean().iloc[-1]
        )
        return round(current / (sma200 + 1e-9), 3)
    except Exception:
        return 1.0