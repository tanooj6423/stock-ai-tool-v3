import yfinance as yf
import pandas as pd
import numpy as np
import ta
import streamlit as st
from universe import NIFTY_INDEX, SECTOR_INDICES

@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="2y"):
    try:
        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def get_weekly_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period="2y", interval="1wk")
        if df.empty:
            return None
        df.dropna(inplace=True)
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def get_nifty_data():
    try:
        df = yf.Ticker(NIFTY_INDEX).history(period="2y")
        df.dropna(inplace=True)
        return df
    except:
        return None

@st.cache_data(ttl=3600)
def get_sector_data(sector_ticker):
    try:
        df = yf.Ticker(sector_ticker).history(period="3mo")
        df.dropna(inplace=True)
        return df
    except:
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
            "Promoter Holding": info.get("heldPercentInsiders", "N/A"),
        }
    except:
        return {}

def add_indicators(df):
    if df is None or len(df) < 50:
        return None
    df = df.copy()
    try:
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
        df["SMA_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
        df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
        df["SMA_200"] = ta.trend.SMAIndicator(df["Close"], window=200).sma_indicator()
        df["EMA_12"] = ta.trend.EMAIndicator(df["Close"], window=12).ema_indicator()
        df["EMA_26"] = ta.trend.EMAIndicator(df["Close"], window=26).ema_indicator()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["BB_mid"] = bb.bollinger_mavg()
        df["BB_width"] = bb.bollinger_wband()
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]).average_true_range()
        df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_SMA"]
        df["ROC"] = ta.momentum.ROCIndicator(df["Close"]).roc()
        df["Stoch"] = ta.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"]).stoch()
        df["Williams_R"] = ta.momentum.WilliamsRIndicator(
            df["High"], df["Low"], df["Close"]).williams_r()
        df["CCI"] = ta.trend.CCIIndicator(
            df["High"], df["Low"], df["Close"]).cci()
        df["Return"] = df["Close"].pct_change()
        df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
        df["Price_to_SMA20"] = (df["Close"] - df["SMA_20"]) / df["SMA_20"]
        df["Price_to_SMA200"] = (df["Close"] - df["SMA_200"]) / df["SMA_200"]
        df["Volatility"] = df["Return"].rolling(window=20).std()
        df.dropna(inplace=True)
        return df
    except:
        return None

def get_support_resistance(df, window=20):
    try:
        highs = df["High"].rolling(window=window, center=True).max()
        lows = df["Low"].rolling(window=window, center=True).min()
        resistance = df["High"][df["High"] == highs].dropna()
        support = df["Low"][df["Low"] == lows].dropna()
        current_price = df["Close"].iloc[-1]
        resistance_levels = sorted(
            [r for r in resistance.values if r > current_price])[:3]
        support_levels = sorted(
            [s for s in support.values if s < current_price],
            reverse=True)[:3]
        return support_levels, resistance_levels
    except:
        return [], []

def get_nifty_correlation(stock_df, nifty_df):
    try:
        stock_returns = stock_df["Close"].pct_change().dropna()
        nifty_returns = nifty_df["Close"].pct_change().dropna()
        aligned = pd.concat(
            [stock_returns, nifty_returns], axis=1).dropna()
        aligned.columns = ["stock", "nifty"]
        return round(aligned["stock"].corr(aligned["nifty"]), 3)
    except:
        return None

def get_relative_strength(stock_df, nifty_df, period=20):
    try:
        stock_return = (stock_df["Close"].iloc[-1] /
                        stock_df["Close"].iloc[-period] - 1) * 100
        nifty_return = (nifty_df["Close"].iloc[-1] /
                        nifty_df["Close"].iloc[-period] - 1) * 100
        return round(stock_return - nifty_return, 2)
    except:
        return None

def get_market_regime(nifty_df):
    try:
        nifty = add_indicators(nifty_df)
        if nifty is None:
            return "unknown"
        price = nifty["Close"].iloc[-1]
        sma50 = nifty["SMA_50"].iloc[-1]
        sma200 = nifty["SMA_200"].iloc[-1]
        if price > sma50 and price > sma200:
            return "bull"
        elif price < sma50 and price < sma200:
            return "bear"
        else:
            return "sideways"
    except:
        return "unknown"

def get_sector_momentum(sector_ticker):
    try:
        df = get_sector_data(sector_ticker)
        if df is None or df.empty:
            return None
        month_return = (df["Close"].iloc[-1] /
                        df["Close"].iloc[-20] - 1) * 100
        return round(month_return, 2)
    except:
        return None

def get_fibonacci_levels(df):
    try:
        high = df["High"].tail(60).max()
        low = df["Low"].tail(60).min()
        diff = high - low
        levels = {
            "0%": low,
            "23.6%": low + 0.236 * diff,
            "38.2%": low + 0.382 * diff,
            "50%": low + 0.5 * diff,
            "61.8%": low + 0.618 * diff,
            "100%": high
        }
        return levels
    except:
        return {}

@st.cache_data(ttl=86400)
def validate_ticker(ticker):
    try:
        ticker = ticker.upper().strip()
        if not ticker.endswith(".NS") and not ticker.endswith("=F"):
            ticker = ticker + ".NS"
        df = yf.Ticker(ticker).history(period="5d")
        return ticker if not df.empty else None
    except:
        return None