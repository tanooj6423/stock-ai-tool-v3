"""
Kite Connect data layer (licensed NSE data).

Design:
- The app OWNER's Kite session powers all internal data
  fetching (historical candles for indicators, models,
  scans). One ₹500/mo subscription serves the whole app.
- Users are only ever shown DERIVED analytics computed
  from this data, never a re-published raw feed.
- Every public fetcher in data.py calls try_kite_history()
  first and silently falls back to yfinance when Kite is
  not configured/connected, so the app always works.

Owner token flow (Kite tokens expire daily by design):
- Log in once a day through the Zerodha panel in the app.
  broker.generate_session() persists the token via
  save_owner_token(); after that, all data flows through
  Kite until the token expires.
"""

import json
import os
from datetime import date, datetime, timedelta

import pandas as pd

from config import DATA_DIR, get_secret

try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

TOKEN_FILE = str(DATA_DIR / "kite_owner_token.json")
INSTRUMENTS_FILE = str(DATA_DIR / "kite_instruments_nse.csv")

# Yahoo-style index tickers -> official NSE index names
# (as they appear in the Kite instruments dump)
INDEX_NAME_MAP = {
    "^NSEI": "NIFTY 50",
    "^INDIAVIX": "INDIA VIX",
    "^NSEBANK": "NIFTY BANK",
    "^CNXIT": "NIFTY IT",
    "^CNXAUTO": "NIFTY AUTO",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXENERGY": "NIFTY ENERGY",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXFINANCE": "NIFTY FINANCIAL SERVICES",
    "^CNXCONSUMER": "NIFTY INDIA CONSUMPTION",
    "^CNXINFRA": "NIFTY INFRASTRUCTURE",
    "^CNXTELECOM": "NIFTY MEDIA",
    "^CNXREALTY": "NIFTY REALTY",
    "^CNXPSUBANK": "NIFTY PSU BANK",
}

_PERIOD_DAYS = {
    "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 365, "2y": 730, "5y": 1825,
    "10y": 2000,  # Kite day-candle request cap
}


# ---------------------------------------------------------
# Owner token persistence
# ---------------------------------------------------------
def save_owner_token(access_token):
    """Persist today's access token (called on login)."""
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({
                "access_token": access_token,
                "date": str(date.today()),
            }, f)
        return True
    except Exception:
        return False


def load_owner_token():
    """Return today's token, or None if absent/expired."""
    try:
        if not os.path.exists(TOKEN_FILE):
            return None
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        if data.get("date") != str(date.today()):
            return None  # Kite tokens expire daily
        return data.get("access_token") or None
    except Exception:
        return None


def get_data_client():
    """
    KiteConnect client authenticated with the owner's
    token, or None if unavailable. Never raises.
    """
    if not KITE_AVAILABLE:
        return None
    api_key = get_secret("ZERODHA_API_KEY")
    token = load_owner_token()
    if not api_key or not token:
        return None
    try:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(token)
        return kite
    except Exception:
        return None


def is_kite_data_active():
    return get_data_client() is not None


# ---------------------------------------------------------
# Instrument lookup
# ---------------------------------------------------------
def _load_instruments(kite):
    """
    NSE instruments dump, cached on disk for the day
    (it's a ~1 MB download refreshed daily by Kite).
    """
    try:
        if os.path.exists(INSTRUMENTS_FILE):
            mtime = datetime.fromtimestamp(
                os.path.getmtime(INSTRUMENTS_FILE)
            )
            if mtime.date() == date.today():
                return pd.read_csv(INSTRUMENTS_FILE)
        rows = kite.instruments("NSE")
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df.to_csv(INSTRUMENTS_FILE, index=False)
        return df
    except Exception:
        try:
            if os.path.exists(INSTRUMENTS_FILE):
                return pd.read_csv(INSTRUMENTS_FILE)
        except Exception:
            pass
        return None


def _resolve_token(kite, ticker):
    """Yahoo-style ticker -> Kite instrument_token."""
    inst = _load_instruments(kite)
    if inst is None or inst.empty:
        return None
    try:
        if ticker.startswith("^"):
            name = INDEX_NAME_MAP.get(ticker)
            if not name:
                return None
            row = inst[
                (inst["segment"] == "INDICES") &
                (inst["tradingsymbol"].str.upper()
                 == name.upper())
            ]
        else:
            symbol = ticker.replace(".NS", "").upper()
            row = inst[
                (inst["tradingsymbol"] == symbol) &
                (inst["instrument_type"] == "EQ")
            ]
        if row.empty:
            return None
        return int(row.iloc[0]["instrument_token"])
    except Exception:
        return None


# ---------------------------------------------------------
# History fetch (drop-in replacement for yfinance frames)
# ---------------------------------------------------------
def try_kite_history(ticker, period="2y", interval="day"):
    """
    Returns a DataFrame indexed by date with columns
    Open/High/Low/Close/Volume (matching yfinance), or
    None if Kite is unavailable — callers then fall back
    to yfinance. Never raises.
    """
    kite = get_data_client()
    if kite is None:
        return None
    try:
        token = _resolve_token(kite, ticker)
        if token is None:
            return None

        days = min(
            _PERIOD_DAYS.get(period, 730), 2000
        )
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)

        candles = kite.historical_data(
            token, from_dt, to_dt, interval
        )
        if not candles:
            return None

        df = pd.DataFrame(candles)
        df["date"] = pd.to_datetime(df["date"])
        # Match yfinance: naive datetime index
        try:
            df["date"] = df["date"].dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        df = df.set_index("date").rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close",
            "volume": "Volume",
        })
        cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.dropna()
        if df.empty:
            return None
        return df
    except Exception:
        return None


def try_kite_weekly(ticker, period="2y"):
    """Weekly candles via resample of daily data."""
    df = try_kite_history(ticker, period=period)
    if df is None or df.empty:
        return None
    try:
        wk = df.resample("W").agg({
            "Open": "first", "High": "max",
            "Low": "min", "Close": "last",
            "Volume": "sum",
        }).dropna()
        return wk if not wk.empty else None
    except Exception:
        return None
