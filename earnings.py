import requests
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta, date

@st.cache_data(ttl=86400)
def get_nse_earnings_calendar():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com"
        }
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers=headers, timeout=10
        )
        url = "https://www.nseindia.com/api/event-calendar"
        response = session.get(
            url, headers=headers, timeout=10
        )
        if response.status_code != 200:
            return {}
        data = response.json()
        earnings_map = {}
        for event in data:
            symbol = event.get("symbol", "")
            purpose = event.get("purpose", "").lower()
            date_str = event.get("date", "")
            if not symbol or not date_str:
                continue
            is_results = any(word in purpose for word in [
                "quarterly", "results", "financial",
                "q1", "q2", "q3", "q4",
                "half year", "annual"
            ])
            if is_results:
                try:
                    event_date = datetime.strptime(
                        date_str, "%d-%b-%Y"
                    ).date()
                    ticker = f"{symbol}.NS"
                    if ticker not in earnings_map:
                        earnings_map[ticker] = event_date
                except Exception:
                    continue
        return earnings_map
    except Exception:
        return {}

@st.cache_data(ttl=86400)
def get_dividend_exdates():
    """
    Fetch upcoming dividend ex-dates from NSE
    and yfinance for all tracked stocks.
    Returns dict of {ticker: (ex_date, dividend_amount)}
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com"
        }
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers=headers, timeout=10
        )
        url = "https://www.nseindia.com/api/event-calendar"
        response = session.get(
            url, headers=headers, timeout=10
        )
        dividend_map = {}
        if response.status_code == 200:
            data = response.json()
            for event in data:
                symbol = event.get("symbol", "")
                purpose = event.get("purpose", "").lower()
                date_str = event.get("date", "")
                if not symbol or not date_str:
                    continue
                is_dividend = any(w in purpose for w in [
                    "dividend", "div", "interim div",
                    "final div", "special div"
                ])
                if is_dividend:
                    try:
                        event_date = datetime.strptime(
                            date_str, "%d-%b-%Y"
                        ).date()
                        ticker = f"{symbol}.NS"
                        dividend_map[ticker] = (
                            event_date, purpose
                        )
                    except Exception:
                        continue
        return dividend_map
    except Exception:
        return {}

def get_dividend_status_yfinance(ticker):
    """
    Fallback: use yfinance to check upcoming
    dividend ex-dates for a specific ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        ex_date_ts = info.get("exDividendDate")
        dividend_amt = info.get("lastDividendValue", 0)
        if not ex_date_ts:
            return None, None
        ex_date = date.fromtimestamp(ex_date_ts)
        today = date.today()
        days_to_ex = (ex_date - today).days
        if -2 <= days_to_ex <= 14:
            return ex_date, dividend_amt
        return None, None
    except Exception:
        return None, None

def get_days_to_earnings(ticker, earnings_map=None):
    if earnings_map is None:
        earnings_map = get_nse_earnings_calendar()
    if ticker not in earnings_map:
        return None
    today = datetime.now().date()
    earnings_date = earnings_map[ticker]
    days_left = (earnings_date - today).days
    if -2 <= days_left <= 30:
        return days_left
    return None

def get_earnings_status(ticker, earnings_map=None):
    days = get_days_to_earnings(ticker, earnings_map)
    if days is None:
        return {
            "has_upcoming": False,
            "days_to_earnings": None,
            "risk_level": "none",
            "flag": False,
            "message": ""
        }
    if days < 0:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "recent",
            "flag": False,
            "message": (
                f"Results announced {abs(days)} days ago"
            )
        }
    elif days <= 3:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "high",
            "flag": True,
            "message": (
                f"⚠️ Results in {days} days — HIGH RISK"
            )
        }
    elif days <= 7:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "medium",
            "flag": True,
            "message": (
                f"⚠️ Results in {days} days — "
                f"ELEVATED RISK"
            )
        }
    else:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "low",
            "flag": False,
            "message": f"Results in {days} days"
        }

def get_dividend_risk(ticker, dividend_map=None):
    """
    Check if a stock has a dividend ex-date
    within the next 10 days.
    An ex-date causes automatic price drop
    equal to dividend amount.
    Returns risk assessment dict.
    """
    today = date.today()

    # First try NSE calendar
    if dividend_map and ticker in dividend_map:
        ex_date, purpose = dividend_map[ticker]
        days_to_ex = (ex_date - today).days
        if 0 <= days_to_ex <= 7:
            return {
                "has_dividend": True,
                "ex_date": ex_date,
                "days_to_ex": days_to_ex,
                "risk_level": "high",
                "flag": True,
                "message": (
                    f"🚨 DIVIDEND EX-DATE in "
                    f"{days_to_ex} days ({ex_date}) — "
                    f"Stock will drop by dividend amount. "
                    f"DO NOT ENTER."
                )
            }
        elif 8 <= days_to_ex <= 14:
            return {
                "has_dividend": True,
                "ex_date": ex_date,
                "days_to_ex": days_to_ex,
                "risk_level": "medium",
                "flag": False,
                "message": (
                    f"⚠️ Dividend ex-date in "
                    f"{days_to_ex} days ({ex_date}). "
                    f"Plan exit before then."
                )
            }

    # Fallback to yfinance
    ex_date, div_amt = get_dividend_status_yfinance(ticker)
    if ex_date:
        days_to_ex = (ex_date - today).days
        if 0 <= days_to_ex <= 7:
            return {
                "has_dividend": True,
                "ex_date": ex_date,
                "days_to_ex": days_to_ex,
                "risk_level": "high",
                "flag": True,
                "message": (
                    f"🚨 DIVIDEND EX-DATE in "
                    f"{days_to_ex} days ({ex_date}). "
                    f"Expected drop: ₹{div_amt:.2f}. "
                    f"DO NOT ENTER."
                )
            }
        elif 8 <= days_to_ex <= 14:
            return {
                "has_dividend": True,
                "ex_date": ex_date,
                "days_to_ex": days_to_ex,
                "risk_level": "medium",
                "flag": False,
                "message": (
                    f"⚠️ Dividend ex-date in "
                    f"{days_to_ex} days. "
                    f"Plan exit before then."
                )
            }

    return {
        "has_dividend": False,
        "ex_date": None,
        "days_to_ex": None,
        "risk_level": "none",
        "flag": False,
        "message": ""
    }

@st.cache_data(ttl=86400)
def get_nse_fii_dii_flow():
    """
    Fetch latest FII/DII data from NSE.
    Returns net FII flow for last 5 days.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com"
        }
        session = requests.Session()
        session.get(
            "https://www.nseindia.com",
            headers=headers, timeout=10
        )
        url = (
            "https://www.nseindia.com/api/"
            "fiidiiTradeReact"
        )
        response = session.get(
            url, headers=headers, timeout=10
        )
        if response.status_code != 200:
            return None
        data = response.json()
        if not data:
            return None

        fii_flows = []
        for entry in data[:5]:
            try:
                net = float(
                    str(entry.get(
                        "fii_net_amount", "0"
                    )).replace(",", "")
                )
                fii_flows.append(net)
            except Exception:
                continue

        if not fii_flows:
            return None

        consecutive_selling = sum(
            1 for f in fii_flows if f < 0
        )
        total_net = sum(fii_flows)

        return {
            "flows": fii_flows,
            "consecutive_selling_days": consecutive_selling,
            "total_net_5d": round(total_net, 2),
            "sentiment": (
                "bearish"
                if consecutive_selling >= 3
                else "bullish"
                if consecutive_selling <= 1
                else "neutral"
            )
        }
    except Exception:
        return None

@st.cache_data(ttl=86400)
def get_batch_earnings_status(tickers):
    earnings_map = get_nse_earnings_calendar()
    results = {}
    for ticker in tickers:
        results[ticker] = get_earnings_status(
            ticker, earnings_map
        )
    return results