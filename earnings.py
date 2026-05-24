import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

@st.cache_data(ttl=86400)
def get_nse_earnings_calendar():
    """
    Fetches upcoming earnings/results dates from NSE.
    Returns a dict of {ticker: results_date}
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com"
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        url = "https://www.nseindia.com/api/event-calendar"
        response = session.get(url, headers=headers, timeout=10)

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
                "quarterly", "results", "financial", "dividend",
                "q1", "q2", "q3", "q4", "half year", "annual"
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

def get_days_to_earnings(ticker, earnings_map=None):
    """
    Returns number of days until next earnings for a ticker.
    Returns None if no upcoming earnings found.
    """
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
    """
    Returns a dict with earnings risk assessment.
    """
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
            "message": f"Results announced {abs(days)} days ago"
        }
    elif days <= 3:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "high",
            "flag": True,
            "message": f"⚠️ Results in {days} days — HIGH RISK"
        }
    elif days <= 7:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "medium",
            "flag": True,
            "message": f"⚠️ Results in {days} days — ELEVATED RISK"
        }
    else:
        return {
            "has_upcoming": True,
            "days_to_earnings": days,
            "risk_level": "low",
            "flag": False,
            "message": f"Results in {days} days"
        }

@st.cache_data(ttl=86400)
def get_batch_earnings_status(tickers):
    """
    Returns earnings status for multiple tickers efficiently.
    """
    earnings_map = get_nse_earnings_calendar()
    results = {}
    for ticker in tickers:
        results[ticker] = get_earnings_status(ticker, earnings_map)
    return results