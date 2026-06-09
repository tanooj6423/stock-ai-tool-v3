"""
broker_alpaca.py — Alpaca paper trading integration for Equitex Intelligence
Equivalent of broker.py but for US stocks via Alpaca Markets.

Setup:
  1. Create free account at alpaca.markets
  2. Go to Paper Trading → API Keys → Generate
  3. Add to HuggingFace Space secrets:
       ALPACA_API_KEY   = your paper trading key
       ALPACA_API_SECRET = your paper trading secret
  4. Base URL for paper trading: https://paper-api.alpaca.markets

alpaca-py install: pip install alpaca-py
"""

import os
import streamlit as st
from datetime import datetime, time as dt_time
import pytz
import yfinance as yf

# Lazy import so app doesn't crash if alpaca-py not installed
def _get_alpaca_clients():
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        return TradingClient, StockHistoricalDataClient, StockLatestQuoteRequest
    except ImportError:
        return None, None, None


def get_alpaca_keys():
    key, secret = None, None
    try:
        key = st.secrets.get("ALPACA_API_KEY", "")
        secret = st.secrets.get("ALPACA_API_SECRET", "")
    except Exception:
        pass
    if not key:
        key = os.getenv("ALPACA_API_KEY", "")
    if not secret:
        secret = os.getenv("ALPACA_API_SECRET", "")
    return key, secret


def is_alpaca_connected() -> bool:
    key, secret = get_alpaca_keys()
    return bool(key and secret and len(key) > 5)


def is_us_market_hours() -> bool:
    """Check if US market is open (9:30–16:00 ET, weekdays)."""
    et = pytz.timezone("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:  # Sat/Sun
        return False
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)
    return market_open <= now.time() <= market_close


@st.cache_data(ttl=60)
def get_us_live_quote(ticker: str) -> float | None:
    """
    Get live quote for a US stock.
    Uses Alpaca data API if connected; falls back to yfinance (15-min delayed).
    """
    if is_alpaca_connected() and is_us_market_hours():
        try:
            TradingClient, StockHistoricalDataClient, StockLatestQuoteRequest = (
                _get_alpaca_clients()
            )
            if StockHistoricalDataClient is None:
                raise ImportError("alpaca-py not installed")
            key, secret = get_alpaca_keys()
            data_client = StockHistoricalDataClient(key, secret)
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quotes = data_client.get_stock_latest_quote(request)
            q = quotes[ticker]
            mid = (q.ask_price + q.bid_price) / 2
            return round(float(mid), 2)
        except Exception:
            pass
    # Fallback: yfinance
    try:
        hist = yf.Ticker(ticker).history(period="2d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 4)
    except Exception:
        pass
    return None


def get_alpaca_account() -> dict | None:
    """Return Alpaca paper account details (cash, equity, P&L)."""
    if not is_alpaca_connected():
        return None
    try:
        TradingClient, _, _ = _get_alpaca_clients()
        if TradingClient is None:
            return None
        key, secret = get_alpaca_keys()
        client = TradingClient(key, secret, paper=True)
        acct = client.get_account()
        return {
            "equity": float(acct.equity),
            "cash": float(acct.cash),
            "buying_power": float(acct.buying_power),
            "pnl": float(acct.equity) - float(acct.last_equity),
            "pnl_pct": (
                (float(acct.equity) - float(acct.last_equity))
                / float(acct.last_equity) * 100
                if float(acct.last_equity) > 0 else 0
            ),
            "status": acct.status,
        }
    except Exception:
        return None


def get_alpaca_positions() -> list:
    """Return current Alpaca paper positions."""
    if not is_alpaca_connected():
        return []
    try:
        TradingClient, _, _ = _get_alpaca_clients()
        if TradingClient is None:
            return []
        key, secret = get_alpaca_keys()
        client = TradingClient(key, secret, paper=True)
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
                "side": p.side,
            }
            for p in positions
        ]
    except Exception:
        return []


def place_alpaca_bracket_order(
    ticker: str,
    qty: int,
    take_profit: float,
    stop_loss: float,
) -> dict:
    """
    Place a paper bracket order:
      - Market BUY entry (fills immediately at market price)
      - Take-profit limit order at target
      - Stop-loss stop order at stop level

    Returns dict with order_id or error message.
    """
    if not is_alpaca_connected():
        return {"success": False, "error": "Alpaca not connected"}
    if qty <= 0:
        return {"success": False, "error": "Quantity must be > 0"}
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import (
            MarketOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        key, secret = get_alpaca_keys()
        client = TradingClient(key, secret, paper=True)
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_loss, 2)),
        )
        order = client.submit_order(order_request)
        return {
            "success": True,
            "order_id": str(order.id),
            "status": str(order.status),
            "symbol": ticker,
            "qty": qty,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def cancel_alpaca_order(order_id: str) -> bool:
    """Cancel an open Alpaca paper order."""
    if not is_alpaca_connected():
        return False
    try:
        TradingClient, _, _ = _get_alpaca_clients()
        if TradingClient is None:
            return False
        key, secret = get_alpaca_keys()
        client = TradingClient(key, secret, paper=True)
        client.cancel_order_by_id(order_id)
        return True
    except Exception:
        return False


def render_alpaca_panel(theme: dict):
    """Render the Alpaca connection panel in Settings tab."""
    st.markdown("### 🇺🇸 Alpaca Paper Trading")
    if not is_alpaca_connected():
        st.markdown(
            f"""<div style='background:{theme['card']};border:1px solid {theme['border']};
            border-radius:8px;padding:16px;margin-bottom:12px;'>
            <b>Connect Alpaca for US paper trading</b><br>
            <small>1. Sign up free at <a href='https://alpaca.markets' target='_blank'>alpaca.markets</a><br>
            2. Go to Paper Trading → API Keys → Generate<br>
            3. Add <code>ALPACA_API_KEY</code> and <code>ALPACA_API_SECRET</code> to HuggingFace Space secrets</small>
            </div>""",
            unsafe_allow_html=True
        )
        # Manual key entry
        with st.expander("Enter keys manually (session only)"):
            k = st.text_input("Alpaca API Key", type="password", key="alp_key_input")
            s = st.text_input("Alpaca Secret Key", type="password", key="alp_sec_input")
            if st.button("Connect", key="alp_connect_btn"):
                if k and s:
                    st.session_state["alpaca_key"] = k
                    st.session_state["alpaca_secret"] = s
                    st.success("Keys saved for this session.")
                    st.rerun()
                else:
                    st.error("Both key and secret are required.")
    else:
        acct = get_alpaca_account()
        if acct:
            col1, col2, col3 = st.columns(3)
            col1.metric("Paper Equity", f"${acct['equity']:,.2f}")
            col2.metric("Cash", f"${acct['cash']:,.2f}")
            pnl_color = "normal" if acct["pnl"] >= 0 else "inverse"
            col3.metric(
                "Today P&L",
                f"${acct['pnl']:+,.2f}",
                f"{acct['pnl_pct']:+.2f}%",
                delta_color=pnl_color
            )
            st.markdown(
                f"<small style='color:{theme['text2']}'>Status: {acct['status']} · "
                f"Buying Power: ${acct['buying_power']:,.2f}</small>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Connected but couldn't fetch account data.")

        positions = get_alpaca_positions()
        if positions:
            st.markdown("**Open Positions**")
            import pandas as pd
            df_pos = pd.DataFrame(positions)[
                ["symbol", "qty", "avg_entry", "current_price",
                 "market_value", "unrealized_pnl", "unrealized_pnl_pct"]
            ]
            df_pos.columns = [
                "Symbol", "Qty", "Avg Entry", "Current",
                "Market Value", "Unrealized P&L", "P&L %"
            ]
            st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")