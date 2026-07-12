import os
import streamlit as st
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

def get_api_key():
    try:
        return st.secrets.get(
            "ZERODHA_API_KEY", ""
        ) or os.getenv("ZERODHA_API_KEY", "")
    except Exception:
        return os.getenv("ZERODHA_API_KEY", "")

def get_api_secret():
    try:
        return st.secrets.get(
            "ZERODHA_API_SECRET", ""
        ) or os.getenv("ZERODHA_API_SECRET", "")
    except Exception:
        return os.getenv("ZERODHA_API_SECRET", "")

def get_kite():
    if not KITE_AVAILABLE:
        return None
    try:
        kite = KiteConnect(api_key=get_api_key())
        token = st.session_state.get("zerodha_token")
        if token:
            kite.set_access_token(token)
        return kite
    except Exception:
        return None

def get_login_url():
    if not KITE_AVAILABLE:
        return None
    try:
        kite = KiteConnect(api_key=get_api_key())
        return kite.login_url()
    except Exception:
        return None

def generate_session(request_token):
    if not KITE_AVAILABLE:
        return None
    try:
        kite = KiteConnect(api_key=get_api_key())
        data = kite.generate_session(
            request_token,
            api_secret=get_api_secret()
        )
        token = data["access_token"]
        # Persist as the app's data-layer token too, so
        # one daily login also powers Kite-sourced data
        # for all fetchers (see kite_data.py).
        try:
            from kite_data import save_owner_token
            save_owner_token(token)
        except Exception:
            pass
        return token
    except Exception as e:
        st.error(f"Zerodha login failed: {e}")
        return None

def is_connected():
    return bool(st.session_state.get("zerodha_token"))

def is_market_hours():
    now = datetime.now()
    market_open = now.replace(
        hour=9, minute=15, second=0
    )
    market_close = now.replace(
        hour=15, minute=30, second=0
    )
    is_weekday = now.weekday() < 5
    return (
        is_weekday and
        market_open <= now <= market_close
    )

@st.cache_data(ttl=60)
def get_live_quote(ticker):
    if not is_connected():
        return None
    try:
        kite = get_kite()
        if kite is None:
            return None
        symbol = ticker.replace(".NS", "")
        instrument = f"NSE:{symbol}"
        quote = kite.quote([instrument])
        data = quote.get(instrument, {})
        return {
            "price": data.get("last_price"),
            "open": data.get("ohlc", {}).get("open"),
            "high": data.get("ohlc", {}).get("high"),
            "low": data.get("ohlc", {}).get("low"),
            "close": data.get("ohlc", {}).get("close"),
            "volume": data.get("volume"),
            "change": data.get("change"),
            "change_pct": data.get(
                "net_change", 0
            )
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_intraday_data(ticker, interval="15minute"):
    if not is_connected():
        return None
    try:
        kite = get_kite()
        if kite is None:
            return None
        symbol = ticker.replace(".NS", "")
        instruments = kite.instruments("NSE")
        instrument_token = None
        for inst in instruments:
            if inst["tradingsymbol"] == symbol:
                instrument_token = inst[
                    "instrument_token"
                ]
                break
        if not instrument_token:
            return None
        to_date = datetime.now()
        from_date = to_date - timedelta(days=60)
        data = kite.historical_data(
            instrument_token,
            from_date, to_date,
            interval
        )
        if not data:
            return None
        import pandas as pd
        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        df.columns = [
            c.capitalize() for c in df.columns
        ]
        return df
    except Exception:
        return None

def get_portfolio():
    if not is_connected():
        return None
    try:
        kite = get_kite()
        if kite is None:
            return None
        holdings = kite.holdings()
        positions = kite.positions()
        return {
            "holdings": holdings,
            "positions": positions.get("net", [])
        }
    except Exception:
        return None

def place_gtt_order(ticker, entry_price,
                    stop_loss, target, quantity):
    if not is_connected():
        return None, "Not connected to Zerodha"
    try:
        kite = get_kite()
        if kite is None:
            return None, "Kite not available"
        symbol = ticker.replace(".NS", "")

        # Stop loss GTT
        sl_gtt = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_SINGLE,
            tradingsymbol=symbol,
            exchange="NSE",
            trigger_values=[stop_loss],
            last_price=entry_price,
            orders=[{
                "transaction_type": (
                    kite.TRANSACTION_TYPE_SELL
                ),
                "quantity": quantity,
                "product": kite.PRODUCT_CNC,
                "order_type": kite.ORDER_TYPE_LIMIT,
                "price": stop_loss * 0.99
            }]
        )

        # Target GTT
        tgt_gtt = kite.place_gtt(
            trigger_type=kite.GTT_TYPE_SINGLE,
            tradingsymbol=symbol,
            exchange="NSE",
            trigger_values=[target],
            last_price=entry_price,
            orders=[{
                "transaction_type": (
                    kite.TRANSACTION_TYPE_SELL
                ),
                "quantity": quantity,
                "product": kite.PRODUCT_CNC,
                "order_type": kite.ORDER_TYPE_LIMIT,
                "price": target * 0.999
            }]
        )
        return {
            "sl_gtt": sl_gtt,
            "target_gtt": tgt_gtt
        }, "GTT orders placed successfully"
    except Exception as e:
        return None, str(e)

def render_zerodha_panel(t):
    st.markdown(
        '<div class="section-label">'
        'Zerodha connection</div>',
        unsafe_allow_html=True
    )

    # Restore today's saved session (one login per day)
    if not st.session_state.get("zerodha_token"):
        try:
            from kite_data import load_owner_token
            saved = load_owner_token()
            if saved:
                st.session_state["zerodha_token"] = saved
        except Exception:
            pass

    params = st.query_params
    if "request_token" in params:
        request_token = params["request_token"]
        if not st.session_state.get("zerodha_token"):
            with st.spinner("Connecting to Zerodha..."):
                token = generate_session(request_token)
            if token:
                st.session_state["zerodha_token"] = token
                st.success("Connected to Zerodha")
                st.rerun()

    # Show which data source is active
    try:
        from kite_data import is_kite_data_active
        src = (
            "Kite Connect (licensed)"
            if is_kite_data_active()
            else "yfinance (fallback — dev only)"
        )
        st.caption(f"Market data source: {src}")
    except Exception:
        pass

    if is_connected():
        st.markdown(
            f'<div style="background:'
            f'rgba({t["green_rgb"]},0.1);'
            f'border:1px solid '
            f'rgba({t["green_rgb"]},0.3);'
            f'border-radius:6px;padding:10px 14px;'
            f'font-size:13px;color:{t["green"]};">'
            f'✓ Connected to Zerodha · '
            f'{"Market open" if is_market_hours() else "Market closed"}'
            f'</div>',
            unsafe_allow_html=True
        )

        portfolio = get_portfolio()
        if portfolio:
            holdings = portfolio.get("holdings", [])
            if holdings:
                st.markdown(
                    '<div class="section-label">'
                    'Your holdings</div>',
                    unsafe_allow_html=True
                )
                import pandas as pd
                holdings_data = []
                total_invested = 0
                total_current = 0
                for h in holdings:
                    if h.get("quantity", 0) == 0:
                        continue
                    invested = (
                        h.get("average_price", 0) *
                        h.get("quantity", 0)
                    )
                    current = (
                        h.get("last_price", 0) *
                        h.get("quantity", 0)
                    )
                    pnl = current - invested
                    pnl_pct = (
                        (pnl / invested * 100)
                        if invested > 0 else 0
                    )
                    total_invested += invested
                    total_current += current
                    holdings_data.append({
                        "Stock": h.get(
                            "tradingsymbol", ""
                        ),
                        "Qty": h.get("quantity", 0),
                        "Avg": f"₹{h.get('average_price',0):,.2f}",
                        "LTP": f"₹{h.get('last_price',0):,.2f}",
                        "P&L": f"₹{pnl:+,.2f}",
                        "Return": f"{pnl_pct:+.1f}%"
                    })

                if holdings_data:
                    total_pnl = total_current - total_invested
                    ph1, ph2, ph3 = st.columns(3)
                    ph1.metric(
                        "Invested",
                        f"₹{total_invested:,.2f}"
                    )
                    ph2.metric(
                        "Current",
                        f"₹{total_current:,.2f}"
                    )
                    ph3.metric(
                        "Total P&L",
                        f"₹{total_pnl:+,.2f}"
                    )
                    st.dataframe(
                        pd.DataFrame(holdings_data),
                        use_container_width=True
                    )

        if st.button("Disconnect Zerodha"):
            del st.session_state["zerodha_token"]
            st.rerun()
    else:
        login_url = get_login_url()
        if login_url:
            st.markdown(
                f'<div style="font-size:13px;'
                f'color:{t["text2"]};'
                f'margin-bottom:12px;">'
                f'Connect your Zerodha account to get '
                f'live prices, one-click GTT orders, '
                f'and real portfolio sync.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<a href="{login_url}" target="_blank">'
                f'<div style="background:{t["accent"]};'
                f'color:{t["bg2"]};'
                f'padding:10px 20px;'
                f'border-radius:5px;'
                f'font-weight:700;font-size:13px;'
                f'display:inline-block;'
                f'cursor:pointer;">'
                f'Login with Zerodha</div></a>',
                unsafe_allow_html=True
            )
        else:
            st.warning(
                "Zerodha API key not configured. "
                "Add ZERODHA_API_KEY to your .env file."
            )