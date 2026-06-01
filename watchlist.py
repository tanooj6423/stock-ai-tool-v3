import json
import os
import streamlit as st
import yfinance as yf
from datetime import datetime

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    try:
        if os.path.exists(WATCHLIST_FILE):
            with open(WATCHLIST_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception:
        return []

def save_watchlist(items):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(items, f, indent=2, default=str)
        return True
    except Exception:
        return False

def add_to_watchlist(ticker, alert_price=None,
                     notes="", direction="above"):
    items = load_watchlist()
    existing = [i for i in items if i["ticker"] == ticker]
    if existing:
        return False, "Already in watchlist"
    items.append({
        "ticker": ticker,
        "added_date": str(datetime.now().date()),
        "alert_price": alert_price,
        "alert_direction": direction,
        "alert_triggered": False,
        "notes": notes
    })
    save_watchlist(items)
    return True, "Added"

def remove_from_watchlist(ticker):
    items = load_watchlist()
    items = [i for i in items if i["ticker"] != ticker]
    save_watchlist(items)

def update_alert(ticker, alert_price, direction):
    items = load_watchlist()
    for item in items:
        if item["ticker"] == ticker:
            item["alert_price"] = alert_price
            item["alert_direction"] = direction
            item["alert_triggered"] = False
    save_watchlist(items)

def get_watchlist_prices():
    items = load_watchlist()
    if not items:
        return []
    results = []
    for item in items:
        try:
            ticker = item["ticker"]
            df = yf.Ticker(ticker).history(period="2d")
            if df is None or df.empty:
                continue
            current = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else current
            change = ((current - prev) / prev) * 100

            alert_hit = False
            alert_msg = ""
            if item.get("alert_price"):
                ap = item["alert_price"]
                direction = item.get(
                    "alert_direction", "above"
                )
                if direction == "above" and current >= ap:
                    alert_hit = True
                    alert_msg = (
                        f"Price ₹{current:,.2f} hit "
                        f"alert ₹{ap:,.2f}"
                    )
                elif direction == "below" and current <= ap:
                    alert_hit = True
                    alert_msg = (
                        f"Price ₹{current:,.2f} hit "
                        f"alert ₹{ap:,.2f}"
                    )

            results.append({
                "ticker": ticker,
                "current_price": round(current, 2),
                "change_pct": round(change, 2),
                "alert_price": item.get("alert_price"),
                "alert_direction": item.get(
                    "alert_direction", "above"
                ),
                "alert_hit": alert_hit,
                "alert_msg": alert_msg,
                "added_date": item.get("added_date"),
                "notes": item.get("notes", "")
            })
        except Exception:
            continue
    return results

def render_watchlist_tab(t):
    st.markdown(
        '<div class="section-label">My watchlist</div>',
        unsafe_allow_html=True
    )

    items = get_watchlist_prices()
    alerts_triggered = [i for i in items if i["alert_hit"]]

    if alerts_triggered:
        st.markdown(
            '<div class="section-label">'
            '🔔 Alerts triggered</div>',
            unsafe_allow_html=True
        )
        for alert in alerts_triggered:
            st.warning(
                f"**{alert['ticker'].replace('.NS','')}** "
                f"— {alert['alert_msg']}"
            )

    if not items:
        st.markdown(
            f'<div style="color:{t["text2"]};'
            f'font-size:13px;">'
            f'Your watchlist is empty. '
            f'Add stocks below or click Add to watchlist '
            f'on any daily pick.</div>',
            unsafe_allow_html=True
        )
    else:
        for item in items:
            change_color = (
                t["green"] if item["change_pct"] >= 0
                else t["red"]
            )
            arrow = (
                "▲" if item["change_pct"] >= 0 else "▼"
            )
            alert_badge = ""
            if item["alert_hit"]:
                alert_badge = (
                    f'<span style="background:'
                    f'rgba({t["green_rgb"]},0.15);'
                    f'color:{t["green"]};'
                    f'border:1px solid '
                    f'rgba({t["green_rgb"]},0.3);'
                    f'padding:2px 8px;border-radius:4px;'
                    f'font-size:11px;margin-left:8px;">'
                    f'🔔 ALERT</span>'
                )

            with st.expander(
                f"{item['ticker'].replace('.NS','')} · "
                f"₹{item['current_price']:,.2f} · "
                f"{arrow} {abs(item['change_pct']):.2f}%",
                expanded=item["alert_hit"]
            ):
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric(
                    "Current price",
                    f"₹{item['current_price']:,.2f}",
                    f"{item['change_pct']:+.2f}%"
                )
                wc2.metric(
                    "Alert level",
                    f"₹{item['alert_price']:,.2f}"
                    if item["alert_price"] else "Not set"
                )
                wc3.metric(
                    "Direction",
                    item["alert_direction"].capitalize()
                    if item["alert_price"] else "—"
                )

                if item["notes"]:
                    st.caption(f"Notes: {item['notes']}")

                with st.form(
                    f"alert_{item['ticker']}"
                ):
                    ac1, ac2, ac3 = st.columns(3)
                    new_alert = ac1.number_input(
                        "Set alert price",
                        min_value=0.0,
                        value=float(
                            item["alert_price"] or 0
                        ),
                        key=f"ap_{item['ticker']}"
                    )
                    new_dir = ac2.selectbox(
                        "Direction",
                        ["above", "below"],
                        index=0 if item.get(
                            "alert_direction"
                        ) == "above" else 1,
                        key=f"ad_{item['ticker']}"
                    )
                    if ac3.form_submit_button(
                        "Update alert"
                    ):
                        update_alert(
                            item["ticker"],
                            new_alert, new_dir
                        )
                        st.rerun()

                if st.button(
                    "Remove",
                    key=f"rm_{item['ticker']}"
                ):
                    remove_from_watchlist(item["ticker"])
                    st.rerun()

    st.markdown(
        '<div class="section-label">'
        'Add to watchlist</div>',
        unsafe_allow_html=True
    )
    with st.form("add_watchlist"):
        wf1, wf2, wf3, wf4 = st.columns(4)
        new_ticker = wf1.text_input(
            "NSE symbol",
            placeholder="e.g. ZOMATO"
        )
        alert_px = wf2.number_input(
            "Alert price (optional)",
            min_value=0.0, value=0.0
        )
        alert_dir = wf3.selectbox(
            "Alert direction",
            ["above", "below"]
        )
        wf_notes = wf4.text_input(
            "Notes (optional)"
        )
        if st.form_submit_button(
            "Add to watchlist", type="primary"
        ):
            if new_ticker:
                ticker = new_ticker.upper()
                if not ticker.endswith(".NS"):
                    ticker += ".NS"
                success, msg = add_to_watchlist(
                    ticker,
                    alert_px if alert_px > 0 else None,
                    wf_notes,
                    alert_dir
                )
                if success:
                    st.success(f"Added {ticker}")
                    st.rerun()
                else:
                    st.warning(msg)