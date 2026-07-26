"""
Paper trading — the free, no-risk engagement hook.

Rationale (from market research): StockGro built a ₹126 Cr
business largely on free virtual trading. Letting users "track"
a setup with fake money is the single best pre-payment
engagement loop — it creates a daily reason to return (checking
your virtual P&L) without any money, any advice, or any SEBI
exposure. It's explicitly a simulation.

Storage: per-user JSON under DATA_DIR/users/<hash>/, same
pattern as journal/watchlist, so users never see each other's
book. Prices are marked to the latest close via the existing
data layer (already timeout-guarded).

Compliance: this is a simulation for learning. No real orders,
no advice. Copy stays in that frame.
"""

import json
import os
from datetime import datetime

import streamlit as st

from config import _user_scoped_file, DATA_DIR

STARTING_CASH = 1_000_000  # ₹10L virtual


def _book_file():
    return _user_scoped_file(
        "paper_book.json", str(DATA_DIR / "paper_book.json")
    )


def _load_book():
    try:
        f = _book_file()
        if os.path.exists(f):
            with open(f) as fh:
                return json.load(fh)
    except Exception:
        pass
    return {"cash": STARTING_CASH, "positions": [],
            "closed": []}


def _save_book(book):
    try:
        with open(_book_file(), "w") as fh:
            json.dump(book, fh, indent=2, default=str)
        return True
    except Exception:
        return False


def track_setup(ticker, entry, stop, target, shares,
                score=None, source="screener"):
    """Open a virtual position. Returns (ok, message)."""
    book = _load_book()
    if any(p["ticker"] == ticker and p.get("open", True)
           for p in book["positions"]):
        return False, "Already tracking this setup."
    cost = entry * shares
    if cost > book["cash"]:
        return False, (
            f"Virtual cash ₹{book['cash']:,.0f} is short of "
            f"₹{cost:,.0f}. Close a position or size down."
        )
    book["cash"] -= cost
    book["positions"].append({
        "ticker": ticker,
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "shares": int(shares),
        "score": score,
        "source": source,
        "opened": datetime.now().isoformat(),
        "open": True,
    })
    _save_book(book)
    return True, "Tracking (virtual)."


def _mark_price(ticker):
    from data import get_stock_data
    df = get_stock_data(ticker, period="5d")
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def close_position(ticker):
    book = _load_book()
    for p in book["positions"]:
        if p["ticker"] == ticker and p.get("open", True):
            px = _mark_price(ticker) or p["entry"]
            pnl = (px - p["entry"]) * p["shares"]
            book["cash"] += px * p["shares"]
            p["open"] = False
            p["exit"] = round(px, 2)
            p["pnl"] = round(pnl, 2)
            p["closed_at"] = datetime.now().isoformat()
            book["closed"].append(p)
            book["positions"] = [
                x for x in book["positions"]
                if not (x["ticker"] == ticker
                        and not x.get("open", True))
            ]
            _save_book(book)
            return True
    return False


def portfolio_summary():
    """Mark-to-market the whole virtual book."""
    book = _load_book()
    open_pos = [p for p in book["positions"]
                if p.get("open", True)]
    invested = mkt_val = unreal = 0.0
    rows = []
    for p in open_pos:
        px = _mark_price(p["ticker"]) or p["entry"]
        val = px * p["shares"]
        cost = p["entry"] * p["shares"]
        pnl = val - cost
        invested += cost
        mkt_val += val
        unreal += pnl
        rows.append({**p, "ltp": round(px, 2),
                     "value": round(val, 2),
                     "pnl": round(pnl, 2),
                     "pnl_pct": round(
                         100 * pnl / cost, 2) if cost else 0})
    realized = sum(c.get("pnl", 0) for c in book["closed"])
    equity = book["cash"] + mkt_val
    return {
        "cash": book["cash"],
        "invested": invested,
        "market_value": mkt_val,
        "unrealized": unreal,
        "realized": realized,
        "equity": equity,
        "total_return_pct": round(
            100 * (equity - STARTING_CASH) / STARTING_CASH, 2),
        "open_rows": rows,
        "closed": book["closed"],
    }


def render_paper_tab(t):
    st.markdown(
        '<div class="section-label">'
        'Paper trading — practice book (virtual money)</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "A risk-free simulation to test the Equitex Score "
        "yourself. Virtual ₹10,00,000 — no real money, no "
        "orders, not advice. Track setups from the Screener "
        "and watch how they'd have played out."
    )

    s = portfolio_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Virtual equity", f"₹{s['equity']:,.0f}",
              f"{s['total_return_pct']:+.2f}%")
    c2.metric("Cash", f"₹{s['cash']:,.0f}")
    c3.metric("Open P&L", f"₹{s['unrealized']:,.0f}")
    c4.metric("Booked P&L", f"₹{s['realized']:,.0f}")

    st.markdown(
        '<div class="section-label">Open positions</div>',
        unsafe_allow_html=True
    )
    if not s["open_rows"]:
        st.markdown(
            f'<div style="color:{t["text2"]};font-size:13px;">'
            f'No open virtual positions yet. Open the '
            f'Screener and hit <b>Track this setup</b> on any '
            f'stock to start your practice book.</div>',
            unsafe_allow_html=True
        )
    else:
        for p in s["open_rows"]:
            col = t["green"] if p["pnl"] >= 0 else t["red"]
            arrow = "▲" if p["pnl"] >= 0 else "▼"
            eqx = (f' · Equitex {p["score"]}'
                   if p.get("score") else "")
            cc1, cc2 = st.columns([5, 1])
            cc1.markdown(
                f'<div style="padding:10px 14px;'
                f'background:{t["card"]};'
                f'border:1px solid {t["border"]};'
                f'border-left:3px solid {col};'
                f'border-radius:10px;margin-bottom:8px;">'
                f'<b>{p["ticker"].replace(".NS","")}</b> · '
                f'{p["shares"]} sh @ ₹{p["entry"]:,.2f} '
                f'→ LTP ₹{p["ltp"]:,.2f}'
                f'<span style="color:{col};font-weight:600;"> '
                f'· {arrow} ₹{p["pnl"]:,.0f} '
                f'({p["pnl_pct"]:+.2f}%)</span><br>'
                f'<span style="font-size:11px;'
                f'color:{t["text2"]};">'
                f'Stop ₹{p["stop"]:,.2f} · '
                f'Target ₹{p["target"]:,.2f}{eqx}'
                f'</span></div>',
                unsafe_allow_html=True
            )
            if cc2.button("Close", key=f"pt_close_{p['ticker']}"):
                close_position(p["ticker"])
                st.rerun()

    if s["closed"]:
        wins = [c for c in s["closed"]
                if c.get("pnl", 0) > 0]
        wr = 100 * len(wins) / len(s["closed"])
        st.markdown(
            f'<div class="section-label">History · '
            f'{len(s["closed"])} closed · '
            f'{wr:.0f}% win rate</div>',
            unsafe_allow_html=True
        )
        for c in reversed(s["closed"][-15:]):
            col = t["green"] if c.get("pnl", 0) >= 0 else t["red"]
            st.markdown(
                f'<div style="font-size:12px;'
                f'color:{t["text2"]};padding:4px 0;">'
                f'{c["ticker"].replace(".NS","")} · '
                f'<span style="color:{col};">'
                f'₹{c.get("pnl",0):,.0f}</span></div>',
                unsafe_allow_html=True
            )
