import json
import os
import streamlit as st
from datetime import datetime, date

from config import JOURNAL_FILE

def load_journal():
    try:
        if os.path.exists(JOURNAL_FILE):
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
        return []
    except Exception:
        return []

def save_journal(trades):
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)
        return True
    except Exception:
        return False

def add_trade(ticker, signal, entry_price, stop_loss,
              target1, target2, shares, capital_at_risk,
              holding_days, score, confidence, sector,
              notes=""):
    trades = load_journal()
    trade = {
        "id": len(trades) + 1,
        "ticker": ticker,
        "signal": signal,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target1": round(target1, 2),
        "target2": round(target2, 2),
        "shares": shares,
        "capital_at_risk": round(capital_at_risk, 2),
        "holding_days": holding_days,
        "score": score,
        "confidence": round(confidence, 4),
        "sector": sector,
        "notes": notes,
        "entry_date": str(date.today()),
        "status": "OPEN",
        "exit_price": None,
        "exit_date": None,
        "pnl": None,
        "pnl_pct": None,
        "outcome": None
    }
    trades.append(trade)
    save_journal(trades)
    return trade

def close_trade(trade_id, exit_price, notes=""):
    trades = load_journal()
    for trade in trades:
        if trade["id"] == trade_id:
            trade["exit_price"] = round(exit_price, 2)
            trade["exit_date"] = str(date.today())
            trade["status"] = "CLOSED"
            trade["notes"] = notes
            pnl = (
                (exit_price - trade["entry_price"]) *
                trade["shares"]
            )
            pnl_pct = (
                (exit_price - trade["entry_price"]) /
                trade["entry_price"] * 100
            )
            trade["pnl"] = round(pnl, 2)
            trade["pnl_pct"] = round(pnl_pct, 2)
            if exit_price >= trade["target1"]:
                trade["outcome"] = "TARGET HIT"
            elif exit_price <= trade["stop_loss"]:
                trade["outcome"] = "STOP HIT"
            else:
                trade["outcome"] = "MANUAL EXIT"
            break
    save_journal(trades)
    return trades

def get_performance_stats(trades):
    closed = [t for t in trades if t["status"] == "CLOSED"]
    if not closed:
        return None

    total_trades = len(closed)
    wins = [t for t in closed if t["pnl"] and t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] and t["pnl"] <= 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    total_pnl = sum(t["pnl"] for t in closed if t["pnl"])
    avg_win = (
        sum(t["pnl"] for t in wins) / len(wins)
        if wins else 0
    )
    avg_loss = (
        sum(t["pnl"] for t in losses) / len(losses)
        if losses else 0
    )
    target_hits = len([
        t for t in closed
        if t["outcome"] == "TARGET HIT"
    ])
    stop_hits = len([
        t for t in closed
        if t["outcome"] == "STOP HIT"
    ])
    avg_confidence = (
        sum(t["confidence"] for t in closed) / total_trades
    )
    rr_achieved = (
        abs(avg_win / avg_loss)
        if avg_loss != 0 else 0
    )

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 3),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "target_hits": target_hits,
        "stop_hits": stop_hits,
        "avg_confidence": round(avg_confidence, 3),
        "rr_achieved": round(rr_achieved, 2),
        "best_trade": max(
            closed, key=lambda x: x["pnl"] or 0
        )["ticker"] if closed else "N/A",
        "worst_trade": min(
            closed, key=lambda x: x["pnl"] or 0
        )["ticker"] if closed else "N/A",
    }

def render_journal_tab(t):
    st.markdown(
        '<div class="section-label">Trade journal</div>',
        unsafe_allow_html=True
    )

    trades = load_journal()
    open_trades = [
        tr for tr in trades if tr["status"] == "OPEN"
    ]
    closed_trades = [
        tr for tr in trades if tr["status"] == "CLOSED"
    ]
    stats = get_performance_stats(trades)

    if stats:
        st.markdown(
            '<div class="section-label">'
            'Performance summary</div>',
            unsafe_allow_html=True
        )
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Total trades", stats["total_trades"])
        p2.metric(
            "Win rate", f"{stats['win_rate']:.1%}"
        )
        p3.metric(
            "Total P&L", f"₹{stats['total_pnl']:+,.2f}"
        )
        p4.metric(
            "Avg win", f"₹{stats['avg_win']:,.2f}"
        )
        p5.metric(
            "Avg loss", f"₹{stats['avg_loss']:,.2f}"
        )
        p6.metric(
            "R/R achieved",
            f"1:{stats['rr_achieved']:.1f}"
        )

        st.markdown(
            '<div class="section-label">'
            'Signal accuracy check</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
        <div style="background:{t['card']};
        border:1px solid {t['border']};
        border-radius:8px;padding:16px;
        font-size:13px;color:{t['text']};">
            Your real win rate is
            <strong>{stats['win_rate']:.1%}</strong>
            across {stats['total_trades']} trades.
            Target hits: {stats['target_hits']} ·
            Stop hits: {stats['stop_hits']}.
            Average confidence on your trades:
            {stats['avg_confidence']:.1%}.
            Best trade: {stats['best_trade']} ·
            Worst: {stats['worst_trade']}.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Open trades</div>',
        unsafe_allow_html=True
    )

    if not open_trades:
        st.markdown(
            f'<div style="color:{t["text2"]}; '
            f'font-size:13px;">No open trades.</div>',
            unsafe_allow_html=True
        )
    else:
        for trade in open_trades:
            with st.expander(
                f"{trade['ticker']} · "
                f"Entered ₹{trade['entry_price']} · "
                f"{trade['entry_date']}",
                expanded=True
            ):
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric(
                    "Entry", f"₹{trade['entry_price']}"
                )
                tc2.metric(
                    "Stop loss",
                    f"₹{trade['stop_loss']}"
                )
                tc3.metric(
                    "Target 1",
                    f"₹{trade['target1']}"
                )
                tc4.metric(
                    "Target 2",
                    f"₹{trade['target2']}"
                )

                with st.form(f"close_{trade['id']}"):
                    exit_px = st.number_input(
                        "Exit price",
                        min_value=0.0,
                        value=float(trade['entry_price']),
                        key=f"ep_{trade['id']}"
                    )
                    exit_note = st.text_input(
                        "Notes (optional)",
                        key=f"en_{trade['id']}"
                    )
                    if st.form_submit_button("Close trade"):
                        close_trade(
                            trade["id"],
                            exit_px,
                            exit_note
                        )
                        st.rerun()

    st.markdown(
        '<div class="section-label">'
        'Log a new trade</div>',
        unsafe_allow_html=True
    )
    with st.form("new_trade"):
        nc1, nc2, nc3 = st.columns(3)
        new_ticker = nc1.text_input(
            "Ticker", placeholder="e.g. RELIANCE"
        )
        new_entry = nc2.number_input(
            "Entry price", min_value=0.0, value=0.0
        )
        new_shares = nc3.number_input(
            "Shares", min_value=0, value=0
        )
        nc4, nc5, nc6 = st.columns(3)
        new_stop = nc4.number_input(
            "Stop loss", min_value=0.0, value=0.0
        )
        new_t1 = nc5.number_input(
            "Target 1", min_value=0.0, value=0.0
        )
        new_t2 = nc6.number_input(
            "Target 2", min_value=0.0, value=0.0
        )
        new_notes = st.text_input("Notes (optional)")
        if st.form_submit_button(
            "Log trade", type="primary"
        ):
            if new_ticker and new_entry > 0:
                add_trade(
                    ticker=new_ticker.upper() + ".NS",
                    signal="BUY",
                    entry_price=new_entry,
                    stop_loss=new_stop,
                    target1=new_t1,
                    target2=new_t2,
                    shares=new_shares,
                    capital_at_risk=(
                        new_shares *
                        abs(new_entry - new_stop)
                    ),
                    holding_days=7,
                    score=0,
                    confidence=0,
                    sector="Manual",
                    notes=new_notes
                )
                st.success(f"Trade logged: {new_ticker}")
                st.rerun()

    if closed_trades:
        st.markdown(
            '<div class="section-label">'
            'Trade history</div>',
            unsafe_allow_html=True
        )
        history_data = []
        for tr in reversed(closed_trades[-20:]):
            history_data.append({
                "Ticker": tr["ticker"].replace(".NS", ""),
                "Entry": f"₹{tr['entry_price']}",
                "Exit": f"₹{tr['exit_price']}",
                "Shares": tr["shares"],
                "P&L": f"₹{tr['pnl']:+,.2f}",
                "Return": f"{tr['pnl_pct']:+.1f}%",
                "Outcome": tr["outcome"],
                "Date": tr["entry_date"]
            })
        import pandas as pd
        st.dataframe(
            pd.DataFrame(history_data),
            use_container_width=True
        )