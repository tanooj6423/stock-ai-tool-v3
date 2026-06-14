"""
tab_us_stocks.py — US Stocks tab for Equitex Intelligence
Renders the full "🇺🇸 US Stocks" tab with:
  - Universe selector (S&P 500 / S&P 500 + Russell 2000)
  - Market regime + VIX display
  - One-click scan
  - USD-formatted pick cards
  - Alpaca paper trade placement
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from screener_us import (
    run_us_scan, get_vix_state, get_us_market_regime,
    get_spy_data, get_us_earnings_risk,
)
from universe_us import get_us_universe
from broker_alpaca import (
    is_alpaca_connected, is_us_market_hours,
    place_alpaca_bracket_order, get_us_live_quote,
    get_alpaca_positions,
)


# ── Main renderer ────────────────────────────────────────────────────────────

def render_us_stocks_tab(theme: dict, settings: dict):
    """
    Called from app.py inside the "US Stocks" tab.

    Parameters
    ----------
    theme    : dict from CSS/theme system (keys: card, border, text1, text2,
               success, danger, warn, bg)
    settings : dict from session_state["settings"] — contains capital,
               risk_pct etc.
    """
    st.markdown(
        "<h2 style='margin-bottom:0'>🇺🇸 US Stocks</h2>"
        "<p style='color:#888;margin-top:2px;font-size:0.92em'>"
        "S&P 500 & Russell 2000 swing trade scanner · Alpaca paper trading</p>",
        unsafe_allow_html=True,
    )

    capital = float(settings.get("capital", 10000))
    risk_pct = float(settings.get("risk_pct", 1.5))

    # ── Market snapshot row ──────────────────────────────────────────────────
    spy_df = get_spy_data()
    regime = get_us_market_regime(spy_df)
    vix = get_vix_state()

    regime_color = {
        "bull": theme.get("success", "#00C853"),
        "bear": theme.get("danger", "#FF1744"),
        "sideways": theme.get("warn", "#FFA726"),
        "unknown": "#888888",
    }.get(regime, "#888888")

    regime_icon = {"bull": "🟢", "bear": "🔴", "sideways": "🟡", "unknown": "⚪"}.get(regime, "⚪")
    vix_icon = "😱" if vix["high_fear"] else "😊" if vix["vix"] and vix["vix"] < 15 else "😐"

    alpaca_connected = is_alpaca_connected()
    mkt_hours = is_us_market_hours()
    mkt_status = "🟢 Market Open" if mkt_hours else "🔴 Market Closed"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "S&P 500 Regime",
        f"{regime_icon} {regime.capitalize()}",
        help="Based on SPY SMA50/200, RSI, MACD",
    )
    col2.metric(
        "VIX (Fear Gauge)",
        f"{vix_icon} {vix['vix'] or 'N/A'}",
        delta=f"{vix['label']}",
        delta_color="inverse" if vix["high_fear"] else "normal",
    )
    col3.metric("Market Status", mkt_status)
    col4.metric(
        "Alpaca",
        "🔗 Connected" if alpaca_connected else "❌ Not connected",
        help="Go to Settings → Alpaca to connect",
    )

    if vix["high_fear"]:
        st.warning(
            f"⚠️ VIX at {vix['vix']} — high fear regime. "
            "Position sizes halved automatically. Consider waiting for calmer conditions.",
            icon="⚠️",
        )
    if regime == "bear":
        st.warning(
            "🐻 Bear market detected. Only stocks outperforming SPY by 3%+ shown. "
            "Extra caution advised.",
            icon="🐻",
        )

    st.divider()

    # ── Scan controls ────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        universe_mode = st.radio(
            "Universe",
            ["S&P 500 (~500 stocks)", "S&P 500 + Russell 2000 (~800 stocks)"],
            horizontal=True,
            key="us_universe_mode",
            help="Larger universe = longer scan (~6–10 min for combined)",
        )
    with col_b:
        min_score = st.slider(
            "Min score", 40, 70, 50, 5, key="us_min_score",
            help="Composite score out of 104. Higher = stricter.",
        )
    with col_c:
        st.write("")
        st.write("")
        scan_btn = st.button("🔍 Scan US Market", type="primary", use_container_width=True)

    # ── Run scan ─────────────────────────────────────────────────────────────
    if scan_btn or "us_picks" in st.session_state:
        if scan_btn:
            # Clear old results
            st.session_state.pop("us_picks", None)
            st.session_state.pop("us_regime", None)

            mode = "sp500_russell" if "Russell" in universe_mode else "sp500"
            tickers = get_us_universe(mode)

            progress_bar = st.progress(0, text="Initialising US scan...")
            status_text = st.empty()

            def _progress(i, total, ticker):
                if total > 0:
                    pct = min(int(i / total * 100), 100)
                    progress_bar.progress(pct, text=f"Scanning {ticker}… ({i}/{total})")
                    status_text.caption(f"Scanning: {ticker}")

            with st.spinner("Running US market scan..."):
                picks, final_regime = run_us_scan(
                    tickers_tuple=tuple(tickers),
                    capital=capital,
                    risk_pct=risk_pct,
                    progress_callback=_progress,
                )

            progress_bar.empty()
            status_text.empty()

            # Apply min score filter
            picks = [p for p in picks if p["score"] >= min_score]
            st.session_state["us_picks"] = picks
            st.session_state["us_regime"] = final_regime
            st.session_state["us_scan_time"] = datetime.now().strftime("%H:%M:%S")

        picks = st.session_state.get("us_picks", [])
        scan_time = st.session_state.get("us_scan_time", "")

        if not picks:
            st.info(
                "No qualifying picks found for current market conditions. "
                "Try lowering the minimum score, or the market may not be favourable right now."
            )
            return

        st.success(
            f"✅ Found **{len(picks)} picks** · Last scan: {scan_time}",
            icon="✅",
        )

        # ── Pick cards ───────────────────────────────────────────────────────
        for idx, pick in enumerate(picks):
            _render_pick_card(pick, idx, theme, capital, risk_pct, alpaca_connected)

    else:
        # Empty state
        _render_empty_state(theme)


# ── Pick card ────────────────────────────────────────────────────────────────

def _render_pick_card(pick: dict, idx: int, theme: dict,
                       capital: float, risk_pct: float,
                       alpaca_connected: bool):
    """Render a single US pick card with all trade details."""
    ticker = pick["ticker"]
    score = pick["score"]
    sector = pick["sector"]
    price = pick["price"]
    entry = pick["entry"]
    sl = pick["stop_loss"]
    t1 = pick["target1"]
    t2 = pick["target2"]
    shares = pick["shares"]
    cost = pick["position_cost"]
    rr1 = pick["rr1"]
    rr2 = pick["rr2"]
    risk_level = pick["risk_level"]
    regime = pick["market_regime"]
    rs = pick["relative_strength"]
    confidence = pick["confidence"]
    holding_days = pick["holding_days"]
    earnings_msg = pick.get("earnings_message", "")
    capital_note = pick.get("capital_note", "")
    key_drivers = pick.get("key_drivers", [])
    score_breakdown = pick.get("score_breakdown", {})
    signal_breakdown = pick.get("signal_breakdown", [])
    max_loss = pick.get("max_loss", 0.0)

    risk_colors = {
        "Low": theme.get("success", "#00C853"),
        "Medium": theme.get("warn", "#FFA726"),
        "High": theme.get("danger", "#FF1744"),
    }
    risk_color = risk_colors.get(risk_level, "#888")
    score_color = (
        theme.get("success", "#00C853") if score >= 70
        else theme.get("warn", "#FFA726") if score >= 55
        else "#888"
    )

    with st.container():
        st.markdown(
            f"""<div style='background:{theme["card"]};border:1px solid {theme["border"]};
            border-radius:12px;padding:20px;margin-bottom:18px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
              <div>
                <span style='font-size:1.5em;font-weight:700;color:{theme["text"]}'>{ticker}</span>
                <span style='margin-left:10px;color:#888;font-size:0.88em'>{sector}</span>
              </div>
              <div style='text-align:right;'>
                <span style='font-size:1.3em;font-weight:700;color:{score_color}'>
                  {score}<span style='font-size:0.6em;color:#888'>/104</span>
                </span>
                <span style='margin-left:12px;padding:3px 9px;border-radius:12px;
                  background:{risk_color}22;color:{risk_color};font-size:0.8em;font-weight:600'>
                  {risk_level} Risk
                </span>
              </div>
            </div>
            <div style='color:{theme["text2"]};font-size:0.85em;margin-top:6px'>
              Current price: <b style='color:{theme["text"]}'>${price:,.2f}</b>
              &nbsp;·&nbsp; Regime: {regime.capitalize()}
              &nbsp;·&nbsp; Confidence: {confidence:.0%}
              {f'&nbsp;·&nbsp; RS vs SPY: {rs:+.1f}%' if rs is not None else ''}
              {f'&nbsp;·&nbsp; {earnings_msg}' if earnings_msg else ''}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # Trade levels
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Entry Zone", f"${entry:,.2f}")
        col2.metric("Stop Loss", f"${sl:,.2f}",
                    delta=f"-{((entry - sl) / entry * 100):.1f}%",
                    delta_color="inverse")
        col3.metric("Target 1", f"${t1:,.2f}",
                    delta=f"+{rr1:.1f}R",
                    delta_color="normal")
        col4.metric("Target 2", f"${t2:,.2f}",
                    delta=f"+{rr2:.1f}R",
                    delta_color="normal")
        col5.metric("Hold Period", f"~{holding_days}d")

        # Position summary
        pct_risk = (entry - sl) / entry * 100
        st.markdown(
            f"""<div style='background:{theme["bg"]};border-radius:8px;padding:10px 14px;
            margin:8px 0;font-size:0.86em;color:{theme["text2"]}'>
            📊 <b>{shares} shares</b> @ ${entry:,.2f}
            = <b>${cost:,.2f}</b> deployed
            &nbsp;|&nbsp; Max loss: <span style='color:{theme["danger"]}'>${max_loss:,.2f}</span>
            ({pct_risk:.1f}% below entry)
            {f'&nbsp;|&nbsp; <span style="color:{theme["warn"]}">{capital_note}</span>' if capital_note else ''}
            </div>""",
            unsafe_allow_html=True,
        )

        # Key drivers
        if key_drivers:
            drivers_html = " &nbsp;·&nbsp; ".join(
                [f"✦ {d}" for d in key_drivers]
            )
            st.markdown(
                f"<p style='font-size:0.84em;color:{theme['text2']};margin:4px 0'>"
                f"{drivers_html}</p>",
                unsafe_allow_html=True,
            )

        # Expandable details
        with st.expander(f"📋 Full analysis — {ticker}"):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Signal Breakdown**")
                if signal_breakdown:
                    df_sb = pd.DataFrame(signal_breakdown)[
                        ["Factor", "Reading", "Status", "Signal"]
                    ]
                    st.dataframe(
                        df_sb, use_container_width=True, hide_index=True
                    )

            with detail_col2:
                st.markdown("**Score Breakdown (out of 104)**")
                if score_breakdown:
                    items = []
                    for k, v in score_breakdown.items():
                        earned = int(v.split("/")[0])
                        maximum = int(v.split("/")[1])
                        pct = earned / maximum * 100
                        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                        items.append({
                            "Layer": k,
                            "Score": v,
                            "Visual": f"{bar} {pct:.0f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(items),
                        use_container_width=True, hide_index=True
                    )

            st.markdown("**Trade Instructions**")
            st.markdown(
                f"""
1. **Wait for price to reach entry zone:** ${entry - 0.10:,.2f} – ${entry + 0.10:,.2f}
2. **Buy {shares} shares of {ticker}** (market or limit order)
3. **Immediately set stop-loss** at **${sl:,.2f}** (−{pct_risk:.1f}%)
4. **Take partial profit** at Target 1: **${t1:,.2f}** (+{rr1:.1f}R) — exit 50%
5. **Trail stop** for remaining shares; final target **${t2:,.2f}** (+{rr2:.1f}R)
6. **Exit in full** if held > {holding_days} trading days
7. Position cost: **${cost:,.2f}** · Max risk: **${max_loss:,.2f}**
"""
            )

            # Alpaca paper trade
            _render_alpaca_trade_button(
                pick=pick,
                idx=idx,
                theme=theme,
                alpaca_connected=alpaca_connected,
            )


# ── Alpaca trade placement ───────────────────────────────────────────────────

def _render_alpaca_trade_button(pick: dict, idx: int,
                                 theme: dict, alpaca_connected: bool):
    """
    Shows Alpaca paper trade placement button inside a pick card.
    Only shown if Alpaca is connected.
    """
    ticker = pick["ticker"]
    shares = pick["shares"]
    t1 = pick["target1"]
    sl = pick["stop_loss"]

    if not alpaca_connected:
        st.info(
            "💡 Connect Alpaca in Settings to paper-trade this pick with one click.",
            icon="💡",
        )
        return

    mkt_open = is_us_market_hours()
    st.markdown("**🤖 Alpaca Paper Trade**")

    if not mkt_open:
        st.caption("⏰ US market is closed — orders will queue for next open.")

    trade_col1, trade_col2, trade_col3, trade_col4 = st.columns(4)
    qty_input = trade_col1.number_input(
        "Shares", min_value=1, max_value=10000,
        value=shares, key=f"us_qty_{idx}_{ticker}"
    )
    tp_input = trade_col2.number_input(
        "Take Profit ($)", min_value=0.01,
        value=float(t1), step=0.01, format="%.2f",
        key=f"us_tp_{idx}_{ticker}"
    )
    sl_input = trade_col3.number_input(
        "Stop Loss ($)", min_value=0.01,
        value=float(sl), step=0.01, format="%.2f",
        key=f"us_sl_{idx}_{ticker}"
    )

    with trade_col4:
        st.write("")  # vertical spacing
        st.write("")
        place_btn = st.button(
            f"📤 Place Paper Order",
            key=f"us_place_{idx}_{ticker}",
            type="primary",
            use_container_width=True,
        )

    order_key = f"us_order_result_{idx}_{ticker}"
    if place_btn:
        if sl_input >= tp_input:
            st.error("Stop loss must be below take profit.")
        else:
            with st.spinner(f"Placing paper bracket order for {ticker}..."):
                result = place_alpaca_bracket_order(
                    ticker=ticker,
                    qty=int(qty_input),
                    take_profit=float(tp_input),
                    stop_loss=float(sl_input),
                )
            st.session_state[order_key] = result

    if order_key in st.session_state:
        result = st.session_state[order_key]
        if result.get("success"):
            st.success(
                f"✅ Paper order placed! "
                f"{result['qty']} × {result['symbol']} "
                f"| TP: ${result['take_profit']:,.2f} "
                f"| SL: ${result['stop_loss']:,.2f} "
                f"| Order ID: `{result['order_id']}`"
            )
        else:
            st.error(f"❌ Order failed: {result.get('error', 'Unknown error')}")

    # Open positions for this ticker
    try:
        positions = get_alpaca_positions()
        ticker_pos = [p for p in positions if p["symbol"] == ticker]
        if ticker_pos:
            p = ticker_pos[0]
            pnl_color = theme.get("success", "#00C853") if p["unrealized_pnl"] >= 0 else theme.get("danger", "#FF1744")
            st.markdown(
                f"<small style='color:{theme['text2']}'>📌 Open position: "
                f"{p['qty']:.0f} shares · Entry ${p['avg_entry']:,.2f} · "
                f"Current ${p['current_price']:,.2f} · "
                f"<span style='color:{pnl_color}'>"
                f"P&L ${p['unrealized_pnl']:+,.2f} ({p['unrealized_pnl_pct']:+.2f}%)"
                f"</span></small>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass


# ── Empty state ──────────────────────────────────────────────────────────────

def _render_empty_state(theme: dict):
    st.markdown(
        f"""<div style='text-align:center;padding:60px 20px;
        background:{theme["card"]};border-radius:12px;
        border:1px dashed {theme["border"]};'>
        <div style='font-size:3em'>🇺🇸</div>
        <h3 style='color:{theme["text"]};margin:12px 0 6px'>US Market Scanner</h3>
        <p style='color:{theme["text2"]};max-width:400px;margin:auto;font-size:0.92em'>
          Scan S&P 500 and Russell 2000 for high-probability swing trade setups.
          Uses the same ML + 10-layer composite scoring as your India picks.
          <br><br>
          Select a universe above and click <b>Scan US Market</b> to begin.
        </p>
        </div>""",
        unsafe_allow_html=True,
    )