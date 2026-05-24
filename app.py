import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import time
from dotenv import load_dotenv
from data import (get_stock_data, add_indicators, get_fundamentals,
                  get_support_resistance, get_nifty_data,
                  get_nifty_correlation, get_relative_strength,
                  get_market_regime, get_fibonacci_levels,
                  validate_ticker)
from model import train_model, get_signal, get_risk_metrics
from sentiment import (get_news_sentiment, NEGATIVE_KEYWORDS,
                       POSITIVE_KEYWORDS)
from ai_explain import explain_signal, generate_pick_thesis
from screener_engine import run_full_scan, calculate_position_size
from universe import ALL_STOCKS, NIFTY_50, COMMODITIES, get_sector
from earnings import get_earnings_status, get_nse_earnings_calendar

load_dotenv()

st.set_page_config(
    page_title="Equitex Intelligence",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

THEMES = {
    "Bloomberg Dark": {
        "bg": "#060608",
        "bg2": "#0d0d12",
        "card": "#12121a",
        "border": "#1e1e2e",
        "green": "#00d4aa",
        "red": "#ff4757",
        "blue": "#5352ed",
        "gold": "#ffd32a",
        "text": "#e8e8f0",
        "text2": "#8888a8",
        "accent": "#00d4aa",
    },
    "Midnight Navy": {
        "bg": "#030711",
        "bg2": "#080f1f",
        "card": "#0d1628",
        "border": "#1a2540",
        "green": "#00e5a0",
        "red": "#ff3860",
        "blue": "#4facfe",
        "gold": "#f9a825",
        "text": "#e2e8f0",
        "text2": "#718096",
        "accent": "#4facfe",
    },
    "Professional Light": {
        "bg": "#f8fafc",
        "bg2": "#ffffff",
        "card": "#ffffff",
        "border": "#e2e8f0",
        "green": "#059669",
        "red": "#dc2626",
        "blue": "#2563eb",
        "gold": "#d97706",
        "text": "#0f172a",
        "text2": "#64748b",
        "accent": "#2563eb",
    }
}

if "theme" not in st.session_state:
    st.session_state.theme = "Bloomberg Dark"

t = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {t['bg']};
    color: {t['text']};
}}

section[data-testid="stSidebar"] {{
    background-color: {t['bg2']};
    border-right: 1px solid {t['border']};
}}

section[data-testid="stSidebar"] * {{
    color: {t['text']} !important;
}}

.main-header {{
    padding: 24px 0 8px 0;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 24px;
}}

.main-title {{
    font-size: 28px;
    font-weight: 700;
    color: {t['text']};
    letter-spacing: -0.5px;
    margin: 0;
}}

.main-subtitle {{
    font-size: 13px;
    color: {t['text2']};
    margin-top: 4px;
}}

.accent-line {{
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, {t['accent']}, transparent);
    margin: 8px 0 0 0;
    border-radius: 2px;
}}

.pick-card {{
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    position: relative;
}}

.pick-card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}}

.pick-rank {{
    font-size: 11px;
    font-weight: 600;
    color: {t['text2']};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.pick-ticker {{
    font-size: 22px;
    font-weight: 700;
    color: {t['text']};
    font-family: 'JetBrains Mono', monospace;
}}

.pick-price {{
    font-size: 14px;
    color: {t['text2']};
    margin-top: 2px;
}}

.signal-badge-buy {{
    background: {t['green']}22;
    color: {t['green']};
    border: 1px solid {t['green']}44;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}}

.signal-badge-sell {{
    background: {t['red']}22;
    color: {t['red']};
    border: 1px solid {t['red']}44;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}}

.score-badge {{
    background: {t['accent']}22;
    color: {t['accent']};
    border: 1px solid {t['accent']}44;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
}}

.metric-card {{
    background: {t['bg2']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}}

.metric-label {{
    font-size: 11px;
    color: {t['text2']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}}

.metric-value {{
    font-size: 18px;
    font-weight: 600;
    color: {t['text']};
    font-family: 'JetBrains Mono', monospace;
}}

.metric-value-green {{
    font-size: 18px;
    font-weight: 600;
    color: {t['green']};
    font-family: 'JetBrains Mono', monospace;
}}

.metric-value-red {{
    font-size: 18px;
    font-weight: 600;
    color: {t['red']};
    font-family: 'JetBrains Mono', monospace;
}}

.score-bar-container {{
    background: {t['border']};
    border-radius: 4px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}}

.score-bar-fill {{
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, {t['accent']}, {t['green']});
}}

.section-header {{
    font-size: 14px;
    font-weight: 600;
    color: {t['text2']};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-header::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {t['border']};
}}

.thesis-card {{
    background: {t['accent']}0d;
    border: 1px solid {t['accent']}33;
    border-left: 3px solid {t['accent']};
    border-radius: 8px;
    padding: 16px;
    font-size: 14px;
    color: {t['text']};
    line-height: 1.7;
}}

.regime-bull {{
    color: {t['green']};
    font-weight: 600;
}}

.regime-bear {{
    color: {t['red']};
    font-weight: 600;
}}

.regime-sideways {{
    color: {t['gold']};
    font-weight: 600;
}}

.regime-unknown {{
    color: {t['text2']};
    font-weight: 600;
}}

.headline-pos {{
    color: {t['green']};
    font-size: 13px;
    padding: 4px 0;
    border-bottom: 1px solid {t['border']};
}}

.headline-neg {{
    color: {t['red']};
    font-size: 13px;
    padding: 4px 0;
    border-bottom: 1px solid {t['border']};
}}

.headline-neu {{
    color: {t['text2']};
    font-size: 13px;
    padding: 4px 0;
    border-bottom: 1px solid {t['border']};
}}

.divider {{
    height: 1px;
    background: {t['border']};
    margin: 24px 0;
}}

[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem !important;
    font-weight: 600;
    color: {t['text']} !important;
}}

[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    color: {t['text2']} !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

[data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background-color: {t['bg2']};
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
    border: 1px solid {t['border']};
}}

.stTabs [data-baseweb="tab"] {{
    background-color: transparent;
    border-radius: 6px;
    color: {t['text2']};
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
}}

.stTabs [aria-selected="true"] {{
    background-color: {t['accent']}22 !important;
    color: {t['accent']} !important;
    border: 1px solid {t['accent']}44;
}}

.stExpander {{
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    margin-bottom: 12px;
}}

.stExpander > div > div > div > div {{
    font-size: 14px;
    font-weight: 500;
    color: {t['text']};
}}

.stButton button {{
    background: {t['accent']};
    color: {t['bg']};
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 20px;
    transition: opacity 0.2s;
}}

.stButton button:hover {{
    opacity: 0.85;
}}

.stSelectbox > div > div {{
    background: {t['bg2']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    color: {t['text']};
}}

.stTextInput > div > div > input {{
    background: {t['bg2']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    color: {t['text']};
}}

.stProgress > div > div > div {{
    background: {t['accent']};
    border-radius: 4px;
}}

.stProgress > div > div {{
    background: {t['border']};
    border-radius: 4px;
}}

.sidebar-logo {{
    font-size: 20px;
    font-weight: 700;
    color: {t['accent']};
    letter-spacing: -0.5px;
    padding: 16px 0 8px 0;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 16px;
}}

.sidebar-section {{
    font-size: 10px;
    font-weight: 600;
    color: {t['text2']};
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 16px 0 8px 0;
}}

.tag {{
    display: inline-block;
    background: {t['bg']};
    border: 1px solid {t['border']};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: {t['text2']};
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}}

.status-bar {{
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
    font-size: 13px;
}}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
    <p class="main-title">📈 Equitex Intelligence</p>
    <p class="main-subtitle">Professional-grade AI equity analysis for Indian markets · Not financial advice</p>
    <div class="accent-line"></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Daily Picks",
    "📊 Stock Analysis",
    "🔮 Price Forecast",
    "📁 Portfolio",
    "⚙️ Settings"
])

with tab5:
    st.markdown('<div class="section-header">Display</div>',
                unsafe_allow_html=True)
    theme_choice = st.selectbox(
        "Theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme)
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown('<div class="section-header">Trading parameters</div>',
                unsafe_allow_html=True)
    capital = st.number_input(
        "Trading capital (₹)",
        min_value=10000, max_value=10000000,
        value=50000, step=5000
    )
    risk_pct = st.slider(
        "Risk per trade (%)",
        min_value=0.5, max_value=3.0,
        value=1.5, step=0.25
    )

    st.markdown('<div class="section-header">Scan settings</div>',
                unsafe_allow_html=True)
    min_score = st.slider(
        "Minimum composite score",
        min_value=40, max_value=85,
        value=55, step=5
    )
    scan_universe = st.selectbox(
        "Scan universe",
        ["Nifty 50 only",
         "Nifty 50 + Next 50",
         "Full universe (150 stocks)"],
        index=1
    )

with tab1:
    st.sidebar.markdown(
        f'<div class="sidebar-logo">Equitex v3</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f'<div class="sidebar-section">Stock selection</div>',
        unsafe_allow_html=True
    )

    mode = st.sidebar.radio(
        "", ["Popular stocks", "Search any NSE stock"],
        label_visibility="collapsed"
    )

    if scan_universe == "Nifty 50 only":
        scan_tickers = NIFTY_50
    elif scan_universe == "Nifty 50 + Next 50":
        from universe import NIFTY_NEXT_50
        scan_tickers = NIFTY_50 + NIFTY_NEXT_50
    else:
        scan_tickers = ALL_STOCKS

    col_h, col_btn = st.columns([5, 1])
    with col_h:
        st.markdown(
            '<div class="section-header">AI screened daily picks</div>',
            unsafe_allow_html=True
        )
    with col_btn:
        run_scan = st.button("🔄 Refresh", type="primary")

    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    start_time = time.time()

    def update_progress(current, total, current_ticker):
        pct = int((current / total) * 100)
        elapsed = time.time() - start_time
        rate = elapsed / max(current, 1)
        remaining = int(rate * (total - current))
        mins = remaining // 60
        secs = remaining % 60
        progress_bar.progress(pct)
        status_text.markdown(
            f'<span style="color:{t["text2"]};font-size:13px;">'
            f'Scanning {current_ticker.replace(".NS", "")} '
            f'({current}/{total})</span>',
            unsafe_allow_html=True
        )
        time_text.markdown(
            f'<span style="color:{t["text2"]};font-size:12px;">'
            f'Est. remaining: {mins}m {secs}s</span>',
            unsafe_allow_html=True
        )

    picks, regime = run_full_scan(
        tuple(scan_tickers),
        capital=capital,
        risk_pct=risk_pct,
        progress_callback=update_progress
    )

    progress_bar.progress(100)
    status_text.empty()
    time_text.empty()

    regime_class = f"regime-{regime}"
    regime_icons = {
        "bull": "🟢", "bear": "🔴",
        "sideways": "🟡", "unknown": "⚪"
    }
    regime_icon = regime_icons.get(regime, "⚪")

    st.markdown(f"""
    <div style="display:flex;gap:24px;align-items:center;
    background:{t['card']};border:1px solid {t['border']};
    border-radius:8px;padding:12px 16px;margin-bottom:20px;
    font-size:13px;">
        <span>
            <span style="color:{t['text2']}">Market regime</span>
            &nbsp;
            <span class="{regime_class}">{regime_icon} {regime.upper()}</span>
        </span>
        <span style="color:{t['border']}">|</span>
        <span style="color:{t['text2']}">
            Stocks scanned: 
            <span style="color:{t['text']};font-weight:600;">
                {len(scan_tickers)}
            </span>
        </span>
        <span style="color:{t['border']}">|</span>
        <span style="color:{t['text2']}">
            Picks found: 
            <span style="color:{t['accent']};font-weight:600;">
                {len(picks)}
            </span>
        </span>
        <span style="color:{t['border']}">|</span>
        <span style="color:{t['text2']}">
            10-layer composite scoring
        </span>
    </div>
    """, unsafe_allow_html=True)

    if not picks:
        st.markdown(f"""
        <div style="background:{t['card']};border:1px solid {t['border']};
        border-radius:10px;padding:32px;text-align:center;color:{t['text2']};">
            <div style="font-size:32px;margin-bottom:12px;">🔍</div>
            <div style="font-size:16px;font-weight:600;
            color:{t['text']};margin-bottom:8px;">
                No picks today
            </div>
            <div style="font-size:13px;">
                No stocks passed all screening criteria. 
                Market conditions may not be favourable 
                for new entries.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, pick in enumerate(picks):
            score = pick["score"]
            score_pct = int((score / 100) * 100)
            signal = pick["signal"]
            badge_class = (
                "signal-badge-buy" if signal == "BUY"
                else "signal-badge-sell"
            )
            trend_icon = (
                "📈" if pick.get("sentiment_trend") == "improving"
                else "📉"
                if pick.get("sentiment_trend") == "deteriorating"
                else "➡️"
            )
            earnings_label = ""
            if pick.get("earnings_message"):
                earnings_label = (
                    f" &nbsp;·&nbsp; 📅 {pick['earnings_message']}"
                )

            with st.expander(
                f"#{i+1}  {pick['ticker'].replace('.NS', '')}  "
                f"·  {signal}  {pick['confidence']:.1%}  "
                f"·  Score {score}/100  "
                f"·  ₹{pick['price']:,.2f}",
                expanded=(i < 3)
            ):
                st.markdown(f"""
                <div style="display:flex;align-items:center;
                gap:12px;margin-bottom:16px;flex-wrap:wrap;">
                    <span class="{badge_class}">{signal}</span>
                    <span class="score-badge">{score}/100</span>
                    <span style="color:{t['text2']};font-size:13px;">
                        {pick['sector']} · 
                        Confidence {pick['confidence']:.1%} · 
                        Sentiment {trend_icon}
                        {earnings_label}
                    </span>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar-fill" 
                    style="width:{score_pct}%"></div>
                </div>
                """, unsafe_allow_html=True)

                if pick.get("earnings_message"):
                    if pick.get("earnings_risk") == "medium":
                        st.warning(
                            f"⚠️ {pick['earnings_message']} — "
                            f"Consider reducing position size."
                        )
                    elif pick.get("earnings_risk") == "low":
                        st.info(f"📅 {pick['earnings_message']}")

                st.markdown(
                    '<div class="section-header">Trade levels</div>',
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Entry", f"₹{pick['entry']:,.2f}")
                c2.metric("Stop loss", f"₹{pick['stop_loss']:,.2f}")
                c3.metric("Target 1", f"₹{pick['target1']:,.2f}")
                c4.metric("Target 2", f"₹{pick['target2']:,.2f}")
                c5.metric("R/R", f"1:{pick['rr1']:.1f}")
                if pick["shares"] > 0:
                    c6.metric("Shares", f"{pick['shares']}")
                else:
                    c6.metric("Shares", "—")
                    if pick.get("capital_note"):
                        st.caption(f"⚠️ {pick['capital_note']}")

                st.markdown(
                    '<div class="section-header">Indicators</div>',
                    unsafe_allow_html=True
                )
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("RSI", f"{pick['rsi']}")
                m2.metric(
                    "Confidence", f"{pick['confidence']:.1%}"
                )
                m3.metric("Buy prob", f"{pick['buy_prob']:.1%}")
                m4.metric(
                    "Sentiment",
                    f"{pick['sentiment'].capitalize()}"
                )
                m5.metric("Sharpe", f"{pick['sharpe']}")
                m6.metric(
                    "Max DD", f"{pick['max_drawdown']}"
                )

                st.markdown(
                    '<div class="section-header">'
                    'Score breakdown</div>',
                    unsafe_allow_html=True
                )
                breakdown = pick["score_breakdown"]
                cols = st.columns(len(breakdown))
                for j, (layer, pts) in enumerate(
                    breakdown.items()
                ):
                    cols[j].metric(layer, pts)

                st.markdown(
                    '<div class="section-header">'
                    'AI trade thesis</div>',
                    unsafe_allow_html=True
                )
                with st.spinner("Generating thesis..."):
                    thesis = generate_pick_thesis(
                        ticker=pick["ticker"],
                        signal=pick["signal"],
                        confidence=pick["confidence"],
                        score=pick["score"],
                        entry=pick["entry"],
                        stop_loss=pick["stop_loss"],
                        target1=pick["target1"],
                        target2=pick["target2"],
                        rr_ratio=pick["rr1"],
                        sentiment=pick["sentiment"],
                        rsi=pick["rsi"],
                        sector=pick["sector"],
                        market_regime=pick["market_regime"],
                        relative_strength=pick["relative_strength"]
                    )
                st.markdown(
                    f'<div class="thesis-card">{thesis}</div>',
                    unsafe_allow_html=True
                )

                if pick["shares"] > 0 and pick["position_cost"] > 0:
                    st.markdown(
                        f'<div style="margin-top:12px;font-size:12px;'
                        f'color:{t["text2"]};">'
                        f'Position: {pick["shares"]} shares &nbsp;·&nbsp; '
                        f'Cost: ₹{pick["position_cost"]:,.2f} '
                        f'&nbsp;·&nbsp; '
                        f'Max loss: ₹'
                        f'{pick["risk_amount"] * pick["shares"]:,.2f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

with tab2:
    st.sidebar.markdown(
        f'<div class="sidebar-section">Stock selection</div>',
        unsafe_allow_html=True
    )
    if mode == "Popular stocks":
        ticker = st.sidebar.selectbox("Select stock", NIFTY_50)
    else:
        raw = st.sidebar.text_input(
            "Enter NSE symbol",
            placeholder="e.g. ZOMATO, IRFC"
        )
        if raw:
            with st.sidebar:
                with st.spinner("Validating..."):
                    result = validate_ticker(raw)
            if result:
                st.sidebar.success(f"✓ Found: {result}")
                ticker = result
            else:
                st.sidebar.error("Symbol not found")
                ticker = NIFTY_50[0]
        else:
            ticker = NIFTY_50[0]

    with st.spinner("Loading market data..."):
        df = get_stock_data(ticker, period="2y")
        if df is not None:
            df = add_indicators(df)
        fundamentals = get_fundamentals(ticker)
        nifty_df = get_nifty_data()
        correlation = (
            get_nifty_correlation(df, nifty_df)
            if df is not None and nifty_df is not None else None
        )
        rs = (
            get_relative_strength(df, nifty_df)
            if df is not None and nifty_df is not None else None
        )
        regime = (
            get_market_regime(nifty_df)
            if nifty_df is not None else "unknown"
        )
        support_levels, resistance_levels = (
            get_support_resistance(df)
            if df is not None else ([], [])
        )

    if df is None or df.empty:
        st.error("Could not load data for this ticker.")
    else:
        price = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        change = ((price - prev) / prev) * 100
        sector = get_sector(ticker)
        change_color = t["green"] if change >= 0 else t["red"]
        change_arrow = "▲" if change >= 0 else "▼"

        st.markdown(f"""
        <div style="margin-bottom:20px;">
            <div style="display:flex;align-items:baseline;
            gap:12px;flex-wrap:wrap;">
                <span style="font-size:28px;font-weight:700;
                color:{t['text']};font-family:'JetBrains Mono',
                monospace;">
                    {ticker.replace('.NS', '')}
                </span>
                <span style="font-size:24px;font-weight:600;
                color:{t['text']};font-family:'JetBrains Mono',
                monospace;">
                    ₹{price:,.2f}
                </span>
                <span style="font-size:16px;font-weight:500;
                color:{change_color};">
                    {change_arrow} {abs(change):.2f}%
                </span>
            </div>
            <div style="font-size:13px;color:{t['text2']};
            margin-top:4px;">
                {fundamentals.get('Sector', sector)} · 
                {fundamentals.get('Industry', 'N/A')} · 
                Regime: 
                <span class="regime-{regime}">
                    {regime.upper()}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Price", f"₹{price:,.2f}", f"{change:+.2f}%")
        c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        c3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        c4.metric(
            "52W High",
            f"₹{fundamentals.get('52W High', 'N/A')}"
        )
        c5.metric(
            "52W Low",
            f"₹{fundamentals.get('52W Low', 'N/A')}"
        )
        c6.metric(
            "Nifty Corr",
            f"{correlation}" if correlation else "N/A"
        )

        st.markdown('<div class="divider"></div>',
                    unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color=t["green"],
            decreasing_line_color=t["red"]
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"], name="SMA 20",
            line=dict(color="#f0a500", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"], name="SMA 50",
            line=dict(color=t["blue"], width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_200"], name="SMA 200",
            line=dict(color="#9b59b6", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color=t["text2"], width=1, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color=t["text2"], width=1, dash="dash"),
            fill="tonexty",
            fillcolor=f"{t['text2']}11"
        ))
        for r in resistance_levels:
            fig.add_hline(
                y=r, line_dash="dot",
                line_color=t["red"], line_width=1,
                annotation_text=f"R ₹{r:,.0f}",
                annotation_position="right",
                annotation_font_color=t["red"],
                annotation_font_size=11
            )
        for s in support_levels:
            fig.add_hline(
                y=s, line_dash="dot",
                line_color=t["green"], line_width=1,
                annotation_text=f"S ₹{s:,.0f}",
                annotation_position="right",
                annotation_font_color=t["green"],
                annotation_font_size=11
            )
        fib_levels = get_fibonacci_levels(df)
        for name, level in fib_levels.items():
            fig.add_hline(
                y=level, line_dash="dot",
                line_color=t["gold"], line_width=0.5,
                annotation_text=f"Fib {name}",
                annotation_position="left",
                annotation_font_color=t["gold"],
                annotation_font_size=10
            )
        fig.update_layout(
            title=dict(
                text=f"{ticker} — Price chart",
                font=dict(color=t["text"], size=14)
            ),
            xaxis_rangeslider_visible=False,
            height=520,
            paper_bgcolor=t["card"],
            plot_bgcolor=t["card"],
            font=dict(color=t["text2"], family="Inter"),
            xaxis=dict(
                gridcolor=t["border"],
                showgrid=True
            ),
            yaxis=dict(
                gridcolor=t["border"],
                showgrid=True
            ),
            legend=dict(
                bgcolor=t["bg2"],
                bordercolor=t["border"],
                borderwidth=1,
                font=dict(size=11)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index, y=df["RSI"],
                line=dict(color=t["accent"], width=1.5),
                fill="tozeroy",
                fillcolor=f"{t['accent']}15",
                name="RSI"
            ))
            fig_rsi.add_hline(
                y=70, line_dash="dash",
                line_color=t["red"], line_width=1,
                annotation_text="Overbought",
                annotation_font_color=t["red"],
                annotation_font_size=10
            )
            fig_rsi.add_hline(
                y=30, line_dash="dash",
                line_color=t["green"], line_width=1,
                annotation_text="Oversold",
                annotation_font_color=t["green"],
                annotation_font_size=10
            )
            fig_rsi.update_layout(
                title=dict(
                    text="RSI (14)",
                    font=dict(color=t["text"], size=13)
                ),
                height=220,
                showlegend=False,
                paper_bgcolor=t["card"],
                plot_bgcolor=t["card"],
                font=dict(color=t["text2"]),
                xaxis=dict(gridcolor=t["border"]),
                yaxis=dict(gridcolor=t["border"],
                           range=[0, 100])
            )
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df.index, y=df["MACD"],
                name="MACD",
                line=dict(color=t["blue"], width=1.5)
            ))
            fig_macd.add_trace(go.Scatter(
                x=df.index, y=df["MACD_signal"],
                name="Signal",
                line=dict(color=t["gold"], width=1.5)
            ))
            fig_macd.add_trace(go.Bar(
                x=df.index, y=df["MACD_hist"],
                name="Histogram",
                marker_color=[
                    t["green"] if v >= 0 else t["red"]
                    for v in df["MACD_hist"]
                ],
                opacity=0.6
            ))
            fig_macd.update_layout(
                title=dict(
                    text="MACD",
                    font=dict(color=t["text"], size=13)
                ),
                height=220,
                paper_bgcolor=t["card"],
                plot_bgcolor=t["card"],
                font=dict(color=t["text2"]),
                xaxis=dict(gridcolor=t["border"]),
                yaxis=dict(gridcolor=t["border"]),
                legend=dict(
                    bgcolor="transparent",
                    font=dict(size=10)
                )
            )
            st.plotly_chart(fig_macd, use_container_width=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=[
                t["green"]
                if df["Close"].iloc[i] >= df["Open"].iloc[i]
                else t["red"]
                for i in range(len(df))
            ],
            opacity=0.7,
            name="Volume"
        ))
        fig_vol.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA"],
            name="SMA 20",
            line=dict(color=t["gold"], width=1.5)
        ))
        fig_vol.update_layout(
            title=dict(
                text="Volume",
                font=dict(color=t["text"], size=13)
            ),
            height=180,
            showlegend=False,
            paper_bgcolor=t["card"],
            plot_bgcolor=t["card"],
            font=dict(color=t["text2"]),
            xaxis=dict(gridcolor=t["border"]),
            yaxis=dict(gridcolor=t["border"])
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown(
            '<div class="section-header">Earnings calendar</div>',
            unsafe_allow_html=True
        )
        earnings_map = get_nse_earnings_calendar()
        earnings = get_earnings_status(ticker, earnings_map)
        if earnings["has_upcoming"]:
            if earnings["risk_level"] == "high":
                st.error(
                    f"🚨 EARNINGS ALERT: {earnings['message']} — "
                    f"Consider waiting until after results."
                )
            elif earnings["risk_level"] == "medium":
                st.warning(
                    f"⚠️ {earnings['message']} — "
                    f"Elevated volatility expected."
                )
            elif earnings["risk_level"] == "recent":
                st.info(f"📅 {earnings['message']}")
            else:
                st.info(f"📅 {earnings['message']}")
        else:
            st.success("✅ No earnings due in the next 30 days")

        st.markdown(
            '<div class="section-header">Model signal</div>',
            unsafe_allow_html=True
        )
        with st.spinner("Running ensemble model..."):
            model, scaler, features, accuracy = train_model(ticker)
            signal, confidence, buy_prob, sell_prob = get_signal(
                model, scaler, df, features
            )
            risk_metrics = get_risk_metrics(df)

        signal_badge = (
            "signal-badge-buy" if signal == "BUY"
            else "signal-badge-sell"
        )
        st.markdown(
            f'<span class="{signal_badge}">'
            f'{signal}</span>&nbsp;&nbsp;'
            f'<span style="color:{t["text2"]};font-size:14px;">'
            f'Confidence {confidence:.1%} · '
            f'Accuracy {accuracy:.1%}</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Buy prob", f"{buy_prob:.1%}")
        r2.metric("Sell prob", f"{sell_prob:.1%}")
        r3.metric("Sharpe", risk_metrics["Sharpe Ratio"])
        r4.metric("Max drawdown", risk_metrics["Max Drawdown"])
        r5.metric("Volatility", risk_metrics["Annual Volatility"])
        r6.metric(
            "Rel. strength",
            f"{rs:+.1f}%" if rs else "N/A"
        )

        if fundamentals:
            st.markdown(
                '<div class="section-header">Fundamentals</div>',
                unsafe_allow_html=True
            )
            keys = [k for k in fundamentals
                    if k not in ["Sector", "Industry"]]
            fcols = st.columns(4)
            for i, k in enumerate(keys):
                v = fundamentals[k]
                if k == "Market Cap" and isinstance(
                    v, (int, float)
                ):
                    fcols[i % 4].metric(k, f"₹{v/1e9:.0f}B")
                elif k in [
                    "Dividend Yield", "ROE",
                    "Revenue Growth", "Promoter Holding"
                ] and isinstance(v, float):
                    fcols[i % 4].metric(k, f"{v:.1%}")
                else:
                    fcols[i % 4].metric(
                        k,
                        f"{round(v, 2)}"
                        if isinstance(v, float) else str(v)
                    )

        st.markdown(
            '<div class="section-header">News sentiment</div>',
            unsafe_allow_html=True
        )
        with st.spinner("Analysing news..."):
            news_data = get_news_sentiment(ticker)

        sentiment = news_data["sentiment"]
        sent_conf = news_data["confidence"]
        distribution = news_data["distribution"]
        headlines = news_data["headlines"]
        trend = news_data["trend"]
        risk_flags = news_data["risk_flags"]
        pos_kw = news_data["positive_keywords"]
        neg_kw = news_data["negative_keywords"]
        sources = news_data["sources"]
        sentiment_score = news_data["sentiment_score"]
        headline_count = news_data["headline_count"]

        trend_icon = (
            "📈" if trend == "improving"
            else "📉" if trend == "deteriorating"
            else "➡️"
        )
        sent_color = (
            t["green"] if sentiment == "positive"
            else t["red"] if sentiment == "negative"
            else t["text2"]
        )

        st.markdown(f"""
        <div style="background:{t['card']};
        border:1px solid {t['border']};
        border-radius:10px;padding:16px;margin-bottom:16px;">
            <div style="display:flex;gap:24px;
            flex-wrap:wrap;margin-bottom:12px;">
                <div>
                    <div style="font-size:11px;
                    color:{t['text2']};
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                    margin-bottom:4px;">Sentiment</div>
                    <div style="font-size:18px;
                    font-weight:600;color:{sent_color};">
                        {sentiment.capitalize()}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;
                    color:{t['text2']};
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                    margin-bottom:4px;">Score</div>
                    <div style="font-size:18px;
                    font-weight:600;
                    color:{t['text']};
                    font-family:'JetBrains Mono',monospace;">
                        {sentiment_score:+.2f}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;
                    color:{t['text2']};
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                    margin-bottom:4px;">Trend</div>
                    <div style="font-size:18px;
                    font-weight:600;color:{t['text']};">
                        {trend_icon} {trend.capitalize()}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;
                    color:{t['text2']};
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                    margin-bottom:4px;">Positive</div>
                    <div style="font-size:18px;
                    font-weight:600;color:{t['green']};
                    font-family:'JetBrains Mono',monospace;">
                        {distribution.get('positive', 0):.1%}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;
                    color:{t['text2']};
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                    margin-bottom:4px;">Negative</div>
                    <div style="font-size:18px;
                    font-weight:600;color:{t['red']};
                    font-family:'JetBrains Mono',monospace;">
                        {distribution.get('negative', 0):.1%}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;
                    color:{t['text2']};
                    text-transform:uppercase;
                    letter-spacing:0.5px;
                    margin-bottom:4px;">Headlines</div>
                    <div style="font-size:18px;
                    font-weight:600;color:{t['text']};
                    font-family:'JetBrains Mono',monospace;">
                        {headline_count}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if risk_flags:
            st.markdown(
                f'<div style="background:{t["red"]}22;'
                f'border:1px solid {t["red"]}44;'
                f'border-radius:6px;padding:8px 12px;'
                f'color:{t["red"]};font-size:13px;'
                f'margin-bottom:8px;">⚠️ Risk flags: '
                f'{", ".join(risk_flags)}</div>',
                unsafe_allow_html=True
            )

        if pos_kw:
            tags = "".join(
                [f'<span class="tag" '
                 f'style="color:{t["green"]};'
                 f'border-color:{t["green"]}44;">'
                 f'{k}</span>' for k in pos_kw]
            )
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'<span style="font-size:12px;'
                f'color:{t["text2"]};">Positive: </span>'
                f'{tags}</div>',
                unsafe_allow_html=True
            )

        if neg_kw:
            tags = "".join(
                [f'<span class="tag" '
                 f'style="color:{t["red"]};'
                 f'border-color:{t["red"]}44;">'
                 f'{k}</span>' for k in neg_kw]
            )
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'<span style="font-size:12px;'
                f'color:{t["text2"]};">Negative: </span>'
                f'{tags}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        if sources:
            st.caption(f"Sources: {' · '.join(sources)}")

        st.markdown("**Recent headlines:**")
        for h in headlines:
            text_lower = h.lower()
            has_neg = any(
                w in text_lower
                for w in NEGATIVE_KEYWORDS[:10]
            )
            has_pos = any(
                w in text_lower
                for w in POSITIVE_KEYWORDS[:10]
            )
            color = (
                t["red"] if has_neg
                else t["green"] if has_pos
                else t["text2"]
            )
            icon = "🔴" if has_neg else "🟢" if has_pos else "⚪"
            st.markdown(
                f'<div style="color:{color};font-size:13px;'
                f'padding:6px 0;border-bottom:1px solid '
                f'{t["border"]};">{icon} {h}</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-header">AI analysis</div>',
            unsafe_allow_html=True
        )
        with st.spinner("Generating investment brief..."):
            explanation = explain_signal(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                sentiment=sentiment,
                rsi=float(df["RSI"].iloc[-1]),
                macd=float(df["MACD"].iloc[-1]),
                accuracy=accuracy,
                buy_prob=buy_prob,
                sell_prob=sell_prob,
                sharpe=risk_metrics["Sharpe Ratio"],
                max_drawdown=risk_metrics["Max Drawdown"],
                pe_ratio=fundamentals.get("PE Ratio"),
                week52_high=fundamentals.get("52W High"),
                week52_low=fundamentals.get("52W Low"),
                current_price=price,
                correlation=correlation,
                sector=sector,
                relative_strength=rs,
                market_regime=regime
            )
        st.markdown(
            f'<div class="thesis-card">{explanation}</div>',
            unsafe_allow_html=True
        )

with tab3:
    st.markdown(
        '<div class="section-header">30-day price forecast</div>',
        unsafe_allow_html=True
    )
    try:
        from prophet import Prophet
        ticker_f = st.selectbox(
            "Select stock", NIFTY_50, key="fc"
        )
        with st.spinner("Running Prophet forecast..."):
            df_f = get_stock_data(ticker_f, period="2y")
            if df_f is not None:
                pf = df_f.reset_index()[
                    ["Date", "Close"]
                ].copy()
                pf.columns = ["ds", "y"]
                pf["ds"] = pf["ds"].dt.tz_localize(None)
                m = Prophet(
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    daily_seasonality=False
                )
                m.fit(pf)
                future = m.make_future_dataframe(periods=30)
                forecast = m.predict(future)

                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    x=pf["ds"], y=pf["y"],
                    name="Actual",
                    line=dict(color=t["text"], width=1.5)
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast["yhat"],
                    name="Forecast",
                    line=dict(color=t["accent"], width=2)
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    line=dict(
                        color=f"{t['accent']}33", width=0
                    ),
                    showlegend=False
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    line=dict(
                        color=f"{t['accent']}33", width=0
                    ),
                    fill="tonexty",
                    fillcolor=f"{t['accent']}22",
                    showlegend=False
                ))
                fig_f.update_layout(
                    title=dict(
                        text=f"{ticker_f} — 30-day forecast",
                        font=dict(color=t["text"], size=14)
                    ),
                    height=500,
                    paper_bgcolor=t["card"],
                    plot_bgcolor=t["card"],
                    font=dict(color=t["text2"]),
                    xaxis=dict(gridcolor=t["border"]),
                    yaxis=dict(gridcolor=t["border"]),
                    legend=dict(
                        bgcolor=t["bg2"],
                        bordercolor=t["border"],
                        borderwidth=1
                    )
                )
                st.plotly_chart(fig_f, use_container_width=True)

                last_forecast = float(forecast["yhat"].iloc[-1])
                last_actual = float(pf["y"].iloc[-1])
                change_f = (
                    (last_forecast - last_actual) /
                    last_actual * 100
                )
                f1, f2, f3 = st.columns(3)
                f1.metric(
                    "Current price", f"₹{last_actual:,.2f}"
                )
                f2.metric(
                    "Forecast (30d)",
                    f"₹{last_forecast:,.2f}"
                )
                f3.metric(
                    "Expected change", f"{change_f:+.1f}%"
                )
                st.caption(
                    "Prophet trend + seasonality model. "
                    "Not financial advice."
                )
    except ImportError:
        st.warning(
            "Run `pip install prophet` to enable forecasting."
        )

with tab4:
    st.markdown(
        '<div class="section-header">Portfolio tracker</div>',
        unsafe_allow_html=True
    )
    with st.form("portfolio_form"):
        holdings = []
        for idx in range(5):
            col1, col2, col3 = st.columns(3)
            t_pick = col1.selectbox(
                f"Stock {idx+1}", ["--"] + NIFTY_50,
                key=f"pt{idx}"
            )
            q = col2.number_input(
                "Quantity",
                min_value=0, value=0, key=f"pq{idx}"
            )
            p = col3.number_input(
                "Avg buy price ₹",
                min_value=0.0, value=0.0, key=f"pp{idx}"
            )
            if t_pick != "--" and q > 0:
                holdings.append((t_pick, q, p))

        submitted = st.form_submit_button(
            "Analyse portfolio", type="primary"
        )

    if submitted and holdings:
        total_invested = 0
        total_current = 0
        portfolio_data = []

        for ticker_p, qty, avg_price in holdings:
            try:
                df_p = get_stock_data(ticker_p, period="2y")
                if df_p is None:
                    continue
                df_p = add_indicators(df_p)
                if df_p is None:
                    continue
                current_p = float(df_p["Close"].iloc[-1])
                invested = (
                    qty * avg_price if avg_price > 0
                    else qty * current_p
                )
                current_val = qty * current_p
                pnl = current_val - invested
                pnl_pct = (
                    (pnl / invested * 100)
                    if invested > 0 else 0
                )
                model_p, sc_p, f_p, acc_p = train_model(
                    ticker_p
                )
                sig_p, conf_p, _, _ = get_signal(
                    model_p, sc_p, df_p, f_p
                )
                total_invested += invested
                total_current += current_val
                portfolio_data.append({
                    "Stock": ticker_p.replace(".NS", ""),
                    "Qty": qty,
                    "Avg price": (
                        f"₹{avg_price:,.2f}"
                        if avg_price > 0 else "N/A"
                    ),
                    "Current": f"₹{current_p:,.2f}",
                    "Value": f"₹{current_val:,.2f}",
                    "P&L": (
                        f"₹{pnl:+,.2f}"
                        if avg_price > 0 else "N/A"
                    ),
                    "Return": (
                        f"{pnl_pct:+.1f}%"
                        if avg_price > 0 else "N/A"
                    ),
                    "Signal": sig_p,
                    "Confidence": f"{conf_p:.1%}"
                })
            except Exception:
                continue

        if portfolio_data:
            total_pnl = total_current - total_invested
            total_pnl_pct = (
                (total_pnl / total_invested * 100)
                if total_invested > 0 else 0
            )
            pnl_color = t["green"] if total_pnl >= 0 else t["red"]
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(
                "Total invested",
                f"₹{total_invested:,.2f}"
            )
            pc2.metric(
                "Current value",
                f"₹{total_current:,.2f}"
            )
            pc3.metric(
                "Total P&L",
                f"₹{total_pnl:+,.2f}",
                f"{total_pnl_pct:+.1f}%"
            )

            st.dataframe(
                pd.DataFrame(portfolio_data),
                use_container_width=True
            )

            fig_pie = px.pie(
                values=[d["Qty"] for d in portfolio_data],
                names=[d["Stock"] for d in portfolio_data],
                title="Portfolio allocation",
                color_discrete_sequence=[
                    t["accent"], t["green"], t["blue"],
                    t["gold"], t["red"]
                ]
            )
            fig_pie.update_layout(
                paper_bgcolor=t["card"],
                font=dict(color=t["text"]),
                title_font_color=t["text"]
            )
            st.plotly_chart(fig_pie, use_container_width=True)