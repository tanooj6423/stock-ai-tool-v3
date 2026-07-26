import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import time
import yfinance as yf
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
import branding
from universe import ALL_STOCKS, NIFTY_50, COMMODITIES, get_sector
from earnings import (get_earnings_status,
                      get_nse_earnings_calendar,
                      get_nse_fii_dii_flow,
                      get_dividend_risk,
                      get_dividend_exdates)
from trade_instructions import (get_entry_instruction,
                                 check_entry_validity)
from journal import render_journal_tab, add_trade, load_journal
from watchlist import (render_watchlist_tab, add_to_watchlist,
                       load_watchlist)
from broker import (render_zerodha_panel, get_live_quote,
                    is_connected, is_market_hours,
                    place_gtt_order)

load_dotenv()

st.set_page_config(
    page_title="Equitex Intelligence",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Koyfin-inspired design system.
# Two skins of one visual language: slate-navy surfaces,
# indigo brand accent, hairline borders, pill buttons,
# underline tabs, mono numerals.
THEMES = {
    "Koyfin Dark": {
        "bg": "#0e1116",
        "bg2": "#12161d",
        "card": "#161b23",
        "border": "#242b37",
        "green": "#00c48c",
        "red": "#f6465d",
        "blue": "#3e7bfa",
        "gold": "#f5a623",
        "text": "#e8ecf3",
        "text2": "#8b94a7",
        "accent": "#6366f1",
        "accent_rgb": "99,102,241",
        "green_rgb": "0,196,140",
        "red_rgb": "246,70,93",
        "btn_text": "#ffffff",
        "success": "#00c48c",
        "danger": "#f6465d",
        "warn": "#f5a623",
    },
    "Koyfin Light": {
        "bg": "#ffffff",
        "bg2": "#f7f8fa",
        "card": "#ffffff",
        "border": "#e4e7ee",
        "green": "#0ba05f",
        "red": "#d92d20",
        "blue": "#2563eb",
        "gold": "#b45309",
        "text": "#101828",
        "text2": "#667085",
        "accent": "#4f46e5",
        "accent_rgb": "79,70,229",
        "green_rgb": "11,160,95",
        "red_rgb": "217,45,32",
        "btn_text": "#ffffff",
        "success": "#0ba05f",
        "danger": "#d92d20",
        "warn": "#b45309",
    },
}

if "theme" not in st.session_state:
    st.session_state.theme = "Koyfin Dark"
if st.session_state.theme not in THEMES:
    # Migrate sessions saved on old theme names
    st.session_state.theme = "Koyfin Dark"

t = THEMES[st.session_state.theme]

def hex_to_rgba(hex_color, alpha=0.1):
    h = hex_color.lstrip('#')
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
/* Hide Streamlit chrome so it looks like a product,
   not a Streamlit demo */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{
    background: transparent; height: 0;
}}
div[data-testid="stToolbar"] {{ display: none; }}
div[data-testid="stDecoration"] {{ display: none; }}
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}
.stApp {{ background-color: {t['bg']}; }}
section[data-testid="stSidebar"] {{
    background-color: {t['bg2']};
    border-right: 1px solid {t['border']};
}}
section[data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
.block-container {{ padding-top: 24px; padding-bottom: 48px; }}
.app-header {{
    display: flex; align-items: baseline;
    gap: 14px; flex-wrap: wrap;
    padding-bottom: 18px;
    margin-bottom: 8px;
    border-bottom: 1px solid {t['border']};
}}
.app-name {{
    font-size: 19px; font-weight: 700;
    color: {t['text']}; letter-spacing: -0.3px;
    margin: 0;
}}
.app-name .brand-dot {{
    color: {t['accent']};
}}
.app-tagline {{ font-size: 12px; color: {t['text2']}; }}
.section-label {{
    font-size: 10px; font-weight: 700;
    color: {t['text2']}; text-transform: uppercase;
    letter-spacing: 2px; margin: 28px 0 12px 0;
}}
.badge-buy {{
    font-size: 11px; font-weight: 600;
    color: {t['green']};
    background: rgba({t['green_rgb']},0.12);
    border: 1px solid rgba({t['green_rgb']},0.28);
    padding: 3px 12px; border-radius: 999px;
    letter-spacing: 0.8px;
}}
.badge-sell {{
    font-size: 11px; font-weight: 600;
    color: {t['red']};
    background: rgba({t['red_rgb']},0.12);
    border: 1px solid rgba({t['red_rgb']},0.28);
    padding: 3px 12px; border-radius: 999px;
    letter-spacing: 0.8px;
}}
.badge-score {{
    font-size: 11px; font-weight: 600;
    color: {t['accent']};
    background: rgba({t['accent_rgb']},0.12);
    border: 1px solid rgba({t['accent_rgb']},0.28);
    padding: 3px 12px; border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
}}
.pick-meta {{ font-size: 12px; color: {t['text2']}; }}
.score-track {{
    height: 2px; background: {t['border']};
    border-radius: 2px; margin: 10px 0 16px 0;
    overflow: hidden;
}}
.score-fill {{
    height: 2px; border-radius: 2px;
    background: linear-gradient(90deg, {t['accent']}, {t['green']});
}}
.level-label {{
    font-size: 10px; color: {t['text2']};
    text-transform: uppercase; letter-spacing: 1.5px;
    margin: 20px 0 8px 0; padding-bottom: 6px;
    border-bottom: 1px solid {t['border']};
}}
.thesis-box {{
    background: rgba({t['accent_rgb']},0.06);
    border-left: 2px solid {t['accent']};
    border-radius: 0 6px 6px 0;
    padding: 14px 16px; font-size: 13px;
    color: {t['text']}; line-height: 1.75;
    margin-top: 8px;
}}
.instruction-box {{
    background: rgba({t['green_rgb']},0.06);
    border: 1px solid rgba({t['green_rgb']},0.25);
    border-left: 3px solid {t['green']};
    border-radius: 0 10px 10px 0;
    padding: 16px 18px; font-size: 13px;
    color: {t['text']}; line-height: 1.85;
    margin: 8px 0;
}}
.instruction-summary {{
    background: rgba({t['accent_rgb']},0.08);
    border: 1px solid rgba({t['accent_rgb']},0.2);
    border-radius: 6px; padding: 10px 14px;
    font-size: 12px; font-weight: 600;
    color: {t['accent']};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 10px;
}}
.validity-valid {{
    background: rgba({t['green_rgb']},0.1);
    border: 1px solid rgba({t['green_rgb']},0.3);
    border-radius: 6px; padding: 10px 14px;
    font-size: 13px; color: {t['green']};
}}
.validity-invalid {{
    background: rgba({t['red_rgb']},0.1);
    border: 1px solid rgba({t['red_rgb']},0.3);
    border-radius: 6px; padding: 10px 14px;
    font-size: 13px; color: {t['red']};
}}
.validity-watch {{
    background: rgba(79,172,254,0.1);
    border: 1px solid rgba(79,172,254,0.3);
    border-radius: 6px; padding: 10px 14px;
    font-size: 13px; color: {t['blue']};
}}
.driver-item {{
    display: flex; gap: 8px;
    align-items: flex-start;
    padding: 5px 0; font-size: 13px;
    color: {t['text']};
}}
.breakdown-table {{
    width: 100%; border-collapse: collapse;
    font-size: 12px; margin: 8px 0;
}}
.breakdown-table th {{
    text-align: left; padding: 6px 10px;
    color: {t['text2']}; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid {t['border']};
}}
.breakdown-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid {t['border']};
    color: {t['text']};
}}
.status-row {{
    display: flex; gap: 20px; align-items: center;
    background: {t['card']};
    border: 1px solid {t['border']};
    border-radius: 10px; padding: 12px 18px;
    margin-bottom: 20px; font-size: 12px;
    flex-wrap: wrap;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}
.status-item {{ color: {t['text2']}; }}
.status-value {{
    color: {t['text']}; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}}
.regime-bull {{ color: {t['green']}; font-weight: 700; }}
.regime-bear {{ color: {t['red']}; font-weight: 700; }}
.regime-sideways {{ color: {t['gold']}; font-weight: 700; }}
.regime-unknown {{ color: {t['text2']}; font-weight: 700; }}
.divider {{ height: 1px; background: {t['border']}; margin: 24px 0; }}
.sentiment-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px; margin-bottom: 12px;
}}
.sentiment-cell {{
    background: {t['bg2']};
    border: 1px solid {t['border']};
    border-radius: 10px; padding: 10px 12px;
}}
.sentiment-cell-label {{
    font-size: 9px; color: {t['text2']};
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 4px;
}}
.sentiment-cell-value {{
    font-size: 16px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    color: {t['text']};
}}
.headline-item {{
    display: flex; gap: 8px;
    align-items: flex-start;
    padding: 6px 0;
    border-bottom: 1px solid {t['border']};
    font-size: 12px;
}}
.pos-text {{ color: {t['green']}; }}
.neg-text {{ color: {t['red']}; }}
.neu-text {{ color: {t['text2']}; }}
.keyword-tag {{
    display: inline-block; font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 7px; border-radius: 3px; margin: 2px;
}}
.keyword-pos {{
    color: {t['green']};
    background: rgba({t['green_rgb']},0.1);
    border: 1px solid rgba({t['green_rgb']},0.25);
}}
.keyword-neg {{
    color: {t['red']};
    background: rgba({t['red_rgb']},0.1);
    border: 1px solid rgba({t['red_rgb']},0.25);
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: {t['text']} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.68rem !important;
    color: {t['text2']} !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}}
/* Koyfin-style underline navigation tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    border: none;
    border-bottom: 1px solid {t['border']};
    border-radius: 0; padding: 0; gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 0; color: {t['text2']};
    font-size: 13px; font-weight: 500;
    padding: 10px 16px; letter-spacing: 0.2px;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {t['text']};
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: {t['text']} !important;
    border: none !important;
    border-bottom: 2px solid {t['accent']} !important;
    font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    background: transparent !important;
}}
.stExpander {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
}}
/* Koyfin-style pill CTA buttons */
.stButton > button {{
    background: {t['accent']} !important;
    color: {t['btn_text']} !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.2px !important;
    padding: 9px 22px !important;
    box-shadow: 0 1px 3px rgba({t['accent_rgb']},0.35) !important;
    transition: filter .15s ease,
        transform .05s ease !important;
}}
.stButton > button:hover {{
    filter: brightness(1.1) !important;
}}
.stButton > button:active {{
    transform: scale(0.98) !important;
}}
.stSelectbox > div > div,
.stTextInput > div > div > input {{
    background: {t['bg2']} !important;
    border: 1px solid {t['border']} !important;
    color: {t['text']} !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus {{
    border-color: rgba({t['accent_rgb']},0.6) !important;
    box-shadow: 0 0 0 3px
        rgba({t['accent_rgb']},0.15) !important;
}}
.stProgress > div > div > div {{
    background: {t['accent']} !important;
}}
.stProgress > div > div {{
    background: {t['border']} !important;
}}
.sidebar-brand {{
    font-size: 15px; font-weight: 700;
    color: {t['text']}; letter-spacing: -0.2px;
    padding: 12px 0 16px 0;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 20px;
}}
.sidebar-brand .brand-dot {{ color: {t['accent']}; }}
.sidebar-label {{
    font-size: 9px; font-weight: 700;
    color: {t['text2']}; text-transform: uppercase;
    letter-spacing: 2px; margin: 16px 0 8px 0;
}}
/* Secondary buttons: quiet outline pills */
.stButton > button[kind="secondary"] {{
    background: transparent !important;
    color: {t['text']} !important;
    border: 1px solid {t['border']} !important;
    box-shadow: none !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: rgba({t['accent_rgb']},0.6) !important;
    color: {t['accent']} !important;
    filter: none !important;
}}
/* Alerts (st.info / success / error / warning) */
div[data-testid="stAlert"] {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    color: {t['text']} !important;
    font-size: 13px !important;
}}
div[data-testid="stAlert"] p {{
    color: {t['text']} !important;
    font-size: 13px !important;
}}
/* Dataframes / tables */
div[data-testid="stDataFrame"] {{
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}}
/* Forms */
div[data-testid="stForm"] {{
    background: {t['card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 12px !important;
    padding: 20px !important;
}}
/* Number inputs & sliders */
.stNumberInput input {{
    background: {t['bg2']} !important;
    border: 1px solid {t['border']} !important;
    color: {t['text']} !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{
    background: {t['accent']} !important;
    border-color: {t['accent']} !important;
}}
/* Checkboxes / radios */
.stCheckbox p, .stRadio p {{
    font-size: 13px !important;
    color: {t['text']} !important;
}}
/* Expander header text */
.stExpander summary p,
.stExpander summary span {{
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {t['text']} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.stExpander summary:hover {{
    color: {t['accent']} !important;
}}
/* Thin themed scrollbars */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: {t['bg']}; }}
::-webkit-scrollbar-thumb {{
    background: {t['border']}; border-radius: 99px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: rgba({t['accent_rgb']},0.5);
}}
/* Text selection in brand color */
::selection {{
    background: rgba({t['accent_rgb']},0.3);
}}
/* Tab bar scrolls instead of wrapping on small screens */
.stTabs [data-baseweb="tab-list"] {{
    overflow-x: auto; flex-wrap: nowrap;
    scrollbar-width: none;
}}
/* Link color */
a, a:visited {{ color: {t['accent']} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Authentication gate — everything below requires login
# ---------------------------------------------------------
import auth
from account_ui import (render_login_gate, render_pricing,
                        render_account_panel,
                        render_upgrade_nudge,
                        render_disclaimer_gate,
                        StagedProgress)

user = render_login_gate(t)
render_disclaimer_gate(t, user)
IS_PRO = auth.is_pro(user)
plan_badge = (
    f'<span style="background:rgba({t["accent_rgb"]},0.15);'
    f'color:{t["accent"]};font-size:10px;font-weight:700;'
    f'padding:3px 10px;border-radius:99px;'
    f'letter-spacing:1px;">PRO</span>'
    if IS_PRO else
    f'<span style="background:{t["bg2"]};'
    f'color:{t["text2"]};font-size:10px;font-weight:700;'
    f'padding:3px 10px;border-radius:99px;'
    f'border:1px solid {t["border"]};'
    f'letter-spacing:1px;">FREE</span>'
)

st.markdown(f"""
<div class="app-header">
    <div class="app-name">equitex<span
        class="brand-dot">.</span></div>
    <div class="app-tagline">
        The Equitex Score for every NSE stock ·
        Quantitative research tool — not investment advice ·
        Not SEBI-registered
    </div>
    <div style="margin-left:auto;display:flex;
         align-items:center;gap:10px;">
        {plan_badge}
        <span style="font-size:11px;
            color:{t['text2']};">{user['email']}</span>
    </div>
</div>
""", unsafe_allow_html=True)

from scan_store import load_scan, save_scan, log_picks
from track_record_tab import render_track_record_tab

(tab1, tab2, tab3, tab4, tab11, tab12, tab5, tab6, tab7,
 tab9, tab10) = st.tabs([
         "Screener",
         "Level Check",
         "Stock Analysis",
         "Scenarios",
         "Macro",
         "Paper Trade",
         "Watchlist",
         "Journal",
         "Settings",
         "Performance",
         "Plans",
     ])

with tab7:
    render_account_panel(t, user)

    st.markdown(
        '<div class="section-label">Display</div>',
        unsafe_allow_html=True
    )
    theme_choice = st.selectbox(
        "Theme", list(THEMES.keys()),
        index=list(THEMES.keys()).index(
            st.session_state.theme
        )
    )
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.markdown(
        '<div class="section-label">'
        'Trading parameters</div>',
        unsafe_allow_html=True
    )
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
    st.markdown(
        '<div class="section-label">'
        'Scan settings</div>',
        unsafe_allow_html=True
    )
    scan_universe = st.selectbox(
        "Scan universe",
        ["Nifty 50 only",
         "Nifty 50 + Next 50",
         "Nifty 50 + Next 50 + Midcap 150",
         "Top 500 curated stocks",
         "All NSE listed stocks (~2000)"],
        index=1
    )
    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )
    render_zerodha_panel(t)

with tab1:
    st.sidebar.markdown(
        f'<div class="sidebar-brand">equitex'
        f'<span class="brand-dot">.</span></div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f'<div class="sidebar-label">Navigation</div>',
        unsafe_allow_html=True
    )
    mode = st.sidebar.radio(
        "", ["Popular stocks", "Search any NSE stock"],
        label_visibility="collapsed"
    )

    from universe import (NIFTY_NEXT_50,
                          NIFTY_MIDCAP_150,
                          ALL_STOCKS,
                          get_nse_all_stocks)
    if scan_universe == "Nifty 50 only":
        scan_tickers = NIFTY_50
    elif scan_universe == "Nifty 50 + Next 50":
        scan_tickers = NIFTY_50 + NIFTY_NEXT_50
    elif scan_universe == (
        "Nifty 50 + Next 50 + Midcap 150"
    ):
        scan_tickers = (
            NIFTY_50 + NIFTY_NEXT_50 +
            NIFTY_MIDCAP_150
        )
    elif scan_universe == "Top 500 curated stocks":
        scan_tickers = ALL_STOCKS
    else:
        with st.spinner(
            "Fetching full NSE stock list..."
        ):
            scan_tickers = get_nse_all_stocks()
        st.caption(
            f"Full NSE universe: "
            f"{len(scan_tickers)} stocks loaded. "
            f"Pre-filter will run first (~90 seconds) "
            f"then full scan (~15-20 minutes)."
        )

    from macro import render_macro_banner, get_macro_risk
    render_macro_banner(t, days_ahead=10)

    top_row_l, top_row_r = st.columns([5, 1])
    with top_row_l:
        st.markdown(
            '<div class="section-label">'
            'Quantitative screen — '
            'model-ranked setups</div>',
            unsafe_allow_html=True
        )
    # Live rescans hammer the server (trains ~100
    # models) — admin-only. Everyone else is always
    # served the precomputed nightly scan.
    run_scan = False
    with top_row_r:
        if auth.is_admin(user):
            run_scan = st.button("↺ Rescan",
                                 type="primary")

    # Serve the freshest precomputed scan available.
    # Weekends/holidays: yesterday's scan stays valid,
    # so fall back up to 4 days before giving up.
    precomputed = None if run_scan else (
        load_scan(max_age_hours=20)
        or load_scan(max_age_hours=96)
    )

    if precomputed:
        picks, regime, scanned_at = precomputed
        st.caption(
            f"Model scan · {scanned_at} · refreshed "
            f"after each market close"
        )
    elif not run_scan:
        # No precomputed results and not admin —
        # never auto-run a 10-minute scan on a page
        # load; show a friendly state instead.
        st.markdown(f"""
        <div style="text-align:center;padding:48px;
        background:{t['card']};
        border:1px solid {t['border']};
        border-radius:12px;color:{t['text2']};">
            <div style="font-size:15px;
            font-weight:600;color:{t['text']};
            margin-bottom:6px;">
                Today's screen is being prepared
            </div>
            <div style="font-size:13px;">
                The screen runs after each market
                close. Check back shortly.
            </div>
        </div>
        """, unsafe_allow_html=True)
        picks, regime = [], "unknown"
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_text = st.empty()
        start_time = time.time()

        def update_progress(current, total,
                            current_ticker):
            pct = int((current / total) * 100)
            elapsed = time.time() - start_time
            rate = elapsed / max(current, 1)
            remaining = int(rate * (total - current))
            mins = remaining // 60
            secs = remaining % 60
            progress_bar.progress(pct)
            status_text.markdown(
                f'<span style="font-size:12px;'
                f'color:{t["text2"]};">'
                f'Scanning '
                f'{current_ticker.replace(".NS","")} '
                f'({current}/{total})</span>',
                unsafe_allow_html=True
            )
            time_text.markdown(
                f'<span style="font-size:11px;'
                f'color:{t["text2"]};">'
                f'Est. remaining: {mins}m {secs}s</span>',
                unsafe_allow_html=True
            )

        picks, regime = run_full_scan(
            tuple(scan_tickers),
            capital=capital,
            risk_pct=risk_pct,
            progress_callback=update_progress
        )
        progress_bar.empty()
        status_text.empty()
        time_text.empty()
        save_scan(picks, regime,
                  universe_name=scan_universe)
        log_picks(picks)

    regime_icons = {
        "bull": "●", "bear": "●",
        "sideways": "●", "unknown": "○"
    }
    regime_icon = regime_icons.get(regime, "○")

    scan_available = (
        precomputed is not None or run_scan
    )

    if scan_available:
        st.markdown(f"""
    <div class="status-row">
        <span class="status-item">Market &nbsp;
            <span class="regime-{regime}">
                {regime_icon} {regime.upper()}
            </span>
        </span>
        <span style="color:{t['border']};">·</span>
        <span class="status-item">Scanned &nbsp;
            <span class="status-value">
                {len(scan_tickers)}
            </span>
        </span>
        <span style="color:{t['border']};">·</span>
        <span class="status-item">Setups &nbsp;
            <span class="status-value"
            style="color:{t['accent']};">
                {len(picks)}
            </span>
        </span>
        <span style="color:{t['border']};">·</span>
        <span class="status-item">
            {"🟢 Zerodha connected" if is_connected() else "Zerodha not connected"}
        </span>
        <span style="color:{t['border']};">·</span>
        <span class="status-item">FII &nbsp;
            <span style="color:{'#c0392b' if picks and picks[0].get('fii_bearish') else '#2d7d5a'};">
                {"● Selling" if picks and picks[0].get('fii_bearish') else "● Neutral"}
            </span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    if scan_available and not picks:
        st.markdown(f"""
        <div style="text-align:center;padding:48px;
        background:{t['card']};
        border:1px solid {t['border']};
        border-radius:8px;color:{t['text2']};">
            <div style="font-size:15px;font-weight:600;
            color:{t['text']};margin-bottom:6px;">
                No qualifying setups today
            </div>
            <div style="font-size:13px;">
                No stocks passed all screening criteria.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        visible_picks = (
            picks if IS_PRO
            else picks[:auth.FREE_PICKS_VISIBLE]
        )
        for i, pick in enumerate(visible_picks):
            score = pick["score"]
            signal = pick["signal"]
            display_signal = (
                "BULLISH" if signal == "BUY"
                else "BEARISH"
            )
            badge_class = (
                "badge-buy" if signal == "BUY"
                else "badge-sell"
            )
            risk_level = pick.get("risk_level", "MEDIUM")
            risk_colors = {
                "LOW": t["green"],
                "MEDIUM": t["gold"],
                "HIGH": t["red"]
            }
            risk_color = risk_colors.get(
                risk_level, t["gold"]
            )

            instr = get_entry_instruction(
                pick, capital, risk_pct
            )

            live_price = None
            if is_connected() and is_market_hours():
                quote = get_live_quote(pick["ticker"])
                if quote:
                    live_price = quote["price"]

            display_price = (
                live_price if live_price
                else pick["price"]
            )
            price_label = (
                "🟢 LIVE" if live_price else "Delayed"
            )

            with st.expander(
                f"#{i+1}  "
                f"{pick['ticker'].replace('.NS','')}  ·  "
                f"Equitex {score} "
                f"({branding.score_tier(score)[0]})  ·  "
                f"{display_signal} "
                f"{pick['confidence']:.0%}  ·  "
                f"₹{display_price:,.2f}  ·  "
                f"Risk: {risk_level}  ·  "
                f"Hold: {pick.get('holding_days',7)}d",
                expanded=(i < 3)
            ):
                st.markdown(
                    branding.score_gauge_html(score, t),
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="display:flex;'
                    f'align-items:center;gap:10px;'
                    f'margin:10px 0 4px;flex-wrap:wrap;">'
                    f'<span class="{badge_class}">'
                    f'{display_signal}</span>'
                    f'<span style="font-size:11px;'
                    f'font-weight:700;color:{risk_color};">'
                    f'● {risk_level} RISK</span>'
                    f'<span class="pick-meta">'
                    f'{pick["sector"]} · '
                    f'Conf {pick["confidence"]:.1%} · '
                    f'Hold {pick.get("holding_days",7)}d · '
                    f'{price_label}'
                    f'</span></div>'
                    f'<div class="score-track">'
                    f'<div class="score-fill" '
                    f'style="width:{score}%;"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if pick.get("earnings_message"):
                    if pick.get("earnings_risk") == "medium":
                        st.warning(
                            f"⚠️ {pick['earnings_message']}"
                        )

                if pick.get("div_message"):
                    st.error(pick["div_message"])

                if pick.get("fii_bearish"):
                    st.warning(
                        f"⚠️ FII net sellers for "
                        f"{pick.get('fii_selling_days', 0)} "
                        f"consecutive days — position "
                        f"size halved automatically."
                    )

                pct_high = pick.get("pct_from_52w_high", 10)
                if pct_high < 5:
                    st.warning(
                        f"⚠️ Only {pct_high:.1f}% below "
                        f"52W high — strong resistance "
                        f"overhead. Tight target."
                    )

                drivers = pick.get("key_drivers", [])
                if drivers:
                    st.markdown(
                        '<div class="level-label">'
                        'Signal drivers</div>',
                        unsafe_allow_html=True
                    )
                    drivers_html = "".join([
                        f'<div class="driver-item">'
                        f'<span style="color:'
                        f'{t["accent"]};">→</span> '
                        f'{d}</div>'
                        for d in drivers
                    ])
                    st.markdown(
                        drivers_html,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="level-label">'
                    'Model scenario</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="instruction-summary">'
                    f'{instr["summary"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="instruction-box">'
                    f'{instr["instruction"]}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="level-label">'
                    'Reference levels</div>',
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric("Entry", f"₹{pick['entry']:,.2f}")
                c2.metric("Stop loss",
                          f"₹{pick['stop_loss']:,.2f}")
                c3.metric("Target 1",
                          f"₹{pick['target1']:,.2f}")
                c4.metric("Target 2",
                          f"₹{pick['target2']:,.2f}")
                c5.metric("R/R", f"1:{pick['rr1']:.1f}")
                c6.metric(
                    "Shares",
                    f"{pick['shares']}"
                    if pick['shares'] > 0 else "—"
                )
                max_loss_val = pick.get("max_loss") or (
                    pick.get("risk_amount", 0) *
                    pick.get("shares", 1)
                )
                c7.metric("Max loss",
                          f"₹{max_loss_val:,.2f}")

                st.markdown(
                    '<div class="level-label">'
                    'Signal breakdown</div>',
                    unsafe_allow_html=True
                )
                breakdown_data = pick.get(
                    "signal_breakdown", []
                )
                if breakdown_data:
                    rows = "".join([
                        f"<tr>"
                        f"<td>{row['Factor']}</td>"
                        f"<td>{row['Reading']}</td>"
                        f"<td>{row['Status']}</td>"
                        f"<td>{row['Signal']}</td>"
                        f"</tr>"
                        for row in breakdown_data
                    ])
                    st.markdown(
                        f'<table class="breakdown-table">'
                        f'<thead><tr>'
                        f'<th>Factor</th>'
                        f'<th>Reading</th>'
                        f'<th>Status</th>'
                        f'<th>Signal</th>'
                        f'</tr></thead>'
                        f'<tbody>{rows}</tbody>'
                        f'</table>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="level-label">'
                    'Composite score</div>',
                    unsafe_allow_html=True
                )
                breakdown = pick["score_breakdown"]
                cols = st.columns(len(breakdown))
                for j, (layer, pts) in enumerate(
                    breakdown.items()
                ):
                    cols[j].metric(layer, pts)

                st.markdown(
                    '<div class="level-label">'
                    'AI trade thesis</div>',
                    unsafe_allow_html=True
                )
                with st.spinner(""):
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
                        relative_strength=pick[
                            "relative_strength"]
                    )
                st.markdown(
                    f'<div class="thesis-box">'
                    f'{thesis}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="level-label">'
                    'Actions</div>',
                    unsafe_allow_html=True
                )
                ab0, ab1, ab2, ab3 = st.columns(4)
                with ab0:
                    if st.button(
                        "▶ Track this setup",
                        key=f"track_{pick['ticker']}",
                        type="primary"
                    ):
                        from paper_trade import track_setup
                        ok_pt, msg_pt = track_setup(
                            pick["ticker"], pick["entry"],
                            pick["stop_loss"], pick["target1"],
                            max(pick["shares"], 1),
                            score=pick["score"]
                        )
                        if ok_pt:
                            st.success(
                                "Tracking in your Paper Trade "
                                "book (virtual)."
                            )
                        else:
                            st.info(msg_pt)
                with ab1:
                    if st.button(
                        "📋 Log this trade",
                        key=f"log_{pick['ticker']}"
                    ):
                        add_trade(
                            ticker=pick["ticker"],
                            signal=pick["signal"],
                            entry_price=pick["entry"],
                            stop_loss=pick["stop_loss"],
                            target1=pick["target1"],
                            target2=pick["target2"],
                            shares=pick["shares"],
                            capital_at_risk=max_loss_val,
                            holding_days=pick.get(
                                "holding_days", 7
                            ),
                            score=pick["score"],
                            confidence=pick["confidence"],
                            sector=pick["sector"]
                        )
                        st.success("Trade logged!")
                with ab2:
                    if st.button(
                        "👁 Add to watchlist",
                        key=f"watch_{pick['ticker']}"
                    ):
                        add_to_watchlist(
                            pick["ticker"],
                            alert_price=pick["target1"],
                            notes=f"Score {pick['score']}"
                        )
                        st.success("Added to watchlist!")
                with ab3:
                    if is_connected():
                        if st.button(
                            "⚡ Place GTT orders",
                            key=f"gtt_{pick['ticker']}"
                        ):
                            result, msg = place_gtt_order(
                                pick["ticker"],
                                pick["entry"],
                                pick["stop_loss"],
                                pick["target1"],
                                pick["shares"]
                            )
                            if result:
                                st.success(msg)
                            else:
                                st.error(msg)

        if not IS_PRO and len(picks) > len(visible_picks):
            hidden = len(picks) - len(visible_picks)
            render_upgrade_nudge(
                t,
                f"{hidden} more ranked pick"
                f"{'s' if hidden > 1 else ''} in today's scan. "
                f"Upgrade to Pro (₹399/mo) to see the full "
                f"ranked list — see the Plans tab."
            )

with tab2:
    st.markdown(
        '<div class="section-label">'
        'Morning entry validity check</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="font-size:13px;'
        f'color:{t["text2"]};margin-bottom:16px;">'
        f'Run this after 9:30am to check if picks '
        f'are still valid given today\'s price. '
        f'{"Live prices via Zerodha." if is_connected() and is_market_hours() else "Using delayed prices."}'
        f'</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🔍 Check all picks now", type="primary"
    ):
        if not picks:
            st.warning(
                "No picks loaded. "
                "Run the daily scan first."
            )
        else:
            st.markdown(
                '<div class="section-label">'
                'Validity results</div>',
                unsafe_allow_html=True
            )
            for pick in picks:
                try:
                    live_quote = (
                        get_live_quote(pick["ticker"])
                        if is_connected() and is_market_hours()
                        else None
                    )
                    if live_quote:
                        current_price = live_quote["price"]
                    else:
                        live = yf.Ticker(
                            pick["ticker"]
                        ).history(period="1d")
                        if live is None or live.empty:
                            continue
                        current_price = float(
                            live["Close"].iloc[-1]
                        )
                    validity = check_entry_validity(
                        pick, current_price
                    )
                    css_class = (
                        "validity-valid"
                        if validity["valid"]
                        else "validity-invalid"
                        if validity["status"] in [
                            "INVALID", "GAPPED UP"
                        ]
                        else "validity-watch"
                    )
                    ticker_name = pick[
                        "ticker"
                    ].replace(".NS", "")
                    st.markdown(
                        f'<div class="{css_class}" '
                        f'style="margin-bottom:8px;">'
                        f'<strong>{ticker_name}</strong>'
                        f' · ₹{current_price:,.2f} · '
                        f'<strong>'
                        f'{validity["status"]}'
                        f'</strong> — '
                        f'{validity["reason"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    continue

    st.markdown(
        '<div class="section-label">'
        'Check a specific stock</div>',
        unsafe_allow_html=True
    )
    with st.form("manual_check"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        check_ticker = mc1.text_input(
            "NSE symbol", placeholder="e.g. NTPC"
        )
        check_entry = mc2.number_input(
            "Your entry price",
            min_value=0.0, value=0.0
        )
        check_stop = mc3.number_input(
            "Your stop loss",
            min_value=0.0, value=0.0
        )
        if mc4.form_submit_button("Check"):
            if check_ticker and check_entry > 0:
                try:
                    ticker_str = check_ticker.upper()
                    if not ticker_str.endswith(".NS"):
                        ticker_str += ".NS"
                    live_quote = (
                        get_live_quote(ticker_str)
                        if is_connected() and is_market_hours()
                        else None
                    )
                    if live_quote:
                        current_price = live_quote["price"]
                    else:
                        live = yf.Ticker(
                            ticker_str
                        ).history(period="1d")
                        current_price = float(
                            live["Close"].iloc[-1]
                        )
                    mock_pick = {
                        "entry": check_entry,
                        "stop_loss": check_stop,
                        "target1": check_entry * 1.05,
                        "target2": check_entry * 1.10
                    }
                    validity = check_entry_validity(
                        mock_pick, current_price
                    )
                    css_class = (
                        "validity-valid"
                        if validity["valid"]
                        else "validity-invalid"
                    )
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<strong>{check_ticker}</strong>'
                        f' · Current ₹{current_price:,.2f}'
                        f' · {validity["status"]} — '
                        f'{validity["reason"]}</div>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.error("Could not fetch price.")

with tab3:
    st.sidebar.markdown(
        f'<div class="sidebar-label">'
        f'Stock selection</div>',
        unsafe_allow_html=True
    )
    if mode == "Popular stocks":
        ticker = st.sidebar.selectbox(
            "Select stock", NIFTY_50
        )
    else:
        raw = st.sidebar.text_input(
            "NSE symbol",
            placeholder="e.g. ZOMATO, IRFC"
        )
        if raw:
            with st.sidebar:
                with st.spinner(""):
                    result = validate_ticker(raw)
            if result:
                st.sidebar.success(f"✓ {result}")
                ticker = result
            else:
                st.sidebar.error("Not found")
                ticker = NIFTY_50[0]
        else:
            ticker = NIFTY_50[0]

    # ---- Free-tier quota: N distinct analyses/day ----
    if "analyzed_today" not in st.session_state:
        st.session_state.analyzed_today = set()
    _new_ticker = ticker not in st.session_state.analyzed_today
    _quota_blocked = (
        not IS_PRO and _new_ticker
        and not auth.can_analyze(user)
    )

    if _quota_blocked:
        render_upgrade_nudge(
            t,
            f"You've used your "
            f"{auth.FREE_ANALYSES_PER_DAY} free analyses "
            f"today. Upgrade to Pro for unlimited stock "
            f"analysis — see the Plans tab."
        )
        df = None
        fundamentals = nifty_df = correlation = rs = None
        regime = "unknown"
        support_levels, resistance_levels = [], []
    else:
        if _new_ticker:
            auth.record_analysis(user)
            st.session_state.analyzed_today.add(ticker)
        prog = StagedProgress(t, [
            ("Fetching 2 years of price history", 2),
            ("Computing 41 technical indicators", 1),
            ("Loading fundamentals", 2),
            ("Benchmarking vs Nifty & VIX", 2),
            ("Mapping support / resistance levels", 1),
        ])
        prog.step()
        df = get_stock_data(ticker, period="2y")
        prog.step()
        if df is not None:
            df = add_indicators(df)
        prog.step()
        fundamentals = get_fundamentals(ticker)
        prog.step()
        nifty_df = get_nifty_data()
        correlation = (
            get_nifty_correlation(df, nifty_df)
            if df is not None and nifty_df is not None
            else None
        )
        rs = (
            get_relative_strength(df, nifty_df)
            if df is not None and nifty_df is not None
            else None
        )
        regime = (
            get_market_regime(nifty_df)
            if nifty_df is not None else "unknown"
        )
        prog.step()
        support_levels, resistance_levels = (
            get_support_resistance(df)
            if df is not None else ([], [])
        )
        prog.done()

    if df is None or df.empty:
        if not _quota_blocked:
            st.error("Could not load data.")
    else:
        live_quote = (
            get_live_quote(ticker)
            if is_connected() and is_market_hours()
            else None
        )
        price = (
            live_quote["price"]
            if live_quote
            else float(df["Close"].iloc[-1])
        )
        prev = float(df["Close"].iloc[-2])
        change = ((price - prev) / prev) * 100
        sector = get_sector(ticker)
        change_color = (
            t["green"] if change >= 0 else t["red"]
        )
        arrow = "▲" if change >= 0 else "▼"
        live_label = (
            " · 🟢 LIVE"
            if live_quote else " · Delayed"
        )

        st.markdown(f"""
        <div style="margin-bottom:20px;">
            <div style="display:flex;
            align-items:baseline;
            gap:14px;flex-wrap:wrap;">
                <span style="font-size:26px;
                font-weight:700;color:{t['text']};
                font-family:'JetBrains Mono',monospace;">
                    {ticker.replace('.NS','')}
                </span>
                <span style="font-size:22px;
                font-weight:600;color:{t['text']};
                font-family:'JetBrains Mono',monospace;">
                    ₹{price:,.2f}
                </span>
                <span style="font-size:14px;
                font-weight:500;color:{change_color};">
                    {arrow} {abs(change):.2f}%
                </span>
            </div>
            <div style="font-size:12px;
            color:{t['text2']};margin-top:4px;">
                {fundamentals.get('Sector', sector)} ·
                {fundamentals.get('Industry','N/A')} ·
                <span class="regime-{regime}">
                    {regime.upper()}
                </span>
                {live_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(
            "Price", f"₹{price:,.2f}",
            f"{change:+.2f}%"
        )
        c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        c3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        c4.metric(
            "52W High",
            f"₹{fundamentals.get('52W High','N/A')}"
        )
        c5.metric(
            "52W Low",
            f"₹{fundamentals.get('52W Low','N/A')}"
        )
        c6.metric(
            "Nifty Corr",
            f"{correlation}" if correlation else "N/A"
        )

        st.markdown(
            '<div class="divider"></div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing_line_color=t["green"],
            decreasing_line_color=t["red"],
            increasing_fillcolor=hex_to_rgba(
                t["green"], 0.7
            ),
            decreasing_fillcolor=hex_to_rgba(
                t["red"], 0.7
            )
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"], name="SMA 20",
            line=dict(color="#b8860b", width=1)
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
            x=df.index, y=df["BB_upper"],
            name="BB Upper",
            line=dict(
                color=hex_to_rgba(t["text2"], 0.5),
                width=1, dash="dash"
            )
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            name="BB Lower",
            line=dict(
                color=hex_to_rgba(t["text2"], 0.5),
                width=1, dash="dash"
            ),
            fill="tonexty",
            fillcolor=hex_to_rgba(t["text2"], 0.04)
        ))
        for r in resistance_levels:
            fig.add_hline(
                y=r, line_dash="dot",
                line_color=t["red"], line_width=1,
                annotation_text=f"R {r:,.0f}",
                annotation_position="right",
                annotation_font_color=t["red"],
                annotation_font_size=10
            )
        for s in support_levels:
            fig.add_hline(
                y=s, line_dash="dot",
                line_color=t["green"], line_width=1,
                annotation_text=f"S {s:,.0f}",
                annotation_position="right",
                annotation_font_color=t["green"],
                annotation_font_size=10
            )
        fib_levels = get_fibonacci_levels(df)
        for name, level in fib_levels.items():
            fig.add_hline(
                y=level, line_dash="dot",
                line_color=t["gold"], line_width=0.5,
                annotation_text=f"Fib {name}",
                annotation_position="left",
                annotation_font_color=t["gold"],
                annotation_font_size=9
            )
        fig.update_layout(
            title=dict(
                text=f"{ticker.replace('.NS','')} · Price",
                font=dict(
                    color=t["text2"], size=12,
                    family="Inter"
                )
            ),
            xaxis_rangeslider_visible=False,
            height=500,
            paper_bgcolor=t["card"],
            plot_bgcolor=t["card"],
            font=dict(color=t["text2"], family="Inter"),
            xaxis=dict(
                gridcolor=t["border"],
                showgrid=True, gridwidth=0.5
            ),
            yaxis=dict(
                gridcolor=t["border"],
                showgrid=True, gridwidth=0.5
            ),
            legend=dict(
                bgcolor=t["bg2"],
                bordercolor=t["border"],
                borderwidth=1, font=dict(size=11)
            ),
            margin=dict(l=0, r=80, t=36, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index, y=df["RSI"],
                line=dict(color=t["accent"], width=1.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(t["accent"], 0.06),
                name="RSI"
            ))
            fig_rsi.add_hline(
                y=70, line_dash="dash",
                line_color=t["red"], line_width=0.8,
                annotation_text="70",
                annotation_font_color=t["red"],
                annotation_font_size=10
            )
            fig_rsi.add_hline(
                y=30, line_dash="dash",
                line_color=t["green"], line_width=0.8,
                annotation_text="30",
                annotation_font_color=t["green"],
                annotation_font_size=10
            )
            fig_rsi.update_layout(
                title=dict(
                    text="RSI (14)",
                    font=dict(
                        color=t["text2"], size=12,
                        family="Inter"
                    )
                ),
                height=200, showlegend=False,
                paper_bgcolor=t["card"],
                plot_bgcolor=t["card"],
                font=dict(color=t["text2"]),
                xaxis=dict(
                    gridcolor=t["border"], gridwidth=0.5
                ),
                yaxis=dict(
                    gridcolor=t["border"],
                    gridwidth=0.5, range=[0, 100]
                ),
                margin=dict(l=0, r=0, t=32, b=0)
            )
            st.plotly_chart(
                fig_rsi, use_container_width=True
            )

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
                name="Hist",
                marker_color=[
                    t["green"] if v >= 0 else t["red"]
                    for v in df["MACD_hist"]
                ],
                opacity=0.5
            ))
            fig_macd.update_layout(
                title=dict(
                    text="MACD",
                    font=dict(
                        color=t["text2"], size=12,
                        family="Inter"
                    )
                ),
                height=200,
                paper_bgcolor=t["card"],
                plot_bgcolor=t["card"],
                font=dict(color=t["text2"]),
                xaxis=dict(
                    gridcolor=t["border"], gridwidth=0.5
                ),
                yaxis=dict(
                    gridcolor=t["border"], gridwidth=0.5
                ),
                legend=dict(
                    bgcolor=t["bg2"],
                    bordercolor=t["border"],
                    borderwidth=1, font=dict(size=10)
                ),
                margin=dict(l=0, r=0, t=32, b=0)
            )
            st.plotly_chart(
                fig_macd, use_container_width=True
            )

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=[
                t["green"]
                if df["Close"].iloc[i] >=
                df["Open"].iloc[i]
                else t["red"]
                for i in range(len(df))
            ],
            opacity=0.6, name="Volume"
        ))
        fig_vol.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA"],
            name="SMA 20",
            line=dict(color=t["gold"], width=1.2)
        ))
        fig_vol.update_layout(
            title=dict(
                text="Volume",
                font=dict(
                    color=t["text2"], size=12,
                    family="Inter"
                )
            ),
            height=160, showlegend=False,
            paper_bgcolor=t["card"],
            plot_bgcolor=t["card"],
            font=dict(color=t["text2"]),
            xaxis=dict(
                gridcolor=t["border"], gridwidth=0.5
            ),
            yaxis=dict(
                gridcolor=t["border"], gridwidth=0.5
            ),
            margin=dict(l=0, r=0, t=32, b=0)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown(
            '<div class="section-label">Earnings</div>',
            unsafe_allow_html=True
        )
        earnings_map = get_nse_earnings_calendar()
        earnings = get_earnings_status(
            ticker, earnings_map
        )
        if earnings["has_upcoming"]:
            if earnings["risk_level"] == "high":
                st.error(
                    f"🚨 {earnings['message']} — "
                    f"Consider waiting."
                )
            elif earnings["risk_level"] == "medium":
                st.warning(
                    f"⚠️ {earnings['message']} — "
                    f"Reduce position size."
                )
            else:
                st.info(f"📅 {earnings['message']}")
        else:
            st.success("No earnings due in 30 days")

        # Macro event risk (Fed / RBI / payrolls nearby)
        from macro import get_macro_risk
        _macro = get_macro_risk(days_ahead=3)
        if _macro["flag"]:
            st.warning(f"🌐 {_macro['message']}")

        st.markdown(
            '<div class="section-label">'
            'Model signal</div>',
            unsafe_allow_html=True
        )
        prog2 = StagedProgress(t, [
            ("Fetching 5 years of training data", 3),
            ("Walk-forward validation (3 folds)", 6),
            ("Training calibrated ensemble "
             "(LightGBM + XGBoost)", 5),
        ])
        prog2.step()
        prog2.step()
        model, scaler, features, accuracy = (
            train_model(ticker)
        )
        prog2.step()
        signal, confidence, buy_prob, sell_prob = (
            get_signal(model, scaler, df, features)
        )
        risk_metrics = get_risk_metrics(df)
        prog2.done()

        sig_badge = (
            "badge-buy" if signal == "BUY"
            else "badge-sell"
        )
        sig_label = (
            "BULLISH" if signal == "BUY" else "BEARISH"
        )
        # Equitex Score for this stock = model's directional
        # probability rendered on the signature 0-100 scale.
        eqx_score = int(round(buy_prob * 100))
        st.markdown(
            branding.score_gauge_html(eqx_score, t),
            unsafe_allow_html=True
        )
        st.markdown("<div style='height:8px'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            f'<span class="{sig_badge}">{sig_label}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="font-size:12px;'
            f'color:{t["text2"]};">'
            f'Model probability {confidence:.1%} · '
            f'Holdout fit {accuracy:.1%} (indicative, '
            f'small sample)</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Buy prob", f"{buy_prob:.1%}")
        r2.metric("Sell prob", f"{sell_prob:.1%}")
        r3.metric("Sharpe", risk_metrics["Sharpe Ratio"])
        r4.metric(
            "Max drawdown",
            risk_metrics["Max Drawdown"]
        )
        r5.metric(
            "Volatility",
            risk_metrics["Annual Volatility"]
        )
        r6.metric(
            "Rel. strength",
            f"{rs:+.1f}%" if rs else "N/A"
        )

        if fundamentals:
            st.markdown(
                '<div class="section-label">'
                'Fundamentals</div>',
                unsafe_allow_html=True
            )
            keys = [
                k for k in fundamentals
                if k not in ["Sector", "Industry"]
            ]
            fcols = st.columns(4)
            for i, k in enumerate(keys):
                v = fundamentals[k]
                if k == "Market Cap" and isinstance(
                    v, (int, float)
                ):
                    fcols[i % 4].metric(
                        k, f"₹{v/1e9:.0f}B"
                    )
                elif k in [
                    "Dividend Yield", "ROE",
                    "Revenue Growth", "Promoter Holding"
                ] and isinstance(v, float):
                    fcols[i % 4].metric(k, f"{v:.1%}")
                else:
                    fcols[i % 4].metric(
                        k,
                        f"{round(v,2)}"
                        if isinstance(v, float)
                        else str(v)
                    )

        st.markdown(
            '<div class="section-label">'
            'News sentiment</div>',
            unsafe_allow_html=True
        )
        with st.spinner(""):
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

        trend_arrow = (
            "↑" if trend == "improving"
            else "↓" if trend == "deteriorating"
            else "→"
        )
        sent_color = (
            t["green"] if sentiment == "positive"
            else t["red"] if sentiment == "negative"
            else t["text2"]
        )

        st.markdown(f"""
        <div class="sentiment-grid">
            <div class="sentiment-cell">
                <div class="sentiment-cell-label">
                    Sentiment</div>
                <div class="sentiment-cell-value"
                style="color:{sent_color};">
                    {sentiment.capitalize()}</div>
            </div>
            <div class="sentiment-cell">
                <div class="sentiment-cell-label">
                    Score</div>
                <div class="sentiment-cell-value">
                    {sentiment_score:+.2f}</div>
            </div>
            <div class="sentiment-cell">
                <div class="sentiment-cell-label">
                    Trend</div>
                <div class="sentiment-cell-value">
                    {trend_arrow} {trend.capitalize()}</div>
            </div>
            <div class="sentiment-cell">
                <div class="sentiment-cell-label">
                    Positive</div>
                <div class="sentiment-cell-value"
                style="color:{t['green']};">
                    {distribution.get('positive',0):.1%}
                </div>
            </div>
            <div class="sentiment-cell">
                <div class="sentiment-cell-label">
                    Negative</div>
                <div class="sentiment-cell-value"
                style="color:{t['red']};">
                    {distribution.get('negative',0):.1%}
                </div>
            </div>
            <div class="sentiment-cell">
                <div class="sentiment-cell-label">
                    Headlines</div>
                <div class="sentiment-cell-value">
                    {headline_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if risk_flags:
            st.error(
                f"Risk flags: {', '.join(risk_flags)}"
            )

        kw_html = ""
        if pos_kw:
            kw_html += "".join([
                f'<span class="keyword-tag keyword-pos">'
                f'{k}</span>' for k in pos_kw
            ])
        if neg_kw:
            kw_html += "".join([
                f'<span class="keyword-tag keyword-neg">'
                f'{k}</span>' for k in neg_kw
            ])
        if kw_html:
            st.markdown(
                f'<div style="margin-bottom:12px;">'
                f'{kw_html}</div>',
                unsafe_allow_html=True
            )

        if sources:
            st.markdown(
                f'<div style="font-size:11px;'
                f'color:{t["text2"]};'
                f'margin-bottom:10px;">'
                f'Sources: {" · ".join(sources)}</div>',
                unsafe_allow_html=True
            )

        headlines_html = ""
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
            css_class = (
                "neg-text" if has_neg
                else "pos-text" if has_pos
                else "neu-text"
            )
            dot = "●" if has_neg or has_pos else "○"
            headlines_html += (
                f'<div class="headline-item">'
                f'<span class="{css_class}">{dot}</span>'
                f'<span class="{css_class}">{h}</span>'
                f'</div>'
            )
        st.markdown(headlines_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-label">'
            'AI analysis</div>',
            unsafe_allow_html=True
        )
        with st.spinner(""):
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
            f'<div class="thesis-box">'
            f'{explanation}</div>',
            unsafe_allow_html=True
        )

with tab4:
    st.markdown(
        '<div class="section-label">'
        '30-day forecast</div>',
        unsafe_allow_html=True
    )
    try:
        from prophet import Prophet
        ticker_f = st.selectbox(
            "Select stock", NIFTY_50, key="fc"
        )
        with st.spinner(""):
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
                future = m.make_future_dataframe(
                    periods=30
                )
                forecast = m.predict(future)

                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(
                    x=pf["ds"], y=pf["y"],
                    name="Actual",
                    line=dict(color=t["text2"], width=1.5)
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast["yhat"],
                    name="Forecast",
                    line=dict(color=t["accent"], width=2)
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    line=dict(width=0),
                    showlegend=False
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=hex_to_rgba(
                        t["accent"], 0.12
                    ),
                    showlegend=False
                ))
                fig_f.update_layout(
                    title=dict(
                        text=(
                            f"{ticker_f.replace('.NS','')}"
                            f" · 30-day forecast"
                        ),
                        font=dict(
                            color=t["text2"], size=12,
                            family="Inter"
                        )
                    ),
                    height=480,
                    paper_bgcolor=t["card"],
                    plot_bgcolor=t["card"],
                    font=dict(color=t["text2"]),
                    xaxis=dict(
                        gridcolor=t["border"],
                        gridwidth=0.5
                    ),
                    yaxis=dict(
                        gridcolor=t["border"],
                        gridwidth=0.5
                    ),
                    legend=dict(
                        bgcolor=t["bg2"],
                        bordercolor=t["border"],
                        borderwidth=1
                    ),
                    margin=dict(l=0, r=0, t=36, b=0)
                )
                st.plotly_chart(
                    fig_f, use_container_width=True
                )

                last_forecast = float(
                    forecast["yhat"].iloc[-1]
                )
                last_actual = float(pf["y"].iloc[-1])
                change_f = (
                    (last_forecast - last_actual) /
                    last_actual * 100
                )
                f1, f2, f3 = st.columns(3)
                f1.metric(
                    "Current", f"₹{last_actual:,.2f}"
                )
                f2.metric(
                    "Forecast (30d)",
                    f"₹{last_forecast:,.2f}"
                )
                f3.metric(
                    "Expected change",
                    f"{change_f:+.1f}%"
                )
                st.caption(
                    "Prophet trend model. "
                    "Not financial advice."
                )
    except ImportError:
        st.info("Install prophet to enable forecasting.")
    except Exception:
        st.warning(
            "Scenario projection is temporarily "
            "unavailable — the data source didn't "
            "respond. Try again in a minute."
        )

with tab11:
    from macro import render_macro_tab
    render_macro_tab(t)

with tab12:
    from paper_trade import render_paper_tab
    render_paper_tab(t)


def _tab_guard(render_fn):
    """
    One tab crashing must never blank the rest of the
    app — show a clean fallback card instead.
    """
    try:
        render_fn()
    except Exception:
        st.warning(
            "This section hit a temporary error — "
            "refresh to try again. If it persists, "
            "it's usually the market-data source "
            "rate-limiting; it recovers on its own."
        )


with tab5:
    _tab_guard(lambda: render_watchlist_tab(
        t,
        max_items=None if IS_PRO
        else auth.FREE_WATCHLIST_MAX
    ))

with tab6:
    _tab_guard(lambda: render_journal_tab(t))

with tab9:
    _tab_guard(lambda: render_track_record_tab(t))

with tab10:
    def _render_plans():
        render_pricing(t, user)
        from legal import render_legal_expanders
        render_legal_expanders(t)
    _tab_guard(_render_plans)
# ---------------------------------------------------------
# Compliance footer (shown on every page)
# ---------------------------------------------------------
from config import DISCLAIMER_LONG

st.markdown(
    f"""<div style="margin-top:48px;padding:16px 18px;
    border-top:1px solid {t['border']};
    font-size:11px;line-height:1.7;
    color:{t['text2']};">{DISCLAIMER_LONG}</div>""",
    unsafe_allow_html=True
)
