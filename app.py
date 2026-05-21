import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from data import (get_stock_data, add_indicators, get_fundamentals,
                  get_support_resistance, get_nifty_data,
                  get_nifty_correlation, get_relative_strength,
                  get_market_regime, get_fibonacci_levels,
                  validate_ticker)
from model import train_model, get_signal, get_risk_metrics
from sentiment import get_news_sentiment
from ai_explain import explain_signal, generate_pick_thesis
from screener_engine import run_full_scan, calculate_position_size
from universe import ALL_STOCKS, NIFTY_50, COMMODITIES, get_sector

st.set_page_config(
    page_title="NSE Stock Intelligence v3",
    layout="wide",
    page_icon="📈"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
[data-testid="stMetricLabel"] { font-size: 0.75rem; }
.buy-badge {
    background: #00c853; color: white;
    padding: 4px 12px; border-radius: 20px;
    font-weight: bold; font-size: 1.1rem;
}
.sell-badge {
    background: #d50000; color: white;
    padding: 4px 12px; border-radius: 20px;
    font-weight: bold; font-size: 1.1rem;
}
.score-high { color: #00c853; font-weight: bold; font-size: 1.3rem; }
.score-mid { color: #ffa000; font-weight: bold; font-size: 1.3rem; }
.score-low { color: #d50000; font-weight: bold; font-size: 1.3rem; }
.pick-card {
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 NSE Stock Intelligence Platform v3")
st.caption("Professional-grade AI equity analysis. For informational purposes only. Not financial advice.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Daily Picks",
    "📊 Stock Analysis",
    "🔮 Price Forecast",
    "📁 Portfolio",
    "⚙️ Settings"
])

with tab5:
    st.subheader("Your settings")
    capital = st.number_input(
        "Trading capital (₹)", min_value=10000,
        max_value=10000000, value=50000, step=5000
    )
    risk_pct = st.slider(
        "Risk per trade (%)", min_value=0.5,
        max_value=3.0, value=1.5, step=0.25
    )
    min_score = st.slider(
        "Minimum score to show", min_value=50,
        max_value=85, value=60, step=5
    )
    scan_universe = st.selectbox(
        "Scan universe",
        ["Nifty 50 only", "Nifty 50 + Next 50", "Full universe (150 stocks)"],
        index=1
    )
    st.caption("Settings are saved for this session.")

with tab1:
    st.subheader("🎯 Daily top picks — AI screened")
    st.caption("Composite 10-layer scoring across the full NSE universe. Auto-cached for 6 hours.")

    col1, col2 = st.columns([3, 1])
    with col2:
        run_scan = st.button("🔄 Run fresh scan", type="primary")

    if scan_universe == "Nifty 50 only":
        scan_tickers = NIFTY_50
    elif scan_universe == "Nifty 50 + Next 50":
        from universe import NIFTY_NEXT_50
        scan_tickers = NIFTY_50 + NIFTY_NEXT_50
    else:
        scan_tickers = ALL_STOCKS

    import time
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
        status_text.text(
            f"Scanning {current_ticker.replace('.NS','')}... "
            f"({current}/{total})"
        )
        time_text.text(f"Estimated time remaining: {mins}m {secs}s")

    picks, regime = run_full_scan(
        tuple(scan_tickers),
        capital=capital,
        risk_pct=risk_pct,
        progress_callback=update_progress
    )

    progress_bar.progress(100)
    status_text.text(f"✅ Scan complete — {len(picks)} picks found")
    time_text.empty()

    regime_color = (
        "green" if regime == "bull"
        else "red" if regime == "bear"
        else "orange"
    )
    st.markdown(
        f"**Market regime:** :{regime_color}[{regime.upper()}] | "
        f"**Stocks scanned:** {len(scan_tickers)} | "
        f"**Top picks found:** {len(picks)}"
    )

    if not picks:
        st.warning("No stocks passed all screening criteria today. Market conditions may not be favourable for new entries.")
    else:
        for i, pick in enumerate(picks):
            score = pick["score"]
            score_class = (
                "score-high" if score >= 75
                else "score-mid" if score >= 65
                else "score-low"
            )
            with st.expander(
                f"#{i+1} {pick['ticker'].replace('.NS','')} — "
                f"Score: {score}/100 | "
                f"{pick['signal']} {pick['confidence']:.1%} | "
                f"₹{pick['price']:,.2f}",
                expanded=(i < 3)
            ):
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Entry", f"₹{pick['entry']:,.2f}")
                c2.metric("Stop loss", f"₹{pick['stop_loss']:,.2f}")
                c3.metric("Target 1", f"₹{pick['target1']:,.2f}")
                c4.metric("Target 2", f"₹{pick['target2']:,.2f}")
                c5.metric("R/R ratio", f"1:{pick['rr1']:.1f}")
                c6.metric("Shares", f"{pick['shares']}")

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("RSI", f"{pick['rsi']}")
                m2.metric("Confidence", f"{pick['confidence']:.1%}")
                m3.metric("Buy prob", f"{pick['buy_prob']:.1%}")
                m4.metric("Sentiment", pick['sentiment'].capitalize())
                m5.metric("Sector", pick['sector'])
                m6.metric("Sharpe", f"{pick['sharpe']}")

                st.markdown("**Score breakdown:**")
                breakdown = pick["score_breakdown"]
                cols = st.columns(len(breakdown))
                for j, (layer, pts) in enumerate(breakdown.items()):
                    max_pts = [20, 15, 10, 10, 10, 8, 8, 8, 5, 10][j]
                    cols[j].metric(layer, f"{pts}/{max_pts}")

                st.markdown("**AI trade thesis:**")
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
                st.info(thesis)

                if pick["position_cost"] > 0:
                    st.caption(
                        f"Position size: {pick['shares']} shares | "
                        f"Cost: ₹{pick['position_cost']:,.2f} | "
                        f"Max loss: ₹{pick['risk_amount'] * pick['shares']:,.2f}"
                    )

with tab2:
    st.sidebar.markdown("## Stock selection")
    mode = st.sidebar.radio(
        "", ["Popular stocks", "Search any NSE stock"],
        label_visibility="collapsed"
    )
    if mode == "Popular stocks":
        ticker = st.sidebar.selectbox("Select stock", NIFTY_50)
    else:
        raw = st.sidebar.text_input(
            "Enter NSE symbol",
            placeholder="e.g. ZOMATO, IRFC, MOTHERSON"
        )
        if raw:
            with st.sidebar:
                with st.spinner("Validating..."):
                    result = validate_ticker(raw)
            if result:
                st.sidebar.success(f"Found: {result}")
                ticker = result
            else:
                st.sidebar.error("Symbol not found.")
                ticker = NIFTY_50[0]
        else:
            ticker = NIFTY_50[0]

    with st.spinner("Loading market data..."):
        df = get_stock_data(ticker, period="2y")
        if df is not None:
            df = add_indicators(df)
        fundamentals = get_fundamentals(ticker)
        nifty_df = get_nifty_data()
        correlation = get_nifty_correlation(df, nifty_df) if (df is not None and nifty_df is not None) else None
        rs = get_relative_strength(df, nifty_df) if (df is not None and nifty_df is not None) else None
        regime = get_market_regime(nifty_df) if nifty_df is not None else "unknown"
        support_levels, resistance_levels = get_support_resistance(df) if df is not None else ([], [])

    if df is None or df.empty:
        st.error("Could not load data for this ticker.")
    else:
        price = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]
        change = ((price - prev) / prev) * 100
        sector = get_sector(ticker)

        st.markdown(f"### {ticker.replace('.NS', '')} — ₹{price:,.2f}")
        st.caption(f"Sector: {fundamentals.get('Sector', sector)} | "
                  f"Industry: {fundamentals.get('Industry', 'N/A')} | "
                  f"Market regime: {regime.upper()}")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Price", f"₹{price:,.2f}", f"{change:+.2f}%")
        c2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
        c3.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        c4.metric("52W High", f"₹{fundamentals.get('52W High', 'N/A')}")
        c5.metric("52W Low", f"₹{fundamentals.get('52W Low', 'N/A')}")
        c6.metric("Nifty Corr", f"{correlation}" if correlation else "N/A")

        st.markdown("---")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_20"], name="SMA 20",
            line=dict(color="orange", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_50"], name="SMA 50",
            line=dict(color="blue", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA_200"], name="SMA 200",
            line=dict(color="purple", width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="gray", width=1, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="gray", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(128,128,128,0.1)"
        ))
        for r in resistance_levels:
            fig.add_hline(
                y=r, line_dash="dot", line_color="red",
                annotation_text=f"R: ₹{r:,.0f}",
                annotation_position="right"
            )
        for s in support_levels:
            fig.add_hline(
                y=s, line_dash="dot", line_color="green",
                annotation_text=f"S: ₹{s:,.0f}",
                annotation_position="right"
            )
        fib_levels = get_fibonacci_levels(df)
        for name, level in fib_levels.items():
            fig.add_hline(
                y=level, line_dash="dot",
                line_color="rgba(255,215,0,0.4)",
                annotation_text=f"Fib {name}",
                annotation_position="left"
            )
        fig.update_layout(
            title=f"{ticker} — Price chart",
            xaxis_rangeslider_visible=False,
            height=550
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=df.index, y=df["RSI"],
                line=dict(color="purple"), name="RSI"
            ))
            fig_rsi.add_hline(
                y=70, line_dash="dash", line_color="red",
                annotation_text="Overbought"
            )
            fig_rsi.add_hline(
                y=30, line_dash="dash", line_color="green",
                annotation_text="Oversold"
            )
            fig_rsi.update_layout(title="RSI", height=250, showlegend=False)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=df.index, y=df["MACD"],
                name="MACD", line=dict(color="blue")
            ))
            fig_macd.add_trace(go.Scatter(
                x=df.index, y=df["MACD_signal"],
                name="Signal", line=dict(color="orange")
            ))
            fig_macd.add_trace(go.Bar(
                x=df.index, y=df["MACD_hist"],
                name="Histogram",
                marker_color=[
                    "green" if v >= 0 else "red"
                    for v in df["MACD_hist"]
                ]
            ))
            fig_macd.update_layout(title="MACD", height=250)
            st.plotly_chart(fig_macd, use_container_width=True)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=[
                "green" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                else "red" for i in range(len(df))
            ],
            name="Volume"
        ))
        fig_vol.add_trace(go.Scatter(
            x=df.index, y=df["Volume_SMA"],
            name="Vol SMA 20",
            line=dict(color="white")
        ))
        fig_vol.update_layout(title="Volume", height=200, showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("---")
        st.subheader("Model signal")
        with st.spinner("Running ensemble model..."):
            model, scaler, features, accuracy = train_model(ticker)
            signal, confidence, buy_prob, sell_prob = get_signal(
                model, scaler, df, features
            )
            risk_metrics = get_risk_metrics(df)

        color = "green" if signal == "BUY" else "red"
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.markdown(f"### :{color}[{signal}]")
        r2.metric("Confidence", f"{confidence:.1%}")
        r3.metric("Buy prob", f"{buy_prob:.1%}")
        r4.metric("Sell prob", f"{sell_prob:.1%}")
        r5.metric("Accuracy", f"{accuracy:.1%}")
        r6.metric("Sharpe", risk_metrics["Sharpe Ratio"])

        r7, r8, r9, r10 = st.columns(4)
        r7.metric("Max drawdown", risk_metrics["Max Drawdown"])
        r8.metric("Ann. volatility", risk_metrics["Annual Volatility"])
        r9.metric("Win rate", risk_metrics["Win Rate"])
        r10.metric("Rel. strength", f"{rs:+.1f}%" if rs else "N/A")

        st.markdown("---")
        if fundamentals:
            st.subheader("Fundamentals")
            keys = [k for k in fundamentals
                   if k not in ["Sector", "Industry"]]
            fcols = st.columns(4)
            for i, k in enumerate(keys):
                v = fundamentals[k]
                if k == "Market Cap" and isinstance(v, (int, float)):
                    fcols[i % 4].metric(k, f"₹{v/1e9:.0f}B")
                elif k == "Dividend Yield" and isinstance(v, float):
                    fcols[i % 4].metric(k, f"{v:.1%}")
                elif k == "ROE" and isinstance(v, float):
                    fcols[i % 4].metric(k, f"{v:.1%}")
                elif k == "Revenue Growth" and isinstance(v, float):
                    fcols[i % 4].metric(k, f"{v:.1%}")
                elif k == "Promoter Holding" and isinstance(v, float):
                    fcols[i % 4].metric(k, f"{v:.1%}")
                else:
                    fcols[i % 4].metric(
                        k, f"{round(v,2)}" if isinstance(v, float) else str(v)
                    )

        st.markdown("---")
        st.subheader("News sentiment")
        with st.spinner("Analysing news..."):
            sentiment, sent_conf, distribution, headlines = get_news_sentiment(ticker)

        color_s = (
            "green" if sentiment == "positive"
            else "red" if sentiment == "negative"
            else "gray"
        )
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Sentiment", sentiment.capitalize())
        sc2.metric("Confidence", f"{sent_conf:.1%}")
        sc3.metric("Positive score", f"{distribution.get('positive', 0):.1%}")
        for h in headlines:
            st.write(f"- {h}")

        st.markdown("---")
        st.subheader("AI analysis")
        with st.spinner("Generating investment brief..."):
            explanation = explain_signal(
                ticker=ticker,
                signal=signal,
                confidence=confidence,
                sentiment=sentiment,
                rsi=df["RSI"].iloc[-1],
                macd=df["MACD"].iloc[-1],
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
        st.info(explanation)

with tab3:
    st.subheader("30-day price forecast")
    try:
        from prophet import Prophet
        ticker_f = st.selectbox("Select stock", NIFTY_50, key="fc")
        with st.spinner("Running forecast..."):
            df_f = get_stock_data(ticker_f, period="2y")
            if df_f is not None:
                pf = df_f.reset_index()[["Date", "Close"]].copy()
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
                    name="Actual", line=dict(color="white")
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast["yhat"],
                    name="Forecast", line=dict(color="orange")
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast["yhat_upper"],
                    line=dict(color="rgba(255,165,0,0.2)"),
                    showlegend=False
                ))
                fig_f.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast["yhat_lower"],
                    line=dict(color="rgba(255,165,0,0.2)"),
                    fill="tonexty",
                    fillcolor="rgba(255,165,0,0.1)",
                    showlegend=False
                ))
                fig_f.update_layout(
                    title=f"{ticker_f} — 30-day forecast",
                    height=500
                )
                st.plotly_chart(fig_f, use_container_width=True)

                last_forecast = forecast["yhat"].iloc[-1]
                last_actual = pf["y"].iloc[-1]
                change_f = ((last_forecast - last_actual) / last_actual) * 100
                f1, f2, f3 = st.columns(3)
                f1.metric("Current price", f"₹{last_actual:,.2f}")
                f2.metric("Forecast (30d)", f"₹{last_forecast:,.2f}")
                f3.metric("Expected change", f"{change_f:+.1f}%")
                st.caption("Prophet model. Not financial advice.")
    except ImportError:
        st.warning("Run `pip install prophet` to enable forecasting.")

with tab4:
    st.subheader("Portfolio tracker")
    with st.form("portfolio_form"):
        st.markdown("Enter your holdings:")
        holdings = []
        for idx in range(5):
            col1, col2, col3 = st.columns(3)
            t = col1.selectbox(
                f"Stock {idx+1}", ["--"] + NIFTY_50,
                key=f"pt{idx}"
            )
            q = col2.number_input(
                "Qty", min_value=0, value=0, key=f"pq{idx}"
            )
            p = col3.number_input(
                "Avg price ₹", min_value=0.0,
                value=0.0, key=f"pp{idx}"
            )
            if t != "--" and q > 0:
                holdings.append((t, q, p))

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
                current_p = df_p["Close"].iloc[-1]
                invested = qty * avg_price if avg_price > 0 else qty * current_p
                current_val = qty * current_p
                pnl = current_val - invested
                pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                model_p, sc_p, f_p, acc_p = train_model(ticker_p)
                sig_p, conf_p, _, _ = get_signal(model_p, sc_p, df_p, f_p)
                total_invested += invested
                total_current += current_val
                portfolio_data.append({
                    "Stock": ticker_p.replace(".NS", ""),
                    "Qty": qty,
                    "Avg price": f"₹{avg_price:,.2f}" if avg_price > 0 else "N/A",
                    "Current": f"₹{current_p:,.2f}",
                    "Value": f"₹{current_val:,.2f}",
                    "P&L": f"₹{pnl:+,.2f}" if avg_price > 0 else "N/A",
                    "Return": f"{pnl_pct:+.1f}%" if avg_price > 0 else "N/A",
                    "Signal": sig_p,
                    "Confidence": f"{conf_p:.1%}"
                })
            except:
                continue

        if portfolio_data:
            total_pnl = total_current - total_invested
            total_pnl_pct = (
                (total_pnl / total_invested * 100)
                if total_invested > 0 else 0
            )
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Total invested", f"₹{total_invested:,.2f}")
            pc2.metric("Current value", f"₹{total_current:,.2f}")
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
                title="Portfolio allocation"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
