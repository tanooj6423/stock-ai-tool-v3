"""Track Record tab — the forward log, shown honestly."""

import pandas as pd
import streamlit as st

from scan_store import (get_track_entries,
                        track_record_stats)


def render_track_record_tab(t):
    st.markdown(
        '<div class="section-label">'
        'Model track record (forward log)</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Every model pick is logged the day it is "
        "published, before the outcome is known, and "
        "evaluated against subsequent market prices. "
        "Wins and losses are both shown. This is a "
        "statistical record of model output, not "
        "investment advice."
    )

    stats = track_record_stats()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Picks logged", stats["total_logged"])
    c2.metric("Open", stats["open"])
    c3.metric("Closed", stats["closed"])
    c4.metric(
        "Win rate",
        f"{stats['win_rate']}%"
        if stats["win_rate"] is not None else "—"
    )
    c5.metric(
        "Avg return/pick",
        f"{stats['avg_return']:+.2f}%"
        if stats["avg_return"] is not None else "—"
    )
    c6.metric(
        "T1 hit rate",
        f"{stats['t1_hit_rate']}%"
        if stats["t1_hit_rate"] is not None else "—"
    )

    entries = get_track_entries()
    if not entries:
        st.info(
            "No picks logged yet. The forward log "
            "starts filling as soon as the nightly "
            "scan runs (or after a manual scan)."
        )
        return

    df = pd.DataFrame(entries)
    cols = ["date", "ticker", "score", "entry",
            "stop", "t1", "t2", "status",
            "exit_price", "exit_date", "return_pct",
            "sector"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.rename(columns={
        "date": "Published", "ticker": "Ticker",
        "score": "Score", "entry": "Ref entry",
        "stop": "Invalidation", "t1": "T1",
        "t2": "T2", "status": "Status",
        "exit_price": "Exit", "exit_date": "Exit date",
        "return_pct": "Return %", "sector": "Sector",
    })
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(
            str
        ).str.replace(".NS", "", regex=False)

    st.dataframe(
        df, use_container_width=True, hide_index=True
    )

    closed = [
        e for e in entries
        if e.get("return_pct") is not None
    ]
    if len(closed) >= 5:
        st.markdown(
            '<div class="section-label">'
            'Cumulative return of closed picks '
            '(equal-weight)</div>',
            unsafe_allow_html=True
        )
        cdf = pd.DataFrame(closed)
        cdf = cdf.sort_values("exit_date")
        cdf["cum"] = (
            (1 + cdf["return_pct"].astype(float) / 100)
            .cumprod() - 1
        ) * 100
        chart_df = cdf.set_index("exit_date")[["cum"]]
        chart_df.columns = ["Cumulative %"]
        st.line_chart(chart_df)
