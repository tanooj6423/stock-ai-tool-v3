"""
Macro / economic-event radar for Equitex.

Design choice: this module is **100% date-based, no network
calls**. Economic-calendar APIs are flaky and would reintroduce
the hanging-fetch problem we just fixed. Instead we compute the
deterministic events (US Non-Farm Payrolls = first Friday of the
month) and carry a small curated table of the scheduled
policy-decision dates (FOMC, RBI MPC), which central banks
publish a year ahead. This is accurate, instant, and can never
break the app.

Sources for the curated dates:
  FOMC 2026: federalreserve.gov FOMC calendar
  RBI MPC FY27: RBI MPC calendar (announced Mar 2026)

Update the tables once a year when the central banks publish the
next calendar.
"""

from datetime import date, datetime, timedelta

# ---------------------------------------------------------
# Curated scheduled policy decisions (announcement day)
# High impact — these move Nifty via rates & global risk.
# ---------------------------------------------------------
FOMC_DECISION_DATES = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
    date(2027, 1, 27),  # next-year buffer
]

RBI_MPC_DATES = [
    date(2026, 4, 8), date(2026, 6, 5), date(2026, 8, 5),
    date(2026, 10, 7), date(2026, 12, 4), date(2027, 2, 5),
]


def _first_friday(year, month):
    d = date(year, month, 1)
    # weekday(): Mon=0 ... Fri=4
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _nfp_dates(start, end):
    """US Non-Farm Payrolls — first Friday of each month."""
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        ff = _first_friday(y, m)
        if start <= ff <= end:
            out.append(ff)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _india_cpi_dates(start, end):
    """
    India CPI inflation — MoSPI releases ~12th of each month
    (5:30pm IST). Dates are approximate (±1 working day).
    """
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        try:
            d = date(y, m, 12)
            if start <= d <= end:
                out.append(d)
        except ValueError:
            pass
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def get_upcoming_macro_events(days_ahead=14, today=None):
    """
    Returns a sorted list of upcoming events:
      {date, days_away, name, region, impact, approx}
    impact: "high" | "medium"
    """
    today = today or date.today()
    horizon = today + timedelta(days=days_ahead)
    events = []

    for d in FOMC_DECISION_DATES:
        if today <= d <= horizon:
            events.append((d, "US Fed rate decision (FOMC)",
                           "US", "high", False))
    for d in RBI_MPC_DATES:
        if today <= d <= horizon:
            events.append((d, "RBI monetary policy (MPC)",
                           "India", "high", False))
    for d in _nfp_dates(today, horizon):
        events.append((d, "US Non-Farm Payrolls",
                       "US", "high", False))
    for d in _india_cpi_dates(today, horizon):
        events.append((d, "India CPI inflation",
                       "India", "medium", True))

    events.sort(key=lambda e: e[0])
    return [
        {
            "date": d,
            "days_away": (d - today).days,
            "name": name,
            "region": region,
            "impact": impact,
            "approx": approx,
        }
        for d, name, region, impact, approx in events
    ]


def get_macro_risk(days_ahead=2, today=None):
    """
    Is a HIGH-impact macro event within `days_ahead` days?
    Used like the earnings-risk flag: setups opened right
    before Fed/RBI/payrolls carry elevated event risk.
    Returns {flag, event, days_away, message}.
    """
    upcoming = get_upcoming_macro_events(
        days_ahead=days_ahead, today=today
    )
    high = [e for e in upcoming if e["impact"] == "high"]
    if not high:
        return {"flag": False, "event": None,
                "days_away": None, "message": ""}
    e = high[0]
    when = ("today" if e["days_away"] == 0
            else "tomorrow" if e["days_away"] == 1
            else f"in {e['days_away']} days")
    return {
        "flag": True,
        "event": e["name"],
        "days_away": e["days_away"],
        "message": (
            f"{e['name']} {when} — markets often see "
            f"elevated volatility around this event."
        ),
    }


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
def render_macro_banner(t, days_ahead=10):
    """Compact horizontal 'macro radar' strip."""
    import streamlit as st

    events = get_upcoming_macro_events(days_ahead=days_ahead)
    if not events:
        return

    chips = []
    for e in events[:5]:
        col = t["red"] if e["impact"] == "high" else t["gold"]
        dot = "●"
        when = ("Today" if e["days_away"] == 0
                else "Tomorrow" if e["days_away"] == 1
                else f"{e['days_away']}d")
        approx = "~" if e["approx"] else ""
        chips.append(
            f'<span style="display:inline-flex;'
            f'align-items:center;gap:6px;'
            f'background:{t["bg2"]};'
            f'border:1px solid {t["border"]};'
            f'border-radius:99px;padding:5px 12px;'
            f'font-size:11.5px;color:{t["text"]};'
            f'margin:0 6px 6px 0;white-space:nowrap;">'
            f'<span style="color:{col};font-size:9px;">'
            f'{dot}</span>{e["name"]} '
            f'<span style="color:{t["text2"]};">'
            f'· {approx}{when}</span></span>'
        )

    st.markdown(
        f'<div style="margin:4px 0 14px 0;">'
        f'<div style="font-size:10px;font-weight:700;'
        f'color:{t["text2"]};text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:8px;">'
        f'Macro radar · next {days_ahead} days</div>'
        f'{"".join(chips)}</div>',
        unsafe_allow_html=True
    )


def render_macro_tab(t):
    """Full economic calendar view."""
    import streamlit as st

    st.markdown(
        '<div class="section-label">'
        'Economic calendar — high-impact events</div>',
        unsafe_allow_html=True
    )
    st.caption(
        "Scheduled events that historically move Indian "
        "equities via rates, global risk sentiment and "
        "institutional flows. Times are informational, "
        "not a prompt to trade."
    )

    events = get_upcoming_macro_events(days_ahead=45)
    if not events:
        st.info("No major scheduled events in the next "
                "45 days.")
        return

    for e in events:
        col = t["red"] if e["impact"] == "high" else t["gold"]
        label = ("HIGH IMPACT" if e["impact"] == "high"
                 else "MEDIUM")
        approx = " (approx.)" if e["approx"] else ""
        when = ("Today" if e["days_away"] == 0
                else "Tomorrow" if e["days_away"] == 1
                else f"in {e['days_away']} days")
        st.markdown(
            f'<div style="display:flex;align-items:center;'
            f'gap:14px;padding:12px 16px;margin-bottom:8px;'
            f'background:{t["card"]};'
            f'border:1px solid {t["border"]};'
            f'border-left:3px solid {col};'
            f'border-radius:10px;">'
            f'<div style="min-width:96px;font-size:12px;'
            f'font-family:monospace;color:{t["text"]};">'
            f'{e["date"].strftime("%a %d %b")}{approx}</div>'
            f'<div style="flex:1;font-size:13px;'
            f'color:{t["text"]};font-weight:500;">'
            f'{e["name"]}'
            f'<span style="color:{t["text2"]};'
            f'font-weight:400;"> · {e["region"]}</span></div>'
            f'<div style="font-size:10px;font-weight:700;'
            f'color:{col};letter-spacing:0.5px;">'
            f'{label}</div>'
            f'<div style="min-width:80px;text-align:right;'
            f'font-size:12px;color:{t["text2"]};">'
            f'{when}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
