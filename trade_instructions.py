import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

# NOTE (compliance): This module describes model-derived
# scenarios and reference levels in plain English.
# It intentionally avoids prescriptive language
# ("buy", "place order", "you should") so the product
# operates as an analytics/research tool, not
# investment advice. Do not reintroduce imperative
# wording without SEBI RA registration.


def get_entry_instruction(pick, capital, risk_pct):
    """
    Generates a plain-English *scenario description*
    for a given model pick: reference levels, the
    model's position-sizing math, and how the setup
    would historically have been managed.
    """
    ticker = pick["ticker"].replace(".NS", "")
    entry = pick["entry"]
    stop = pick["stop_loss"]
    t1 = pick["target1"]
    t2 = pick["target2"]
    shares = pick["shares"]
    cost = pick["position_cost"]
    max_loss = pick.get("max_loss") or (
        pick.get("risk_amount", 0) * pick.get("shares", 1)
    )
    holding = pick.get("holding_days", 7)
    rr = pick.get("rr1", 2.0)
    risk_level = pick.get("risk_level", "MEDIUM")
    confidence = pick.get("confidence", 0.6)
    regime = pick.get("market_regime", "unknown")

    # Reference zone — 0.5% above model entry level
    entry_low = round(entry * 0.998, 2)
    entry_high = round(entry * 1.005, 2)

    # Scenario time window
    today = datetime.now()
    exit_date = today + timedelta(days=holding)
    # Skip weekends
    while exit_date.weekday() >= 5:
        exit_date += timedelta(days=1)
    exit_date_str = exit_date.strftime("%A %d %b")

    # Half position for the partial-exit scenario
    half_shares = max(1, shares // 2)
    remaining_shares = shares - half_shares

    if shares == 0:
        return {
            "summary": (
                "Setup exceeds the configured capital"
            ),
            "instruction": (
                f"{ticker} meets the model's screening "
                f"criteria, but at the configured capital "
                f"the risk-based sizing rounds to zero "
                f"shares. Minimum capital for one share "
                f"at this risk setting: "
                f"{pick.get('capital_note', 'N/A')}. "
                f"The scenario below is shown for "
                f"reference only."
            ),
            "valid": False,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "exit_date": exit_date_str
        }

    instruction = (
        f"Model reference zone for {ticker}: "
        f"₹{entry_low:,.2f}–₹{entry_high:,.2f}. "
        f"At your configured capital "
        f"(₹{capital:,.0f}) and {risk_pct}% risk per "
        f"position, the model's sizing works out to "
        f"{shares} shares (≈₹{cost:,.2f} notional).\n\n"
        f"Invalidation level: ₹{stop:,.2f}. If price "
        f"traded there, the modelled downside on this "
        f"sizing would be ₹{max_loss:,.2f} — "
        f"{(max_loss/capital*100):.1f}% of the "
        f"configured capital.\n\n"
        f"Upside reference levels: T1 ₹{t1:,.2f} and "
        f"T2 ₹{t2:,.2f} (reward/risk 1:{rr:.1f}). "
        f"In backtests, this setup type performed best "
        f"when half the position ({half_shares} shares) "
        f"was reduced at T1 with the invalidation level "
        f"raised to the entry zone, and the remainder "
        f"({remaining_shares} shares) held toward T2 — "
        f"with the scenario closed by {exit_date_str} "
        f"if neither level was reached. The model's "
        f"scenario window is {holding} trading days."
    )

    if regime == "bear":
        instruction += (
            f"\n\nContext: the market regime model reads "
            f"current conditions as bearish. Historically, "
            f"setups of this type had weaker outcomes and "
            f"wider drawdowns in bear regimes."
        )

    if risk_level == "HIGH":
        instruction += (
            f"\n\nRisk flag: the model classifies this as "
            f"a high-volatility setup. The full modelled "
            f"downside is ₹{max_loss:,.2f}."
        )

    instruction += (
        "\n\nThis is a statistical scenario generated "
        "from historical data, not a recommendation "
        "or advice to trade."
    )

    return {
        "summary": (
            f"Zone ₹{entry_low:,.2f}–₹{entry_high:,.2f} · "
            f"Invalidation ₹{stop:,.2f} · "
            f"Model sizing {shares} sh · "
            f"Window ends {exit_date_str}"
        ),
        "instruction": instruction,
        "valid": True,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "exit_date": exit_date_str,
        "half_shares": half_shares,
        "remaining_shares": remaining_shares
    }


def check_entry_validity(pick, current_price):
    """
    Compares the current market price against the
    model's reference zone and reports the setup's
    statistical status.
    """
    entry_low = pick["entry"] * 0.998
    entry_high = pick["entry"] * 1.008
    stop = pick["stop_loss"]

    if current_price < stop:
        return {
            "valid": False,
            "status": "INVALIDATED",
            "reason": (
                f"Price ₹{current_price:,.2f} is below "
                f"the invalidation level ₹{stop:,.2f}. "
                f"The model no longer considers this "
                f"setup active."
            ),
            "color": "red"
        }
    elif current_price > entry_high * 1.02:
        gap_pct = (
            (current_price - pick["entry"]) /
            pick["entry"] * 100
        )
        return {
            "valid": False,
            "status": "EXTENDED",
            "reason": (
                f"Price ₹{current_price:,.2f} is "
                f"{gap_pct:.1f}% above the reference "
                f"zone. The original reward/risk profile "
                f"of this setup no longer applies."
            ),
            "color": "orange"
        }
    elif entry_low <= current_price <= entry_high:
        return {
            "valid": True,
            "status": "IN ZONE",
            "reason": (
                f"Price ₹{current_price:,.2f} is within "
                f"the model's reference zone "
                f"₹{entry_low:,.2f}–₹{entry_high:,.2f}."
            ),
            "color": "green"
        }
    else:
        return {
            "valid": True,
            "status": "NEAR ZONE",
            "reason": (
                f"Price ₹{current_price:,.2f} is near "
                f"but not inside the reference zone "
                f"(starts at ₹{entry_low:,.2f})."
            ),
            "color": "blue"
        }
