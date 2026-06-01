import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

def get_entry_instruction(pick, capital, risk_pct):
    """
    Generates a plain English trade instruction
    for a given pick.
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
    holding = pick["holding_days"]
    rr = pick["rr1"]
    risk_level = pick["risk_level"]
    confidence = pick["confidence"]
    regime = pick["market_regime"]

    # Entry zone — allow 0.5% above entry
    entry_low = round(entry * 0.998, 2)
    entry_high = round(entry * 1.005, 2)

    # Exit date
    today = datetime.now()
    exit_date = today + timedelta(days=holding)
    # Skip weekends
    while exit_date.weekday() >= 5:
        exit_date += timedelta(days=1)
    exit_date_str = exit_date.strftime("%A %d %b")

    # Half position for partial exit
    half_shares = max(1, shares // 2)
    remaining_shares = shares - half_shares

    # Build instruction
    if shares == 0:
        return {
            "summary": "Capital too low for this trade",
            "instruction": (
                f"{ticker} is a valid setup but requires "
                f"more capital. Minimum needed: "
                f"{pick.get('capital_note', 'N/A')}. "
                f"Skip this trade or increase your "
                f"capital allocation in Settings."
            ),
            "valid": False,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "exit_date": exit_date_str
        }

    instruction = (
        f"If {ticker} opens or trades between "
        f"₹{entry_low:,.2f} and ₹{entry_high:,.2f}, "
        f"buy {shares} shares. "
        f"Your total outlay will be approximately "
        f"₹{cost:,.2f}. "
        f"Immediately place a stop loss order at "
        f"₹{stop:,.2f}. "
        f"If the stop is triggered, your loss will be "
        f"₹{max_loss:,.2f} — "
        f"that is {(max_loss/capital*100):.1f}% of your "
        f"₹{capital:,.0f} capital.\n\n"
        f"Your first target is ₹{t1:,.2f}. "
        f"When Target 1 is reached, sell {half_shares} "
        f"shares and move your stop loss up to your "
        f"entry price of ₹{entry:,.2f}. "
        f"This locks in profit and makes the remaining "
        f"{remaining_shares} shares risk-free.\n\n"
        f"Let the remaining {remaining_shares} shares "
        f"run to Target 2 at ₹{t2:,.2f} "
        f"(R/R ratio 1:{rr:.1f}). "
        f"If Target 2 is not reached, exit the full "
        f"position by {exit_date_str} regardless of price."
    )

    if regime == "bear":
        instruction += (
            f"\n\nNote: Market is in a bear regime. "
            f"Consider reducing position size by half "
            f"and targeting only Target 1."
        )

    if risk_level == "HIGH":
        instruction += (
            f"\n\nRisk warning: This is a high-risk setup. "
            f"Only enter if you are comfortable with the "
            f"full ₹{max_loss:,.2f} loss scenario."
        )

    return {
        "summary": (
            f"Buy {shares} shares between "
            f"₹{entry_low:,.2f}–₹{entry_high:,.2f} · "
            f"Stop ₹{stop:,.2f} · "
            f"Exit by {exit_date_str}"
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
    Checks if current market price is still
    within valid entry zone.
    """
    entry_low = pick["entry"] * 0.998
    entry_high = pick["entry"] * 1.008
    stop = pick["stop_loss"]

    if current_price < stop:
        return {
            "valid": False,
            "status": "INVALID",
            "reason": (
                f"Price ₹{current_price:,.2f} has broken "
                f"below stop loss ₹{stop:,.2f}. "
                f"Do not enter this trade."
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
            "status": "GAPPED UP",
            "reason": (
                f"Price ₹{current_price:,.2f} has gapped "
                f"{gap_pct:.1f}% above entry zone. "
                f"Risk/reward no longer favourable. "
                f"Wait for a pullback or skip."
            ),
            "color": "orange"
        }
    elif entry_low <= current_price <= entry_high:
        return {
            "valid": True,
            "status": "VALID ENTRY NOW",
            "reason": (
                f"Price ₹{current_price:,.2f} is within "
                f"entry zone ₹{entry_low:,.2f}–"
                f"₹{entry_high:,.2f}. "
                f"Good time to enter."
            ),
            "color": "green"
        }
    else:
        return {
            "valid": True,
            "status": "WATCH",
            "reason": (
                f"Price ₹{current_price:,.2f} is near "
                f"but not yet in entry zone. "
                f"Set alert at ₹{entry_low:,.2f}."
            ),
            "color": "blue"
        }