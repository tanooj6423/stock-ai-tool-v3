"""
Nightly scan job — run after market close (e.g. 18:30 IST
on weekdays). Precomputes the full scan so the app serves
results instantly, logs every pick to the forward track
record, and evaluates outcomes of previous picks.

Usage:
    python nightly_scan.py

Cron (VPS):
    30 18 * * 1-5  cd /path/to/app && python nightly_scan.py >> data/nightly.log 2>&1

Env:
    SCAN_UNIVERSE = n50 | n100 | n250   (default n100)
    SCAN_CAPITAL, SCAN_RISK_PCT         (sizing defaults)
"""

import os
import sys
import traceback
from datetime import datetime


def log(msg):
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        f" {msg}",
        flush=True,
    )


def main():
    log("Nightly scan starting")

    # Imports here so a broken dependency is logged,
    # not silently fatal before logging starts.
    from universe import (NIFTY_50, NIFTY_NEXT_50,
                          NIFTY_MIDCAP_150)
    from screener_engine import run_full_scan
    from data import get_stock_data
    from scan_store import (save_scan, log_picks,
                            evaluate_track_record,
                            track_record_stats)
    from kite_data import is_kite_data_active

    choice = os.getenv("SCAN_UNIVERSE", "n100").lower()
    if choice == "n50":
        tickers, name = NIFTY_50, "Nifty 50"
    elif choice == "n250":
        tickers = (NIFTY_50 + NIFTY_NEXT_50 +
                   NIFTY_MIDCAP_150)
        name = "Nifty 50 + Next 50 + Midcap 150"
    else:
        tickers = NIFTY_50 + NIFTY_NEXT_50
        name = "Nifty 50 + Next 50"

    capital = float(os.getenv("SCAN_CAPITAL", "100000"))
    risk_pct = float(os.getenv("SCAN_RISK_PCT", "1.0"))

    log(f"Universe: {name} ({len(tickers)} stocks)")
    log(
        "Data source: "
        + ("Kite Connect" if is_kite_data_active()
           else "yfinance fallback")
    )

    # 1) Evaluate outcomes of previously logged picks
    #    (do this first — uses fresh EOD candles)
    try:
        n = evaluate_track_record(get_stock_data)
        log(f"Track record: {n} entries closed/updated")
    except Exception:
        log("Track record evaluation failed:")
        traceback.print_exc()

    # 2) Run the full scan
    def progress(current, total, ticker):
        if current % 25 == 0 or current == total:
            log(f"  scan {current}/{total} ({ticker})")

    picks, regime = run_full_scan(
        tuple(tickers),
        capital=capital,
        risk_pct=risk_pct,
        progress_callback=progress,
    )
    log(
        f"Scan complete: {len(picks)} picks, "
        f"regime={regime}"
    )

    # 3) Persist for the app + append to forward log
    if not save_scan(picks, regime, universe_name=name):
        log("ERROR: failed to save scan results")
        sys.exit(1)
    added = log_picks(picks)
    log(f"Track record: {added} new picks logged")

    stats = track_record_stats()
    log(f"Track record stats: {stats}")
    log("Nightly scan done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL: nightly scan crashed")
        traceback.print_exc()
        sys.exit(1)
