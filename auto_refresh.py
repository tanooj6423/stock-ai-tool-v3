"""
In-app background scan refresh — zero-ops scheduling.

Hugging Face Spaces (and most simple hosts) have no cron. Rather
than depend on your laptop being on, the app refreshes its own
scan from inside the running container:

  * one background thread per process (guarded so it starts once),
  * on startup: if the scan is missing or older than ~20h, run it,
  * then every day shortly after the NSE close (18:30 IST /
    13:00 UTC), run it again,
  * each run also evaluates the forward track record.

This means: **you don't have to open or start anything.** As long
as the Space is running, it keeps its own picks fresh. On a Space
rebuild the data resets, but the thread regenerates it on the next
startup.

Enable/disable with env var AUTO_REFRESH (default "1"). Universe is
kept small (Nifty 50) so a background run is quick and gentle on
the data source. Set AUTO_REFRESH_UNIVERSE=n100 for the wider scan.

Note: on a proper always-on host you can instead run
`nightly_scan.py` from real cron — that's more robust for
production. This module is the no-server convenience path.
"""

import os
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone


def _log(msg):
    print(f"[auto_refresh {datetime.utcnow():%Y-%m-%d %H:%M:%S}] "
          f"{msg}", flush=True)


def _seconds_until_next_run(hour_utc=13, minute_utc=0):
    """Seconds until the next HH:MM UTC (13:00 UTC = 18:30 IST)."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour_utc, minute=minute_utc,
                         second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(60, (target - now).total_seconds())


def _run_scan():
    """One scan cycle: evaluate track record, scan, save."""
    from universe import NIFTY_50, NIFTY_NEXT_50
    from screener_engine import run_full_scan
    from data import get_stock_data
    from scan_store import (save_scan, log_picks,
                            evaluate_track_record)

    universe = os.getenv("AUTO_REFRESH_UNIVERSE", "n50").lower()
    if universe == "n100":
        tickers, name = NIFTY_50 + NIFTY_NEXT_50, \
            "Nifty 50 + Next 50"
    else:
        tickers, name = NIFTY_50, "Nifty 50"

    try:
        evaluate_track_record(get_stock_data)
    except Exception:
        _log("track-record eval failed")
        traceback.print_exc()

    _log(f"scanning {len(tickers)} stocks ({name})...")
    picks, regime = run_full_scan(
        tuple(tickers), capital=100000, risk_pct=1.0,
        run_prefilter=False,
    )

    # Defensive: a rate-limited / failing data source makes
    # run_full_scan return 0 picks. NEVER overwrite an existing
    # good scan with an empty one — a dated screen beats a
    # blank one. Only save when we got setups, or when there's
    # no scan at all yet.
    from scan_store import load_scan
    have_existing = load_scan(max_age_hours=10_000_000) is not None
    if picks or not have_existing:
        save_scan(picks, regime, universe_name=name)
        log_picks(picks)
        _log(f"saved: {len(picks)} setups, regime={regime}")
    else:
        _log("0 setups (likely data-source throttling) — "
             "keeping existing scan, not overwriting")


def _scan_is_stale(max_age_hours=20):
    from scan_store import load_scan
    return load_scan(max_age_hours=max_age_hours) is None


def _loop():
    # Startup catch-up: refresh immediately if stale.
    try:
        if _scan_is_stale():
            _log("scan stale on startup — refreshing now")
            _run_scan()
        else:
            _log("scan is fresh on startup")
    except Exception:
        _log("startup refresh failed")
        traceback.print_exc()

    # Daily loop, shortly after NSE close.
    while True:
        try:
            wait = _seconds_until_next_run()
            _log(f"next refresh in {wait/3600:.1f}h")
            time.sleep(wait)
            _run_scan()
        except Exception:
            _log("scheduled refresh failed; retry in 1h")
            traceback.print_exc()
            time.sleep(3600)


_STARTED = False


def start_background_refresh():
    """Idempotent: starts the daemon thread once per process."""
    global _STARTED
    if _STARTED:
        return
    if os.getenv("AUTO_REFRESH", "1") != "1":
        _log("disabled via AUTO_REFRESH=0")
        _STARTED = True
        return
    _STARTED = True
    t = threading.Thread(target=_loop, daemon=True,
                         name="equitex-auto-refresh")
    t.start()
    _log("background refresh thread started")
