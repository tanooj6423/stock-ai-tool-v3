"""
Precomputed scan storage + forward track record.

- save_scan / load_scan: the nightly job writes scan
  results; the app serves them instantly instead of
  training models in the request path.
- Track record: every pick is logged the day it is
  published, then evaluated daily against subsequent
  prices. This forward log (published BEFORE outcomes
  are known) is the honest performance evidence the
  product is built on.
"""

import json
import os
from datetime import datetime, date

import pandas as pd

from config import DATA_DIR

SCAN_FILE = str(DATA_DIR / "scan_results.json")
TRACK_FILE = str(DATA_DIR / "track_record.json")


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


# ---------------------------------------------------------
# Precomputed scan
# ---------------------------------------------------------
def save_scan(picks, regime, universe_name=""):
    try:
        payload = {
            "generated_at": datetime.now().isoformat(
                timespec="seconds"
            ),
            "regime": regime,
            "universe": universe_name,
            "picks": picks,
        }
        with open(SCAN_FILE, "w") as f:
            json.dump(payload, f, indent=1,
                      default=_json_default)
        return True
    except Exception:
        return False


def load_scan(max_age_hours=20):
    """
    Returns (picks, regime, generated_at_str) if a scan
    exists and is fresh enough, else None.
    """
    try:
        if not os.path.exists(SCAN_FILE):
            return None
        with open(SCAN_FILE) as f:
            payload = json.load(f)
        gen = datetime.fromisoformat(
            payload["generated_at"]
        )
        age_h = (
            (datetime.now() - gen).total_seconds() / 3600
        )
        if age_h > max_age_hours:
            return None
        return (
            payload.get("picks", []),
            payload.get("regime", "unknown"),
            gen.strftime("%d %b %Y, %H:%M"),
        )
    except Exception:
        return None


# ---------------------------------------------------------
# Track record — forward log
# ---------------------------------------------------------
def _load_track():
    try:
        if os.path.exists(TRACK_FILE):
            with open(TRACK_FILE) as f:
                return json.load(f)
        return []
    except Exception:
        return []


def _save_track(entries):
    try:
        with open(TRACK_FILE, "w") as f:
            json.dump(entries, f, indent=1,
                      default=_json_default)
        return True
    except Exception:
        return False


def log_picks(picks):
    """
    Append today's picks to the forward log (once per
    ticker per day). Returns number of new entries.
    """
    entries = _load_track()
    today = str(date.today())
    existing = {
        (e["date"], e["ticker"]) for e in entries
    }
    added = 0
    for p in picks:
        key = (today, p.get("ticker"))
        if key in existing:
            continue
        try:
            entries.append({
                "date": today,
                "ticker": p["ticker"],
                "signal": p.get("signal", "BUY"),
                "score": p.get("score"),
                "confidence": float(
                    p.get("confidence", 0) or 0
                ),
                "entry": float(p["entry"]),
                "stop": float(p["stop_loss"]),
                "t1": float(p["target1"]),
                "t2": float(p["target2"]),
                "holding_days": int(
                    p.get("holding_days", 7) or 7
                ),
                "sector": p.get("sector", "N/A"),
                "status": "open",
                "t1_touched": False,
                "exit_price": None,
                "exit_date": None,
                "return_pct": None,
            })
            added += 1
        except (KeyError, TypeError, ValueError):
            continue
    _save_track(entries)
    return added


def evaluate_track_record(price_fetcher):
    """
    Walk open entries forward using daily candles.
    price_fetcher(ticker, period) -> OHLCV DataFrame
    (pass data.get_stock_data).

    Day-by-day, conservative ordering (stop checked
    before targets when both hit intra-day):
      Low <= stop        -> closed 'stopped' at stop
      High >= t2         -> closed 'target2' at t2
      High >= t1         -> t1_touched (position runs on)
      window exhausted   -> closed 'expired' at close
    Returns number of entries updated.
    """
    entries = _load_track()
    updated = 0
    for e in entries:
        if e.get("status") != "open":
            continue
        try:
            df = price_fetcher(e["ticker"], "3mo")
            if df is None or df.empty:
                continue
            entry_date = pd.Timestamp(e["date"])
            idx = pd.DatetimeIndex(df.index)
            try:
                idx = idx.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            df = df.copy()
            df.index = idx
            path = df[df.index > entry_date]
            if path.empty:
                continue

            horizon = int(e.get("holding_days", 7))
            entry_px = float(e["entry"])
            closed = False

            for i, (day, row) in enumerate(
                path.iterrows(), start=1
            ):
                if float(row["Low"]) <= e["stop"]:
                    e["status"] = "stopped"
                    e["exit_price"] = e["stop"]
                    e["exit_date"] = str(day.date())
                    closed = True
                elif float(row["High"]) >= e["t2"]:
                    e["status"] = "target2"
                    e["exit_price"] = e["t2"]
                    e["exit_date"] = str(day.date())
                    closed = True
                else:
                    if float(row["High"]) >= e["t1"]:
                        e["t1_touched"] = True
                    if i >= horizon:
                        e["status"] = "expired"
                        e["exit_price"] = float(
                            row["Close"]
                        )
                        e["exit_date"] = str(day.date())
                        closed = True
                if closed:
                    e["return_pct"] = round(
                        (float(e["exit_price"]) -
                         entry_px) / entry_px * 100, 2
                    )
                    updated += 1
                    break
        except Exception:
            continue
    _save_track(entries)
    return updated


def track_record_stats():
    """Aggregate stats over closed entries."""
    entries = _load_track()
    closed = [
        e for e in entries
        if e.get("status") != "open"
        and e.get("return_pct") is not None
    ]
    open_n = sum(
        1 for e in entries if e.get("status") == "open"
    )
    if not closed:
        return {
            "total_logged": len(entries),
            "open": open_n, "closed": 0,
            "win_rate": None, "avg_return": None,
            "t1_hit_rate": None, "best": None,
            "worst": None,
        }
    rets = [float(e["return_pct"]) for e in closed]
    wins = sum(1 for r in rets if r > 0)
    t1_hits = sum(
        1 for e in closed
        if e.get("t1_touched") or
        e.get("status") == "target2"
    )
    return {
        "total_logged": len(entries),
        "open": open_n,
        "closed": len(closed),
        "win_rate": round(wins / len(closed) * 100, 1),
        "avg_return": round(
            sum(rets) / len(rets), 2
        ),
        "t1_hit_rate": round(
            t1_hits / len(closed) * 100, 1
        ),
        "best": max(rets),
        "worst": min(rets),
    }


def get_track_entries():
    entries = _load_track()
    return sorted(
        entries,
        key=lambda e: (e.get("date", ""),
                       e.get("ticker", "")),
        reverse=True,
    )
