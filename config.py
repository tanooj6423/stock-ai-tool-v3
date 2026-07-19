"""
Central configuration for Equitex Intelligence.

All deployment-specific values come from environment
variables so the app can run anywhere (local, VPS,
Railway/Render, Docker) without code changes.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------
# Set DATA_DIR to a mounted volume in production, e.g.
#   DATA_DIR=/data   (docker-compose mounts ./data there)
# Falls back to ./data next to the code for local dev.
DATA_DIR = Path(
    os.getenv("DATA_DIR", Path(__file__).parent / "data")
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

JOURNAL_FILE = str(DATA_DIR / "trade_journal.json")
WATCHLIST_FILE = str(DATA_DIR / "watchlist.json")


def _user_scoped_file(base_name, legacy_path):
    """
    Per-user storage path. When someone is logged in,
    their journal/watchlist live in
    DATA_DIR/users/<hash-of-email>/<base_name> so users
    never see each other's data. Falls back to the
    legacy single-user file when there's no session
    (dev mode with AUTH_DISABLED=1, cron scripts).
    """
    try:
        import hashlib

        import streamlit as st
        email = st.session_state.get("user_email")
        if email:
            uid = hashlib.sha256(
                email.strip().lower().encode()
            ).hexdigest()[:16]
            d = DATA_DIR / "users" / uid
            d.mkdir(parents=True, exist_ok=True)
            return str(d / base_name)
    except Exception:
        pass
    return legacy_path


def journal_file():
    return _user_scoped_file(
        "trade_journal.json", JOURNAL_FILE
    )


def watchlist_file():
    return _user_scoped_file(
        "watchlist.json", WATCHLIST_FILE
    )


# ---------------------------------------------------------
# Secrets
# ---------------------------------------------------------
def get_secret(name, default=""):
    """
    Read a secret from Streamlit secrets first (if the
    runtime provides them), then from the environment.
    Works on HF Spaces, Docker, and bare VPS alike.
    """
    try:
        import streamlit as st
        val = st.secrets.get(name, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(name, default)


# ---------------------------------------------------------
# App metadata / compliance
# ---------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "Equitex Intelligence")

DISCLAIMER_SHORT = (
    "Analytics & research tool — not investment advice. "
    "Not a SEBI-registered Research Analyst or "
    "Investment Adviser."
)

DISCLAIMER_LONG = (
    "Equitex Intelligence is a data analytics and "
    "research platform. All scores, probabilities, "
    "levels and scenarios shown are statistical outputs "
    "of quantitative models applied to historical data. "
    "They are not recommendations, tips, or advice to "
    "buy or sell any security. Past model performance "
    "does not guarantee future results. Securities "
    "markets are subject to market risk. Consult a "
    "SEBI-registered investment professional before "
    "making investment decisions."
)
