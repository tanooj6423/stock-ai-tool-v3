"""
Built-in authentication + subscription tiers for Equitex.

- Users stored in SQLite (DATA_DIR/equitex.db) — survives restarts,
  works on Docker volumes / Railway volumes / HF persistent storage.
- Passwords hashed with PBKDF2-HMAC-SHA256 (260k iterations, per-user salt).
- Tiers: "free" and "pro". Pro is granted by Razorpay payment
  (see billing.py / webhook_server.py) or manually by the admin.
- Free-tier limits are enforced through helpers in this module so the
  UI code stays clean.

Environment variables:
  ADMIN_EMAIL   — this account is always Pro and sees the admin panel.
  AUTH_DISABLED — set to "1" to bypass login entirely (local dev).
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, date

from config import DATA_DIR

DB_PATH = str(DATA_DIR / "equitex.db")

PBKDF2_ITERATIONS = 260_000

# ---------------------------------------------------------
# Free-tier limits (single source of truth)
# ---------------------------------------------------------
FREE_ANALYSES_PER_DAY = 3
FREE_WATCHLIST_MAX = 5
FREE_PICKS_VISIBLE = 1     # free users see the top pick only


# ---------------------------------------------------------
# DB
# ---------------------------------------------------------
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            pro_until TEXT,
            razorpay_customer_id TEXT,
            razorpay_subscription_id TEXT
        );
        CREATE TABLE IF NOT EXISTS usage (
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,
            analyses INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, day)
        );
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            email TEXT,
            provider TEXT DEFAULT 'razorpay',
            payment_id TEXT UNIQUE,
            amount_inr REAL,
            plan TEXT,
            status TEXT,
            raw TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL
        );
        """)
        # Migration: disclaimer acknowledgment timestamp
        try:
            c.execute("ALTER TABLE users ADD COLUMN "
                      "disclaimer_ack TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


init_db()


# ---------------------------------------------------------
# Password hashing
# ---------------------------------------------------------
def _hash_pw(password: str, salt: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt),
        PBKDF2_ITERATIONS
    )
    return dk.hex()


def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


# ---------------------------------------------------------
# Signup / login
# ---------------------------------------------------------
def create_user(email: str, password: str):
    """Returns (ok, message)."""
    email = email.strip().lower()
    if not _valid_email(email):
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt = secrets.token_hex(16)
    pw_hash = _hash_pw(password, salt)
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users (email, pw_hash, salt, created_at) "
                "VALUES (?,?,?,?)",
                (email, pw_hash, salt, datetime.utcnow().isoformat())
            )
        return True, "Account created — you're signed in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def verify_login(email: str, password: str):
    """Returns user dict on success, None on failure."""
    email = email.strip().lower()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE email=?", (email,)
        ).fetchone()
    if row is None:
        return None
    if hmac.compare_digest(_hash_pw(password, row["salt"]), row["pw_hash"]):
        return dict(row)
    return None


def get_user(email: str):
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE email=?",
            (email.strip().lower(),)
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------
# Persistent sessions (survive Streamlit reruns/restarts)
# ---------------------------------------------------------
SESSION_TTL = 30 * 24 * 3600  # 30 days


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (?,?,?,?)", (token, user_id, now, now + SESSION_TTL)
        )
        c.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    return token


def get_session_user(token: str):
    if not token:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token=? AND s.expires_at > ?",
            (token, time.time())
        ).fetchone()
    return dict(row) if row else None


def destroy_session(token: str):
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


# ---------------------------------------------------------
# Tiers
# ---------------------------------------------------------
def is_admin(user) -> bool:
    admin = os.getenv("ADMIN_EMAIL", "").strip().lower()
    return bool(user) and bool(admin) and user["email"] == admin


def is_pro(user) -> bool:
    """Pro if: admin, tier='pro' with unexpired pro_until (or none set)."""
    if not user:
        return False
    if is_admin(user):
        return True
    if user.get("tier") != "pro":
        return False
    pro_until = user.get("pro_until")
    if pro_until:
        try:
            return datetime.fromisoformat(pro_until) >= datetime.utcnow()
        except ValueError:
            return False
    return True


def record_disclaimer_ack(email: str):
    """Timestamped proof the user acknowledged the
    research-tool disclaimer (kept for compliance)."""
    with _conn() as c:
        c.execute(
            "UPDATE users SET disclaimer_ack=? WHERE email=?",
            (datetime.utcnow().isoformat(),
             email.strip().lower())
        )


def set_tier(email: str, tier: str, pro_until: str = None):
    with _conn() as c:
        c.execute(
            "UPDATE users SET tier=?, pro_until=? WHERE email=?",
            (tier, pro_until, email.strip().lower())
        )


# ---------------------------------------------------------
# Free-tier usage limits
# ---------------------------------------------------------
def analyses_used_today(user) -> int:
    if not user:
        return 0
    with _conn() as c:
        row = c.execute(
            "SELECT analyses FROM usage WHERE user_id=? AND day=?",
            (user["id"], date.today().isoformat())
        ).fetchone()
    return row["analyses"] if row else 0


def can_analyze(user) -> bool:
    if is_pro(user):
        return True
    return analyses_used_today(user) < FREE_ANALYSES_PER_DAY


def record_analysis(user):
    if not user or is_pro(user):
        return
    with _conn() as c:
        c.execute(
            "INSERT INTO usage (user_id, day, analyses) VALUES (?,?,1) "
            "ON CONFLICT(user_id, day) "
            "DO UPDATE SET analyses = analyses + 1",
            (user["id"], date.today().isoformat())
        )


# ---------------------------------------------------------
# Payments bookkeeping (called by billing / webhook)
# ---------------------------------------------------------
def record_payment(email, payment_id, amount_inr, plan, status, raw=""):
    user = get_user(email) if email else None
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO payments (user_id, email, payment_id, "
                "amount_inr, plan, status, raw, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (user["id"] if user else None, email, payment_id,
                 amount_inr, plan, status, raw,
                 datetime.utcnow().isoformat())
            )
        return True
    except sqlite3.IntegrityError:
        return False  # duplicate payment_id — already processed


def list_users(limit=200):
    with _conn() as c:
        rows = c.execute(
            "SELECT email, tier, pro_until, created_at FROM users "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
