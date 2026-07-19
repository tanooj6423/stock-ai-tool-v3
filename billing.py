"""
Razorpay billing for Equitex.

Two supported modes — the app picks automatically:

1. **API mode** (recommended): set RAZORPAY_KEY_ID + RAZORPAY_KEY_SECRET
   in .env. The app creates a Razorpay Payment Link per checkout with the
   user's email attached, and the webhook server (webhook_server.py)
   upgrades the account automatically when payment completes.

2. **Static-link mode** (zero code, launch today): create Payment Pages
   in the Razorpay dashboard and set RAZORPAY_LINK_MONTHLY /
   RAZORPAY_LINK_YEARLY to their URLs. Users pay with their email; you
   upgrade them in the built-in Admin panel (Settings → Admin), or let
   the webhook do it if configured.

Plans (INR):
  PRO_MONTHLY  ₹399 / month
  PRO_YEARLY   ₹2,999 / year
"""

import json
import os

import requests

from config import get_secret

PLANS = {
    "pro_monthly": {"label": "Pro · Monthly", "amount_inr": 399,
                    "period_days": 31},
    "pro_yearly": {"label": "Pro · Yearly", "amount_inr": 2999,
                   "period_days": 366},
}

RAZORPAY_API = "https://api.razorpay.com/v1"


def keys():
    return (get_secret("RAZORPAY_KEY_ID"),
            get_secret("RAZORPAY_KEY_SECRET"))


def api_configured() -> bool:
    kid, ksec = keys()
    return bool(kid and ksec)


def static_link(plan: str) -> str:
    return get_secret(
        "RAZORPAY_LINK_MONTHLY" if plan == "pro_monthly"
        else "RAZORPAY_LINK_YEARLY"
    )


def billing_configured() -> bool:
    return api_configured() or bool(
        static_link("pro_monthly") or static_link("pro_yearly")
    )


def create_payment_link(plan: str, email: str):
    """
    Create a Razorpay Payment Link for the plan, tagged with the
    user's email so the webhook can upgrade the right account.
    Returns (url, error).
    """
    if plan not in PLANS:
        return None, "Unknown plan."
    if not api_configured():
        url = static_link(plan)
        if url:
            return url, None
        return None, ("Billing isn't configured yet. Add Razorpay keys "
                      "or payment-page links to .env.")
    p = PLANS[plan]
    try:
        r = requests.post(
            f"{RAZORPAY_API}/payment_links",
            auth=keys(),
            json={
                "amount": p["amount_inr"] * 100,  # paise
                "currency": "INR",
                "description": f"Equitex {p['label']}",
                "customer": {"email": email},
                "notify": {"email": True},
                "notes": {"email": email, "plan": plan},
                "callback_url": get_secret(
                    "APP_URL", "https://example.com"
                ),
                "callback_method": "get",
            },
            timeout=15,
        )
        data = r.json()
        if r.status_code in (200, 201) and data.get("short_url"):
            return data["short_url"], None
        return None, data.get("error", {}).get(
            "description", f"Razorpay error {r.status_code}"
        )
    except requests.RequestException as e:
        return None, f"Network error contacting Razorpay: {e}"


def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """Validate X-Razorpay-Signature on webhook calls."""
    import hashlib
    import hmac as _hmac
    secret = get_secret("RAZORPAY_WEBHOOK_SECRET")
    if not secret:
        return False
    expected = _hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return _hmac.compare_digest(expected, signature or "")


def extract_payment(event: dict):
    """
    Pull (email, payment_id, amount_inr, plan) out of a Razorpay
    webhook event (payment_link.paid or payment.captured).
    """
    payload = event.get("payload", {})
    pl = payload.get("payment_link", {}).get("entity", {})
    pay = payload.get("payment", {}).get("entity", {})
    notes = pl.get("notes") or pay.get("notes") or {}
    email = (notes.get("email") or pay.get("email") or "").lower()
    payment_id = pay.get("id") or pl.get("id")
    amount = (pay.get("amount") or pl.get("amount_paid") or 0) / 100
    plan = notes.get("plan") or (
        "pro_yearly" if amount >= 2500 else "pro_monthly"
    )
    return email, payment_id, amount, plan
