"""
Razorpay webhook receiver — runs alongside the Streamlit app.

    uvicorn webhook_server:app --host 0.0.0.0 --port 8600

Razorpay dashboard → Webhooks → add:
    URL:    https://yourdomain.com/razorpay/webhook
    Secret: same value as RAZORPAY_WEBHOOK_SECRET in .env
    Events: payment_link.paid, payment.captured

On a valid event it marks the payment in SQLite and upgrades the
user's account to Pro for the plan period. Idempotent — replayed
events are ignored.
"""

import json
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Response

import auth
from billing import (PLANS, extract_payment,
                     verify_webhook_signature)

app = FastAPI(title="Equitex webhooks")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/razorpay/webhook")
async def razorpay_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Razorpay-Signature", "")
    if not verify_webhook_signature(body, sig):
        return Response(status_code=400, content="bad signature")

    event = json.loads(body)
    if event.get("event") not in ("payment_link.paid",
                                  "payment.captured"):
        return {"ignored": event.get("event")}

    email, payment_id, amount, plan = extract_payment(event)
    if not email or not payment_id:
        return Response(status_code=422,
                        content="missing email/payment id")

    fresh = auth.record_payment(
        email, payment_id, amount, plan, "captured",
        raw=body.decode(errors="replace")[:5000]
    )
    if not fresh:
        return {"ok": True, "duplicate": True}

    days = PLANS.get(plan, PLANS["pro_monthly"])["period_days"]
    user = auth.get_user(email)
    base = datetime.utcnow()
    if user and user.get("pro_until"):
        try:
            cur = datetime.fromisoformat(user["pro_until"])
            base = max(base, cur)   # extend, don't overwrite
        except ValueError:
            pass
    auth.set_tier(email, "pro",
                  (base + timedelta(days=days)).isoformat())
    return {"ok": True, "upgraded": email, "plan": plan}
