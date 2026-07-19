---
title: Stock AI Tool V3
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.57.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Equitex Intelligence

Quantitative equity analytics for NSE markets, with user accounts,
Free/Pro subscription tiers, and a forward-verified track record.
Research/analytics tool — **not investment advice, not SEBI-registered**.

> The YAML front-matter above is only used by Hugging Face Spaces and can
> be deleted once the app is fully migrated off HF.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
streamlit run app.py
```

Tip for development: set `AUTH_DISABLED=1` in `.env` to skip the login
screen. Set `ADMIN_EMAIL=you@example.com` to make your account Pro with
an admin panel (Settings tab).

User accounts, journal and watchlist data live under `./data/`
(configurable via `DATA_DIR`).

## Accounts & monetization

- Built-in email/password auth (PBKDF2-hashed, SQLite) — no external
  service needed. Sessions persist 30 days.
- Tiers: **Free** (3 analyses/day, top pick only, watchlist ≤ 5) and
  **Pro** ₹399/mo or ₹2,999/yr (everything). Limits are defined in
  `auth.py`.
- **Razorpay** (not yet activated): checkout + auto-upgrade are wired in
  `billing.py` / `webhook_server.py`. To go live, add
  `RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET` (or zero-code Payment Page
  links) to `.env`, then run the webhook receiver:
  ```bash
  uvicorn webhook_server:app --host 0.0.0.0 --port 8600
  ```
  and register `https://yourdomain.com/razorpay/webhook` in the
  Razorpay dashboard. Until then, upgrade users manually from the
  admin panel.
- Terms of Service and Privacy Policy live in `legal.py` (Plans tab).
  Have a lawyer review before charging.

## Nightly scan (precomputed picks + track record)

```bash
python nightly_scan.py            # after market close
# cron: 30 18 * * 1-5
```

Precomputes the Daily Picks scan, logs every pick to the forward track
record, and evaluates past picks. The app serves these instantly.

## Deploy with Docker (VPS, Railway, Render, Fly.io)

```bash
cp .env.example .env   # fill in your keys
docker compose up -d --build
```

The app listens on port **8501**. All user data persists in the
`equitex_data` volume mounted at `/data`.

Recommended production setup on a VPS:

1. Point your domain's DNS at the server (Cloudflare proxy recommended).
2. Put Caddy or nginx in front of port 8501 for HTTPS, e.g. Caddyfile:
   ```
   yourdomain.com {
       reverse_proxy localhost:8501
   }
   ```
3. Set all secrets in `.env` (never commit it — it's gitignored).

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | AI-generated data commentaries |
| `ZERODHA_API_KEY/SECRET` | Optional personal Zerodha link |
| `ADMIN_EMAIL` | Always-Pro admin account with user management |
| `AUTH_DISABLED` | `1` = skip login (dev only) |
| `RAZORPAY_*` | Payments (see Accounts & monetization) |
| `DATA_DIR` | Persistent storage path (default `./data`) |

## Model (v3.1)

Per-stock gradient-boosted classifiers (LightGBM + XGBoost ensemble)
on a Triple-Barrier target (ATR-scaled take-profit/stop within a 7-day
horizon). 41 features including trend strength (ADX), money flow (MFI),
OBV slope, 60-day momentum, distance from 52-week high, and downside
deviation. Evaluated with purged walk-forward splits (3 folds +
terminal split, 7-day embargo); probabilities isotonic-calibrated on a
held-out tail. Reported scores are out-of-sample.

## Compliance posture

The product is positioned as a **data analytics/research tool**: it
publishes model scores, probabilities, reference levels and scenario
statistics. It deliberately avoids prescriptive buy/sell instructions.
Do not add recommendation language without SEBI Research Analyst
registration. See `config.py` for the canonical disclaimers.

## Known limitations / next steps

- Market data currently comes from yfinance — fine for personal use,
  **must be replaced with a licensed feed (e.g. Kite Connect) before
  commercial launch**. (`kite_data.py` is ready — add keys.)
- Razorpay checkout is scaffolded but not activated (add keys).
- Run `nightly_scan.py` on a scheduler so picks are always fresh.
