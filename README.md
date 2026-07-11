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
Quantitative equity analytics for NSE markets (with a US stocks module).
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

Journal/watchlist data is stored under `./data/` (configurable via
`DATA_DIR`).

## Deploy with Docker (VPS, Railway, Render, Fly.io)

```bash
cp .env.example .env   # fill in your keys
docker compose up -d --build
```

The app listens on port **8501**. Journal/watchlist persist in the
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
| `ALPACA_API_KEY/SECRET` | Optional US stocks module |
| `DATA_DIR` | Persistent storage path (default `./data`) |

## Compliance posture

The product is positioned as a **data analytics/research tool**: it
publishes model scores, probabilities, reference levels and scenario
statistics. It deliberately avoids prescriptive buy/sell instructions.
Do not add recommendation language without SEBI Research Analyst
registration. See `config.py` for the canonical disclaimers.

## Known limitations / next steps

- Market data currently comes from yfinance — fine for personal use,
  **must be replaced with a licensed feed (e.g. Kite Connect) before
  commercial launch**.
- Single-user: no authentication or per-user storage yet.
- Models train on demand; move to precomputed nightly scans before
  taking real traffic.
