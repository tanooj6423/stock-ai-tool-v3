# Equitex — Step-by-Step Guide to Monetization-Ready
*Prepared July 2026. Work top to bottom; each phase builds on the last.*

**Status legend:** ✅ already done in your codebase · ⬜ to do

---

## Phase 0 — What's already done ✅

- ✅ Compliance reframe: app presents model scenarios/reference levels, not buy/sell advice (Level-1 analytics posture, no SEBI RA needed)
- ✅ Legal disclaimers in header + footer (`config.py`)
- ✅ Model fixes: delivery-feature bug removed, calibration leak fixed, honest metric labels
- ✅ Koyfin-style UI (Koyfin Dark / Koyfin Light themes), Streamlit chrome hidden
- ✅ Portable deployment: working `Dockerfile`, `docker-compose.yml`, `.env.example`, persistent `DATA_DIR`

---

## Phase 1 — See it and ship it to your Space (today, ₹0)

1. ⬜ Run locally to verify everything:
   ```bash
   cd ~/Desktop/stock-ai-tool-v3
   pip install -r requirements.txt   # if not already
   streamlit run app.py
   ```
   Opens at http://localhost:8501. Check every tab.

2. ⬜ Push the new code to your existing HF Space (it deploys from git):
   ```bash
   git add -A
   git commit -m "Level 1 compliance + Koyfin UI + deploy prep"
   git push
   ```
   The Space rebuilds in ~5 min. Use it as your free staging site from now on.

> **About the UI:** the Hugging Face frame around your app disappears
> automatically on your own domain — no code change needed. The app itself
> is Streamlit and looks identical everywhere; its look comes from your
> code (which now carries the Koyfin design).

---

## Phase 2 — Your own domain + hosting (this week, ~₹1,000–1,700/mo)

3. ⬜ Buy a domain (~₹700–1,000/yr): Namecheap, Porkbun, or GoDaddy.
   Ideas: `equitex.in`, `getequitex.com`. Add it to a free Cloudflare
   account (SSL + caching + DDoS protection).

4. ⬜ Pick a host — either is fine:

   **Option A — Railway (easiest, ~$5–10/mo):**
   1. Push your repo to GitHub (private repo is fine).
   2. railway.app → New Project → Deploy from GitHub repo.
   3. It detects the Dockerfile and builds automatically.
   4. Settings → Networking → add your custom domain; copy the CNAME
      Railway gives you into Cloudflare DNS.
   5. Variables tab → add everything from `.env.example`
      (GEMINI_API_KEY etc.).
   6. Add a Volume mounted at `/data` (this keeps journal/watchlist
      across deploys).

   **Option B — VPS (Hetzner/DigitalOcean, ~₹400–800/mo, more control):**
   1. Create the smallest Ubuntu server, SSH in.
   2. Install Docker: `curl -fsSL https://get.docker.com | sh`
   3. `git clone` your repo, `cp .env.example .env`, fill in keys.
   4. `docker compose up -d --build`
   5. Install Caddy for HTTPS:
      ```
      yourdomain.com {
          reverse_proxy localhost:8501
      }
      ```
   6. Point your domain's A record at the server IP in Cloudflare.

5. ⬜ Once your domain works, keep the HF Space as staging or delete it.
   (You can then also delete the YAML front-matter block at the top of
   README.md.)

---

## Phase 3 — Licensed data (before charging anyone, ₹500/mo)

yfinance is fine for personal use but prohibited for commercial products,
and it's your most fragile dependency.

6. ⬜ Open a Zerodha account if needed → developers.kite.trade →
   create an app → subscribe to Kite Connect (₹500/mo, includes live +
   10y historical NSE/BSE data).
7. ⬜ Replace the yfinance calls in `data.py` (`get_stock_data`,
   `get_nifty_data`, `get_india_vix`, quotes) with Kite Connect's
   historical/quote APIs, keeping yfinance as dev fallback.
   *(Ask me — this is a focused change to one file plus a token-refresh
   helper.)*
8. ⬜ Alternatives if you prefer: Upstox / Dhan / Fyers APIs (free or
   cheap). Same swap, different client library.

---

## Phase 4 — Precompute nightly (performance + cost)

Right now models train on demand — slow and won't survive real traffic.

9. ⬜ Create `nightly_scan.py`: after market close, train models and run
   `run_full_scan` for the whole universe, write results to
   `DATA_DIR/scan_results.json` (or the DB from Phase 5).
10. ⬜ Schedule it: `cron` on the VPS (`30 18 * * 1-5`) or Railway cron
    service. The app then just reads the precomputed file — instant loads.
11. ⬜ Start logging every nightly pick with timestamp — this becomes your
    **public track record**, your single best marketing asset.

---

## Phase 5 — Accounts + payments (the actual monetization, weeks 3–6)

12. ⬜ Auth: create a free Supabase project → enable email + Google
    login. Gate the app behind login (`streamlit-supabase-auth` or a
    simple session check).
13. ⬜ Move journal/watchlist from JSON files to Supabase Postgres,
    keyed by user id (schema: `users`, `journal_trades`, `watchlist_items`,
    `subscriptions`).
14. ⬜ Payments: sign up for **Razorpay** (individual/proprietorship is
    fine to start; you'll need PAN + bank account). Create subscription
    plans:
    - Free: 3 analyses/day, top-3 scan blurred, watchlist ≤5
    - Pro ₹399/mo or ₹2,999/yr: everything
15. ⬜ Wire Razorpay webhooks → set `subscriptions.status` in Supabase →
    app checks status to unlock Pro features.
16. ⬜ Add Terms of Service + Privacy Policy pages (required by Razorpay
    KYC and by law). One securities-lawyer consult (~₹5–15k) to bless
    your copy is strongly recommended before charging.

---

## Phase 6 — Credibility + launch (weeks 4–8)

17. ⬜ Build the backtest page: simulate the pick methodology 3–5 years
    back **including brokerage/STT/slippage**, publish equity curve, win
    rate, max drawdown. Show live-forward picks from step 11 alongside.
18. ⬜ Landing page on your domain root (hero, 3 screenshots, track
    record, pricing, FAQ). Can be a simple static page in front of the
    Streamlit app.
19. ⬜ Distribution engine:
    - Telegram channel: daily free scan teaser (top pick only, delayed)
    - X/Twitter + r/IndianStockMarket: weekly "every pick, wins AND
      losses" post
    - Broker affiliate links (Upstox/Angel One/Dhan: ~₹300–600 per
      funded account) in onboarding
20. ⬜ Launch: soft-launch to the Telegram community first, iterate a
    week, then wider posts.

---

## Ongoing monthly costs at launch

| Item | Cost |
|---|---|
| Hosting (Railway or VPS) | ₹400–900 |
| Kite Connect data | ₹500 |
| Domain (amortized) | ~₹70 |
| Gemini API | ₹300–1,000 |
| Supabase, Cloudflare | ₹0 (free tiers) |
| Razorpay | ~2% of revenue |
| **Total** | **~₹1,500–2,500/mo** ✅ within budget |

---

## Rules to not break

1. No prescriptive buy/sell language anywhere in the paid product until
   you hold SEBI RA registration (the code now enforces this framing —
   don't revert it).
2. No charging users while the data layer is still yfinance.
3. Never publish accuracy claims you can't back with the cost-adjusted
   backtest.
4. `.env` never goes in git (already ignored).

*Next actions I can do for you right now: the Kite Connect data-layer swap
(step 7), the nightly scan job (step 9), or the Supabase auth gate
(step 12). Just say which.*
