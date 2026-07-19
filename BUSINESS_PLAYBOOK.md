# Equitex — Business Playbook (India, no SEBI RA)
*July 2026. This replaces guesswork with a sequence. Not legal advice — one lawyer review (~₹10–15k) before charging is the cheapest insurance you'll ever buy.*

## 1. The legal position you're monetizing from

You are selling **software access**, not advice. That's the entire business model, and every decision flows from it. A SEBI Research Analyst sells *recommendations*; you sell a *terminal* — screeners, analytics, scenario math, a journal. Screener.in, Trendlyne, Tickertape, and Chartink all charge subscriptions on exactly this posture without RA registration.

What the app now enforces (don't undo any of it):
- No prescriptive language anywhere: "setups", "reference levels", "scenarios" — never "buy", "picks as advice", "you should".
- Disclaimers at signup, in the header, in the footer, in Terms.
- **Timestamped disclaimer acknowledgment** stored per user at first login — your audit trail that every paying customer was informed.
- ToS + Privacy Policy in-app (DPDP Act aware).

What stays on your side of the fence:
1. **Marketing is the #1 risk, not the app.** The moment your Telegram/Twitter says "today's winning stock" or "95% accurate tips", you're an unregistered advisor regardless of what the app says. Market the *tool*: "our screener flagged X at score 71 — here's the full methodology."
2. **Data licensing before revenue.** yfinance's terms prohibit commercial use. Kite Connect (₹500/mo) fixes this — `kite_data.py` is already built, just add keys. Do this before the first paid subscriber.
3. **Business hygiene:** operate as sole proprietor to start (PAN + current account is enough for Razorpay). GST registration only becomes mandatory at ₹20L annual revenue — register then, not now.
4. **Never publish performance claims** you can't back with the cost-adjusted forward track record in the app. The Performance tab is your only permitted marketing number.

## 2. Positioning

**"A quant terminal for retail swing traders — see what the models see."**
Competitor prices anchor you: Trendlyne ₹583/mo, Tickertape Pro ₹245/mo, StockEdge ₹499/mo. Your differentiators: per-stock ML with honest out-of-sample scores, ATR-based scenario levels with position-sizing math, and a public forward track record that includes the losses. Radical transparency *is* the brand — nobody else shows their misses.

## 3. Pricing & funnel (already wired in-app)

- **Free**: 3 analyses/day, top setup only, watchlist of 5. The free tier is a demo, not a product — its job is to show the locked rows.
- **Pro ₹399/mo / ₹2,999/yr** (annual = 2 months free; push annual hard, it's retention in disguise).
- Launch offer: first 50 users "founding member" ₹199/mo locked for a year. Scarcity + early revenue + testimonials.
- Later (only after 3+ months of live track record): ₹999/mo tier with API/export access for power users.

## 4. Distribution engine (₹0 budget)

1. **Telegram channel** — daily post after the nightly scan: market regime + the #1 setup with score, *levels blurred*, "full screen on Equitex". This is the top of your funnel; the app link is in every post.
2. **Weekly transparency post** on X + r/IndianStreetBets + r/IndianStockMarket: every setup from last week, wins AND losses, net return after costs. Losing weeks get posted too — that's what makes it credible and shareable.
3. **Broker affiliates** — Upstox/Angel One/Dhan pay ₹300–600 per funded account. Put "open a broker account" links in onboarding and the Journal tab. This monetizes free users who never subscribe, and it's SEBI-safe (you're referring accounts, not advising).
4. **SEO later**: one page per NSE-500 stock ("TATAMOTORS technical snapshot") is the long-term compounding channel once you're off Streamlit or front it with a static site.

## 5. Cost stack & break-even

| Item | ₹/mo |
|---|---|
| Railway/VPS hosting | 500–900 |
| Kite Connect data | 500 |
| Domain + Cloudflare | ~70 |
| Gemini API | 300–800 |
| **Total** | **~₹1,400–2,300** |

Break-even: **6 Pro subscribers**. 100 subscribers = ~₹40k/mo (~85% margin). At founding-member pricing you need 12. These are hobby-scale numbers to reach — the real game is the conversion rate from Telegram followers, benchmark 1–3%.

## 6. Sequence (do in order)

1. Buy domain, deploy on Railway with `/data` volume, `ADMIN_EMAIL` set. HF Space stays as staging.
2. Kite Connect keys in → licensed data. Cron `nightly_scan.py` at 18:30 IST weekdays.
3. Razorpay account + keys in `.env` (scaffolding is done — reminder is on the task list).
4. Lawyer review of ToS/Privacy/marketing copy.
5. Start the Telegram channel NOW, before launch — the track record and audience should age together. Launch paid when you have ~8 weeks of forward record and ~500 followers.

## 7. KPIs (weekly)

Signups, free→pro conversion (target 3–5%), churn (target <8%/mo), Telegram followers, affiliate clicks, forward-record net return after costs.

## 8. Red lines (never cross)

- No "tips", "calls", "targets achieved 🎯" language anywhere public.
- No DMs answering "should I buy X?" — canned reply pointing to the tool.
- No paid subscriber while data is yfinance.
- No accuracy marketing beyond the in-app Performance tab.
