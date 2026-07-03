---
title: Crypto Signals (Research)
emoji: 🪙
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.57.0"
python_version: "3.11"
app_file: crypto/app.py
pinned: false
---

# Crypto market analysis — research dashboard

Decision-support tool: probabilistic 7-day direction signals for the top-30
liquid Binance universe, with walk-forward-validated success rates, honest
confidence intervals, backtests, and explicit AVOID/LONG/NO-SIGNAL stances.

**Not financial advice. No trade execution. No API keys.**

Copy this file to `README.md` at the root of the Hugging Face Space repo.
Update data by running locally, then pushing:

```bash
python -m crypto.pipeline daily     # refresh -> features -> train -> backtest -> recommend
git add crypto/data/raw crypto/artifacts
git commit -m "daily data refresh"
git push space main                 # 'space' = the HF Space remote
```
