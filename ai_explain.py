import os
import streamlit as st
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

def get_api_key():
    key = None
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except:
        pass
    if not key:
        key = os.getenv("GEMINI_API_KEY", "")
    return key

@st.cache_data(ttl=3600)
def explain_signal(ticker, signal, confidence, sentiment,
                   rsi, macd, accuracy, buy_prob, sell_prob,
                   sharpe=None, max_drawdown=None,
                   pe_ratio=None, week52_high=None,
                   week52_low=None, current_price=None,
                   correlation=None, sector=None,
                   relative_strength=None,
                   market_regime=None):

    technicals = f"""
- Signal: {signal} (confidence: {confidence:.1%}, model accuracy: {accuracy:.1%})
- Buy probability: {buy_prob:.1%} | Sell probability: {sell_prob:.1%}
- RSI: {rsi:.1f} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})
- MACD: {macd:.2f}
- News sentiment: {sentiment}
- Market regime: {market_regime or 'unknown'}
- Relative strength vs Nifty: {relative_strength or 'N/A'}"""

    risk = ""
    if sharpe:
        risk += f"\n- Sharpe Ratio: {sharpe}"
    if max_drawdown:
        risk += f"\n- Max Drawdown: {max_drawdown}"

    fundamentals = ""
    if pe_ratio and pe_ratio != "N/A":
        fundamentals += f"\n- P/E Ratio: {round(pe_ratio, 1)}"
    if week52_high and week52_high != "N/A":
        fundamentals += f"\n- 52W Range: ₹{week52_low} — ₹{week52_high}"
    if current_price and week52_high and week52_high != "N/A":
        pct = ((current_price - week52_high) / week52_high) * 100
        fundamentals += f"\n- vs 52W High: {pct:.1f}%"
    if correlation:
        fundamentals += f"\n- Nifty 50 correlation: {correlation}"
    if sector and sector != "N/A":
        fundamentals += f"\n- Sector: {sector}"

    prompt = f"""You are a senior equity analyst at a top-tier Indian 
investment bank specialising in NSE-listed stocks.

Analyse {ticker} based on the following data and produce a structured 
investment brief.

TECHNICAL DATA:{technicals}
{f'RISK METRICS:{risk}' if risk else ''}
{f'FUNDAMENTAL DATA:{fundamentals}' if fundamentals else ''}

Write a professional 5-point analysis:
1. Technical outlook
2. Momentum and sentiment
3. Risk assessment
4. Fundamental context
5. Summary verdict — one decisive sentence

Use precise financial language.
End with: "This is not financial advice."
"""
    try:
        client = genai.Client(api_key=get_api_key())
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        try:
            client = genai.Client(api_key=get_api_key())
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except Exception:
            return "AI analysis temporarily unavailable."

@st.cache_data(ttl=3600)
def generate_pick_thesis(ticker, signal, confidence, score,
                          entry, stop_loss, target1, target2,
                          rr_ratio, sentiment, rsi, sector,
                          market_regime, relative_strength):

    prompt = f"""You are a professional swing trader and analyst.

Generate a concise 3-sentence trade thesis for {ticker}.

Data:
- Signal: {signal} | Score: {score}/100 | Confidence: {confidence:.1%}
- Entry: ₹{entry:,.2f} | Stop: ₹{stop_loss:,.2f} | T1: ₹{target1:,.2f} | T2: ₹{target2:,.2f}
- R/R ratio: 1:{rr_ratio:.1f}
- RSI: {rsi:.1f} | Sentiment: {sentiment}
- Sector: {sector} | Market regime: {market_regime}
- Relative strength vs Nifty: {relative_strength}

Write:
1. Why this setup is compelling right now
2. The key catalyst or technical trigger
3. The main risk to watch

Be specific, concise, professional.
End with: "Not financial advice."
"""
    try:
        client = genai.Client(api_key=get_api_key())
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception:
        try:
            client = genai.Client(api_key=get_api_key())
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text
        except Exception:
            return "Thesis generation unavailable."