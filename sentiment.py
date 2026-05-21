import yfinance as yf
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_finbert():
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

def get_sentiment(texts):
    try:
        model = load_finbert()
        if not texts:
            return "neutral", 0.0, {}
        results = model(texts[:5], truncation=True, max_length=512)
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for r in results:
            scores[r["label"].lower()] += r["score"]
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant] / len(results)
        distribution = {
            k: round(v / len(results), 3)
            for k, v in scores.items()
        }
        return dominant, round(confidence, 3), distribution
    except:
        return "neutral", 0.0, {}

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        headlines = []
        for n in news[:8]:
            title = (
                n.get("title") or
                (n.get("content") or {}).get("title") or ""
            )
            if title:
                headlines.append(title)
        if not headlines:
            return "neutral", 0.0, {}, []
        sentiment, confidence, distribution = get_sentiment(headlines)
        return sentiment, confidence, distribution, headlines
    except:
        return "neutral", 0.0, {}, []

@st.cache_data(ttl=3600)
def get_batch_sentiment(tickers):
    results = {}
    for ticker in tickers:
        try:
            sentiment, confidence, _, _ = get_news_sentiment(ticker)
            results[ticker] = {
                "sentiment": sentiment,
                "confidence": confidence
            }
        except:
            results[ticker] = {
                "sentiment": "neutral",
                "confidence": 0.0
            }
    return results