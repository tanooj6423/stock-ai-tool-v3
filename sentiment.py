import yfinance as yf
import streamlit as st
import requests
import re
from datetime import datetime, timedelta

# transformers/torch are heavy optional deps: FinBERT is
# used when available, keyword scoring is the fallback.
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

NEGATIVE_KEYWORDS = [
    "fraud", "scam", "investigation", "probe", "scandal", "lawsuit",
    "default", "bankruptcy", "insolvency", "downgrade", "loss", "decline",
    "warning", "recall", "penalty", "fine", "raid", "arrest", "resign",
    "layoff", "cut", "miss", "disappoint", "weak", "fall", "drop",
    "crash", "plunge", "slump", "concern", "risk", "threat", "selloff"
]

POSITIVE_KEYWORDS = [
    "profit", "growth", "record", "beat", "upgrade", "buy", "outperform",
    "expansion", "acquisition", "deal", "contract", "order", "launch",
    "partnership", "approval", "dividend", "buyback", "strong", "rise",
    "rally", "surge", "gain", "high", "positive", "win", "award",
    "breakthrough", "innovation", "revenue", "margin", "guidance"
]

@st.cache_resource
def load_finbert():
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers not installed — "
            "keyword fallback will be used"
        )
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

def analyze_keywords(text):
    text_lower = text.lower()
    neg_found = [w for w in NEGATIVE_KEYWORDS if w in text_lower]
    pos_found = [w for w in POSITIVE_KEYWORDS if w in text_lower]
    return pos_found, neg_found

def get_sentiment_trend(headlines_with_dates):
    if not headlines_with_dates or len(headlines_with_dates) < 2:
        return "stable"
    try:
        model = load_finbert()
        recent = headlines_with_dates[:3]
        older = headlines_with_dates[3:]
        if not older:
            return "stable"

        def avg_score(items):
            texts = [h["title"] for h in items]
            results = model(texts, truncation=True, max_length=512)
            scores = []
            for r in results:
                if r["label"].lower() == "positive":
                    scores.append(r["score"])
                elif r["label"].lower() == "negative":
                    scores.append(-r["score"])
                else:
                    scores.append(0)
            return sum(scores) / len(scores) if scores else 0

        recent_score = avg_score(recent)
        older_score = avg_score(older)
        diff = recent_score - older_score

        if diff > 0.15:
            return "improving"
        elif diff < -0.15:
            return "deteriorating"
        else:
            return "stable"
    except Exception:
        return "stable"

def get_sentiment_scores(texts):
    try:
        model = load_finbert()
        if not texts:
            return "neutral", 0.0, {}, [], []
        results = model(texts[:8], truncation=True, max_length=512)
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for r in results:
            scores[r["label"].lower()] += r["score"]
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant] / len(results)
        distribution = {
            k: round(v / len(results), 3)
            for k, v in scores.items()
        }

        all_pos_keywords = []
        all_neg_keywords = []
        for text in texts:
            pos, neg = analyze_keywords(text)
            all_pos_keywords.extend(pos)
            all_neg_keywords.extend(neg)

        pos_unique = list(set(all_pos_keywords))[:5]
        neg_unique = list(set(all_neg_keywords))[:5]

        return dominant, round(confidence, 3), distribution, pos_unique, neg_unique
    except Exception:
        return "neutral", 0.0, {}, [], []

@st.cache_data(ttl=1800)
def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news or []

        headlines_with_dates = []
        for n in news[:10]:
            title = (
                n.get("title") or
                (n.get("content") or {}).get("title") or ""
            )
            pub_time = n.get("providerPublishTime") or n.get("pubDate") or 0
            if title:
                headlines_with_dates.append({
                    "title": title,
                    "time": pub_time,
                    "source": n.get("publisher") or
                              (n.get("content") or {}).get("provider", {}).get("displayName", "Unknown")
                })

        headlines_with_dates.sort(
            key=lambda x: x["time"], reverse=True
        )

        headlines = [h["title"] for h in headlines_with_dates]

        if not headlines:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "distribution": {},
                "headlines": [],
                "sources": [],
                "positive_keywords": [],
                "negative_keywords": [],
                "trend": "stable",
                "headline_count": 0,
                "risk_flags": [],
                "sentiment_score": 0.0
            }

        sentiment, confidence, distribution, pos_kw, neg_kw = get_sentiment_scores(headlines)
        trend = get_sentiment_trend(headlines_with_dates)

        risk_flags = []
        all_text = " ".join(headlines).lower()
        high_risk_terms = [
            "fraud", "investigation", "probe", "default",
            "bankruptcy", "scam", "raid", "arrest", "penalty"
        ]
        for term in high_risk_terms:
            if term in all_text:
                risk_flags.append(term.upper())

        pos_score = distribution.get("positive", 0)
        neg_score = distribution.get("negative", 0)
        sentiment_score = round(pos_score - neg_score, 3)

        sources = list(set([
            h["source"] for h in headlines_with_dates
            if h["source"] != "Unknown"
        ]))[:5]

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "distribution": distribution,
            "headlines": headlines,
            "sources": sources,
            "positive_keywords": pos_kw,
            "negative_keywords": neg_kw,
            "trend": trend,
            "headline_count": len(headlines),
            "risk_flags": risk_flags,
            "sentiment_score": sentiment_score
        }

    except Exception:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "distribution": {},
            "headlines": [],
            "sources": [],
            "positive_keywords": [],
            "negative_keywords": [],
            "trend": "stable",
            "headline_count": 0,
            "risk_flags": [],
            "sentiment_score": 0.0
        }

@st.cache_data(ttl=1800)
def get_batch_sentiment(tickers):
    results = {}
    for ticker in tickers:
        try:
            data = get_news_sentiment(ticker)
            results[ticker] = {
                "sentiment": data["sentiment"],
                "confidence": data["confidence"],
                "sentiment_score": data["sentiment_score"],
                "risk_flags": data["risk_flags"],
                "trend": data["trend"]
            }
        except Exception:
            results[ticker] = {
                "sentiment": "neutral",
                "confidence": 0.0,
                "sentiment_score": 0.0,
                "risk_flags": [],
                "trend": "stable"
            }
    return results