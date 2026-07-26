"""
The Equitex Score — the product's signature metric.

Rationale (from market research): the durable tool businesses
(Trendlyne's DVM, Tickertape's scores) don't sell "buy this",
they sell a *branded, memorable number* users check daily. We
already compute a 0-100 composite in the screener; this module
gives it an identity, a tiered label, and one consistent render
everywhere so it reads like a product, not a debug value.

Compliance: the score is explicitly an analytics rating of setup
quality, never a recommendation. Labels avoid buy/sell verbs.
"""


def score_tier(score):
    """Map 0-100 → (label, meaning). Analytics language only."""
    if score >= 75:
        return ("Prime", "Exceptional setup quality")
    if score >= 65:
        return ("Strong", "High setup quality")
    if score >= 55:
        return ("Constructive", "Above-average setup")
    if score >= 45:
        return ("Neutral", "Average / mixed signals")
    return ("Weak", "Below-average setup")


def score_color(score, t):
    if score >= 65:
        return t["green"]
    if score >= 55:
        return t["accent"]
    if score >= 45:
        return t["gold"]
    return t["red"]


def score_gauge_html(score, t, size="lg"):
    """
    A compact circular-style gauge card for the Equitex Score.
    Pure HTML/CSS (SVG ring) — no JS, renders in st.markdown.
    """
    label, meaning = score_tier(score)
    col = score_color(score, t)
    r = 34
    import math
    circ = 2 * math.pi * r
    dash = circ * (score / 100.0)
    dim = 92 if size == "lg" else 64
    fs_num = 26 if size == "lg" else 18
    return f"""
    <div style="display:flex;align-items:center;gap:16px;">
      <svg width="{dim}" height="{dim}" viewBox="0 0 92 92">
        <circle cx="46" cy="46" r="{r}" fill="none"
          stroke="{t['border']}" stroke-width="7"/>
        <circle cx="46" cy="46" r="{r}" fill="none"
          stroke="{col}" stroke-width="7" stroke-linecap="round"
          stroke-dasharray="{dash} {circ}"
          transform="rotate(-90 46 46)"/>
        <text x="46" y="52" text-anchor="middle"
          font-family="JetBrains Mono, monospace"
          font-size="{fs_num}" font-weight="600"
          fill="{t['text']}">{score}</text>
      </svg>
      <div>
        <div style="font-size:10px;font-weight:700;
          letter-spacing:2px;text-transform:uppercase;
          color:{t['text2']};">Equitex Score</div>
        <div style="font-size:16px;font-weight:700;
          color:{col};margin-top:2px;">{label}</div>
        <div style="font-size:11px;color:{t['text2']};">
          {meaning} · {score}/100</div>
      </div>
    </div>
    """


def score_chip_html(score, t):
    """Inline pill for lists/expander headers."""
    label, _ = score_tier(score)
    col = score_color(score, t)
    return (
        f'<span style="display:inline-flex;align-items:center;'
        f'gap:6px;font-family:JetBrains Mono, monospace;'
        f'font-size:11px;font-weight:600;color:{col};'
        f'background:{t["bg2"]};border:1px solid {t["border"]};'
        f'border-radius:99px;padding:3px 11px;">'
        f'<span style="font-size:9px;">●</span>'
        f'Equitex {score} · {label}</span>'
    )
