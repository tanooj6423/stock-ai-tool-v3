"""Crypto analysis dashboard (decision-support only — no execution).

Run:  streamlit run crypto/app.py
Reads only pipeline/model/backtest artifacts; renders them without
massaging. Every page carries the disclaimer banner (hard rule).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from crypto.config import DATA_DIR, HORIZON_DAYS, PROB_THRESHOLD, RAW_DIR
from crypto.model import ARTIFACT_DIR, WARMUP_FOLDS

st.set_page_config(page_title="Crypto Signals (Research)", layout="wide")

DISCLAIMER = (
    "**Research / decision-support tool — not financial advice.** Signals are "
    "probabilistic model outputs with substantial uncertainty; the backtested "
    "portfolio Sharpe confidence interval includes zero. Past performance "
    "does not predict future results. Alt/memecoin backtests carry "
    "survivorship-by-liquidity bias (see REQUIREMENTS §11). Nothing here "
    "executes trades."
)


@st.cache_data(ttl=6 * 3600)
def load():
    a = {}
    a["signals"] = pd.read_parquet(ARTIFACT_DIR / "signals.parquet")
    a["buckets"] = pd.read_parquet(ARTIFACT_DIR / "bucket_stats.parquet")
    a["oof"] = pd.read_parquet(ARTIFACT_DIR / "oof.parquet")
    a["importance"] = pd.read_parquet(ARTIFACT_DIR / "importance.parquet")
    a["curves"] = pd.read_parquet(ARTIFACT_DIR / "equity_curves.parquet")
    a["port"] = pd.read_csv(ARTIFACT_DIR / "backtest_portfolio.csv")
    a["assets_bt"] = pd.read_csv(ARTIFACT_DIR / "backtest_assets.csv")
    a["regimes"] = pd.read_csv(ARTIFACT_DIR / "backtest_regimes.csv")
    a["threshold"] = pd.read_csv(ARTIFACT_DIR / "backtest_threshold.csv")
    return a


def bucket_for(p: float, buckets: pd.DataFrame) -> pd.Series | None:
    for _, b in buckets.iterrows():
        lo, hi = b["bucket"].strip("(]").split(",")
        if float(lo) < p <= float(hi):
            return b
    return None


def page_signals(a):
    st.subheader(f"Current signals — P(asset up over next {HORIZON_DAYS} days)")
    sig = a["signals"].copy()
    latest = pd.to_datetime(sig["date"].iloc[0])
    age = (pd.Timestamp.utcnow().tz_localize(None).normalize() - latest).days
    if age > 2:
        st.error(f"Data is {age} days old (latest bar {latest.date()}). "
                 "Run `python -m crypto.pipeline refresh` then retrain.")
    rows = []
    for _, s in sig.iterrows():
        b = bucket_for(s["p_cal"], a["buckets"])
        rows.append({
            "asset": s["asset"], "family": s["family"],
            "close": s["close"], "P(up,7d)": s["p_cal"],
            "OOS hit rate of this bucket":
                (f"{b['hit_rate']:.0%} (95% CI {b['ci_lo']:.0%}–"
                 f"{b['ci_hi']:.0%}, n={int(b['n'])})") if b is not None else "n/a",
            "history_days": int(s["history_days"]),
            "in training set": bool(s["trained_on_asset"])})
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", height=600,
                 column_config={"P(up,7d)": st.column_config.NumberColumn(
                     format="%.3f")})
    st.caption(
        f"Bar date {latest.date()} (UTC close). Long/flat strategy threshold "
        f"is {PROB_THRESHOLD}; signals in the 0.45–0.65 buckets have been "
        "statistical coin-flips out of sample — treat them as 'no signal'. "
        "Assets not in the training set (short history) are scored by the "
        "pooled model and are lower-trust.")
    fig = px.bar(sig.sort_values("p_cal"), x="p_cal", y="asset",
                 orientation="h", color="family",
                 labels={"p_cal": "calibrated P(up, 7d)"})
    fig.add_vline(x=0.5, line_dash="dot")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")


def page_backtest(a):
    st.subheader("Backtest — walk-forward, 15 bps/side, t+1-open execution")
    st.markdown("**Portfolio (equal-weight across assets with signals)** — "
                "in-sample (hyperparameter warmup folds) and out-of-sample "
                "are separate rows and must never be blended.")
    st.dataframe(a["port"].round(3), width="stretch")
    curves = a["curves"]
    for split in ("OOS", "IS"):
        c = curves[curves["split"] == split]
        daily = c.groupby("date")[["strat_ret", "bh_ret"]].mean()
        eq = (1 + daily).cumprod()
        fig = go.Figure()
        fig.add_scatter(x=eq.index, y=eq["strat_ret"], name="strategy")
        fig.add_scatter(x=eq.index, y=eq["bh_ret"], name="buy & hold")
        fig.update_layout(title=f"{split} equity (1 = start)", height=350)
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Per-fold regime breakdown** (regime labeled from BTC "
                "trend t-stat inside each window):")
    st.dataframe(a["regimes"].round(3), width="stretch")
    st.markdown("**Threshold sensitivity (OOS):**")
    st.dataframe(a["threshold"].round(3), width="stretch")
    st.markdown("**Per asset (OOS):**")
    oos = a["assets_bt"].query("split == 'OOS'").round(3)
    st.dataframe(oos, width="stretch", height=500)


def page_model(a):
    st.subheader("Model internals")
    imp = a["importance"].head(20)
    fig = px.bar(imp[::-1], x="perm_auc_drop", y="feature", orientation="h",
                 title="Permutation importance (OOS AUC drop, last fold)")
    fig.update_layout(height=550)
    st.plotly_chart(fig, width="stretch")
    oof = a["oof"]
    oos = oof[(oof["fold"] > WARMUP_FOLDS) & oof["p_cal"].notna()].copy()
    st.markdown("**Calibration (OOS):** predicted probability vs realized "
                "up-rate; the diagonal is perfect calibration.")
    oos["bin"] = pd.cut(oos["p_cal"], np.arange(0.2, 0.85, 0.05))
    cal = oos.groupby("bin", observed=True).agg(
        p=("p_cal", "mean"), real=("y", "mean"), n=("y", "size")).dropna()
    fig = go.Figure()
    fig.add_scatter(x=cal["p"], y=cal["real"], mode="markers+lines",
                    name="model", text=[f"n={n}" for n in cal["n"]])
    fig.add_scatter(x=[0.2, 0.8], y=[0.2, 0.8], mode="lines",
                    line_dash="dot", name="perfect")
    fig.update_layout(height=400, xaxis_title="predicted P(up)",
                      yaxis_title="realized up rate")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Model confidence and accuracy over time (OOS):**")
    oos["correct"] = (oos["p_cal"] > 0.5).astype(int) == oos["y"]
    daily = oos.groupby("date").agg(conf=("p_cal", "mean"),
                                    acc=("correct", "mean"))
    daily["acc_90d"] = daily["acc"].rolling(90).mean()
    fig = go.Figure()
    fig.add_scatter(x=daily.index, y=daily["conf"], name="mean P(up)")
    fig.add_scatter(x=daily.index, y=daily["acc_90d"],
                    name="90d rolling accuracy")
    fig.add_hline(y=0.5, line_dash="dot")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")


def page_data(a):
    st.subheader("Data health")
    from crypto.pipeline import status
    st.dataframe(status(), width="stretch", height=600)
    st.caption("macro 'gaps' are weekends/holidays (forward-filled at "
               "feature time); fear_greed gaps are missing days at the "
               "source. Refresh: `python -m crypto.pipeline refresh`, then "
               "`python -m crypto.features` and `python -m crypto.model "
               "train` and `python -m crypto.backtest`.")


st.title("Crypto market analysis — research dashboard")
st.warning(DISCLAIMER)
try:
    artifacts = load()
except FileNotFoundError as err:
    st.error(f"Artifacts missing ({err}). Run the pipeline first: refresh -> "
             "features -> model train -> backtest.")
    st.stop()

def page_recommendations():
    st.subheader("Recommendations — explicit, sized, with historical success rates")
    rec_path = ARTIFACT_DIR / "recommendations.parquet"
    if not rec_path.exists():
        st.info("Run `python -m crypto.recommend` first.")
        return
    rec = pd.read_parquet(rec_path)
    budget = st.number_input("Hypothetical crypto budget (USD)", 100.0,
                             10_000_000.0, 10_000.0, step=1000.0)
    rec["alloc_usd"] = (rec["weight"] * budget).round(0)
    st.markdown(
        f"**Stance counts:** LONG {int((rec['stance'] == 'LONG').sum())} · "
        f"AVOID/EXIT {int((rec['stance'] == 'AVOID/EXIT').sum())} · "
        f"NO SIGNAL {int((rec['stance'] == 'NO SIGNAL').sum())} — "
        f"cash/stables share of budget: "
        f"**{1 - rec['weight'].sum():.0%}**")
    show = rec[["asset", "family", "stance", "p_up_7d", "hist_success",
                "success_ci_lo", "success_ci_hi", "bucket_n", "weight",
                "alloc_usd", "why", "low_trust"]].copy()
    st.dataframe(show, width="stretch", height=650, column_config={
        "p_up_7d": st.column_config.NumberColumn("P(up,7d)", format="%.3f"),
        "hist_success": st.column_config.NumberColumn(
            "hist. success", format="percent"),
        "success_ci_lo": st.column_config.NumberColumn("CI lo", format="percent"),
        "success_ci_hi": st.column_config.NumberColumn("CI hi", format="percent"),
        "weight": st.column_config.NumberColumn(format="percent")})
    st.caption(
        "'hist. success' = out-of-sample hit rate of this signal's "
        "probability bucket during 2.4 years of walk-forward validation — a "
        "historical frequency with a 95% CI, never a promise. For AVOID "
        "rows, success means the asset indeed did not rise over the next 7 "
        "days. Sizing is an inverse-volatility conviction heuristic capped "
        "at 20%/asset, long/flat only — not an optimized portfolio.")


tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Recommendations", "Signals", "Backtest", "Model", "Data health"])
with tab0:
    page_recommendations()
with tab1:
    page_signals(artifacts)
with tab2:
    page_backtest(artifacts)
with tab3:
    page_model(artifacts)
with tab4:
    page_data(artifacts)
