"""Recommendation layer: turn calibrated signals into explicit, honest,
actionable guidance — stance, reasoning, sizing, historical success rate.

Stances (bucket-driven; skill lives in the tails, so the coin-flip middle is
explicitly NO SIGNAL):
  LONG        p_cal >= 0.55  — model expects up-move; historically the >0.65
                               bucket is the only long bucket with real skill,
                               so conviction matters.
  AVOID/EXIT  p_cal <= 0.45  — model expects weakness; the 0.35-0.45 bucket
                               hit 61% OOS. "Success" = staying out of a
                               falling asset.
  NO SIGNAL   otherwise      — 0.45-0.65 buckets were OOS coin-flips.

Sizing (heuristic, stated openly, long/flat only, no leverage):
  weight_i ∝ (p_cal_i − 0.5) / realized_vol30_i over LONG assets,
  each capped at 20% of the crypto risk budget; remainder = cash/stables.
  This is inverse-volatility scaled conviction, NOT an optimized portfolio.

Success % = the OOS hit rate (with Wilson 95% CI) of the signal's
probability bucket from walk-forward validation — a historical frequency,
never a promise.

CLI: python -m crypto.recommend [budget_usd]
Artifacts: crypto/artifacts/recommendations.parquet
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from crypto.config import DATA_DIR, MIN_TRAIN_HISTORY_DAYS
from crypto.features import FEATURE_COLS
from crypto.model import ARTIFACT_DIR, winsorize_apply

MAX_WEIGHT = 0.20
LONG_THR, AVOID_THR = 0.55, 0.45

# terse human labels for the most decision-relevant features
NICE = {
    "stable_g7": "stablecoin supply 7d growth", "stable_g30": "stablecoin supply 30d growth",
    "fng": "Fear&Greed level", "fng_z90": "Fear&Greed z-score", "fng_ch7": "Fear&Greed 7d change",
    "dxy_r7": "dollar index 7d move", "spx_r7": "S&P500 7d move", "us10y_ch7": "US 10y yield 7d change",
    "breadth_sma50": "market breadth (>50d SMA)", "altseason": "alt-season gauge",
    "mkt_disp7": "cross-market dispersion", "mkt_ret7_med": "median market 7d return",
    "mkt_btc_mvrv": "BTC MVRV (cycle valuation)", "mkt_btc_netflow_z": "BTC exchange netflow z",
    "mkt_btc_adract_z": "BTC active addresses z", "rsi_14": "RSI(14)",
    "macd_hist": "MACD histogram", "dd_90": "drawdown from 90d high",
    "dist_sma50": "distance from 50d SMA", "dist_sma200": "distance from 200d SMA",
    "funding_z90": "funding rate z-score", "funding_7m": "funding rate 7d mean",
    "funding_cum30": "30d cumulative funding", "fund_trend_int": "funding-vs-trend crowding",
    "basis": "perp-spot basis", "basis_z90": "perp basis z-score",
    "cs_rank_r7": "7d momentum rank in universe", "cs_rank_r30": "30d momentum rank",
    "cs_rank_chg7": "momentum-rank rotation", "vol_ratio": "vol regime (7d/30d)",
    "taker_ratio_7": "taker buy pressure", "mvrv": "MVRV (cycle valuation)",
    "exnetflow_z90": "exchange netflow z", "ret_7": "7d return", "ret_30": "30d return",
    "beta_btc_60": "beta to BTC", "vol_30": "30d volatility", "vol_7": "7d volatility",
    "listing_age": "listing age", "dist_ath": "distance from ATH",
    "amihud_30": "illiquidity", "log_qvol30": "liquidity (volume)",
}


def _bucket_row(p: float, buckets: pd.DataFrame) -> dict:
    for _, b in buckets.iterrows():
        lo, hi = (float(x) for x in b["bucket"].strip("(]").split(","))
        if lo < p <= hi:
            return b.to_dict()
    return {}


def _stance(p: float) -> str:
    if p >= LONG_THR:
        return "LONG"
    if p <= AVOID_THR:
        return "AVOID/EXIT"
    return "NO SIGNAL"


def top_drivers(lgbm, X_row: pd.DataFrame, k: int = 4) -> str:
    """Signed top feature contributions (LightGBM pred_contrib, log-odds)."""
    contrib = lgbm.booster_.predict(X_row, pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(contrib))
    parts = []
    for i in order:
        col = FEATURE_COLS[i]
        if col in ("asset", "family") or contrib[i] == 0:
            continue
        name = NICE.get(col, col)
        parts.append(f"{name} ({'+' if contrib[i] > 0 else '−'})")
        if len(parts) == k:
            break
    return "; ".join(parts)


def build_recommendations(budget: float = 10_000.0) -> pd.DataFrame:
    sig = pd.read_parquet(ARTIFACT_DIR / "signals.parquet")
    buckets = pd.read_parquet(ARTIFACT_DIR / "bucket_stats.parquet")
    panel = pd.read_parquet(DATA_DIR / "features.parquet")
    panel["asset"] = panel["asset"].astype("category")
    panel["family"] = panel["family"].astype("category")
    latest = panel[panel["date"] == panel["date"].max()].set_index("asset")
    lgbm = joblib.load(ARTIFACT_DIR / "model_lgbm.joblib")
    cal = joblib.load(ARTIFACT_DIR / "calibrator.joblib")

    rows = []
    for _, s in sig.iterrows():
        asset = s["asset"]
        b = _bucket_row(s["p_cal"], buckets)
        row_feats = latest.loc[[asset]].reset_index()
        X = winsorize_apply(row_feats, cal["clips"])[FEATURE_COLS]
        stance = _stance(s["p_cal"])
        rows.append({
            "asset": asset, "family": s["family"], "close": s["close"],
            "stance": stance, "p_up_7d": s["p_cal"],
            "hist_success": b.get("hit_rate", np.nan),
            "success_ci_lo": b.get("ci_lo", np.nan),
            "success_ci_hi": b.get("ci_hi", np.nan),
            "bucket_n": b.get("n", np.nan),
            "bucket_mean_7d_ret": b.get("mean_fwd_ret", np.nan),
            "vol30": latest.loc[asset, "vol_30"],
            "why": top_drivers(lgbm, X),
            "low_trust": not bool(s["trained_on_asset"]),
        })
    rec = pd.DataFrame(rows).sort_values("p_up_7d", ascending=False)

    longs = rec[rec["stance"] == "LONG"].copy()
    rec["weight"], rec["alloc_usd"] = 0.0, 0.0
    if not longs.empty:
        raw = (longs["p_up_7d"] - 0.5) / (longs["vol30"] + 1e-9)
        w = (raw / raw.sum()).clip(upper=MAX_WEIGHT)
        rec.loc[w.index, "weight"] = w
        rec.loc[w.index, "alloc_usd"] = (w * budget).round(0)
    rec["cash_note"] = ""
    cash_w = 1 - rec["weight"].sum()
    rec.attrs["cash_weight"] = float(cash_w)
    rec.attrs["budget"] = budget
    return rec.reset_index(drop=True)


def render_text(rec: pd.DataFrame) -> str:
    budget = rec.attrs.get("budget", 10_000)
    cash_w = rec.attrs.get("cash_weight", 1.0)
    lines = [
        f"RECOMMENDATIONS — {pd.Timestamp.utcnow().date()} "
        f"(hypothetical crypto budget ${budget:,.0f}; long/flat; heuristic "
        "sizing; historical frequencies, not promises)", ""]
    for stance in ("LONG", "AVOID/EXIT", "NO SIGNAL"):
        g = rec[rec["stance"] == stance]
        if g.empty:
            continue
        lines.append(f"--- {stance} ({len(g)}) ---")
        for _, r in g.iterrows():
            succ = (f"{r['hist_success']:.0%} (CI {r['success_ci_lo']:.0%}–"
                    f"{r['success_ci_hi']:.0%}, n={int(r['bucket_n'])})"
                    if pd.notna(r["hist_success"]) else "n/a")
            alloc = (f" | size ${r['alloc_usd']:,.0f} ({r['weight']:.0%})"
                     if r["weight"] > 0 else "")
            trust = " | LOW TRUST (short history)" if r["low_trust"] else ""
            if stance == "LONG":
                verb = "hist. success of similar signals"
            elif stance == "AVOID/EXIT":
                verb = "hist. success of avoiding (signal correct)"
            else:
                verb = "bucket was a coin-flip OOS"
            lines.append(
                f"{r['asset']:>6} P(up,7d)={r['p_up_7d']:.2f} | {verb}: {succ}"
                f"{alloc}{trust}\n        why: {r['why']}")
        lines.append("")
    lines.append(f"Cash/stables: {cash_w:.0%} of budget (no LONG signals -> "
                 "capital preservation is the position).")
    return "\n".join(lines)


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000.0
    rec = build_recommendations(budget)
    rec.to_parquet(ARTIFACT_DIR / "recommendations.parquet", index=False)
    print(render_text(rec))


if __name__ == "__main__":
    main()
