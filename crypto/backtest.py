"""Backtesting engine (REQUIREMENTS §4 + §10/§11 amendments).

Mechanics (fully causal):
  - Signal p(t) is computed at the close of day t (walk-forward OOF preds).
  - Position for day t+1 is set from p(t): long iff p(t) > threshold.
  - Execution at the OPEN of t+1; P&L accrues open(t+1) -> open(t+2).
  - Cost = (FEE_BPS + SLIPPAGE_BPS) per side, charged on every state change.
  - Long/flat only. No leverage. Equal-weight portfolio across assets that
    have a prediction that day.

Honesty rules baked in:
  - IS  = warmup folds 1-2 (hyperparameters were tuned on them), raw probs.
  - OOS = folds 3+, fold-causally calibrated probs.
    The two are computed and reported separately; there is deliberately no
    function that produces a blended number.
  - Portfolio Sharpe ships with a moving-block-bootstrap 95% CI (labels of a
    7d target overlap across days; naive t-stats would overstate certainty).
  - Regime labels are data-derived from BTC within each fold window and
    reported per fold, so bad regimes are visible, not averaged away.

Artifacts -> crypto/artifacts/: backtest_assets.csv, backtest_portfolio.csv,
backtest_regimes.csv, backtest_threshold.csv, equity_curves.parquet.

CLI: python -m crypto.backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.config import (FEE_BPS, PROB_THRESHOLD, RAW_DIR, SLIPPAGE_BPS,
                           asset_family)
from crypto.model import ARTIFACT_DIR, WARMUP_FOLDS, wilson_interval

COST_PER_SIDE = (FEE_BPS + SLIPPAGE_BPS) / 1e4
TRADING_DAYS = 365  # crypto trades every day


# ---------------------------------------------------------------------------
# Core mechanics
# ---------------------------------------------------------------------------

def asset_backtest(opens: pd.Series, prob: pd.Series,
                   threshold: float = PROB_THRESHOLD,
                   cost: float = COST_PER_SIDE) -> pd.DataFrame:
    """Daily strategy returns for one asset.

    `opens` is the full daily open series (indexed by date); `prob` holds the
    signal computed at each date's CLOSE. Position for day d is decided from
    prob(d-1) and P&L for day d is open(d) -> open(d+1).
    """
    df = pd.DataFrame({"open": opens})
    df["r_oo"] = df["open"].shift(-1) / df["open"] - 1
    df["prob"] = prob.reindex(df.index)
    df["pos"] = (df["prob"].shift(1) > threshold).astype(float)
    df["turnover"] = df["pos"].diff().abs().fillna(df["pos"])
    df["strat_ret"] = df["pos"] * df["r_oo"] - df["turnover"] * cost
    df = df[df["r_oo"].notna()]
    # restrict to the evaluation window: first to last date with a signal
    # (shifted by one day because execution is next-day)
    valid = df.index[df["prob"].shift(1).notna()]
    if len(valid) == 0:
        return df.iloc[0:0]
    return df.loc[valid.min():valid.max()]


def extract_trades(bt: pd.DataFrame, cost: float = COST_PER_SIDE) -> list[float]:
    """Net compounded return of each round-trip long trade."""
    trades, cur, in_pos = [], 1.0, False
    for _, row in bt.iterrows():
        if row["pos"] == 1:
            if not in_pos:
                in_pos, cur = True, 1.0 - cost
            cur *= 1 + row["r_oo"]
        elif in_pos:
            trades.append(cur * (1 - cost) - 1)
            in_pos = False
    if in_pos:
        trades.append(cur * (1 - cost) - 1)
    return trades


def sharpe(returns: pd.Series) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    eq = (1 + returns).cumprod()
    return float((eq / eq.cummax() - 1).min())


def block_bootstrap_sharpe_ci(returns: pd.Series, block: int = 10,
                              n_boot: int = 2000, seed: int = 42,
                              alpha: float = 0.05) -> tuple[float, float]:
    """Moving-block bootstrap CI for the annualized Sharpe (handles the
    serial correlation induced by overlapping 7d labels)."""
    r = returns.to_numpy()
    n = len(r)
    if n < 2 * block:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    sh = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.concatenate([r[s:s + block] for s in starts[i]])[:n]
        sd = sample.std()
        sh[i] = sample.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    lo, hi = np.nanpercentile(sh, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def metrics(bt: pd.DataFrame, cost: float = COST_PER_SIDE) -> dict:
    r, n = bt["strat_ret"], len(bt)
    trades = extract_trades(bt, cost)
    years = n / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    bh = float((1 + bt["r_oo"]).prod() - 1)
    wins = sum(t > 0 for t in trades)
    ci = wilson_interval(wins, len(trades)) if trades else (np.nan, np.nan)
    return {
        "days": n, "sharpe": sharpe(r), "max_dd": max_drawdown(r),
        "total_ret": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan,
        "exposure": float(bt["pos"].mean()), "n_trades": len(trades),
        "win_rate": wins / len(trades) if trades else np.nan,
        "win_ci_lo": ci[0], "win_ci_hi": ci[1],
        "bh_ret": bh, "bh_sharpe": sharpe(bt["r_oo"]),
        "bh_max_dd": max_drawdown(bt["r_oo"]),
    }


# ---------------------------------------------------------------------------
# Orchestration over the OOF predictions
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, dict]:
    oof = pd.read_parquet(ARTIFACT_DIR / "oof.parquet")
    opens = {}
    for asset in oof["asset"].unique():
        o = pd.read_parquet(RAW_DIR / f"ohlcv_{asset}.parquet")
        opens[asset] = o.set_index("date")["open"]
    return oof, opens


def _prob_frame(oof: pd.DataFrame, split: str) -> pd.DataFrame:
    """IS: warmup folds, raw probs. OOS: later folds, calibrated probs."""
    if split == "IS":
        g = oof[oof["fold"] <= WARMUP_FOLDS].copy()
        g["p"] = g["p_raw"]
    else:
        g = oof[oof["fold"] > WARMUP_FOLDS].copy()
        g["p"] = g["p_cal"]
    return g


def run_split(oof: pd.DataFrame, opens: dict, split: str,
              threshold: float = PROB_THRESHOLD
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-asset metrics, the daily returns matrix, and per-asset curves."""
    g = _prob_frame(oof, split)
    rows, ret_cols, curve_frames = [], {}, []
    for asset, ga in g.groupby("asset", observed=True):
        prob = ga.set_index("date")["p"]
        bt = asset_backtest(opens[asset], prob, threshold)
        if bt.empty:
            continue
        m = metrics(bt)
        rows.append({"asset": asset, "family": asset_family(asset),
                     "split": split, **m})
        ret_cols[asset] = bt["strat_ret"]
        curve_frames.append(pd.DataFrame({
            "date": bt.index, "asset": asset, "split": split,
            "strat_ret": bt["strat_ret"].to_numpy(),
            "bh_ret": bt["r_oo"].to_numpy()}))
    assets_df = (pd.DataFrame(rows)
                 .sort_values("sharpe", ascending=False)
                 .reset_index(drop=True))
    ret_mat = pd.DataFrame(ret_cols).sort_index()
    curves = pd.concat(curve_frames, ignore_index=True)
    return assets_df, ret_mat, curves


def portfolio_metrics(ret_mat: pd.DataFrame, oof: pd.DataFrame, opens: dict,
                      split: str) -> dict:
    port = ret_mat.mean(axis=1)  # equal-weight over assets active that day
    g = _prob_frame(oof, split)
    bh = pd.DataFrame({
        a: asset_backtest(opens[a], ga.set_index("date")["p"], -np.inf)["r_oo"]
        for a, ga in g.groupby("asset", observed=True)}).sort_index().mean(axis=1)
    lo, hi = block_bootstrap_sharpe_ci(port)
    eq = (1 + port).cumprod()
    return {"split": split, "days": len(port), "sharpe": sharpe(port),
            "sharpe_ci_lo": lo, "sharpe_ci_hi": hi,
            "max_dd": max_drawdown(port),
            "total_ret": float(eq.iloc[-1] - 1),
            "bh_sharpe": sharpe(bh), "bh_total_ret": float((1 + bh).prod() - 1),
            "bh_max_dd": max_drawdown(bh)}


# ---------------------------------------------------------------------------
# Regime analysis and threshold sensitivity
# ---------------------------------------------------------------------------

def label_regime(btc_close: pd.Series) -> str:
    """Trend t-stat within the window: |sum(r)| / (std(r) * sqrt(n))."""
    r = np.log(btc_close).diff().dropna()
    if len(r) < 10 or r.std() == 0:
        return "n/a"
    t = r.sum() / (r.std() * np.sqrt(len(r)))
    if t > 1.0:
        return "trending-up"
    if t < -1.0:
        return "trending-down"
    return "choppy/ranging"


def regime_table(oof: pd.DataFrame, ret_mat_by_split: dict,
                 opens: dict) -> pd.DataFrame:
    rows = []
    for fold, gf in oof.groupby("fold"):
        split = "IS" if fold <= WARMUP_FOLDS else "OOS"
        start, end = gf["date"].min(), gf["date"].max()
        btc = opens["BTC"].loc[start:end]
        port = ret_mat_by_split[split].loc[start:end].mean(axis=1)
        btc_ret = float(btc.iloc[-1] / btc.iloc[0] - 1)
        rows.append({"fold": fold, "split": split,
                     "start": start.date(), "end": end.date(),
                     "regime": label_regime(btc), "btc_ret": btc_ret,
                     "sharpe": sharpe(port), "max_dd": max_drawdown(port),
                     "total_ret": float((1 + port).prod() - 1),
                     "exposure": float((ret_mat_by_split[split]
                                        .loc[start:end] != 0).mean().mean())})
    return pd.DataFrame(rows)


def threshold_sensitivity(oof: pd.DataFrame, opens: dict) -> pd.DataFrame:
    rows = []
    for thr in (0.50, 0.55, 0.60, 0.65):
        _, ret_mat, _ = run_split(oof, opens, "OOS", threshold=thr)
        port = ret_mat.mean(axis=1)
        lo, hi = block_bootstrap_sharpe_ci(port)
        rows.append({"threshold": thr, "sharpe": sharpe(port),
                     "ci_lo": lo, "ci_hi": hi,
                     "total_ret": float((1 + port).prod() - 1),
                     "max_dd": max_drawdown(port),
                     "exposure": float((ret_mat > 0).mean().mean())})
    return pd.DataFrame(rows)


def main() -> None:
    oof, opens = load_inputs()
    out = {}
    all_assets, all_curves, ret_mats = [], [], {}
    for split in ("IS", "OOS"):
        assets_df, ret_mat, curves = run_split(oof, opens, split)
        all_assets.append(assets_df)
        all_curves.append(curves)
        ret_mats[split] = ret_mat
        out[split] = portfolio_metrics(ret_mat, oof, opens, split)
    assets = pd.concat(all_assets, ignore_index=True)
    port = pd.DataFrame([out["IS"], out["OOS"]])
    regimes = regime_table(oof, ret_mats, opens)
    thr = threshold_sensitivity(oof, opens)
    assets.to_csv(ARTIFACT_DIR / "backtest_assets.csv", index=False)
    port.to_csv(ARTIFACT_DIR / "backtest_portfolio.csv", index=False)
    regimes.to_csv(ARTIFACT_DIR / "backtest_regimes.csv", index=False)
    thr.to_csv(ARTIFACT_DIR / "backtest_threshold.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_parquet(
        ARTIFACT_DIR / "equity_curves.parquet", index=False)

    pd.set_option("display.width", 220)
    print("=== PORTFOLIO (equal-weight, long/flat, threshold "
          f"{PROB_THRESHOLD}, {FEE_BPS + SLIPPAGE_BPS:.0f}bps/side) ===")
    print(port.round(3).to_string(index=False))
    print("\n=== PER-FOLD REGIME BREAKDOWN ===")
    print(regimes.round(3).to_string(index=False))
    print("\n=== THRESHOLD SENSITIVITY (OOS) ===")
    print(thr.round(3).to_string(index=False))
    print("\n=== PER-FAMILY (OOS) ===")
    fam = (assets[assets["split"] == "OOS"]
           .groupby("family")[["sharpe", "total_ret", "max_dd", "exposure",
                               "bh_ret"]].mean().round(3))
    print(fam.to_string())
    print("\n=== TOP/BOTTOM ASSETS BY OOS SHARPE ===")
    oos_a = assets[assets["split"] == "OOS"]
    cols = ["asset", "family", "sharpe", "total_ret", "max_dd", "exposure",
            "n_trades", "win_rate", "bh_ret"]
    print(pd.concat([oos_a.head(6), oos_a.tail(4)])[cols]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
