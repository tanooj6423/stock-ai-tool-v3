"""Model layer: pooled XGBoost + LightGBM ensemble, walk-forward validation.

Design (REQUIREMENTS §3, §10, §11):
  - Binary target: forward 7-day log return > 0.
  - Walk-forward expanding window, folds split globally by DATE across the
    pooled panel; between train end and test start there is a
    (purge + embargo) = 14-day gap so no training label window overlaps any
    test date and the shared market factor can't leak.
  - Hyperparameters: small grid scored on folds 1-2 only, then FROZEN.
  - Calibration is fold-causal: the isotonic calibrator applied to fold k is
    fit only on out-of-fold predictions from folds < k (folds 1-2 stay raw
    and are treated as warmup: they are excluded from headline OOS metrics).
  - Assets with < MIN_TRAIN_HISTORY_DAYS of history are never trained on
    (they are still scored at signal time, flagged).
  - Sample weights: sqrt-inverse of per-asset row count, so 2,300-day majors
    don't drown 400-day listings.

Artifacts -> crypto/artifacts/: oof.parquet (all OOF predictions),
fold_metrics.csv, importance.parquet, bucket_stats.parquet, signals.parquet,
model_xgb.joblib / model_lgbm.joblib / calibrator.joblib.

CLI: python -m crypto.model train
"""
from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd
# xgboost MUST be imported before lightgbm: with this conda env's bundled
# libomp copies, loading lightgbm's OpenMP runtime first makes any later
# XGBoost fit segfault (verified 2026-07-02). Keep this order.
from xgboost import XGBClassifier  # noqa: I001  isort: skip
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from crypto.config import (DATA_DIR, HORIZON_DAYS, MIN_TRAIN_HISTORY_DAYS,
                           PACKAGE_DIR)
from crypto.features import FEATURE_COLS

ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
PURGE = HORIZON_DAYS          # no train label window may touch test dates
EMBARGO = HORIZON_DAYS        # extra gap for shared-market-factor decay
N_FOLDS = 8
WARMUP_FOLDS = 2              # tuning + calibration warmup; not headline OOS
NUMERIC_FEATS = [c for c in FEATURE_COLS if c not in ("asset", "family")]

BASE_PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "subsample": 0.8,
               "colsample": 0.8}
GRID = [{"max_depth": d, "reg_lambda": l} for d in (3, 5) for l in (1.0, 10.0)]


# ---------------------------------------------------------------------------
# Panel preparation and fold construction
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(DATA_DIR / "features.parquet")
    panel["asset"] = panel["asset"].astype("category")
    panel["family"] = panel["family"].astype("category")
    return panel


def trainable_mask(panel: pd.DataFrame) -> pd.Series:
    """Rows usable for training: labeled + asset has enough history."""
    counts = panel.groupby("asset", observed=True)["date"].transform("size")
    return panel["y"].notna() & (counts >= MIN_TRAIN_HISTORY_DAYS)


def make_folds(dates: pd.DatetimeIndex, n_folds: int = N_FOLDS,
               initial_frac: float = 0.5, purge: int = PURGE,
               embargo: int = EMBARGO) -> list[dict]:
    """Expanding-window folds split on unique dates. Train covers everything
    up to (test_start - purge - embargo); test windows tile the rest."""
    udates = pd.DatetimeIndex(sorted(dates.unique()))
    start_i = int(len(udates) * initial_frac)
    test_dates = udates[start_i:]
    bounds = np.array_split(np.arange(len(test_dates)), n_folds)
    folds = []
    for k, idx in enumerate(bounds, start=1):
        ts, te = test_dates[idx[0]], test_dates[idx[-1]]
        folds.append({"fold": k,
                      "train_end": ts - pd.Timedelta(days=purge + embargo),
                      "test_start": ts, "test_end": te})
    return folds


def sample_weights(assets: pd.Series) -> np.ndarray:
    counts = assets.value_counts()
    w = assets.map(np.sqrt(1.0 / counts)).astype(float).to_numpy()
    return w * len(w) / w.sum()


def winsorize_fit(train: pd.DataFrame) -> dict:
    q = train[NUMERIC_FEATS].quantile([0.01, 0.99])
    return {"lo": q.loc[0.01], "hi": q.loc[0.99]}


def winsorize_apply(df: pd.DataFrame, clips: dict) -> pd.DataFrame:
    out = df.copy()
    out[NUMERIC_FEATS] = out[NUMERIC_FEATS].clip(clips["lo"], clips["hi"],
                                                 axis=1)
    return out


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def _make_models(params: dict) -> tuple:
    xgb = XGBClassifier(
        n_estimators=BASE_PARAMS["n_estimators"],
        learning_rate=BASE_PARAMS["learning_rate"],
        max_depth=params["max_depth"], reg_lambda=params["reg_lambda"],
        subsample=BASE_PARAMS["subsample"],
        colsample_bytree=BASE_PARAMS["colsample"],
        tree_method="hist", enable_categorical=True, eval_metric="logloss",
        n_jobs=-1, random_state=42, verbosity=0)
    lgbm = LGBMClassifier(
        n_estimators=BASE_PARAMS["n_estimators"],
        learning_rate=BASE_PARAMS["learning_rate"],
        max_depth=params["max_depth"], reg_lambda=params["reg_lambda"],
        num_leaves=2 ** params["max_depth"] - 1,
        subsample=BASE_PARAMS["subsample"], subsample_freq=1,
        colsample_bytree=BASE_PARAMS["colsample"],
        n_jobs=-1, random_state=42, verbosity=-1)
    return xgb, lgbm


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                params: dict) -> tuple[np.ndarray, tuple]:
    clips = winsorize_fit(train)
    tr = winsorize_apply(train, clips)
    te = winsorize_apply(test, clips)
    X_tr, y_tr = tr[FEATURE_COLS], tr["y"].astype(int)
    w = sample_weights(tr["asset"])
    xgb, lgbm = _make_models(params)
    xgb.fit(X_tr, y_tr, sample_weight=w)
    lgbm.fit(X_tr, y_tr, sample_weight=w)
    p = (xgb.predict_proba(te[FEATURE_COLS])[:, 1]
         + lgbm.predict_proba(te[FEATURE_COLS])[:, 1]) / 2
    return p, (xgb, lgbm, clips)


# ---------------------------------------------------------------------------
# Walk-forward training
# ---------------------------------------------------------------------------

def tune(panel: pd.DataFrame, folds: list[dict]) -> dict:
    """Pick grid params by mean AUC on the warmup folds only."""
    train_ok = trainable_mask(panel)
    scores = []
    for params in GRID:
        aucs = []
        for f in folds[:WARMUP_FOLDS]:
            tr = panel[train_ok & (panel["date"] <= f["train_end"])]
            te = panel[train_ok & panel["date"].between(f["test_start"],
                                                        f["test_end"])]
            p, _ = fit_predict(tr, te, params)
            aucs.append(roc_auc_score(te["y"].astype(int), p))
        scores.append((float(np.mean(aucs)), params))
        print(f"  grid {params} -> warmup AUC {np.mean(aucs):.4f}")
    best = max(scores, key=lambda s: s[0])
    print(f"frozen params: {best[1]} (warmup AUC {best[0]:.4f})")
    return best[1]


def walk_forward(panel: pd.DataFrame, params: dict,
                 folds: list[dict]) -> pd.DataFrame:
    """OOF predictions for every fold, with fold-causal calibration."""
    train_ok = trainable_mask(panel)
    oof_frames = []
    for f in folds:
        tr = panel[train_ok & (panel["date"] <= f["train_end"])]
        te = panel[train_ok & panel["date"].between(f["test_start"],
                                                    f["test_end"])]
        p_raw, _ = fit_predict(tr, te, params)
        prior = (pd.concat(oof_frames) if oof_frames else None)
        if prior is not None and f["fold"] > WARMUP_FOLDS:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01,
                                     y_max=0.99)
            iso.fit(prior["p_raw"], prior["y"])
            p_cal = iso.predict(p_raw)
        else:
            p_cal = np.full(len(p_raw), np.nan)
        oof_frames.append(pd.DataFrame({
            "date": te["date"].to_numpy(), "asset": te["asset"].to_numpy(),
            "family": te["family"].to_numpy(), "fold": f["fold"],
            "p_raw": p_raw, "p_cal": p_cal,
            "y": te["y"].astype(int).to_numpy(),
            "fwd_ret": te["fwd_ret"].to_numpy(),
            "close": te["close"].to_numpy(),
            "next_open": te["next_open"].to_numpy()}))
        print(f"  fold {f['fold']}: train<= {f['train_end'].date()} "
              f"({len(tr):,} rows) test {f['test_start'].date()}"
              f"->{f['test_end'].date()} ({len(te):,} rows)")
    return pd.concat(oof_frames, ignore_index=True)


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in oof.groupby("fold"):
        p = g["p_cal"].where(g["p_cal"].notna(), g["p_raw"])
        rows.append({
            "fold": k, "n": len(g), "base_rate": g["y"].mean(),
            "auc": roc_auc_score(g["y"], g["p_raw"]) if g["y"].nunique() > 1 else np.nan,
            "brier": brier_score_loss(g["y"], p),
            "logloss": log_loss(g["y"], p.clip(1e-6, 1 - 1e-6)),
            "acc@0.5": ((p > 0.5).astype(int) == g["y"]).mean(),
            "warmup": k <= WARMUP_FOLDS})
    # weekly non-overlapping subsample sanity check (label-overlap effect)
    oos = oof[oof["fold"] > WARMUP_FOLDS]
    weekly = oos[oos["date"].dt.dayofweek == 0]
    rows.append({
        "fold": "OOS(3+)", "n": len(oos), "base_rate": oos["y"].mean(),
        "auc": roc_auc_score(oos["y"], oos["p_raw"]),
        "brier": brier_score_loss(oos["y"], oos["p_cal"]),
        "logloss": log_loss(oos["y"], oos["p_cal"].clip(1e-6, 1 - 1e-6)),
        "acc@0.5": ((oos["p_cal"] > 0.5).astype(int) == oos["y"]).mean(),
        "warmup": False})
    rows.append({
        "fold": "OOS-weekly", "n": len(weekly), "base_rate": weekly["y"].mean(),
        "auc": roc_auc_score(weekly["y"], weekly["p_raw"]),
        "brier": brier_score_loss(weekly["y"], weekly["p_cal"]),
        "logloss": log_loss(weekly["y"], weekly["p_cal"].clip(1e-6, 1 - 1e-6)),
        "acc@0.5": ((weekly["p_cal"] > 0.5).astype(int) == weekly["y"]).mean(),
        "warmup": False})
    return pd.DataFrame(rows)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (center - half, center + half)


def bucket_stats(oof: pd.DataFrame) -> pd.DataFrame:
    """OOS hit-rate by calibrated-probability bucket with Wilson 95% CI.
    This is the uncertainty statement shown next to every signal."""
    oos = oof[(oof["fold"] > WARMUP_FOLDS) & oof["p_cal"].notna()].copy()
    edges = [0, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 1.0]
    oos["bucket"] = pd.cut(oos["p_cal"], edges)
    rows = []
    for b, g in oos.groupby("bucket", observed=True):
        pred_up = g["p_cal"] > 0.5
        hits = int((pred_up == g["y"].astype(bool)).sum())
        lo, hi = wilson_interval(hits, len(g))
        rows.append({"bucket": str(b), "n": len(g),
                     "hit_rate": hits / len(g) if len(g) else np.nan,
                     "ci_lo": lo, "ci_hi": hi,
                     "realized_up_rate": g["y"].mean(),
                     "mean_fwd_ret": g["fwd_ret"].mean()})
    return pd.DataFrame(rows)


def importance(panel: pd.DataFrame, params: dict,
               folds: list[dict]) -> pd.DataFrame:
    """Gain importance (last-fold model) + manual permutation importance on
    the LAST fold's test window (OOS only)."""
    rng = np.random.default_rng(42)
    train_ok = trainable_mask(panel)
    f = folds[-1]
    tr = panel[train_ok & (panel["date"] <= f["train_end"])]
    te = panel[train_ok & panel["date"].between(f["test_start"], f["test_end"])]
    p_base, (xgb, lgbm, clips) = fit_predict(tr, te, params)
    y = te["y"].astype(int).to_numpy()
    base_auc = roc_auc_score(y, p_base)
    te_w = winsorize_apply(te, clips)
    rows = []
    gain = pd.Series(lgbm.booster_.feature_importance("gain"),
                     index=FEATURE_COLS)
    for col in FEATURE_COLS:
        X = te_w[FEATURE_COLS].copy()
        shuffled = X[col].sample(frac=1.0, random_state=int(rng.integers(1e9)))
        X[col] = pd.Series(shuffled.to_numpy(), index=X.index,
                           dtype=X[col].dtype)
        p = (xgb.predict_proba(X)[:, 1] + lgbm.predict_proba(X)[:, 1]) / 2
        rows.append({"feature": col, "gain_lgbm": float(gain[col]),
                     "perm_auc_drop": base_auc - roc_auc_score(y, p)})
    return (pd.DataFrame(rows)
            .sort_values("perm_auc_drop", ascending=False)
            .reset_index(drop=True))


def final_signals(panel: pd.DataFrame, params: dict,
                  oof: pd.DataFrame) -> pd.DataFrame:
    """Train on all eligible data, calibrate on all non-warmup OOF, score the
    latest date for EVERY asset (short-history assets flagged)."""
    train_ok = trainable_mask(panel)
    tr = panel[train_ok]
    latest = panel[panel["date"] == panel["date"].max()]
    p_raw, (xgb, lgbm, clips) = fit_predict(tr, latest, params)
    oos = oof[oof["fold"] > WARMUP_FOLDS]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(oos["p_raw"], oos["y"])
    counts = panel.groupby("asset", observed=True)["date"].size()
    sig = pd.DataFrame({
        "date": latest["date"].to_numpy(), "asset": latest["asset"].to_numpy(),
        "family": latest["family"].to_numpy(),
        "close": latest["close"].to_numpy(),
        "p_raw": p_raw, "p_cal": iso.predict(p_raw),
        "history_days": latest["asset"].map(counts).to_numpy()})
    sig["trained_on_asset"] = sig["history_days"] >= MIN_TRAIN_HISTORY_DAYS
    joblib.dump(xgb, ARTIFACT_DIR / "model_xgb.joblib")
    joblib.dump(lgbm, ARTIFACT_DIR / "model_lgbm.joblib")
    joblib.dump({"iso": iso, "clips": clips, "params": params},
                ARTIFACT_DIR / "calibrator.joblib")
    return sig.sort_values("p_cal", ascending=False).reset_index(drop=True)


def train() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    panel = load_panel()
    eligible = panel[trainable_mask(panel)]
    folds = make_folds(pd.DatetimeIndex(eligible["date"]))
    print(f"panel {len(panel):,} rows | trainable {len(eligible):,} | "
          f"{N_FOLDS} folds (warmup {WARMUP_FOLDS})")
    params = tune(panel, folds)
    oof = walk_forward(panel, params, folds)
    oof.to_parquet(ARTIFACT_DIR / "oof.parquet", index=False)
    fm = fold_metrics(oof)
    fm.to_csv(ARTIFACT_DIR / "fold_metrics.csv", index=False)
    print(); print(fm.round(4).to_string(index=False))
    bs = bucket_stats(oof)
    bs.to_parquet(ARTIFACT_DIR / "bucket_stats.parquet", index=False)
    print(); print(bs.round(4).to_string(index=False))
    imp = importance(panel, params, folds)
    imp.to_parquet(ARTIFACT_DIR / "importance.parquet", index=False)
    print(); print("top 15 features (permutation AUC drop, OOS):")
    print(imp.head(15).round(5).to_string(index=False))
    sig = final_signals(panel, params, oof)
    sig.to_parquet(ARTIFACT_DIR / "signals.parquet", index=False)
    print(); print("current signals (top/bottom 5 by P(up,7d)):")
    print(pd.concat([sig.head(5), sig.tail(5)]).round(4).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "train":
        raise SystemExit("usage: python -m crypto.model train")
    train()
