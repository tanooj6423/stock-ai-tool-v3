"""Model-layer tests: fold construction (purge/embargo), weights, Wilson CI,
and a miniature end-to-end walk-forward on synthetic data checking
calibration causality."""
import numpy as np
import pandas as pd
import pytest

import crypto.model as M
from crypto.config import HORIZON_DAYS
from crypto.features import build_features
from crypto.tests.test_features import _synthetic_raw


def test_make_folds_purge_and_embargo():
    dates = pd.DatetimeIndex(pd.date_range("2021-01-01", periods=1000))
    folds = M.make_folds(dates, n_folds=6)
    assert len(folds) == 6
    prev_end = None
    for f in folds:
        gap = (f["test_start"] - f["train_end"]).days
        assert gap == M.PURGE + M.EMBARGO
        # no train label window (train_end, train_end+7] touches test dates
        assert f["train_end"] + pd.Timedelta(days=HORIZON_DAYS) < f["test_start"]
        assert f["test_end"] >= f["test_start"]
        if prev_end is not None:  # expanding, contiguous test tiling
            assert f["test_start"] > prev_end
        prev_end = f["test_end"]
    # expanding window: train_end strictly increases
    ends = [f["train_end"] for f in folds]
    assert all(b > a for a, b in zip(ends, ends[1:]))


def test_sample_weights_dampen_long_histories():
    assets = pd.Series(["BTC"] * 900 + ["NEW"] * 100)
    w = M.sample_weights(assets)
    assert w.mean() == pytest.approx(1.0)
    w_btc, w_new = w[0], w[-1]
    assert w_new > w_btc                       # short history upweighted
    assert w_new / w_btc == pytest.approx(3.0)  # sqrt(900/100)


def test_wilson_interval():
    lo, hi = M.wilson_interval(55, 100)
    assert 0.44 < lo < 0.55 < hi < 0.65
    assert M.wilson_interval(0, 0) == (pytest.approx(np.nan, nan_ok=True),) * 2


def test_winsorize_clips_to_train_quantiles():
    n = 500
    tr = pd.DataFrame({c: np.random.default_rng(1).normal(size=n)
                       for c in M.NUMERIC_FEATS})
    te = tr.copy()
    te.iloc[0, 0] = 1e9
    clips = M.winsorize_fit(tr)
    out = M.winsorize_apply(te, clips)
    assert out.iloc[0, 0] == pytest.approx(clips["hi"].iloc[0])


@pytest.fixture(scope="module")
def mini_setup(monkeypatch_module=None):
    panel = build_features(_synthetic_raw())
    panel["asset"] = panel["asset"].astype("category")
    panel["family"] = panel["family"].astype("category")
    return panel


def test_mini_walk_forward_calibration_causality(mini_setup, monkeypatch):
    panel = mini_setup
    monkeypatch.setattr(M, "MIN_TRAIN_HISTORY_DAYS", 100)
    monkeypatch.setitem(M.BASE_PARAMS, "n_estimators", 20)
    folds = M.make_folds(
        pd.DatetimeIndex(panel.loc[panel["y"].notna(), "date"]), n_folds=4)
    oof = M.walk_forward(panel, {"max_depth": 3, "reg_lambda": 1.0}, folds)
    assert set(oof["fold"]) == {1, 2, 3, 4}
    # warmup folds carry no calibrated probability; later folds do
    assert oof.loc[oof["fold"] <= M.WARMUP_FOLDS, "p_cal"].isna().all()
    assert oof.loc[oof["fold"] > M.WARMUP_FOLDS, "p_cal"].notna().all()
    assert oof["p_raw"].between(0, 1).all()
    # every OOF row lies inside its fold's test window (never in train)
    for f in folds:
        g = oof[oof["fold"] == f["fold"]]
        assert (g["date"] >= f["test_start"]).all()
        assert (g["date"] <= f["test_end"]).all()


def test_trainable_mask_excludes_short_history(mini_setup, monkeypatch):
    panel = mini_setup
    monkeypatch.setattr(M, "MIN_TRAIN_HISTORY_DAYS", 10_000)
    assert M.trainable_mask(panel).sum() == 0
    monkeypatch.setattr(M, "MIN_TRAIN_HISTORY_DAYS", 100)
    m = M.trainable_mask(panel)
    assert m.sum() > 0
    assert panel.loc[m, "y"].notna().all()
