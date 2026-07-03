"""Backtest engine tests: hand-computed P&L, cost application, causality."""
import numpy as np
import pandas as pd
import pytest

from crypto.backtest import (asset_backtest, block_bootstrap_sharpe_ci,
                             extract_trades, label_regime, max_drawdown,
                             metrics, sharpe)

DATES = pd.date_range("2024-01-01", periods=8, freq="D")


def test_hand_computed_pnl():
    """Signal fires at close of day 1 -> enter open day 2, P&L open2->open3,
    exit when signal drops at close of day 3 (flat from open day 4)."""
    opens = pd.Series([100, 100, 100, 110, 121, 121, 121, 121.0], index=DATES)
    prob = pd.Series([np.nan, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, np.nan],
                     index=DATES)
    cost = 0.0015
    bt = asset_backtest(opens, prob, threshold=0.55, cost=cost)
    # day 2 (2024-01-03): long, r_oo = 110/100-1 = 0.10, entry cost charged
    d3 = bt.loc["2024-01-03"]
    assert d3["pos"] == 1 and d3["r_oo"] == pytest.approx(0.10)
    assert d3["strat_ret"] == pytest.approx(0.10 - cost)
    # day 3 (2024-01-04): still long (signal from close day 3 exits NEXT day)
    d4 = bt.loc["2024-01-04"]
    assert d4["pos"] == 1 and d4["strat_ret"] == pytest.approx(0.10)
    # day 4 (2024-01-05): flat, exit cost charged
    d5 = bt.loc["2024-01-05"]
    assert d5["pos"] == 0 and d5["strat_ret"] == pytest.approx(-cost)
    # afterwards: flat, zero P&L
    assert (bt.loc["2024-01-06":, "strat_ret"] == 0).all()
    # trade extraction: one round trip, compounded net of both sides
    trades = extract_trades(bt, cost)
    assert len(trades) == 1
    assert trades[0] == pytest.approx((1 - cost) * 1.10 * 1.10 * (1 - cost) - 1)


def test_costs_only_on_state_change():
    opens = pd.Series(100.0, index=DATES)  # flat prices -> only costs matter
    prob = pd.Series([np.nan, 0.9, 0.9, 0.9, 0.1, 0.9, np.nan, np.nan],
                     index=DATES)
    cost = 0.0015
    bt = asset_backtest(opens, prob, threshold=0.55, cost=cost)
    # state changes: entry (day3), exit (day6), re-entry (day7) = 3 sides
    assert bt["turnover"].sum() == pytest.approx(3.0)
    assert bt["strat_ret"].sum() == pytest.approx(-3 * cost)


def test_no_lookahead_position_lags_signal():
    """A perfect-foresight signal must NOT capture same-day returns."""
    opens = pd.Series([100, 100, 200, 200, 200, 200, 200, 200.0], index=DATES)
    # signal spikes on the day BEFORE the jump is already priced at next open:
    # prob at close of day2 (index 1) -> position on day3 (index 2), which
    # earns open4/open3 - 1 = 0, NOT the 100% jump from open2->open3.
    prob = pd.Series([np.nan, np.nan, 0.9, np.nan, np.nan, np.nan, np.nan,
                      np.nan], index=DATES)
    bt = asset_backtest(opens, prob, threshold=0.55, cost=0.0)
    # the signal day itself is not evaluated (no lagged signal exists yet)
    assert pd.Timestamp("2024-01-03") not in bt.index
    assert bt.loc["2024-01-04", "pos"] == 1
    assert bt.loc["2024-01-04", "strat_ret"] == pytest.approx(0.0)
    assert bt["strat_ret"].sum() == pytest.approx(0.0)


def test_metrics_and_drawdown():
    r = pd.Series([0.10, -0.50, 0.20])
    assert max_drawdown(r) == pytest.approx(-0.50)
    assert np.isnan(sharpe(pd.Series([0.01])))
    bt = pd.DataFrame({"strat_ret": [0.01, 0.0, -0.02],
                       "r_oo": [0.01, 0.03, -0.02],
                       "pos": [1.0, 0.0, 1.0]})
    m = metrics(bt, cost=0.0)
    assert m["exposure"] == pytest.approx(2 / 3)
    assert m["n_trades"] == 2
    assert m["bh_ret"] == pytest.approx(1.01 * 1.03 * 0.98 - 1)


def test_block_bootstrap_ci_brackets_sharpe():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 500))
    lo, hi = block_bootstrap_sharpe_ci(r, n_boot=500)
    s = sharpe(r)
    assert lo < s < hi
    assert hi - lo > 0


def test_label_regime():
    up = pd.Series(np.exp(np.linspace(0, 1, 100) +
                          np.random.default_rng(1).normal(0, 0.01, 100)))
    flat = pd.Series(np.exp(np.random.default_rng(2).normal(0, 0.02, 100).cumsum() * 0.05))
    assert label_regime(up) == "trending-up"
    assert label_regime(1 / up) == "trending-down"
    assert label_regime(flat) == "choppy/ranging"
