"""Feature-engineering tests. The mutation test is the load-bearing one:
it proves no feature at date t depends on any raw data after t.
"""
import numpy as np
import pandas as pd
import pytest

from crypto.config import HORIZON_DAYS
from crypto.features import FEATURE_COLS, build_features

RNG = np.random.default_rng(7)
N_DAYS = 320


def _synthetic_ohlcv(n=N_DAYS, seed=0, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    op = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n))
    op[0] = close[0]
    hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.01, n)))
    vol = abs(rng.normal(1000, 200, n))
    return pd.DataFrame({
        "date": dates, "open": op, "high": hi, "low": lo, "close": close,
        "volume": vol, "quote_volume": vol * close,
        "trades": (vol * 10).astype("int64"), "taker_buy_base": vol * 0.5,
    })


def _synthetic_raw(n=N_DAYS):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    raw = {}
    for i, base in enumerate(["BTC", "ETH", "AAA", "MMM"]):
        o = _synthetic_ohlcv(n, seed=i)
        raw[f"ohlcv_{base}"] = o
        raw[f"funding_{base}"] = pd.DataFrame({
            "date": dates,
            "funding_sum": RNG.normal(3e-4, 2e-4, n),
            "funding_mean": RNG.normal(1e-4, 1e-4, n),
            "n_settlements": 3})
        raw[f"perp_{base}"] = pd.DataFrame({
            "date": dates,
            "perp_close": o["close"] * (1 + RNG.normal(0, 1e-3, n)),
            "perp_volume": o["volume"] * 2,
            "perp_quote_volume": o["quote_volume"] * 2,
            "perp_trades": o["trades"] * 2})
    for base in ["BTC", "ETH"]:
        raw[f"onchain_{base}"] = pd.DataFrame({
            "date": dates,
            "AdrActCnt": abs(RNG.normal(7e5, 1e5, n)),
            "TxTfrCnt": abs(RNG.normal(9e5, 1e5, n)),
            "FeeTotNtv": abs(RNG.normal(4, 1, n)),
            "CapMVRVCur": abs(RNG.normal(1.5, 0.2, n)),
            "FlowInExUSD": abs(RNG.normal(1.5e9, 3e8, n)),
            "FlowOutExUSD": abs(RNG.normal(1.5e9, 3e8, n)),
            "SplyExUSD": abs(RNG.normal(1.5e11, 1e10, n))})
    raw["stablecoins"] = pd.DataFrame({
        "date": dates, "usdt_supply": np.linspace(1e11, 2e11, n),
        "usdc_supply": np.linspace(5e10, 7e10, n)})
    raw["fear_greed"] = pd.DataFrame({
        "date": dates, "fng_value": RNG.uniform(5, 95, n).round(),
        "fng_class": "Neutral"})
    raw["macro"] = pd.DataFrame({
        "date": dates, "dxy": 100 + np.cumsum(RNG.normal(0, 0.3, n)),
        "spx": 4000 + np.cumsum(RNG.normal(0, 20, n)),
        "us10y": 4 + np.cumsum(RNG.normal(0, 0.02, n))})
    return raw


@pytest.fixture(scope="module")
def raw():
    return _synthetic_raw()


@pytest.fixture(scope="module")
def panel(raw):
    return build_features(raw)


NUMERIC_FEATS = [c for c in FEATURE_COLS if c not in ("asset", "family")]


def test_schema_and_shape(panel):
    assert set(FEATURE_COLS).issubset(panel.columns)
    assert panel["asset"].nunique() == 4
    assert not panel.duplicated(["date", "asset"]).any()


def test_no_lookahead_mutation(raw, panel):
    """Corrupt ALL raw data strictly after the cutoff; every feature value
    on or before the cutoff must be bit-identical."""
    cutoff = pd.Timestamp("2023-01-01") + pd.Timedelta(days=N_DAYS - 60)
    mutated = {}
    for k, df in raw.items():
        df = df.copy()
        after = df["date"] > cutoff
        for col in df.columns:
            if col == "date" or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            df.loc[after, col] = df.loc[after, col] * RNG.uniform(
                1.5, 3.0, int(after.sum()))
        mutated[k] = df
    p2 = build_features(mutated)
    key = ["date", "asset"]
    a = panel[panel["date"] <= cutoff].sort_values(key).reset_index(drop=True)
    b = p2[p2["date"] <= cutoff].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + NUMERIC_FEATS], b[key + NUMERIC_FEATS])


def test_target_uses_only_future(raw, panel):
    """fwd_ret at t must equal log(close[t+7]/close[t]) and y its sign."""
    btc = raw["ohlcv_BTC"].set_index("date")["close"]
    p = panel[panel["asset"] == "BTC"].set_index("date")
    t = btc.index[100]
    expected = np.log(btc.iloc[100 + HORIZON_DAYS] / btc.iloc[100])
    assert p.loc[t, "fwd_ret"] == pytest.approx(expected)
    assert p.loc[t, "y"] == float(expected > 0)
    # last HORIZON_DAYS rows have no label
    assert p["y"].tail(HORIZON_DAYS).isna().all()


def test_onchain_shift(raw, panel):
    """MVRV published for day t must appear in features at t+1."""
    oc = raw["onchain_BTC"].set_index("date")["CapMVRVCur"]
    p = panel[panel["asset"] == "BTC"].set_index("date")["mvrv"]
    t = oc.index[50]
    assert p.loc[t + pd.Timedelta(days=1)] == pytest.approx(oc.loc[t])
    assert not np.isclose(p.loc[t], oc.loc[t]) or oc.loc[t] == oc.iloc[49]


def test_sentiment_shift(raw, panel):
    fg = raw["fear_greed"].set_index("date")["fng_value"]
    p = panel[panel["asset"] == "BTC"].set_index("date")["fng"]
    t = fg.index[50]
    assert p.loc[t + pd.Timedelta(days=1)] == pytest.approx(fg.loc[t])


def test_next_open_is_next_bar(raw, panel):
    o = raw["ohlcv_BTC"]
    p = panel[panel["asset"] == "BTC"].reset_index(drop=True)
    assert p.loc[10, "next_open"] == pytest.approx(o.loc[11, "open"])


def test_cross_sectional_ranks_in_unit_interval(panel):
    for col in ["cs_rank_r7", "cs_rank_r30", "cs_rank_funding",
                "cs_rank_amihud", "donchian_55", "breadth_sma50"]:
        s = panel[col].dropna()
        assert len(s) > 0 and s.between(-1e-9, 1 + 1e-9).all(), col


def test_rs_btc_is_zero_for_btc(panel):
    s = panel.loc[panel["asset"] == "BTC", "rs_btc_7"].dropna()
    assert np.allclose(s, 0)


def test_no_infs(panel):
    num = panel[NUMERIC_FEATS].to_numpy(dtype=float)
    assert np.isfinite(num[~np.isnan(num)]).all()
