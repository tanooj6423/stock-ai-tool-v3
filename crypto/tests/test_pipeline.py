"""Pipeline tests: parsing, aggregation, incremental merge, gap detection,
plus data-quality checks that run against real stored parquet when present.

All unit tests are network-free.
"""
import pandas as pd
import pytest

from crypto.config import RAW_DIR, SEED_BASES
from crypto.pipeline import (aggregate_funding_daily, merge_incremental,
                             parse_klines, report_gaps)

DAY_MS = 86_400_000


def _kline(open_ms: int, close: float) -> list:
    return [open_ms, "100.0", "110.0", "90.0", str(close), "1000.0",
            open_ms + DAY_MS - 1, "100000.0", 5000, "600.0", "60000.0", "0"]


class TestParseKlines:
    def test_parses_types_and_values(self):
        t0 = 1_600_000_000_000 - (1_600_000_000_000 % DAY_MS)
        df = parse_klines([_kline(t0, 105.0), _kline(t0 + DAY_MS, 106.0)],
                          now_ms=t0 + 2 * DAY_MS)
        assert list(df.columns) == ["date", "open", "high", "low", "close",
                                    "volume", "quote_volume", "trades",
                                    "taker_buy_base"]
        assert len(df) == 2
        assert df["close"].tolist() == [105.0, 106.0]
        assert df["date"].dt.tz is None  # tz-naive UTC convention
        assert (df["date"].dt.normalize() == df["date"]).all()

    def test_drops_still_open_bar(self):
        t0 = 1_600_000_000_000 - (1_600_000_000_000 % DAY_MS)
        # "now" is midway through the second bar -> it must be dropped
        df = parse_klines([_kline(t0, 105.0), _kline(t0 + DAY_MS, 106.0)],
                          now_ms=t0 + DAY_MS + 1000)
        assert len(df) == 1
        assert df["close"].iloc[0] == 105.0


class TestFundingAggregation:
    def test_daily_sum_mean_count(self):
        times = pd.to_datetime([
            "2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00",
            "2024-01-02 00:00",
        ])
        df = pd.DataFrame({"funding_time": times,
                           "funding_rate": [0.0001, 0.0002, 0.0003, -0.0001]})
        out = aggregate_funding_daily(df)
        assert len(out) == 2
        d1 = out[out["date"] == "2024-01-01"].iloc[0]
        assert d1["funding_sum"] == pytest.approx(0.0006)
        assert d1["funding_mean"] == pytest.approx(0.0002)
        assert d1["n_settlements"] == 3
        d2 = out[out["date"] == "2024-01-02"].iloc[0]
        assert d2["funding_sum"] == pytest.approx(-0.0001)
        assert d2["n_settlements"] == 1


class TestMergeIncremental:
    def test_dedup_newest_wins_and_sorted(self):
        old = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                            "close": [1.0, 2.0]})
        new = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                            "close": [2.5, 3.0]})  # 01-02 revised upstream
        out = merge_incremental(old, new)
        assert len(out) == 3
        assert out["date"].is_monotonic_increasing
        assert out["date"].is_unique
        assert out.loc[out["date"] == "2024-01-02", "close"].iloc[0] == 2.5

    def test_none_and_empty_old(self):
        new = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "x": [1]})
        assert len(merge_incremental(None, new)) == 1
        assert len(merge_incremental(new.iloc[:0], new)) == 1

    def test_idempotent(self):
        new = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                            "x": [1, 2]})
        once = merge_incremental(None, new)
        twice = merge_incremental(once, new)
        pd.testing.assert_frame_equal(once, twice)


class TestReportGaps:
    def test_finds_missing_days(self):
        df = pd.DataFrame({"date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-05"])})
        gaps = report_gaps(df)
        assert list(gaps) == list(pd.to_datetime(["2024-01-03", "2024-01-04"]))

    def test_no_gaps_and_empty(self):
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5)})
        assert len(report_gaps(df)) == 0
        assert len(report_gaps(df.iloc[:0])) == 0


# ---------------------------------------------------------------------------
# Data-quality checks on real stored data (skipped until first refresh runs)
# ---------------------------------------------------------------------------

def _stored(name):
    path = RAW_DIR / f"{name}.parquet"
    if not path.exists():
        pytest.skip(f"{name}.parquet not fetched yet")
    return pd.read_parquet(path)


@pytest.mark.parametrize("asset", SEED_BASES)
def test_stored_ohlcv_quality(asset):
    df = _stored(f"ohlcv_{asset}")
    assert df["date"].is_unique and df["date"].is_monotonic_increasing
    assert df["date"].dt.tz is None
    assert (df[["open", "high", "low", "close", "volume"]] >= 0).all().all()
    assert (df["high"] >= df["low"]).all()
    assert (df["taker_buy_base"] <= df["volume"] * 1.0001).all()
    assert len(report_gaps(df)) == 0  # Binance daily bars must be continuous
    # the still-open UTC day must never be stored
    assert df["date"].max() < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


@pytest.mark.parametrize("asset", SEED_BASES)
def test_stored_funding_quality(asset):
    df = _stored(f"funding_{asset}")
    assert df["date"].is_unique and df["date"].is_monotonic_increasing
    # Sanity bound only: real extremes exist (SOL hit -17%/day with hourly
    # settlements on 2022-11-10, the FTX collapse).
    assert df["funding_sum"].abs().max() < 0.5


def test_stored_onchain_quality():
    df = _stored("onchain_BTC")
    assert df["date"].is_unique and df["date"].is_monotonic_increasing
    assert df["AdrActCnt"].dropna().gt(0).all()


def test_stored_sentiment_quality():
    df = _stored("fear_greed")
    assert df["date"].is_unique and df["date"].is_monotonic_increasing
    assert df["fng_value"].between(0, 100).all()
