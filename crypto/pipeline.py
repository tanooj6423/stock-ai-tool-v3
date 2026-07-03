"""Data pipeline: fetch, store and incrementally refresh all raw sources.

Sources (all free, keyless):
  - Binance spot klines            -> raw/ohlcv_{ASSET}.parquet
  - Binance futures funding rates  -> raw/funding_{ASSET}.parquet
  - Binance futures open interest  -> raw/oi_{ASSET}.parquet   (accumulates;
      Binance only retains ~30d of history, so OI is NOT a v1 model feature)
  - Coin Metrics Community on-chain -> raw/onchain_{ASSET}.parquet
  - Coin Metrics stablecoin supply  -> raw/stablecoins.parquet
  - Alternative.me Fear & Greed     -> raw/fear_greed.parquet
  - yfinance macro closes           -> raw/macro.parquet

Conventions: every table has a tz-naive `date` column meaning the UTC
calendar day the row describes; rows are unique and sorted by date; the
still-in-progress current UTC day is never stored. Publication lags
(on-chain, sentiment) are handled downstream in features.py, not here.

CLI:  python -m crypto.pipeline refresh | status
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import requests

from crypto.config import (BINANCE_FUTURES_HOST, BINANCE_SPOT_HOSTS,
                           CM_METRICS, CM_ONCHAIN, CM_STABLE_METRIC,
                           CM_STABLECOINS, COINMETRICS_HOST, EXCLUDED_BASES,
                           FEAR_GREED_URL, MACRO_TICKERS, RAW_DIR,
                           REFRESH_OVERLAP_DAYS, SEED_BASES, START_DATE,
                           UNIVERSE_SIZE)

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
OHLCV_SCHEMA = ["date", "open", "high", "low", "close", "volume",
                "quote_volume", "trades", "taker_buy_base"]


def _utc_today() -> pd.Timestamp:
    """Current UTC calendar day, tz-naive (matches the `date` column dtype)."""
    return pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)


def _get(url: str, params: dict | None = None, retries: int = 3,
         timeout: int = 30):
    last_err: Exception = RuntimeError("unreachable")
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:  # rate limited: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as err:
            last_err = err
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status in (400, 451):  # bad request / geo-block: no point retrying
                raise
            time.sleep(2 ** attempt)
    raise last_err


# ---------------------------------------------------------------------------
# Universe selection (dynamic: top-N liquid USDT spot pairs + seed majors)
# ---------------------------------------------------------------------------

def select_universe(tickers: list, size: int = UNIVERSE_SIZE) -> pd.DataFrame:
    """Rank USDT spot pairs by 24h quote volume; exclude stables/wrapped/
    leveraged; always include the seed majors.

    KNOWN LIMITATION (documented in REQUIREMENTS §10): membership is a
    present-day snapshot, so alt/memecoin backtest results carry
    survivorship-by-liquidity bias. Mitigations downstream: per-family
    result reporting + training-entry rules.
    """
    rows = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in EXCLUDED_BASES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            continue
        rows.append({"base": base, "symbol": sym,
                     "quote_volume_24h": float(t["quoteVolume"])})
    df = (pd.DataFrame(rows)
            .sort_values("quote_volume_24h", ascending=False)
            .reset_index(drop=True))
    top = df.head(size)
    seeds = df[df["base"].isin(SEED_BASES)]
    out = (pd.concat([top, seeds]).drop_duplicates("base")
             .sort_values("quote_volume_24h", ascending=False)
             .reset_index(drop=True))
    out["rank"] = range(1, len(out) + 1)
    out["snapshot_date"] = _utc_today()
    return out


def fetch_universe() -> pd.DataFrame:
    last_err: Exception | None = None
    for host in BINANCE_SPOT_HOSTS:
        try:
            return select_universe(_get(f"{host}/api/v3/ticker/24hr"))
        except requests.RequestException as err:
            last_err = err
    raise last_err


def get_universe() -> pd.DataFrame:
    """Stored universe snapshot; fetched (and persisted) if absent."""
    path = RAW_DIR / "universe.parquet"
    if path.exists():
        return pd.read_parquet(path)
    uni = fetch_universe()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    uni.to_parquet(path, index=False)
    return uni


# ---------------------------------------------------------------------------
# Binance spot OHLCV
# ---------------------------------------------------------------------------

def parse_klines(rows: list, now_ms: int) -> pd.DataFrame:
    """Parse raw kline rows; drops the still-open bar (close_time >= now)."""
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df = df[df["close_time"].astype("int64") < now_ms]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["open_time"].astype("int64"), unit="ms"),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "quote_volume": df["quote_volume"].astype(float),
        "trades": df["trades"].astype("int64"),
        "taker_buy_base": df["taker_buy_base"].astype(float),
    })
    return out.reset_index(drop=True)


def fetch_binance_klines(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    rows: list = []
    last_err: Exception | None = None
    for host in BINANCE_SPOT_HOSTS:
        try:
            cursor = start_ms
            rows = []
            while True:
                batch = _get(f"{host}/api/v3/klines",
                             {"symbol": symbol, "interval": "1d",
                              "startTime": cursor, "limit": 1000})
                rows.extend(batch)
                if len(batch) < 1000:
                    break
                cursor = int(batch[-1][0]) + 1
                time.sleep(0.15)
            last_err = None
            break
        except requests.RequestException as err:
            last_err = err
    if last_err is not None:
        raise last_err
    return parse_klines(rows, now_ms)


# ---------------------------------------------------------------------------
# Binance futures: funding rates (full history) and open interest (~30d only)
# ---------------------------------------------------------------------------

def aggregate_funding_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 8-hourly funding settlements to one row per UTC day."""
    day = df["funding_time"].dt.normalize()
    g = df.groupby(day)["funding_rate"]
    out = pd.DataFrame({
        "date": g.sum().index,
        "funding_sum": g.sum().to_numpy(),
        "funding_mean": g.mean().to_numpy(),
        "n_settlements": g.count().to_numpy(),
    })
    return out.reset_index(drop=True)


def _perp_candidates(symbol: str) -> list[str]:
    """Perp contract symbols to try: sub-cent coins trade as 1000-prefixed
    contracts on Binance futures (e.g. spot PEPEUSDT -> perp 1000PEPEUSDT)."""
    return [symbol, f"1000{symbol}"]


def fetch_binance_funding(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    records: list = []
    for sym in _perp_candidates(symbol):
        cursor = start_ms
        records = []
        try:
            while True:
                batch = _get(f"{BINANCE_FUTURES_HOST}/fapi/v1/fundingRate",
                             {"symbol": sym, "startTime": cursor, "limit": 1000})
                if not batch:
                    break
                records.extend(batch)
                if len(batch) < 1000:
                    break
                cursor = int(batch[-1]["fundingTime"]) + 1
                time.sleep(0.15)
        except requests.HTTPError:
            records = []
        if records:
            break
    if not records:
        return pd.DataFrame(columns=["date", "funding_sum", "funding_mean",
                                     "n_settlements"])
    df = pd.DataFrame({
        "funding_time": pd.to_datetime(
            [int(r["fundingTime"]) for r in records], unit="ms"),
        "funding_rate": [float(r["fundingRate"]) for r in records],
    })
    daily = aggregate_funding_daily(df)
    return daily[daily["date"] < _utc_today()].reset_index(drop=True)


def fetch_perp_klines(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    """Daily perp-futures klines (for perp-spot basis and perp/spot volume
    ratio). Assets without a perp contract return an empty frame."""
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    empty = pd.DataFrame(columns=["date", "perp_close", "perp_volume",
                                  "perp_quote_volume", "perp_trades"])
    for sym in _perp_candidates(symbol):
        rows: list = []
        try:
            cursor = start_ms
            while True:
                batch = _get(f"{BINANCE_FUTURES_HOST}/fapi/v1/klines",
                             {"symbol": sym, "interval": "1d",
                              "startTime": cursor, "limit": 1000})
                rows.extend(batch)
                if len(batch) < 1000:
                    break
                cursor = int(batch[-1][0]) + 1
                time.sleep(0.15)
        except requests.HTTPError:
            continue
        if not rows:
            continue
        df = parse_klines(rows, now_ms)
        if sym.startswith("1000"):  # contract priced on 1000 units of spot
            df["close"] = df["close"] / 1000.0
        df = df.rename(columns={"close": "perp_close", "volume": "perp_volume",
                                "quote_volume": "perp_quote_volume",
                                "trades": "perp_trades"})
        return df[["date", "perp_close", "perp_volume", "perp_quote_volume",
                   "perp_trades"]]
    return empty


def fetch_binance_oi(symbol: str) -> pd.DataFrame:
    """Daily open interest. Binance retains ~30 days; we accumulate over time."""
    batch = _get(f"{BINANCE_FUTURES_HOST}/futures/data/openInterestHist",
                 {"symbol": symbol, "period": "1d", "limit": 500})
    if not batch:
        return pd.DataFrame(columns=["date", "open_interest",
                                     "open_interest_usd"])
    df = pd.DataFrame({
        "date": pd.to_datetime([int(r["timestamp"]) for r in batch],
                               unit="ms").normalize(),
        "open_interest": [float(r["sumOpenInterest"]) for r in batch],
        "open_interest_usd": [float(r["sumOpenInterestValue"]) for r in batch],
    })
    df = df[df["date"] < _utc_today()]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Coin Metrics Community
# ---------------------------------------------------------------------------

def _cm_query(asset: str, metrics: list[str], start: pd.Timestamp) -> list:
    url = f"{COINMETRICS_HOST}/timeseries/asset-metrics"
    params: dict | None = {
        "assets": asset, "metrics": ",".join(metrics), "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"), "page_size": 10000,
    }
    data: list = []
    while url:
        j = _get(url, params)
        data.extend(j.get("data", []))
        url = j.get("next_page_url")
        params = None
        if url:
            time.sleep(0.15)
    return data


def _cm_to_frame(data: list, metrics: list[str]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame(columns=["date"] + metrics)
    df = pd.DataFrame(data)
    out = pd.DataFrame({"date": pd.to_datetime(df["time"]).dt.tz_localize(None)})
    for m in metrics:
        out[m] = pd.to_numeric(df[m], errors="coerce") if m in df else float("nan")
    return out.reset_index(drop=True)


def fetch_coinmetrics(asset: str, start: pd.Timestamp,
                      metrics: list[str] | None = None) -> pd.DataFrame:
    """Fetch on-chain metrics; unsupported metrics are dropped per asset."""
    metrics = metrics or CM_METRICS
    try:
        return _cm_to_frame(_cm_query(asset, metrics, start), metrics)
    except requests.HTTPError:
        pass  # some requested metric unsupported for this asset -> retry singly
    frames = []
    ok_metrics = []
    for m in metrics:
        try:
            part = _cm_to_frame(_cm_query(asset, [m], start), [m])
            frames.append(part.set_index("date"))
            ok_metrics.append(m)
        except requests.HTTPError:
            continue
    if not frames:
        return pd.DataFrame(columns=["date"] + metrics)
    merged = pd.concat(frames, axis=1, join="outer").sort_index().reset_index()
    for m in metrics:  # keep a stable schema even for unsupported metrics
        if m not in merged:
            merged[m] = float("nan")
    return merged[["date"] + metrics]


def fetch_stablecoin_supply(start: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for cm_id, col in CM_STABLECOINS.items():
        df = fetch_coinmetrics(cm_id, start, metrics=[CM_STABLE_METRIC])
        df = df.rename(columns={CM_STABLE_METRIC: col}).set_index("date")
        frames.append(df)
    out = pd.concat(frames, axis=1, join="outer").sort_index().reset_index()
    return out


# ---------------------------------------------------------------------------
# Sentiment and macro
# ---------------------------------------------------------------------------

def fetch_fear_greed() -> pd.DataFrame:
    j = _get(FEAR_GREED_URL, {"limit": 0})
    df = pd.DataFrame(j["data"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["timestamp"].astype("int64"), unit="s"),
        "fng_value": df["value"].astype(float),
        "fng_class": df["value_classification"].astype(str),
    })
    out = out[out["date"] < _utc_today()]
    return out.sort_values("date").reset_index(drop=True)


def fetch_macro(start: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(list(MACRO_TICKERS), start=start.strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)["Close"]
    raw = raw.rename(columns=MACRO_TICKERS)
    out = raw.reset_index().rename(columns={"Date": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out[out["date"] < _utc_today()]
    cols = ["date"] + list(MACRO_TICKERS.values())
    return out[[c for c in cols if c in out]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def merge_incremental(old: pd.DataFrame | None,
                      new: pd.DataFrame) -> pd.DataFrame:
    """Combine stored + freshly fetched rows: unique dates, newest wins, sorted."""
    if old is None or old.empty:
        df = new.copy()
    else:
        df = pd.concat([old, new], ignore_index=True)
    df = (df.drop_duplicates(subset="date", keep="last")
            .sort_values("date").reset_index(drop=True))
    return df


def report_gaps(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Calendar days missing between the first and last stored date."""
    if df.empty:
        return pd.DatetimeIndex([])
    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    return expected.difference(pd.DatetimeIndex(df["date"]))


def _load(name: str) -> pd.DataFrame | None:
    path = RAW_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _fetch_start(existing: pd.DataFrame | None) -> pd.Timestamp:
    if existing is None or existing.empty:
        return pd.Timestamp(START_DATE)
    return existing["date"].max() - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)


def _upsert(name: str, fetch_fn) -> dict:
    """Incrementally refresh one table; returns a status-row dict."""
    existing = _load(name)
    try:
        new = fetch_fn(_fetch_start(existing))
        df = merge_incremental(existing, new)
        if df.empty:
            return {"table": name, "status": "empty", "rows": 0, "new": 0,
                    "first": "-", "last": "-", "gaps": "-"}
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RAW_DIR / f"{name}.parquet", index=False)
        added = len(df) - (0 if existing is None else len(existing))
        return {"table": name, "status": "ok", "rows": len(df),
                "new": added, "first": str(df["date"].min().date()),
                "last": str(df["date"].max().date()),
                "gaps": len(report_gaps(df))}
    except Exception as err:  # a dead source degrades, never blocks the run
        rows = 0 if existing is None else len(existing)
        return {"table": name, "status": f"FAILED: {type(err).__name__}",
                "rows": rows, "new": 0, "first": "-", "last": "-", "gaps": "-"}


def refresh_all() -> pd.DataFrame:
    universe = get_universe()
    results = []
    for row in universe.itertuples():
        base, sym = row.base, row.symbol
        results.append(_upsert(
            f"ohlcv_{base}", lambda s, x=sym: fetch_binance_klines(x, s)))
        results.append(_upsert(
            f"funding_{base}", lambda s, x=sym: fetch_binance_funding(x, s)))
        results.append(_upsert(
            f"perp_{base}", lambda s, x=sym: fetch_perp_klines(x, s)))
        results.append(_upsert(
            f"oi_{base}", lambda s, x=sym: fetch_binance_oi(x)))
        if base in CM_ONCHAIN:
            results.append(_upsert(
                f"onchain_{base}",
                lambda s, x=CM_ONCHAIN[base]: fetch_coinmetrics(x, s)))
    results.append(_upsert("stablecoins", fetch_stablecoin_supply))
    results.append(_upsert("fear_greed", lambda s: fetch_fear_greed()))
    results.append(_upsert("macro", fetch_macro))
    return pd.DataFrame(results)


def status() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        if "date" not in df.columns or df.empty:  # e.g. universe snapshot
            continue
        rows.append({"table": path.stem, "rows": len(df),
                     "first": str(df["date"].min().date()),
                     "last": str(df["date"].max().date()),
                     "gaps": len(report_gaps(df)),
                     "columns": ", ".join(c for c in df.columns if c != "date")})
    return pd.DataFrame(rows)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "refresh"
    if cmd == "daily":
        # full daily update: data -> features -> model -> backtest -> recs
        print(refresh_all().to_string(index=False))
        from crypto import features
        features.main()
        from crypto import model  # imports xgboost before lightgbm
        model.train()
        from crypto import backtest, recommend
        backtest.main()
        recommend.main()
        return
    if cmd == "refresh":
        out = refresh_all()
    elif cmd == "status":
        out = status()
    else:
        raise SystemExit(f"unknown command: {cmd} (use refresh|status|daily)")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.max_colwidth", 60):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
