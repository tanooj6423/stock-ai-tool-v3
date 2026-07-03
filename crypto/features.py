"""Feature engineering: raw parquet tables -> one long panel (date, asset).

Anti-lookahead conventions (tested in tests/test_features.py):
  - Every rolling computation is trailing (pandas default).
  - A row (t, asset) may only use data knowable at the close of UTC day t:
    on-chain, stablecoin-supply and sentiment sources describe day t but are
    published during t+1, so they are shifted +1 day here.
  - Macro closes print before the crypto UTC day ends -> no shift; weekend
    rows forward-fill Friday (past data only).
  - Cross-sectional features use same-date values across assets only.
  - The target and the backtest helper `next_open` are the ONLY forward-
    looking columns, and they are not features.

Output: crypto/data/features.parquet
CLI:    python -m crypto.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crypto.config import (CM_ONCHAIN, DATA_DIR, HORIZON_DAYS, RAW_DIR,
                           asset_family)

ONCHAIN_SHIFT = 1    # days: publication lag for on-chain + stablecoin supply
SENTIMENT_SHIFT = 1  # days: conservative lag for Fear & Greed
EPS = 1e-12

# Columns the model may use (asset/family are categoricals; everything else
# numeric). Kept as an explicit registry so the model layer can't silently
# pick up helper/target columns.
FEATURE_COLS = [
    # technical
    "ret_1", "ret_7", "ret_30", "vol_7", "vol_30", "vol_ratio", "rsi_14",
    "macd_hist", "atr_pct", "dist_sma50", "dist_sma200", "dd_90",
    "donchian_55", "dist_ath", "vol_z30", "ret_skew_30", "adx_14",
    "beta_btc_60", "corr_btc_60", "beta_instab", "amihud_30",
    "range_pctl_30", "path_eff_7", "taker_ratio_7", "taker_div",
    "trade_size_z30", "tc_vol_spread", "weekend_vol_share",
    # derivatives
    "funding_7m", "funding_z90", "funding_cum30", "funding_vol30",
    "fund_trend_int", "basis", "basis_z90", "basis_mom5", "fund_basis_disl",
    "perp_spot_vratio",
    # asset-level on-chain (BTC/ETH only; NaN elsewhere)
    "adract_z90", "adract_g7", "txtfr_g7", "fee_g7", "mvrv", "mvrv_z365",
    "exnetflow_z90", "splyex_ch30",
    # market-level
    "mkt_btc_mvrv", "mkt_btc_netflow_z", "mkt_btc_adract_z",
    "mkt_btc_fee_impulse", "mkt_eth_fee_impulse", "stable_g7", "stable_g30",
    "fng", "fng_ch7", "fng_z90", "dxy_r7", "spx_r7", "us10y_ch7",
    "btc_spx_corr30", "breadth_sma50", "mkt_ret7_med", "mkt_disp7",
    "altseason",
    # cross-sectional
    "rs_btc_7", "rs_med_7", "cs_rank_r7", "cs_rank_r30", "cs_rank_chg7",
    "cs_rank_funding", "cs_rank_amihud",
    # meta
    "listing_age", "log_qvol30", "asset", "family",
]
META_COLS = ["date", "close", "next_open", "fwd_ret", "y"]


def _z(s: pd.Series, w: int) -> pd.Series:
    m, sd = s.rolling(w).mean(), s.rolling(w).std()
    return (s - m) / (sd + EPS)


def _rsi(close: pd.Series, w: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / w, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / w, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, w: int = 14) -> pd.Series:
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / w, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=h.index).ewm(alpha=1 / w, adjust=False).mean() / (atr + EPS)
    mdi = 100 * pd.Series(minus, index=h.index).ewm(alpha=1 / w, adjust=False).mean() / (atr + EPS)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    return dx.ewm(alpha=1 / w, adjust=False).mean()


def technical_features(o: pd.DataFrame) -> pd.DataFrame:
    """Per-asset technical features from spot OHLCV. Index: date."""
    f = pd.DataFrame(index=o.index)
    c, h, l, v, qv = o["close"], o["high"], o["low"], o["volume"], o["quote_volume"]
    logc = np.log(c)
    r1 = logc.diff()
    f["ret_1"], f["ret_7"], f["ret_30"] = r1, logc.diff(7), logc.diff(30)
    f["vol_7"], f["vol_30"] = r1.rolling(7).std(), r1.rolling(30).std()
    f["vol_ratio"] = f["vol_7"] / (f["vol_30"] + EPS)
    f["rsi_14"] = _rsi(c)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c + EPS)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / (c + EPS)
    f["dist_sma50"] = c / (c.rolling(50).mean() + EPS) - 1
    f["dist_sma200"] = c / (c.rolling(200).mean() + EPS) - 1
    f["dd_90"] = c / (c.rolling(90).max() + EPS) - 1
    lo55, hi55 = l.rolling(55).min(), h.rolling(55).max()
    f["donchian_55"] = (c - lo55) / (hi55 - lo55 + EPS)
    f["dist_ath"] = c / (c.cummax() + EPS) - 1
    f["vol_z30"] = _z(np.log(qv + 1), 30)
    f["ret_skew_30"] = r1.rolling(30).skew()
    f["adx_14"] = _adx(h, l, c)
    f["amihud_30"] = np.log((r1.abs() / (qv + EPS)).rolling(30).mean() + EPS)
    rng = (h - l) / (c + EPS)
    f["range_pctl_30"] = rng.rolling(30).rank(pct=True)
    f["path_eff_7"] = ((o["close"] - o["open"]).abs() / (h - l + EPS)).rolling(7).mean()
    taker = o["taker_buy_base"] / (v + EPS)
    f["taker_ratio_7"] = taker.rolling(7).mean()
    f["taker_div"] = _z(f["taker_ratio_7"].diff(5), 90) - _z(logc.diff(5), 90)
    f["trade_size_z30"] = _z(np.log(qv / (o["trades"] + 1) + EPS), 30)
    f["tc_vol_spread"] = _z(np.log(o["trades"] + 1), 30) - f["vol_z30"]
    wk = o.index.dayofweek >= 5
    wvol = qv.where(wk, 0.0)
    f["weekend_vol_share"] = wvol.rolling(30).sum() / (qv.rolling(30).sum() + EPS)
    f["listing_age"] = np.log1p(np.arange(len(o)))
    f["log_qvol30"] = np.log(qv.rolling(30).mean() + 1)
    return f


def derivatives_features(f: pd.DataFrame, funding: pd.DataFrame | None,
                         perp: pd.DataFrame | None,
                         o: pd.DataFrame) -> pd.DataFrame:
    """Funding + perp-basis features aligned to the spot index."""
    idx = o.index
    if funding is not None and not funding.empty:
        fs = funding.set_index("date")["funding_sum"].reindex(idx)
        f["funding_7m"] = fs.rolling(7).mean()
        f["funding_z90"] = _z(fs, 90)
        f["funding_cum30"] = fs.rolling(30).sum()
        f["funding_vol30"] = fs.rolling(30).std()
        f["fund_trend_int"] = f["funding_z90"] * np.sign(f["ret_7"])
    if perp is not None and not perp.empty:
        p = perp.set_index("date").reindex(idx)
        basis = p["perp_close"] / (o["close"] + EPS) - 1
        f["basis"] = basis
        f["basis_z90"] = _z(basis, 90)
        f["basis_mom5"] = basis.diff(5)
        f["perp_spot_vratio"] = _z(
            np.log(p["perp_quote_volume"] / (o["quote_volume"] + EPS) + EPS), 30)
        if "funding_7m" in f:
            fs = f["funding_7m"]
            f["fund_basis_disl"] = (
                (np.sign(fs) != np.sign(basis)) & fs.notna() & basis.notna()
            ).astype(float).where(fs.notna() & basis.notna())
    return f


def onchain_features(oc: pd.DataFrame) -> pd.DataFrame:
    """Asset-level on-chain features (BTC/ETH). Shifted +1d for publication
    lag; returned indexed by the date the data is USABLE."""
    oc = oc.set_index("date").shift(ONCHAIN_SHIFT, freq="D")
    g = pd.DataFrame(index=oc.index)
    la = np.log(oc["AdrActCnt"] + 1)
    g["adract_z90"], g["adract_g7"] = _z(la, 90), la.diff(7)
    g["txtfr_g7"] = np.log(oc["TxTfrCnt"] + 1).diff(7)
    g["fee_g7"] = np.log(oc["FeeTotNtv"] + EPS).diff(7)
    g["mvrv"], g["mvrv_z365"] = oc["CapMVRVCur"], _z(oc["CapMVRVCur"], 365)
    netflow = (oc["FlowInExUSD"] - oc["FlowOutExUSD"]) / (oc["SplyExUSD"] + EPS)
    g["exnetflow_z90"] = _z(netflow, 90)
    g["splyex_ch30"] = oc["SplyExUSD"].pct_change(30)
    g["fee_impulse"] = np.log(
        oc["FeeTotNtv"] / (oc["TxTfrCnt"] + 1) + EPS).diff(7)
    return g


def market_features(raw: dict) -> pd.DataFrame:
    """Date-indexed market-level features shared by all assets."""
    frames = []
    for base, pref in [("BTC", "mkt_btc_"), ("ETH", "mkt_eth_")]:
        key = f"onchain_{base}"
        if key in raw:
            g = onchain_features(raw[key])
            cols = {"fee_impulse": f"{pref}fee_impulse"}
            if base == "BTC":
                cols |= {"mvrv": "mkt_btc_mvrv",
                         "exnetflow_z90": "mkt_btc_netflow_z",
                         "adract_z90": "mkt_btc_adract_z"}
            frames.append(g[list(cols)].rename(columns=cols))
    if "stablecoins" in raw:
        st = raw["stablecoins"].set_index("date").shift(ONCHAIN_SHIFT, freq="D")
        tot = np.log(st.fillna(0).sum(axis=1) + EPS)
        frames.append(pd.DataFrame(
            {"stable_g7": tot.diff(7), "stable_g30": tot.diff(30)}))
    if "fear_greed" in raw:
        fg = raw["fear_greed"].set_index("date").shift(SENTIMENT_SHIFT, freq="D")
        v = fg["fng_value"]
        frames.append(pd.DataFrame(
            {"fng": v, "fng_ch7": v.diff(7), "fng_z90": _z(v, 90)}))
    if "macro" in raw and "ohlcv_BTC" in raw:
        btc_idx = pd.DatetimeIndex(raw["ohlcv_BTC"]["date"])
        m = (raw["macro"].set_index("date")
             .reindex(pd.date_range(btc_idx.min(), btc_idx.max(), freq="D"))
             .ffill())
        mk = pd.DataFrame(index=m.index)
        mk["dxy_r7"] = np.log(m["dxy"] + EPS).diff(7)
        mk["spx_r7"] = np.log(m["spx"] + EPS).diff(7)
        mk["us10y_ch7"] = m["us10y"].diff(7)
        btc_r = np.log(raw["ohlcv_BTC"].set_index("date")["close"]).diff()
        spx_r = np.log(m["spx"] + EPS).diff()
        mk["btc_spx_corr30"] = btc_r.reindex(m.index).rolling(30).corr(spx_r)
        frames.append(mk)
    out = pd.concat(frames, axis=1, join="outer").sort_index()
    return out


def cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Same-date cross-asset features added to the long panel."""
    g = panel.groupby("date")
    btc_r7 = panel.loc[panel["asset"] == "BTC", ["date", "ret_7"]].set_index("date")["ret_7"]
    btc_r30 = panel.loc[panel["asset"] == "BTC", ["date", "ret_30"]].set_index("date")["ret_30"]
    panel["rs_btc_7"] = panel["ret_7"] - panel["date"].map(btc_r7)
    med7 = g["ret_7"].transform("median")
    panel["rs_med_7"] = panel["ret_7"] - med7
    panel["mkt_ret7_med"] = med7
    panel["mkt_disp7"] = g["ret_7"].transform("std")
    panel["breadth_sma50"] = g["dist_sma50"].transform(lambda s: (s > 0).mean())
    nonbtc_med30 = (panel[panel["asset"] != "BTC"].groupby("date")["ret_30"]
                    .median())
    panel["altseason"] = panel["date"].map(nonbtc_med30 - btc_r30)
    panel["cs_rank_r7"] = g["ret_7"].rank(pct=True)
    panel["cs_rank_r30"] = g["ret_30"].rank(pct=True)
    panel["cs_rank_chg7"] = (panel.groupby("asset")["cs_rank_r7"]
                             .transform(lambda s: s.diff(7)))
    panel["cs_rank_funding"] = g["funding_7m"].rank(pct=True)
    panel["cs_rank_amihud"] = g["amihud_30"].rank(pct=True)
    return panel


def build_features(raw: dict) -> pd.DataFrame:
    """Assemble the full long panel from a dict of raw tables."""
    mkt = market_features(raw)
    btc_r1 = np.log(raw["ohlcv_BTC"].set_index("date")["close"]).diff()
    bases = sorted(k[len("ohlcv_"):] for k in raw if k.startswith("ohlcv_"))
    frames = []
    for base in bases:
        o = raw[f"ohlcv_{base}"].set_index("date")
        f = technical_features(o)
        b = btc_r1.reindex(f.index)
        f["corr_btc_60"] = f["ret_1"].rolling(60).corr(b)
        beta60 = f["ret_1"].rolling(60).cov(b) / (b.rolling(60).var() + EPS)
        beta90 = f["ret_1"].rolling(90).cov(b) / (b.rolling(90).var() + EPS)
        f["beta_btc_60"], f["beta_instab"] = beta60, beta60 - beta90
        f = derivatives_features(f, raw.get(f"funding_{base}"),
                                 raw.get(f"perp_{base}"), o)
        if base in CM_ONCHAIN and f"onchain_{base}" in raw:
            g = onchain_features(raw[f"onchain_{base}"])
            f = f.join(g.drop(columns=["fee_impulse"]))
        f = f.join(mkt)
        f["asset"], f["family"] = base, asset_family(base)
        # helper + target columns (never features)
        f["close"] = o["close"]
        f["next_open"] = o["open"].shift(-1)
        f["fwd_ret"] = np.log(o["close"]).diff(HORIZON_DAYS).shift(-HORIZON_DAYS)
        f["y"] = (f["fwd_ret"] > 0).astype(float).where(f["fwd_ret"].notna())
        frames.append(f.reset_index().rename(columns={"index": "date"}))
    panel = pd.concat(frames, ignore_index=True)
    panel = cross_sectional_features(panel)
    for col in FEATURE_COLS + META_COLS:  # stable schema
        if col not in panel:
            panel[col] = np.nan
    return panel[["date"] + [c for c in FEATURE_COLS if c != "date"]
                 + [c for c in META_COLS if c != "date"]]


def load_raw() -> dict:
    return {p.stem: pd.read_parquet(p) for p in RAW_DIR.glob("*.parquet")
            if p.stem != "universe"}


def main() -> None:
    panel = build_features(load_raw())
    out = DATA_DIR / "features.parquet"
    panel.to_parquet(out, index=False)
    n_feat = len([c for c in FEATURE_COLS if c not in ("asset", "family")])
    print(f"panel: {len(panel):,} rows x {n_feat} numeric features "
          f"+ asset/family | assets: {panel['asset'].nunique()} | "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    cov = (panel.groupby("asset")
           .agg(rows=("date", "size"),
                labeled=("y", lambda s: int(s.notna().sum()))))
    print(cov.sort_values("rows", ascending=False).to_string())


if __name__ == "__main__":
    main()
