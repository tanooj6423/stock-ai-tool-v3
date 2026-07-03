"""Central configuration for the crypto analysis system.

All timestamps in this project are UTC. A row dated D describes calendar
day D (00:00-23:59:59.999 UTC) and is knowable at the close of day D.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Earliest date we attempt to fetch. Assets listed later simply start later.
START_DATE = "2020-01-01"

# Dynamic universe: top-N most liquid Binance USDT spot pairs (snapshot at
# refresh time), always including the seed majors. Memecoins and newer
# listings are deliberately in scope (user directive 2026-07-02); assets with
# < MIN_TRAIN_HISTORY_DAYS of history are scored but excluded from training.
SEED_BASES = ["BTC", "ETH", "SOL", "BNB"]
UNIVERSE_SIZE = 30
MIN_TRAIN_HISTORY_DAYS = 400

# Quote/base assets that are not tradable "coins" for our purposes.
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "DAI", "USDP", "USD1", "USDE", "BUSD", "EUR",
    "RLUSD", "PYUSD", "GUSD",
    "EURI", "GBP", "TRY", "BRL", "JPY", "AEUR", "XUSD", "USTC",
    "WBTC", "WBETH", "BETH", "STETH", "WSTETH", "CBBTC", "PAXG", "XAUT",
}

# Coin Metrics Community has usable on-chain coverage for BTC/ETH only
# (checked 2026-07-02; e.g. SOL: none, BNB: ended 2019). Other assets get NaN
# asset-level on-chain features; BTC on-chain doubles as market-level state.
CM_ONCHAIN = {"BTC": "btc", "ETH": "eth"}

# Coin Metrics Community metrics requested per asset (unsupported ones are
# dropped gracefully per asset at fetch time). The community tier no longer
# carries NVTAdj/CapRealUSD/TxTfrValAdjUSD; it does carry MVRV and exchange
# flows, which are stronger cycle/flow signals anyway.
CM_METRICS = ["AdrActCnt", "TxTfrCnt", "FeeTotNtv", "CapMVRVCur",
              "FlowInExUSD", "FlowOutExUSD", "SplyExUSD"]

# Stablecoins used for the market-level "dry powder" supply feature.
CM_STABLECOINS = {"usdt": "usdt_supply", "usdc": "usdc_supply"}
CM_STABLE_METRIC = "SplyCur"

# Macro tickers via yfinance -> column name. (Gold dropped per REQUIREMENTS §10.)
MACRO_TICKERS = {"DX-Y.NYB": "dxy", "^GSPC": "spx", "^TNX": "us10y"}

# API hosts. data-api.binance.vision is the official market-data-only mirror,
# tried as fallback if api.binance.com is unreachable/geo-blocked.
BINANCE_SPOT_HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
BINANCE_FUTURES_HOST = "https://fapi.binance.com"
COINMETRICS_HOST = "https://community-api.coinmetrics.io/v4"
FEAR_GREED_URL = "https://api.alternative.me/fng/"

# When incrementally refreshing, re-fetch this many trailing days to pick up
# any source-side corrections.
REFRESH_OVERLAP_DAYS = 5

# Asset family (categorical model feature; per Gemini review, family
# generalizes better than per-ticker identity for short-history assets).
# Any base not listed is "alt".
FAMILY_MAJOR = {"BTC", "ETH"}
FAMILY_MEME = {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "MEME", "BOME",
               "PENGU", "TRUMP", "PNUT", "FARTCOIN", "1000SATS", "1000CHEEMS",
               "1MBABYDOGE", "ACT", "NEIRO", "TURBO", "MOODENG", "POPCAT",
               "BRETT", "MEW", "DOGS", "HMSTR", "BABYDOGE", "KOMA", "SPX"}


def asset_family(base: str) -> str:
    if base in FAMILY_MAJOR:
        return "major"
    if base in FAMILY_MEME:
        return "meme"
    return "alt"


# Modeling / backtest constants (used by later components).
HORIZON_DAYS = 7
PROB_THRESHOLD = 0.55
FEE_BPS = 10.0       # per side
SLIPPAGE_BPS = 5.0   # per side
