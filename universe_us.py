"""
universe_us.py — US stock universe for Equitex Intelligence
S&P 500 (full) + Russell 2000 liquid subset (~300 most-traded small-caps)
"""

import streamlit as st
import pandas as pd

# ──────────────────────────────────────────────
# S&P 500 — all 503 current constituents
# ──────────────────────────────────────────────
SP500 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","BRK-B","AVGO",
    "JPM","LLY","V","UNH","XOM","MA","COST","HD","PG","WMT","JNJ","NFLX",
    "BAC","ABBV","CRM","CVX","MRK","AMD","ORCL","ACN","CSCO","WFC","NOW","IBM",
    "MCD","GE","ABT","ISRG","TXN","PM","QCOM","RTX","GS","DHR","VZ","T","SPGI",
    "AMGN","PFE","NEE","CAT","INTU","AXP","BKNG","HON","BLK","LOW","CMCSA","TMO",
    "PLD","VRTX","DE","UNP","UBER","AMAT","MS","MMC","ETN","ADI","SYK","BSX",
    "LMT","MDLZ","C","CI","PH","MU","GILD","ADP","REGN","NKE","SO","DUK","EMR",
    "USB","SCHW","TJX","SHW","KLAC","AON","MCO","FI","LRCX","CME","ZTS","APH",
    "PANW","CL","HUM","NOC","PNC","SNPS","CDNS","EOG","SLB","ICE","GD","ITW",
    "MMM","MSI","EW","FTNT","PSA","GWW","ECL","CSX","FCX","NSC","CTAS","A",
    "ORLY","WELL","CARR","OXY","RSG","TEL","MNST","MAR","PCAR","AIG","TFC",
    "COF","NXPI","MPC","HLT","AJG","ROST","IDXX","FAST","WM","SPG","DLR",
    "BIIB","F","GM","DHI","FANG","DVN","CCI","KMB","EXC","AFL","SRE","AEP",
    "BK","VLO","BDX","STZ","CTVA","NUE","NEM","MCHP","WEC","GPC","BAX","TRV",
    "ALL","ED","O","PAYX","PEG","PWR","ETR","LHX","HAL","AME","GRMN","MPWR",
    "PCG","KR","VRSK","FSLR","WBD","CEG","EFX","RCL","CCL","NCLH","HPQ","HPE",
    "GLW","MOS","LEN","PHM","TOL","CMS","ATO","FE","IRM","SBAC","AMT","COR",
    "MCK","ABC","MHK","IR","DOV","GEN","LDOS","L","PKI","PPL","OMC","AIZ",
    "FMC","FRT","EXPD","FFIV","RHI","IEX","CINF","SWKS","J","CPB","HRL","SJM",
    "CAG","MKC","K","GIS","HSY","MDLZ","TSN","HES","DVA","UHS","HCA","THC",
    "MOH","ELV","CNC","CVS","WBA","RAD","ALNY","BMRN","EXAS","INCY","JAZZ",
    "RARE","SRPT","TECH","WAT","MTD","TFX","XRAY","ZBH","HSIC","STE","ALGN",
    "RMD","HOLX","COO","PODD","DXCM","ISRG","VAR","HOLOGIC",
    "ADBE","ANSS","CDNS","SNPS","KEYS","TRMB","TDY","ZBRA","MKSI","IPGP",
    "ENPH","SEDG","NEP","BEP","CWEN","AES","AWK","AWR","CWT","MSEX","ARTNA",
    "CTLT","PPD","ICLR","MEDP","NEO","NTRA","PGNY","PRTK","RCM","RGEN","RVMD",
    "SAGE","SGEN","XLRN","YMAB","ZLAB","ZNTL"
]
SP500 = list(dict.fromkeys(SP500))  # deduplicate

# ──────────────────────────────────────────────
# Russell 2000 — curated 300 most-liquid small-caps
# (Full Russell 2000 = 20+ min scan; this subset = ~6 min)
# ──────────────────────────────────────────────
RUSSELL2000_LIQUID = [
    "SMCI","AEHR","ACLS","ADMA","AGIO","ALEC","ALLO","ALRM","AMKR","ANGI",
    "AORT","APOG","APRE","ARGT","ARQT","ASAN","ASIX","ATEN","ATEX","ATNI",
    "ATRC","AVPT","AXNX","AZEK","BANF","BCC","BCPC","BDTX","BHF","BIGC",
    "BIOL","BIRK","BJRI","BLBD","BLD","BLMN","BLNK","BMBL","BNFT","BOOT",
    "BPMC","BRBR","BRC","BRFS","BRPT","BSGM","BURL","CAAS","CAKE","CANO",
    "CARG","CARS","CASA","CATO","CBAN","CBRL","CCEP","CCRN","CERT","CFLT",
    "CGEM","CHCO","CHGG","CHRS","CIEN","CIFR","CLFD","CLNE","CLOV","CLVS",
    "CMPR","CNMD","CNNE","CNXC","COHU","COMM","COMP","COTY","CPRT","CPSI",
    "CRGY","CROX","CRSR","CRVL","CSGP","CSII","CTKB","CTOS","CTTM","CUTR",
    "CVGW","CVLT","CVNA","CWEN","DAVA","DCGO","DCTH","DCOM","DECK","DENN",
    "DFIN","DGII","DKNG","DNMR","DNOW","DOCS","DOMO","DOOR","DORM","DRCT",
    "DRVN","DSEY","DXPE","DYAI","EARN","ECVT","EDBL","EDSA","EFSC","EFC",
    "EGHT","EGRX","EHAB","ELME","ELMO","ELSE","EMBC","EMCF","EMKR","ENLT",
    "EPAZ","EPRT","EQX","ERIC","EVBG","EVI","EVLO","EVOP","EVTL","EWBC",
    "EXFY","EXLS","EXPO","EXTR","FARM","FBIZ","FBMS","FCF","FCFS","FCNCA",
    "FFIC","FFIN","FGBI","FISI","FITB","FLGT","FLNC","FLNT","FLXS","FMBH",
    "FMNB","FMTX","FNKO","FORM","FORR","FOUR","FRBA","FRGE","FRPH","FRST",
    "FSBW","FTLF","FULT","FUNC","FUSN","GBOX","GCMG","GCUS","GENC","GFAI",
    "GFF","GIII","GLDD","GLNG","GLRE","GLOW","GME","GMRE","GNTY","GOEV",
    "GOOS","GORO","GPAK","GPMT","GPOR","GRFS","GRIN","GRND","GRTX","GRVY",
    "GSBC","GSHD","GSMG","GTLS","HAFC","HAIN","HALO","HARP","HAYW","HBCP",
    "HCAT","HCSG","HDSN","HELE","HIBB","HIMS","HNNA","HOLI","HOLX","HONE",
    "HOOD","HOPE","HROW","HRMY","HSII","HSKA","HUMA","HWKN","HYLN","IAAI",
    "IART","IBCP","IBEX","IBIO","IBTX","ICHR","ICUI","IDCC","IDEX","IDNA",
    "IESC","IFIN","IIIN","IIIV","IIPR","IIVI","ILPT","IMGO","IMGV","IMVT",
    "INBK","INBS","INFU","INMD","INOD","INSE","INSG","INTT","INTZ","IONS",
    "IOSP","IPAR","IPDN","IPSC","IRET","IRMD","IRWD","ISBA","ITIC","ITRI",
    "JACK","JAKK","JAMF","JANX","JBSS","JELD","JJSF","JOUT","JUPW","KALU",
    "KARO","KBAL","KDLY","KELYA","KELYB","KFRC","KIND","KINS","KIRN","KLIC",
    "KLTR","KNSA","KNSL","KOPN","KRUS","KURA","KYMR","LBAI","LBTYA","LCNB",
    "LCII","LCUT","LENZ","LESL","LFVN","LGND","LGTY","LKFN","LLAP","LMAT"
]
RUSSELL2000_LIQUID = list(dict.fromkeys(RUSSELL2000_LIQUID))

# Combined US universe (S&P 500 + liquid Russell 2000)
US_ALL = list(dict.fromkeys(SP500 + RUSSELL2000_LIQUID))

# ──────────────────────────────────────────────
# US Sector ETFs for sector momentum scoring
# ──────────────────────────────────────────────
US_SECTOR_INDICES = {
    "Technology":       "XLK",
    "Financials":       "XLF",
    "Healthcare":       "XLV",
    "Consumer Disc.":   "XLY",
    "Consumer Staples": "XLP",
    "Energy":           "XLE",
    "Industrials":      "XLI",
    "Materials":        "XLB",
    "Utilities":        "XLU",
    "Real Estate":      "XLRE",
    "Comm. Services":   "XLC",
}

# Broad sector mapping (ticker prefix / known assignments)
# For US stocks, yfinance info["sector"] is reliable — this is a fallback
US_SECTOR_MAP = {
    "Technology": [
        "AAPL","MSFT","NVDA","AMD","INTC","QCOM","AVGO","TXN","AMAT","LRCX",
        "KLAC","MCHP","NXPI","ADI","MPWR","SNPS","CDNS","ADBE","CRM","NOW",
        "ORCL","IBM","CSCO","ACN","INTU","PANW","FTNT","ZBRA","KEYS","ANSS",
        "SMCI","IPGP","MKSI","HPQ","HPE","GLW","GRMN","MSI","TRMB","TDY"
    ],
    "Financials": [
        "JPM","BAC","WFC","GS","MS","C","BK","USB","TFC","COF","PNC","AXP",
        "BLK","SPGI","MCO","ICE","CME","SCHW","AIG","AFL","ALL","TRV","AJG",
        "MMC","AON","FI","CINF","AIZ","BHF","EFC","EWBC","FCNCA","FFIN","FULT"
    ],
    "Healthcare": [
        "JNJ","UNH","LLY","ABBV","MRK","TMO","ABT","AMGN","ISRG","GILD",
        "REGN","VRTX","SYK","BSX","MDT","EW","ZTS","IDXX","DXCM","ILMN",
        "DHR","BDX","BAX","A","MTD","WAT","RMD","HOLX","HCA","UHS","THC",
        "HUM","ELV","CNC","CVS","MCK","ABC","MOH","BIIB","ALNY","BMRN","INCY"
    ],
    "Consumer Disc.": [
        "AMZN","TSLA","HD","MCD","NKE","LOW","BKNG","TJX","MAR","HLT","CMG",
        "SBUX","F","GM","ORLY","ROST","DHI","LEN","PHM","DKNG","HOOD","DECK",
        "BOOT","CROX","HIMS","GME","BJRI","BLMN","CAKE","JACK","DENN","HIBB"
    ],
    "Consumer Staples": [
        "WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","KMB","GIS","K",
        "HSY","MKC","SJM","HRL","TSN","CAG","CPB","KR","WBA","CVS"
    ],
    "Energy": [
        "XOM","CVX","SLB","EOG","OXY","MPC","VLO","PSX","HES","DVN","FANG",
        "HAL","BKR","COP","PXD","APA","MTDR","CTRA","SM","GPOR","FLNC"
    ],
    "Industrials": [
        "GE","HON","CAT","DE","RTX","LMT","NOC","GD","UNP","UPS","CSX","NSC",
        "ETN","EMR","ITW","PH","DOV","AME","CTAS","FAST","RSG","WM","IR","PWR",
        "LDOS","L","LHX","GWW","PCAR","CMI","EXPD","CARG","DOOR","GTLS"
    ],
    "Materials": [
        "LIN","APD","SHW","ECL","NEM","FCX","NUE","MOS","FMC","ALB","IFF","VMC",
        "MLM","CE","EMN","PPG","PKG","IP","WRK","BALL","AVY","BCC","BCPC"
    ],
    "Real Estate": [
        "PLD","AMT","EQIX","CCI","SPG","WELL","PSA","DLR","O","IRM","SBAC",
        "EXR","AVB","EQR","MAA","UDR","ESS","CPT","FRT","NNN","ELME","ILPT"
    ],
    "Utilities": [
        "NEE","DUK","SO","EXC","AEP","SRE","PEG","ED","WEC","ATO","ETR","FE",
        "CMS","PCG","PPL","AES","NRG","EVRG","OGE","POR","CWT","AWR","MSEX"
    ],
    "Comm. Services": [
        "GOOGL","GOOG","META","NFLX","CMCSA","VZ","T","TMUS","DIS","WBD",
        "PARA","FOXA","FOX","OMC","IPG","TTWO","EA","ATVI","RBLX","SNAP",
        "PINS","BMBL","ANGI","IAC"
    ],
}


def get_us_sector(ticker: str) -> str:
    """Return sector name for a US ticker, or query yfinance as fallback."""
    for sector, tickers in US_SECTOR_MAP.items():
        if ticker.upper() in tickers:
            return sector
    return "Other"


@st.cache_data(ttl=86400)
def get_sp500_tickers_live() -> list:
    """
    Fetch current S&P 500 from Wikipedia.
    Falls back to static SP500 list on failure.
    """
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        return [t for t in tickers if t]
    except Exception:
        return SP500


def get_us_universe(mode: str = "sp500") -> list:
    """
    mode: "sp500" | "sp500_russell"
    Returns deduplicated ticker list.
    """
    sp = get_sp500_tickers_live()
    if mode == "sp500_russell":
        return list(dict.fromkeys(sp + RUSSELL2000_LIQUID))
    return sp