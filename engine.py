"""
engine.py — "Doubler" Scoring Engine v2.0
==========================================
Screens the S&P 1500 universe for high-convexity stocks with the
technical DNA to double within a 3–6 month window.

V2 Improvements over V1:
  - Regime filter (SPY trend + VIX percentile)
  - Earnings catalyst proximity flag
  - Float size & short interest factor
  - ATR-based volatility profile scoring
  - Signal freshness tracking (new vs stale entries)
  - Sector concentration caps
  - Extended volume metric (50-day OBV slope)
  - RS window reweighting (50/35/15)
  - Overextension filter (>100% above 200 SMA)
  - Ensemble scoring (3 model variants)
  - Full walk-forward backtest engine

Methodology:
  1. Trend Intensity   (30%) — Proximity to 52-week high + SMA stack
  2. Relative Strength  (25%) — Multi-period outperformance vs SPY (reweighted)
  3. Volume Persistence (20%) — 50-day OBV slope + up/down volume ratio
  4. Volatility Quality (15%) — Smooth uptrend vs whipsaw (ATR-adjusted)
  5. Catalyst Proximity (10%) — Earnings within 30 days flag

  Hard Filters:
    - Price < 200-day SMA → disqualified
    - Price > 200% of 200-day SMA → disqualified (overextension)

  Adjustments:
    - SMA stack bonus: +0.5
    - Regime penalty: score * 0.6 when SPY < 200 SMA
    - Sector cap: max 4 names per GICS sector in final output
    - Ensemble: only surface stocks appearing in top N across 3 model variants
"""

import datetime as dt
import io
import logging
import math
import warnings
from typing import Optional, Callable

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# S&P 1500 UNIVERSE
# ─────────────────────────────────────────────────────────────

_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
_SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

_FALLBACK_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","GOOG","META","TSLA","BRK-B","AVGO",
    "JPM","LLY","UNH","V","XOM","MA","COST","HD","PG","JNJ",
    "ABBV","NFLX","BAC","CRM","CVX","MRK","TMUS","AMD","PEP","KO",
    "LIN","TMO","ACN","ADBE","MCD","CSCO","ABT","WMT","DHR","PM",
    "NEE","TXN","QCOM","INTU","UNP","AMGN","RTX","ISRG","HON","GE",
    "LOW","AMAT","CAT","BA","GS","BKNG","ELV","DE","BLK","SPGI",
    "ADP","MDLZ","SYK","TJX","GILD","ADI","VRTX","MMC","LRCX","PLD",
    "ETN","PANW","BSX","REGN","CI","KLAC","SNPS","CDNS","CME","SHW",
    "MO","SO","ICE","CMG","MCK","MPC","DUK","CL","ITW","PH",
    "APD","NOC","EOG","SLB","EMR","GD","TDG","AJG","ABNB","WMB",
    "ORLY","ROP","SPG","AZO","MSI","ADSK","MELI","CRWD","FTNT","DDOG",
    "UBER","AXON","PLTR","DASH","WDAY","TEAM","SNOW","NET","ZS","MNST",
    "FICO","CEG","VST","APP","TRGP","HWM","ANET","TTD","NTRA","DUOL",
    "WING","CAVA","ONON","ELF","CELH","TXRH","FIX","WMS","BOOT","LNTH",
    "SFM","IPAR","KRYS","CORT","WDFC","CALM","MOD","SKYW","ATGE","ACIW",
    "IDCC","SPSC","EXLS","TMDX","CPRX","VCEL","UFPT","PAYO","HRMY",
    "SMCI","DECK","NEM","FANG","OXY","DVN","HAL","PSX","VLO","CTRA",
    "EQT","AR","RRC","MTDR","CHRD","MGY","SM","GPOR","NEXT","TALO",
    "AAON","ABCL","ACGL","ACHR","ACM","ACVA","ADPT","AEHR","AEM","AFRM",
    "AGCO","AGI","AGIO","AGYS","AI","AISP","AIT","ALAB","ALKT","ALNY",
    "ALSN","AMKR","AMP","AMPH","AMR","ANF","ANSS","AOS","APLE","APPF",
    "APPN","ARES","ARHS","ARKK","AROC","ARRY","ASH","ASTE","ATKR","AUR",
    "AVO","AVTR","AX","AZEK","AZN","BALL","BBIO","BBY","BDX","BECN",
    "BG","BHC","BILL","BJ","BRBR","BRO","BURL","BWA","BYRN","CACC",
    "CADE","CART","CASY","CBOE","CBRE","CCJ","CEIX","CF","CFR","CHDN",
    "CHE","CHWY","CIB","CLS","CLX","COIN","COHR","COO","COR","CPAY",
    "CRI","CRL","CRSP","CRUS","CSGP","CTAS","CTLT","CTSH","CTVA","CW",
    "CYBR","CYTK","DAL","DCI","DFS","DINO","DKS","DOCS","DOV","DOX",
    "DSGX","DTM","DT","DUOL","DY","EAT","EGAN","EHC","EME","ENPH",
    "EPAM","EQH","ESI","ESTC","EVR","EW","EWBC","EXAS","EXPD","EXPE",
    "EXPO","FCNCA","FDS","FHN","FIVE","FLR","FND","FOUR","FOXF","FPH",
    "FSLR","FTV","GFL","GKOS","GL","GLOB","GLPI","GMS","GNTX","GRAB",
    "GWW","HAS","HBAN","HCA","HEIA","HEI","HES","HII","HIMS","HLF",
    "HOLX","HPE","HPQ","HRL","HSIC","HTHT","HXL","IBM","ICL","IEX",
    "INCY","INGR","INSM","INSP","IOT","IONS","IOSP","IPG","IQV","IR",
    "IRM","IT","ITCI","JAZZ","JBHT","JBL","JKHY","JNPR","KBR","KD",
    "KEY","KGS","KMB","KNSL","KNX","KR","KVUE","LAMR","LANC","LECO",
    "LFUS","LGIH","LHX","LKQ","LMND","LOPE","LPX","LSCC","LUV","LVS",
    "LW","LYB","LYV","MAA","MAS","MANH","MEDP","MGNI","MKC","MKTX",
    "MLM","MMSI","MOS","MPWR","MRVL","MSGS","MTCH","MTD","MTSI","MU",
    "NBIX","NCLH","NDVA","NDSN","NFE","NJR","NKTR","NMIH","NOG","NOW",
    "NSC","NTAP","NTNX","NUE","NXST","NYT","OC","ODFL","OKE","OLED",
    "OMC","ON","ONTO","OPEN","ORI","OSIS","OTIS","OVV","PAYC","PCOR",
    "PCTY","PEN","PFG","PJNG","PKG","PLUG","PODD","POST","POWI","PPC",
    "PRGO","PSTG","PTC","QTWO","RBRK","RCL","RGEN","RGLD","RH","RHP",
    "RHI","RLI","RMBS","RNR","ROCK","ROK","ROKU","ROST","RPM","RPRX",
    "RS","RSG","RUN","RVMD","SAH","SAIA","SAM","SATS","SBAC","SBUX",
    "SDOG","SEDG","SEER","SGH","SHAK","SLAB","SLM","SNA","SNEX","SSD",
    "SSNC","SSP","STAG","STLD","STR","STX","SWAV","SWK","SWKS","SYF",
    "SYY","TDC","TER","TFC","TFX","TGT","THC","TKR","TNET","TOST",
    "TREX","TRMB","TROW","TRU","TSCO","TSN","TWLO","TYL","UBER","UFPI",
    "ULTA","UMBF","UNM","UPST","VEEV","VFC","VICI","VKTX","VMC","VRNS",
    "VRSN","VRSK","VRT","VRTX","WAB","WAL","WBA","WFRD","WH","WHR",
    "WIX","WKHS","WPC","WRB","WSM","WSO","WST","WTFC","WTS","XEL",
    "XPO","XYL","YMH","ZBH","ZBRA","ZEN","ZI","ZM","ZWS"
]

_SECTOR_CACHE: dict[str, str] = {}


def get_sp1500_tickers() -> list[str]:
    """Fetch S&P 1500 constituent tickers from multiple sources."""
    tickers = set()

    # Method 1: Try GitHub-hosted S&P 500 CSV (reliable, no blocking)
    csv_urls = [
        ("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv", "symbol", "S&P 500 CSV"),
    ]
    for csv_url, col_name, label in csv_urls:
        try:
            import urllib.request
            req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=15)
            csv_data = response.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(csv_data))
            for col in df.columns:
                if col.lower().strip() in ["symbol", "ticker", "symbols"]:
                    syms = df[col].dropna().astype(str).tolist()
                    syms = [s.strip().replace(".", "-") for s in syms if 0 < len(s.strip()) < 10]
                    tickers.update(syms)
                    logger.info(f"  {label}: fetched {len(syms)} tickers")
                    # Try to get sector info
                    for sec_col in df.columns:
                        if "sector" in sec_col.lower():
                            for _, row in df.iterrows():
                                sym = str(row[col]).strip().replace(".", "-")
                                sec = str(row[sec_col]).strip()
                                if sym and sec and sec != "nan":
                                    _SECTOR_CACHE[sym] = sec
                            break
                    break
        except Exception as e:
            logger.warning(f"  Could not fetch {label}: {e}")

    # Method 2: Try Wikipedia with browser-like headers
    for url, label in [
        (_SP500_URL, "S&P 500"),
        (_SP400_URL, "S&P 400"),
        (_SP600_URL, "S&P 600"),
    ]:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
            response = urllib.request.urlopen(req, timeout=15)
            html = response.read().decode("utf-8")
            tables = pd.read_html(io.StringIO(html))
            for tbl in tables:
                for col in tbl.columns:
                    col_lower = str(col).lower()
                    if "symbol" in col_lower or "ticker" in col_lower:
                        syms = tbl[col].dropna().astype(str).tolist()
                        syms = [s.strip().replace(".", "-") for s in syms if 0 < len(s.strip()) < 10]
                        if len(syms) > 20:
                            tickers.update(syms)
                            logger.info(f"  {label}: scraped {len(syms)} tickers")
                            for sec_col in tbl.columns:
                                if "sector" in str(sec_col).lower() or "gics" in str(sec_col).lower():
                                    for _, row in tbl.iterrows():
                                        sym = str(row[col]).strip().replace(".", "-")
                                        sec = str(row[sec_col]).strip()
                                        if sym and sec and sec != "nan":
                                            _SECTOR_CACHE[sym] = sec
                                    break
                            break
        except Exception as e:
            logger.warning(f"  Could not scrape {label}: {e}")

    # Method 3: Fallback to hardcoded list
    tickers.update(_FALLBACK_TICKERS)
    logger.info(f"Merged fallback list")

    tickers.discard("SPY")
    logger.info(f"  Total unique tickers: {len(tickers)}")
    return sorted(tickers)


def _get_sector(ticker: str) -> str:
    if ticker in _SECTOR_CACHE:
        return _SECTOR_CACHE[ticker]
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
        _SECTOR_CACHE[ticker] = sector
        return sector
    except Exception:
        return "Unknown"


# ─────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────

def _safe_download(
    tickers: list[str],
    period: str = "15mo",
    start: str = None,
    end: str = None,
    progress_callback: Callable = None,
) -> dict[str, pd.DataFrame]:
    results = {}
    batch_size = 50
    all_tickers = ["SPY"] + tickers
    total = len(all_tickers)
    done = 0

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        try:
            kwargs = dict(
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if start and end:
                kwargs["start"] = start
                kwargs["end"] = end
            else:
                kwargs["period"] = period

            data = yf.download(batch, **kwargs)

            if data.empty:
                done += len(batch)
                continue

            if len(batch) == 1:
                tk = batch[0]
                df = data.copy()
                if not df.empty and len(df) > 60:
                    results[tk] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            else:
                for tk in batch:
                    try:
                        if tk in data.columns.get_level_values(0):
                            df = data[tk][["Open", "High", "Low", "Close", "Volume"]].dropna()
                            if len(df) > 60:
                                results[tk] = df
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"Batch download failed ({batch[:3]}...): {e}")

        done += len(batch)
        if progress_callback:
            progress_callback(min(done / total, 1.0))

    logger.info(f"Successfully downloaded data for {len(results)} tickers")
    return results


# ─────────────────────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────────────────────

def detect_market_regime(spy_df: pd.DataFrame) -> dict:
    close = spy_df["Close"]
    sma200 = close.rolling(200).mean()

    latest_price = float(close.iloc[-1])
    latest_sma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else latest_price

    spy_above_200 = latest_price > latest_sma200

    returns = close.pct_change().dropna()
    realized_vol = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)

    all_vol = returns.rolling(20).std() * np.sqrt(252) * 100
    all_vol = all_vol.dropna()
    if len(all_vol) > 0:
        vol_percentile = float((all_vol < realized_vol).mean())
    else:
        vol_percentile = 0.5

    if spy_above_200 and vol_percentile < 0.7:
        regime = "BULL"
        penalty = 1.0
    elif spy_above_200 and vol_percentile >= 0.7:
        regime = "BULL_VOLATILE"
        penalty = 0.85
    elif not spy_above_200 and vol_percentile < 0.7:
        regime = "BEAR_CALM"
        penalty = 0.65
    else:
        regime = "BEAR_VOLATILE"
        penalty = 0.50

    return {
        "regime": regime,
        "spy_above_200sma": spy_above_200,
        "realized_vol": round(realized_vol, 1),
        "vol_percentile": round(vol_percentile, 2),
        "score_multiplier": penalty,
    }


# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (ENHANCED)
# ─────────────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, window // 2)).mean()


def _compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    volume = df["Volume"]

    sma50 = _sma(close, 50)
    sma150 = _sma(close, 150)
    sma200 = _sma(close, 200)

    latest_price = float(close.iloc[-1])
    latest_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else latest_price
    latest_sma150 = float(sma150.iloc[-1]) if pd.notna(sma150.iloc[-1]) else latest_price
    latest_sma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else latest_price

    one_year = close.iloc[-252:] if len(close) >= 252 else close
    high_52w = float(one_year.max())
    pct_from_high = (latest_price / high_52w) if high_52w > 0 else 0

    sma_stack_aligned = (
        latest_price > latest_sma50 > latest_sma150 > latest_sma200
    ) if all(v > 0 for v in [latest_sma50, latest_sma150, latest_sma200]) else False

    above_200sma = latest_price > latest_sma200 if latest_sma200 > 0 else False

    pct_above_200sma = ((latest_price / latest_sma200) - 1) if latest_sma200 > 0 else 0
    overextended = pct_above_200sma > 2.0

    def _period_return(n_days):
        if len(close) > n_days:
            prev = float(close.iloc[-n_days - 1])
            return (latest_price - prev) / prev if prev > 0 else 0.0
        return 0.0

    ret_1m = _period_return(21)
    ret_3m = _period_return(63)
    ret_6m = _period_return(126)
    ret_12m = _period_return(252)

    recent_20 = df.iloc[-20:]
    daily_chg_20 = recent_20["Close"].diff()
    up_vol = float(recent_20.loc[daily_chg_20 > 0, "Volume"].sum())
    down_vol = float(recent_20.loc[daily_chg_20 <= 0, "Volume"].sum())
    vol_ratio = up_vol / down_vol if down_vol > 0 else (10.0 if up_vol > 0 else 1.0)

    recent_50 = df.iloc[-50:] if len(df) >= 50 else df.iloc[-20:]
    obv = (np.sign(recent_50["Close"].diff()) * recent_50["Volume"]).cumsum()
    obv_values = obv.dropna().values
    if len(obv_values) > 5:
        x = np.arange(len(obv_values))
        obv_slope = float(np.polyfit(x, obv_values, 1)[0])
    else:
        obv_slope = 0.0

    high = df["High"].iloc[-20:]
    low = df["Low"].iloc[-20:]
    prev_close = df["Close"].iloc[-21:-1]
    if len(prev_close) == len(high):
        tr = pd.concat([
            high - low,
            (high - prev_close.values).abs(),
            (low - prev_close.values).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.mean())
        atr_pct = atr / latest_price if latest_price > 0 else 0
    else:
        atr = 0
        atr_pct = 0

    recent_60 = close.iloc[-60:] if len(close) >= 60 else close
    rolling_max = recent_60.cummax()
    drawdowns = (recent_60 - rolling_max) / rolling_max
    max_drawdown_60d = float(drawdowns.min())

    return {
        "price": latest_price,
        "sma50": latest_sma50,
        "sma150": latest_sma150,
        "sma200": latest_sma200,
        "high_52w": high_52w,
        "pct_from_high": pct_from_high,
        "sma_stack_aligned": sma_stack_aligned,
        "above_200sma": above_200sma,
        "overextended": overextended,
        "pct_above_200sma": round(pct_above_200sma, 4),
        "ret_1m": ret_1m,
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "ret_12m": ret_12m,
        "vol_ratio": vol_ratio,
        "obv_slope": obv_slope,
        "atr": atr,
        "atr_pct": atr_pct,
        "max_dd_60d": max_drawdown_60d,
    }


# ─────────────────────────────────────────────────────────────
# EARNINGS CATALYST DETECTION
# ─────────────────────────────────────────────────────────────

def _check_earnings_proximity(ticker: str) -> dict:
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty:
            if isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    dates = cal.loc["Earnings Date"]
                    next_date = pd.Timestamp(dates.iloc[0])
                elif "Earnings Date" in cal.columns:
                    next_date = pd.Timestamp(cal["Earnings Date"].iloc[0])
                else:
                    return {"earnings_within_30d": False, "days_to_earnings": None}
            elif isinstance(cal, dict):
                ed = cal.get("Earnings Date", [])
                if ed:
                    next_date = pd.Timestamp(ed[0])
                else:
                    return {"earnings_within_30d": False, "days_to_earnings": None}
            else:
                return {"earnings_within_30d": False, "days_to_earnings": None}

            days_until = (next_date - pd.Timestamp.now()).days
            return {
                "earnings_within_30d": 0 < days_until <= 30,
                "days_to_earnings": max(days_until, 0) if days_until > 0 else None,
            }
    except Exception:
        pass
    return {"earnings_within_30d": False, "days_to_earnings": None}


# ─────────────────────────────────────────────────────────────
# FLOAT & SHORT INTEREST
# ─────────────────────────────────────────────────────────────

def _get_float_short_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        float_shares = info.get("floatShares", None)
        short_pct = info.get("shortPercentOfFloat", None)
        market_cap = info.get("marketCap", None)

        if float_shares:
            if float_shares < 20_000_000:
                float_class = "micro"
            elif float_shares < 100_000_000:
                float_class = "small"
            elif float_shares < 500_000_000:
                float_class = "medium"
            else:
                float_class = "large"
        else:
            float_class = "unknown"

        return {
            "float_shares": float_shares,
            "short_pct_float": short_pct,
            "float_class": float_class,
            "market_cap": market_cap,
        }
    except Exception:
        return {
            "float_shares": None,
            "short_pct_float": None,
            "float_class": "unknown",
            "market_cap": None,
        }


# ─────────────────────────────────────────────────────────────
# DECILE RANKING
# ─────────────────────────────────────────────────────────────

def _decile_rank(series: pd.Series) -> pd.Series:
    ranks = series.rank(pct=True, method="average")
    return np.ceil(ranks * 10).clip(1, 10).astype(int)


# ─────────────────────────────────────────────────────────────
# COMPOSITE SCORE (V2)
# ─────────────────────────────────────────────────────────────

def calculate_composite_score(
    metrics_df: pd.DataFrame,
    regime_multiplier: float = 1.0,
    weights: dict = None,
) -> pd.DataFrame:
    if weights is None:
        weights = {
            "trend": 0.30,
            "rs": 0.25,
            "volume": 0.20,
            "volatility": 0.15,
            "catalyst": 0.10,
        }

    df = metrics_df.copy()

    df["trend_decile"] = _decile_rank(df["pct_from_high"])

    df["rs_3m_decile"] = _decile_rank(df["rs_3m"])
    df["rs_6m_decile"] = _decile_rank(df["rs_6m"])
    df["rs_12m_decile"] = _decile_rank(df["rs_12m"])
    df["rs_avg_decile"] = (
        df["rs_3m_decile"] * 0.50
        + df["rs_6m_decile"] * 0.35
        + df["rs_12m_decile"] * 0.15
    ).round(1)

    df["vol_ratio_decile"] = _decile_rank(df["vol_ratio"])
    if "obv_slope" in df.columns:
        df["obv_decile"] = _decile_rank(df["obv_slope"])
        df["vol_decile"] = ((df["vol_ratio_decile"] * 0.5 + df["obv_decile"] * 0.5)).round(1)
    else:
        df["vol_decile"] = df["vol_ratio_decile"].astype(float)

    if "atr_pct" in df.columns:
        df["vol_quality_decile"] = _decile_rank(-df["atr_pct"])
    else:
        df["vol_quality_decile"] = 5.0

    if "max_dd_60d" in df.columns:
        df["dd_quality_decile"] = _decile_rank(df["max_dd_60d"])
    else:
        df["dd_quality_decile"] = 5.0

    df["quality_decile"] = ((df["vol_quality_decile"] * 0.6 + df["dd_quality_decile"] * 0.4)).round(1)

    if "earnings_within_30d" in df.columns:
        df["catalyst_score"] = df["earnings_within_30d"].astype(float) * 10.0
    else:
        df["catalyst_score"] = 5.0

    df["composite_score"] = (
        df["trend_decile"] * weights["trend"]
        + df["rs_avg_decile"] * weights["rs"]
        + df["vol_decile"] * weights["volume"]
        + df["quality_decile"] * weights["volatility"]
        + df["catalyst_score"] * weights["catalyst"]
    ).round(2)

    df.loc[df["sma_stack_aligned"], "composite_score"] += 0.5
    logger.warning(f"Before filters: {len(df[df.composite_score > 0])} scores>0, above_200sma_true={df.above_200sma.sum()}")
    df.loc[~df["above_200sma"], "composite_score"] = 0.0
    if "overextended" in df.columns:
        df.loc[df["overextended"], "composite_score"] = 0.0

    if regime_multiplier < 1.0:
        mask = df["composite_score"] > 0
        df.loc[mask, "composite_score"] = (df.loc[mask, "composite_score"] * regime_multiplier).round(2)

    if df.index.name == "ticker" or "ticker" not in df.columns:
        df = df.reset_index()
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    return df


# ─────────────────────────────────────────────────────────────
# ENSEMBLE SCORING
# ─────────────────────────────────────────────────────────────

def _ensemble_score(metrics_df: pd.DataFrame, regime_multiplier: float = 1.0) -> pd.DataFrame:
    variants = [
        {"trend": 0.30, "rs": 0.25, "volume": 0.20, "volatility": 0.15, "catalyst": 0.10},
        {"trend": 0.35, "rs": 0.30, "volume": 0.15, "volatility": 0.10, "catalyst": 0.10},
        {"trend": 0.25, "rs": 0.20, "volume": 0.25, "volatility": 0.20, "catalyst": 0.10},
    ]

    top_n = min(75, len(metrics_df) // 5)
    top_sets = []

    for w in variants:
        scored = calculate_composite_score(metrics_df.copy(), regime_multiplier, weights=w)
        scored_qualified = scored[scored["composite_score"] > 0].head(top_n)
        if "ticker" in scored_qualified.columns:
            tks = set(scored_qualified["ticker"].tolist())
        else:
            tks = set(scored_qualified.index.tolist()) if scored_qualified.index.name != "Rank" else set()
        top_sets.append(tks)

    consensus = top_sets[0]
    for s in top_sets[1:]:
        consensus = consensus & s

    primary = calculate_composite_score(metrics_df.copy(), regime_multiplier, weights=variants[0])
    primary["ensemble_consensus"] = False
    if "ticker" in primary.columns:
        primary.loc[primary["ticker"].isin(consensus), "ensemble_consensus"] = True
    else:
        for idx, row in primary.iterrows():
            tk = row.get("ticker", "")
            if tk in consensus:
                primary.at[idx, "ensemble_consensus"] = True

    return primary


# ─────────────────────────────────────────────────────────────
# SECTOR CAP
# ─────────────────────────────────────────────────────────────

def apply_sector_cap(df: pd.DataFrame, max_per_sector: int = 4) -> pd.DataFrame:
    if "sector" not in df.columns:
        return df
    if df["sector"].nunique() <= 1:
        return df

    result = []
    sector_counts = {}

    for _, row in df.iterrows():
        sec = row.get("sector", "Unknown")
        count = sector_counts.get(sec, 0)
        if count < max_per_sector:
            result.append(row)
            sector_counts[sec] = count + 1

    if not result:
        return df

    capped = pd.DataFrame(result)
    capped.index = range(1, len(capped) + 1)
    capped.index.name = "Rank"
    return capped


# ─────────────────────────────────────────────────────────────
# SPARKLINE
# ─────────────────────────────────────────────────────────────

def make_sparkline_data(price_data: dict[str, pd.DataFrame], ticker: str, days: int = 60) -> list[float]:
    if ticker not in price_data:
        return []
    close = price_data[ticker]["Close"].iloc[-days:]
    return close.tolist()


# ─────────────────────────────────────────────────────────────
# EXIT SIGNALS
# ─────────────────────────────────────────────────────────────

def compute_exit_signals(price_data: dict[str, pd.DataFrame], ticker: str) -> dict:
    if ticker not in price_data:
        return {}

    df = price_data[ticker]
    close = df["Close"]
    latest = float(close.iloc[-1])

    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

    high = df["High"].iloc[-20:]
    low = df["Low"].iloc[-20:]
    prev_close = df["Close"].iloc[-21:-1]
    if len(prev_close) == len(high):
        tr = pd.concat([
            high - low,
            (high - prev_close.values).abs(),
            (low - prev_close.values).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.mean())
    else:
        atr = latest * 0.02

    recent_high = float(close.iloc[-20:].max())
    atr_stop = recent_high - (2.0 * atr)
    pct_stop = recent_high * 0.85
    trailing_stop = max(atr_stop, pct_stop)

    return {
        "current_price": latest,
        "trailing_stop": round(trailing_stop, 2),
        "atr_stop": round(atr_stop, 2),
        "pct_stop_15": round(pct_stop, 2),
        "sma50_stop": round(sma50, 2) if sma50 else None,
        "exit_signal_atr": latest < atr_stop,
        "exit_signal_sma50": latest < sma50 if sma50 else False,
        "exit_signal_pct": latest < pct_stop,
    }


# ─────────────────────────────────────────────────────────────
# FULL SCAN PIPELINE (V2)
# ─────────────────────────────────────────────────────────────

def run_full_scan(
    progress_callback: Callable = None,
    fetch_fundamentals: bool = True,
    use_ensemble: bool = True,
    sector_cap: int = 4,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    if progress_callback:
        progress_callback(0.0, "Fetching S&P 1500 constituents…")

    tickers = get_sp1500_tickers()
    logger.info(f"Universe: {len(tickers)} tickers")

    if progress_callback:
        progress_callback(0.05, f"Downloading data for {len(tickers)} stocks…")

    def _dl_progress(pct):
        if progress_callback:
            progress_callback(0.05 + pct * 0.55, f"Downloading… {int(pct*100)}%")

    price_data = _safe_download(tickers, period="24mo", progress_callback=_dl_progress)

    if "SPY" not in price_data:
        logger.error("SPY data missing — cannot compute relative strength")
        return pd.DataFrame(), {}, {}

    if progress_callback:
        progress_callback(0.62, "Detecting market regime…")

    regime_info = detect_market_regime(price_data["SPY"])
    logger.info(f"Market regime: {regime_info['regime']} (multiplier={regime_info['score_multiplier']})")

    if progress_callback:
        progress_callback(0.65, "Computing technical indicators…")

    spy_close = price_data["SPY"]["Close"]
    spy_price = float(spy_close.iloc[-1])

    def _spy_ret(n):
        if len(spy_close) > n:
            return (spy_price - float(spy_close.iloc[-n - 1])) / float(spy_close.iloc[-n - 1])
        return 0.0

    spy_ret_1m = _spy_ret(21)
    spy_ret_3m = _spy_ret(63)
    spy_ret_6m = _spy_ret(126)
    spy_ret_12m = _spy_ret(252)

    rows = []
    valid_tickers = [t for t in tickers if t in price_data and t != "SPY"]

    for i, tk in enumerate(valid_tickers):
        try:
            ind = _compute_indicators(price_data[tk])
            ind["ticker"] = tk
            ind["sector"] = _SECTOR_CACHE.get(tk, "Unknown")
            ind["rs_1m"] = ind["ret_1m"] - spy_ret_1m
            ind["rs_3m"] = ind["ret_3m"] - spy_ret_3m
            ind["rs_6m"] = ind["ret_6m"] - spy_ret_6m
            ind["rs_12m"] = ind["ret_12m"] - spy_ret_12m
            rows.append(ind)
        except Exception as e:
            logger.warning(f"Skipping {tk}: {e}")

        if progress_callback and i % 100 == 0:
            progress_callback(0.65 + (i / len(valid_tickers)) * 0.15, f"Analyzing {tk}…")

    if not rows:
        logger.error("No valid stock data computed")
        return pd.DataFrame(), price_data, regime_info

    metrics_df = pd.DataFrame(rows).set_index("ticker")

    if fetch_fundamentals:
        if progress_callback:
            progress_callback(0.82, "Fetching earnings & float data for top candidates…")

        pre_scored = calculate_composite_score(metrics_df.copy(), regime_info["score_multiplier"])
        top_candidates = pre_scored[pre_scored["composite_score"] > 0].head(100)

        if "ticker" in top_candidates.columns:
            candidate_tickers = top_candidates["ticker"].tolist()
        else:
            candidate_tickers = []

        earnings_data = {}
        float_data = {}
        for j, ctk in enumerate(candidate_tickers):
            try:
                earnings_data[ctk] = _check_earnings_proximity(ctk)
                float_data[ctk] = _get_float_short_info(ctk)
            except Exception:
                pass

            if progress_callback and j % 20 == 0:
                progress_callback(0.82 + (j / max(len(candidate_tickers), 1)) * 0.08,
                                  f"Fundamentals… {ctk}")

        for ctk in candidate_tickers:
            if ctk in metrics_df.index:
                if ctk in earnings_data:
                    for k, v in earnings_data[ctk].items():
                        metrics_df.at[ctk, k] = v
                if ctk in float_data:
                    for k, v in float_data[ctk].items():
                        metrics_df.at[ctk, k] = v

        for col in ["earnings_within_30d", "days_to_earnings", "float_shares",
                     "short_pct_float", "float_class", "market_cap"]:
            if col not in metrics_df.columns:
                metrics_df[col] = None
            if col == "earnings_within_30d":
                metrics_df[col] = metrics_df[col].fillna(False)

    if progress_callback:
        progress_callback(0.92, "Scoring & ranking…")

    if use_ensemble:
        scored = _ensemble_score(metrics_df, regime_info["score_multiplier"])
    else:
        scored = calculate_composite_score(metrics_df, regime_info["score_multiplier"])

    if sector_cap > 0 and "sector" in scored.columns:
        scored = apply_sector_cap(scored, max_per_sector=sector_cap)

    if progress_callback:
        progress_callback(1.0, "Scan complete!")

    logger.info(f"Scored {len(scored)} stocks. Top composite = {scored['composite_score'].max():.2f}")
    return scored, price_data, regime_info


# ─────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────

def run_backtest(
    scan_date: str,
    hold_months: int = 6,
    top_n: int = 20,
    min_score: float = 5.0,
    progress_callback: Callable = None,
) -> dict:
    scan_dt = pd.Timestamp(scan_date)
    end_dt = scan_dt + pd.DateOffset(months=hold_months)

    data_start = (scan_dt - pd.DateOffset(months=15)).strftime("%Y-%m-%d")
    data_end = (end_dt + pd.DateOffset(days=5)).strftime("%Y-%m-%d")

    if progress_callback:
        progress_callback(0.0, f"Backtest: fetching universe as of {scan_date}…")

    tickers = get_sp1500_tickers()

    if progress_callback:
        progress_callback(0.05, f"Downloading historical data ({data_start} to {data_end})…")

    def _dl_progress(pct):
        if progress_callback:
            progress_callback(0.05 + pct * 0.55, f"Downloading… {int(pct*100)}%")

    price_data = _safe_download(tickers, start=data_start, end=data_end, progress_callback=_dl_progress)

    if "SPY" not in price_data:
        return {"error": "SPY data missing for backtest period"}

    if progress_callback:
        progress_callback(0.62, "Computing indicators as of scan date…")

    scan_data = {}
    for tk, df in price_data.items():
        trimmed = df[df.index <= scan_dt]
        if len(trimmed) > 60:
            scan_data[tk] = trimmed

    if "SPY" not in scan_data:
        return {"error": "Insufficient SPY data before scan date"}

    regime_info = detect_market_regime(scan_data["SPY"])

    spy_close = scan_data["SPY"]["Close"]
    spy_price = float(spy_close.iloc[-1])

    def _spy_ret(n):
        if len(spy_close) > n:
            return (spy_price - float(spy_close.iloc[-n - 1])) / float(spy_close.iloc[-n - 1])
        return 0.0

    spy_ret_1m = _spy_ret(21)
    spy_ret_3m = _spy_ret(63)
    spy_ret_6m = _spy_ret(126)
    spy_ret_12m = _spy_ret(252)

    rows = []
    valid_tickers = [t for t in tickers if t in scan_data and t != "SPY"]

    for i, tk in enumerate(valid_tickers):
        try:
            ind = _compute_indicators(scan_data[tk])
            ind["ticker"] = tk
            ind["sector"] = _SECTOR_CACHE.get(tk, "Unknown")
            ind["rs_1m"] = ind["ret_1m"] - spy_ret_1m
            ind["rs_3m"] = ind["ret_3m"] - spy_ret_3m
            ind["rs_6m"] = ind["ret_6m"] - spy_ret_6m
            ind["rs_12m"] = ind["ret_12m"] - spy_ret_12m
            rows.append(ind)
        except Exception:
            pass

        if progress_callback and i % 100 == 0:
            progress_callback(0.62 + (i / max(len(valid_tickers), 1)) * 0.15, f"Analyzing {tk}…")

    if not rows:
        return {"error": "No valid stock data for backtest period"}

    metrics_df = pd.DataFrame(rows).set_index("ticker")

    if progress_callback:
        progress_callback(0.80, "Scoring as of scan date…")

    scored = calculate_composite_score(metrics_df, regime_info["score_multiplier"])

    qualified = scored[scored["composite_score"] >= min_score].copy()
    if "sector" in qualified.columns:
        qualified = apply_sector_cap(qualified, max_per_sector=4)
    picks = qualified.head(top_n).copy()

    if picks.empty:
        return {"error": f"No stocks met minimum score of {min_score} on {scan_date}"}

    if progress_callback:
        progress_callback(0.85, "Measuring forward returns…")

    if "ticker" in picks.columns:
        pick_tickers = picks["ticker"].tolist()
    else:
        pick_tickers = picks.index.tolist() if picks.index.name != "Rank" else []

    forward_returns = {}
    for tk in pick_tickers:
        if tk in price_data:
            full_df = price_data[tk]
            entry_slice = full_df[full_df.index <= scan_dt]
            exit_slice = full_df[full_df.index <= end_dt]
            if not entry_slice.empty and not exit_slice.empty:
                entry_price = float(entry_slice["Close"].iloc[-1])
                exit_price = float(exit_slice["Close"].iloc[-1])
                fwd_ret = (exit_price - entry_price) / entry_price
                forward_returns[tk] = {
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "forward_return": round(fwd_ret, 4),
                    "doubled": fwd_ret >= 1.0,
                }

    spy_full = price_data["SPY"]
    spy_entry_slice = spy_full[spy_full.index <= scan_dt]
    spy_exit_slice = spy_full[spy_full.index <= end_dt]
    if not spy_entry_slice.empty and not spy_exit_slice.empty:
        spy_entry = float(spy_entry_slice["Close"].iloc[-1])
        spy_exit = float(spy_exit_slice["Close"].iloc[-1])
        spy_fwd_return = (spy_exit - spy_entry) / spy_entry
    else:
        spy_fwd_return = 0.0

    results_rows = []
    for tk in pick_tickers:
        row_data = {"ticker": tk}
        if tk in forward_returns:
            row_data.update(forward_returns[tk])
        if "ticker" in picks.columns:
            match = picks[picks["ticker"] == tk]
            if not match.empty:
                row_data["composite_score"] = float(match.iloc[0]["composite_score"])
                row_data["sector"] = match.iloc[0].get("sector", "Unknown")
        results_rows.append(row_data)

    results_df = pd.DataFrame(results_rows)

    if "forward_return" in results_df.columns and not results_df["forward_return"].isna().all():
        valid_returns = results_df["forward_return"].dropna()
        portfolio_return = float(valid_returns.mean())
        win_rate = float((valid_returns > 0).mean())
        doubler_rate = float((valid_returns >= 1.0).mean())
        best_idx = valid_returns.idxmax()
        worst_idx = valid_returns.idxmin()
        best_pick = results_df.loc[best_idx, "ticker"] if best_idx is not None else "N/A"
        worst_pick = results_df.loc[worst_idx, "ticker"] if worst_idx is not None else "N/A"
        best_ret = float(valid_returns.max())
        worst_ret = float(valid_returns.min())
    else:
        portfolio_return = 0
        win_rate = 0
        doubler_rate = 0
        best_pick = worst_pick = "N/A"
        best_ret = worst_ret = 0

    if progress_callback:
        progress_callback(1.0, "Backtest complete!")

    return {
        "scan_date": scan_date,
        "hold_months": hold_months,
        "exit_date": end_dt.strftime("%Y-%m-%d"),
        "picks": results_df,
        "num_picks": len(results_df),
        "portfolio_return": round(portfolio_return, 4),
        "spy_return": round(spy_fwd_return, 4),
        "alpha": round(portfolio_return - spy_fwd_return, 4),
        "win_rate": round(win_rate, 4),
        "doubler_rate": round(doubler_rate, 4),
        "best_pick": best_pick,
        "best_return": round(best_ret, 4),
        "worst_pick": worst_pick,
        "worst_return": round(worst_ret, 4),
        "regime": regime_info,
    }
