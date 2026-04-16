"""
update_data.py — Daily Data Updater
=====================================
Downloads the latest price data and appends to the stored CSV.
Run by GitHub Actions daily, or manually.
"""

import os
import logging
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"
TICKERS_DIR = os.path.join(DATA_DIR, "tickers")
TICKERS_FILE = os.path.join(DATA_DIR, "tickers.csv")


def get_tickers() -> list[str]:
    """Get ticker list from stored file or fetch fresh."""
    if os.path.exists(TICKERS_FILE):
        df = pd.read_csv(TICKERS_FILE)
        return df["ticker"].tolist()

    # Fetch from GitHub CSV + Wikipedia
    tickers = set()
    try:
        import urllib.request, io
        req = urllib.request.Request(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response = urllib.request.urlopen(req, timeout=15)
        csv_data = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_data))
        for col in df.columns:
            if col.lower().strip() in ["symbol", "ticker"]:
                syms = df[col].dropna().astype(str).tolist()
                syms = [s.strip().replace(".", "-") for s in syms if 0 < len(s.strip()) < 10]
                tickers.update(syms)
                break
    except Exception as e:
        logger.warning(f"Could not fetch S&P 500 CSV: {e}")

    for url in [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    ]:
        try:
            import urllib.request, io
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
            response = urllib.request.urlopen(req, timeout=15)
            html = response.read().decode("utf-8")
            tables = pd.read_html(io.StringIO(html))
            for tbl in tables:
                for col in tbl.columns:
                    if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                        syms = tbl[col].dropna().astype(str).tolist()
                        syms = [s.strip().replace(".", "-") for s in syms if 0 < len(s.strip()) < 10]
                        if len(syms) > 20:
                            tickers.update(syms)
                            break
        except Exception:
            pass

    tickers.discard("SPY")
    tickers.add("SPY")
    ticker_list = sorted(tickers)

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame({"ticker": ticker_list}).to_csv(TICKERS_FILE, index=False)
    logger.info(f"Saved {len(ticker_list)} tickers to {TICKERS_FILE}")
    return ticker_list


def initial_download(tickers: list[str]):
    """Download full 24-month history for all tickers."""
    logger.info(f"Initial download: {len(tickers)} tickers, 24 months")
    all_tickers = tickers if "SPY" in tickers else ["SPY"] + tickers

    all_data = []
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="max", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if data.empty:
                continue
            if len(batch) == 1:
                tk = batch[0]
                df = data[["Open","High","Low","Close","Volume"]].dropna().copy()
                df["ticker"] = tk
                all_data.append(df)
            else:
                for tk in batch:
                    try:
                        if tk in data.columns.get_level_values(0):
                            df = data[tk][["Open","High","Low","Close","Volume"]].dropna().copy()
                            df["ticker"] = tk
                            all_data.append(df)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Batch failed: {e}")
        logger.info(f"  Downloaded {i+len(batch)}/{len(all_tickers)}")

    if all_data:
        combined = pd.concat(all_data)
        combined.index.name = "Date"
        combined = combined.reset_index()
        combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime("%Y-%m-%d")
        os.makedirs(TICKERS_DIR, exist_ok=True)
        count = 0
        for tk in combined["ticker"].unique():
            tk_df = combined[combined["ticker"] == tk][["Date","Open","High","Low","Close","Volume"]]
            safe = tk.replace("/", "_")
            tk_df.to_csv(os.path.join(TICKERS_DIR, f"{safe}.csv"), index=False)
            count += 1
        logger.info(f"Saved {count} individual ticker files to {TICKERS_DIR}")
    else:
        logger.error("No data downloaded")


def daily_update(tickers: list[str]):
    """Download only the latest 5 days and append new rows."""
    if not os.path.isdir(TICKERS_DIR):
        logger.info("No existing data — running initial download")
        initial_download(tickers)
        return

    # Find last date from SPY file
    spy_file = os.path.join(TICKERS_DIR, "SPY.csv")
    if os.path.exists(spy_file):
        spy_df = pd.read_csv(spy_file)
        spy_df["Date"] = pd.to_datetime(spy_df["Date"])
        last_date = spy_df["Date"].max()
    else:
        last_date = pd.Timestamp.now() - pd.Timedelta(days=7)
    logger.info(f"Last date in files: {last_date.strftime('%Y-%m-%d')}")

    all_tickers = tickers if "SPY" in tickers else ["SPY"] + tickers
    new_data = []
    batch_size = 50

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period="5d", group_by="ticker", auto_adjust=True, threads=True, progress=False)
            if data.empty:
                continue
            if len(batch) == 1:
                tk = batch[0]
                df = data[["Open","High","Low","Close","Volume"]].dropna().copy()
                df["ticker"] = tk
                new_data.append(df)
            else:
                for tk in batch:
                    try:
                        if tk in data.columns.get_level_values(0):
                            df = data[tk][["Open","High","Low","Close","Volume"]].dropna().copy()
                            df["ticker"] = tk
                            new_data.append(df)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Batch failed: {e}")

    if new_data:
        combined = pd.concat(new_data)
        combined.index.name = "Date"
        combined = combined.reset_index()
        combined["Date"] = pd.to_datetime(combined["Date"])

        new_rows = combined[combined["Date"] > last_date]
        if not new_rows.empty:
            os.makedirs(TICKERS_DIR, exist_ok=True)
            updated_count = 0
            for tk in new_rows["ticker"].unique():
                tk_new = new_rows[new_rows["ticker"] == tk][["Date","Open","High","Low","Close","Volume"]].copy()
                tk_new["Date"] = tk_new["Date"].dt.strftime("%Y-%m-%d")
                safe = tk.replace("/", "_")
                tk_file = os.path.join(TICKERS_DIR, f"{safe}.csv")
                if os.path.exists(tk_file):
                    existing_tk = pd.read_csv(tk_file)
                    updated_tk = pd.concat([existing_tk, tk_new], ignore_index=True)
                    updated_tk.to_csv(tk_file, index=False)
                else:
                    tk_new.to_csv(tk_file, index=False)
                updated_count += 1
            logger.info(f"Updated {updated_count} ticker files with new data")
        else:
            logger.info("No new data to append — already up to date")
    else:
        logger.warning("Could not download any new data")


if __name__ == "__main__":
    import sys
    tickers = get_tickers()

    if "--init" in sys.argv:
        initial_download(tickers)
    else:
        daily_update(tickers)
