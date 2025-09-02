# seed_companies.py
import os
import argparse
import pandas as pd
import yfinance as yf
from sqlalchemy import text
import logging

from db import get_engine  # ✅ centralized DB connection

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
logging.basicConfig(
    filename="logs/seed_companies.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Helper: fetch company info from yfinance
# ----------------------------------------------------------
def fetch_company_info(ticker: str) -> dict | None:
    try:
        tk = yf.Ticker(ticker)
        meta = {}
        try:
            meta = tk.get_info() or {}
        except Exception:
            meta = {}

        fast = {}
        try:
            fast = tk.fast_info or {}
        except Exception:
            fast = {}

        if not meta and not fast:
            return None

        return {
            "ticker": ticker.upper(),
            "name": meta.get("longName") or meta.get("shortName") or fast.get("shortName"),
            "sector": meta.get("sector"),
            "industry": meta.get("industry"),
            "exchange": meta.get("exchange") or fast.get("exchange"),
        }
    except Exception as e:
        logger.error(f"{ticker}: failed to fetch info ({e})")
        return None

# ----------------------------------------------------------
# Insert into DB
# ----------------------------------------------------------
def upsert_company(row: dict):
    sql = text("""
        INSERT INTO companies (ticker, name, sector, industry, exchange, delisted, delisted_date, added_at)
        VALUES (:ticker, :name, :sector, :industry, :exchange, FALSE, NULL, NOW())
        ON CONFLICT (ticker) DO NOTHING
    """)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(sql, row)

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed companies table from tickers.csv")
    parser.add_argument("--file", type=str, default="tickers.csv", help="CSV file with ticker column")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"CSV file not found: {args.file}")

    df = pd.read_csv(args.file, header=None, names=["ticker"])
    if "ticker" not in df.columns:
        raise ValueError("CSV must contain a 'ticker' column")

    tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
    print(f"📄 Loaded {len(tickers)} tickers from {args.file}")

    inserted = 0
    for t in tickers:
        info = fetch_company_info(t)
        if info:
            upsert_company(info)
            inserted += 1
            print(f"✅ {t} inserted")
            logger.info(f"{t}: inserted")
        else:
            print(f"⚠️ {t}: no info found")
            logger.warning(f"{t}: no info found")

    print(f"🎉 Done. Inserted {inserted}/{len(tickers)} tickers.")
