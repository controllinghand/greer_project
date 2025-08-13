# fetch_financials.py
import os
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from datetime import datetime
from pytz import timezone
import logging

# ----------------------------------------------------------
# DB helpers
# ----------------------------------------------------------
from db import get_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "fetch_financials.log"),
    level=logging.INFO,  # INFO for progress
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Initialize DB Connection (global engine)
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def parse_tickers_arg(raw: str | None) -> list[str]:
    """
    Parse a --tickers string into a list. Accepts comma and/or whitespace.
    Example: "AAPL, MSFT TSLA" -> ["AAPL","MSFT","TSLA"]
    """
    if not raw:
        return []
    # Replace commas with spaces, split, strip, upper, dedup while preserving order
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out

# Function: Extract a financial metric from dataframe by tag
def get_financial_value(df, possible_labels, year):
    df.index = df.index.str.strip().str.lower()
    for label in possible_labels:
        label_clean = label.strip().lower()
        if label_clean in df.index:
            return df.loc[label_clean].get(year, np.nan)
    return np.nan

# Function: Fetch financial data for a ticker using yfinance
def fetch_financial_data(ticker: str, retries: int = 3, delay: int = 2) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            # Fetch all financials in one call
            financials = stock.get_financials()
            balance_sheet = stock.get_balance_sheet()
            cashflow = stock.get_cashflow()

            if financials.empty or balance_sheet.empty or cashflow.empty:
                raise ValueError("Missing financial statement(s)")

            data = {
                "BOOK_VALUE_PER_SHARE": [],
                "FREE_CASH_FLOW": [],
                "NET_MARGIN": [],
                "TOTAL_REVENUE": [],
                "NET_INCOME": [],
                "DILUTED_SHARES_OUTSTANDING": [],
                "dates": []
            }

            for year in financials.columns:
                year_str = year.strftime("%Y-%m-%d") if isinstance(year, pd.Timestamp) else str(year)

                total_assets = get_financial_value(balance_sheet, ["total assets"], year)
                total_liab = get_financial_value(balance_sheet, ["total liabilities", "total liabilities net minority interest", "liabilities"], year)
                shares_out = get_financial_value(balance_sheet, ["ordinary shares number", "common stock shares outstanding"], year)
                op_cash_flow = get_financial_value(cashflow, ["operating cash flow", "total cash from operating activities"], year)
                capex = get_financial_value(cashflow, ["capital expenditure"], year)
                revenue = get_financial_value(financials, ["total revenue", "revenue", "operating revenue"], year)
                net_income = get_financial_value(financials, ["net income", "net income common stockholders", "net income from continuing operations"], year)

                book_value = (total_assets - total_liab) / shares_out if all(not np.isnan(x) and x != 0 for x in [total_assets, total_liab, shares_out]) else np.nan
                free_cash_flow = op_cash_flow + capex if all(not np.isnan(x) for x in [op_cash_flow, capex]) else np.nan
                net_margin = (net_income / revenue) * 100 if all(not np.isnan(x) and revenue != 0 for x in [net_income, revenue]) else np.nan

                if all(np.isnan(x) for x in [book_value, free_cash_flow, net_margin, revenue, net_income]):
                    continue

                data["BOOK_VALUE_PER_SHARE"].append(book_value)
                data["FREE_CASH_FLOW"].append(free_cash_flow)
                data["NET_MARGIN"].append(net_margin)
                data["TOTAL_REVENUE"].append(revenue)
                data["NET_INCOME"].append(net_income)
                data["DILUTED_SHARES_OUTSTANDING"].append(shares_out)
                data["dates"].append(year_str)

            df = pd.DataFrame(data)
            df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
            df["ticker"] = ticker
            return df.sort_values("dates").reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error fetching data for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error(f"Failed to fetch data for {ticker}")
    return pd.DataFrame()

# Function: Load tickers from file or companies table, with explicit override list
def load_tickers(file_path: str | None, explicit_tickers: list[str] | None) -> list[str]:
    """
    Priority:
      1) explicit_tickers (from --tickers)
      2) file_path (CSV with 'ticker' column)
      3) companies table (default)
    """
    if explicit_tickers:
        print(f"🎯 Using explicit tickers: {', '.join(explicit_tickers)}")
        logger.info(f"Using explicit tickers: {explicit_tickers}")
        return explicit_tickers

    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Could not read file {file_path}: {e}")
        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .str.strip()
            .unique()
            .tolist()
        )
        logger.info(f"Loaded {len(tickers)} tickers from file")
        return tickers

    print("🗃️ Loading tickers from companies table…")
    with engine.connect() as conn:
        tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
    logger.info(f"Loaded {len(tickers)} tickers from companies table")
    return tickers

# Function: Fetch existing dates from financials table for all tickers
def get_existing_dates():
    query = text("SELECT ticker, report_date FROM financials")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.groupby("ticker")["report_date"].apply(list).to_dict()

# Function: Process tickers in parallel
def process_tickers(tickers, max_workers=10):
    start_time = time.time()
    existing_dates = get_existing_dates()
    new_rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_financial_data, ticker): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if df.empty:
                    print(f"⚠️ Skipping {ticker}: no data.")
                    logger.info(f"Skipping {ticker}: no data")
                    continue

                # Filter out existing dates
                ticker_existing_dates = pd.to_datetime(existing_dates.get(ticker, []))
                df["dates"] = pd.to_datetime(df["dates"])
                new_rows_df = df[~df["dates"].isin(ticker_existing_dates)]

                if new_rows_df.empty:
                    print(f"✅ No new data for {ticker}.")
                    logger.info(f"No new data for {ticker}")
                else:
                    new_rows.append(new_rows_df)
                    print(f"✅ Collected {len(new_rows_df)} new row(s) for {ticker}.")
                    logger.info(f"Collected {len(new_rows_df)} new row(s) for {ticker}")

            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                logger.error(f"Error processing {ticker}: {e}")

    # Combine all new rows and insert in bulk
    if new_rows:
        all_new_rows = pd.concat(new_rows, ignore_index=True)
        query = text("""
            INSERT INTO financials (
                ticker, report_date, book_value_per_share, free_cash_flow, 
                net_margin, total_revenue, net_income, shares_outstanding
            )
            VALUES (:ticker, :report_date, :book_value_per_share, :free_cash_flow, 
                    :net_margin, :total_revenue, :net_income, :shares_outstanding)
            ON CONFLICT (ticker, report_date) DO NOTHING
        """)

        # Use a transaction block
        with engine.begin() as conn:
            conn.execute(
                query,
                [
                    {
                        "ticker": row["ticker"],
                        "report_date": row["dates"],
                        "book_value_per_share": row["BOOK_VALUE_PER_SHARE"],
                        "free_cash_flow": row["FREE_CASH_FLOW"],
                        "net_margin": row["NET_MARGIN"],
                        "total_revenue": row["TOTAL_REVENUE"],
                        "net_income": row["NET_INCOME"],
                        "shares_outstanding": row["DILUTED_SHARES_OUTSTANDING"]
                    }
                    for _, row in all_new_rows.iterrows()
                ]
            )
        print(f"✅ Inserted {len(all_new_rows)} new rows into financials.")
        logger.info(f"Inserted {len(all_new_rows)} new rows into financials")

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")
    logger.info(f"Total processing time: {elapsed:.2f} seconds")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch financials from yfinance and store in PostgreSQL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV file of tickers (expects 'ticker' column)")
    group.add_argument(
        "--tickers",
        type=str,
        help="Optional comma/space separated list of tickers (e.g., \"AAPL,MSFT TSLA\")"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max parallel workers (default: 10)"
    )
    args = parser.parse_args()

    explicit_tickers = parse_tickers_arg(args.tickers)
    tickers = load_tickers(args.file, explicit_tickers)

    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    process_tickers(tickers, max_workers=args.workers)
