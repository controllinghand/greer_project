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

# Import shared DB connections
from db import get_engine

# Logging Setup
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "fetch_financials.log"),
    level=logging.INFO,  # Changed to INFO to log progress
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Initialize DB Connection
engine = get_engine()

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

# Function: Load tickers from file or companies table
def load_tickers(file_path=None):
    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = df["ticker"].dropna().str.upper().unique().tolist()
    else:
        print("🗃️ Loading tickers from companies table...")
        with engine.connect() as conn:
            tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
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
        with engine.connect() as conn:
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
            conn.commit()
        print(f"✅ Inserted {len(all_new_rows)} new rows into financials.")
        logger.info(f"Inserted {len(all_new_rows)} new rows into financials")

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch financials from yFinance and store in PostgreSQL")
    parser.add_argument("--file", type=str, help="Optional path to CSV file of tickers")
    args = parser.parse_args()

    tickers = load_tickers(args.file)
    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    process_tickers(tickers, max_workers=10)