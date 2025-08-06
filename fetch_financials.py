# fetch_financials.py

import os
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime
from sqlalchemy import create_engine
import psycopg2
import logging

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "fetch_financials.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Connect to PostgreSQL using SQLAlchemy engine
# ----------------------------------------------------------
engine = create_engine("postgresql+psycopg2://greer_user@localhost/yfinance_db")

# ----------------------------------------------------------
# Connect to PostgreSQL database
# ----------------------------------------------------------
conn = psycopg2.connect(
    dbname="yfinance_db",
    user="greer_user",
    password="",
    host="localhost",
    port=5432
)

# ----------------------------------------------------------
# Function: Extract a financial metric from dataframe by tag
# ----------------------------------------------------------
def get_financial_value(df, possible_labels, year):
    df.index = df.index.str.strip().str.lower()
    for label in possible_labels:
        label_clean = label.strip().lower()
        if label_clean in df.index:
            return df.loc[label_clean].get(year, np.nan)
    return np.nan

# ----------------------------------------------------------
# Function: Fetch financial data for a ticker using yfinance
# ----------------------------------------------------------
def fetch_financial_data(ticker: str, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            income_stmt = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow

            if income_stmt.empty or balance_sheet.empty or cashflow.empty:
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

            for year in income_stmt.columns:
                year_str = year.strftime("%Y-%m-%d") if isinstance(year, pd.Timestamp) else str(year)

                total_assets = get_financial_value(balance_sheet, ["total assets"], year)
                total_liab = get_financial_value(balance_sheet, ["total liabilities", "total liabilities net minority interest", "liabilities"], year)
                shares_out = get_financial_value(balance_sheet, ["ordinary shares number", "common stock shares outstanding"], year)
                op_cash_flow = get_financial_value(cashflow, ["operating cash flow", "total cash from operating activities"], year)
                capex = get_financial_value(cashflow, ["capital expenditure"], year)
                revenue = get_financial_value(income_stmt, ["total revenue", "revenue", "operating revenue"], year)
                net_income = get_financial_value(income_stmt, ["net income", "net income common stockholders", "net income from continuing operations"], year)

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
            return df.sort_values("dates").reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error fetching data for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error(f"Failed to fetch data for {ticker}")
    return pd.DataFrame()

# ----------------------------------------------------------
# Function: Insert a row of financials into PostgreSQL
# ----------------------------------------------------------
def insert_financial_row(conn, ticker, row):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO financials (
                ticker, report_date, book_value_per_share, free_cash_flow, 
                net_margin, total_revenue, net_income, shares_outstanding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, report_date) DO NOTHING
        """, (
            ticker,
            row["dates"],
            row["BOOK_VALUE_PER_SHARE"],
            row["FREE_CASH_FLOW"],
            row["NET_MARGIN"],
            row["TOTAL_REVENUE"],
            row["NET_INCOME"],
            row["DILUTED_SHARES_OUTSTANDING"]
        ))
    conn.commit()

# ----------------------------------------------------------
# Function: Load tickers from file or companies table
# ----------------------------------------------------------
def load_tickers(file_path=None):
    if file_path:
        print(f"📄 Loading tickers from file: {file_path}")
        df = pd.read_csv(file_path)
        tickers = df["ticker"].dropna().str.upper().unique().tolist()
    else:
        print("🗃️  Loading tickers from companies table...")
        with engine.begin() as conn:
            tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
    return tickers

# ----------------------------------------------------------
# Main Execution
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch financials from yFinance and store in PostgreSQL")
    parser.add_argument("--file", type=str, help="Optional path to CSV file of tickers")
    args = parser.parse_args()

    tickers = load_tickers(args.file)
    print(f"✅ Loaded {len(tickers)} tickers\n")

    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        try:
            df = fetch_financial_data(ticker)
            if df.empty:
                print(f"⚠️ Skipping {ticker}: no data.")
                continue

            with conn.cursor() as cur:
                cur.execute("SELECT report_date FROM financials WHERE ticker = %s", (ticker,))
                existing_dates = [row[0] for row in cur.fetchall() if row[0] is not None]
                existing_dates = pd.to_datetime(existing_dates)

            df["dates"] = pd.to_datetime(df["dates"])
            new_rows = df[~df["dates"].isin(existing_dates)]

            if new_rows.empty:
                print(f"✅ No new data for {ticker}.")
            else:
                for _, row in new_rows.iterrows():
                    insert_financial_row(conn, ticker, row)
                print(f"✅ Inserted {len(new_rows)} new row(s) for {ticker}.")

        except Exception as e:
            logger.error(f"❌ Error processing {ticker}: {e}")
