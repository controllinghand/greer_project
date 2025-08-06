# greer_value_score.py

import os
import time
import argparse
import numpy as np
import pandas as pd
import psycopg2
import logging

from datetime import datetime
from sqlalchemy import create_engine

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "greer_value_score.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Connect to PostgreSQL using SQLAlchemy engine
# ----------------------------------------------------------
engine = create_engine("postgresql+psycopg2://greer_user@localhost/yfinance_db")

# ----------------------------------------------------------
# Connect to PostgreSQL using psycopg2
# ----------------------------------------------------------
conn = psycopg2.connect(
    dbname="yfinance_db",
    user="greer_user",
    password="",  # Add password if set
    host="localhost",
    port=5432
)

# ----------------------------------------------------------
# Function: Calculate Greer Value % based on metric trend
# ----------------------------------------------------------
def calculate_greer_value(data: pd.DataFrame, metric: str, weighting: str = "Exponential", exp_alpha: float = 0.6) -> float:
    if data.empty or len(data) < 2:
        return np.nan

    changes = []
    prev_value = None
    for value in data[metric]:
        if prev_value is not None and not np.isnan(value) and not np.isnan(prev_value):
            if metric == "DILUTED_SHARES_OUTSTANDING":
                changes.append(1.0 if value < prev_value and value > 0 else 0.0)
            else:
                changes.append(1.0 if value > prev_value and value > 0 else 0.0)
        prev_value = value

    if not changes:
        return np.nan

    def simple_avg(arr): return np.mean(arr) * 100.0
    def linear_weighted_avg(arr): 
        weights = np.arange(1, len(arr) + 1)
        return np.sum(np.array(arr) * weights) / np.sum(weights) * 100.0
    def exp_weighted_avg(arr, alpha): 
        weights = np.array([alpha ** (len(arr) - 1 - i) for i in range(len(arr))])
        return np.sum(np.array(arr) * weights) / np.sum(weights) * 100.0

    return (exp_weighted_avg(changes, exp_alpha) if weighting == "Exponential" else
            linear_weighted_avg(changes) if weighting == "Linear" else
            simple_avg(changes))

# ----------------------------------------------------------
# Function: Compute Greer Value Score from multiple metrics
# ----------------------------------------------------------
def compute_greer_value_score(ticker: str, data: pd.DataFrame) -> dict:
    metrics = ["BOOK_VALUE_PER_SHARE", "FREE_CASH_FLOW", "NET_MARGIN", 
               "TOTAL_REVENUE", "NET_INCOME", "DILUTED_SHARES_OUTSTANDING"]
    results = {}
    valid_metrics = 0
    total_pct = 0.0
    above_50_count = 0

    for metric in metrics:
        pct = calculate_greer_value(data, metric)
        results[metric] = {"pct": pct}
        if not np.isnan(pct):
            valid_metrics += 1
            total_pct += pct
            if pct >= 50:
                above_50_count += 1

    greer_value = total_pct / valid_metrics if valid_metrics > 0 else np.nan
    results["GreerValue"] = {"score": greer_value, "above_50_count": above_50_count}
    return {ticker: results}

# ----------------------------------------------------------
# Function: Load a list of tickers from a given file
# ----------------------------------------------------------
def load_tickers_from_file(filename):
    tickers = []
    with open(filename, "r") as f:
        for line in f:
            ticker = line.strip().upper()
            if ticker:
                tickers.append(ticker)
    return tickers

# ----------------------------------------------------------
# Function: Load historical financial data for a ticker
# ----------------------------------------------------------
def load_data_from_db(conn, ticker):
    query = """
        SELECT report_date, 
               book_value_per_share, 
               free_cash_flow, 
               net_margin, 
               total_revenue, 
               net_income, 
               shares_outstanding
        FROM financials
        WHERE ticker = %s
        ORDER BY report_date ASC
    """
    df = pd.read_sql(query, engine, params=(ticker,))
    df.rename(columns={
        "report_date": "dates",
        "book_value_per_share": "BOOK_VALUE_PER_SHARE",
        "free_cash_flow": "FREE_CASH_FLOW",
        "net_margin": "NET_MARGIN",
        "total_revenue": "TOTAL_REVENUE",
        "net_income": "NET_INCOME",
        "shares_outstanding": "DILUTED_SHARES_OUTSTANDING"
    }, inplace=True)
    return df

# ----------------------------------------------------------
# Function: Insert or update Greer scores into DB
# ----------------------------------------------------------
def insert_greer_score(conn, row, report_date):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO greer_scores (
                ticker, report_date, greer_score, above_50_count,
                book_pct, fcf_pct, margin_pct, revenue_pct, income_pct, shares_pct
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, report_date) DO UPDATE
            SET
                greer_score = EXCLUDED.greer_score,
                above_50_count = EXCLUDED.above_50_count,
                book_pct = EXCLUDED.book_pct,
                fcf_pct = EXCLUDED.fcf_pct,
                margin_pct = EXCLUDED.margin_pct,
                revenue_pct = EXCLUDED.revenue_pct,
                income_pct = EXCLUDED.income_pct,
                shares_pct = EXCLUDED.shares_pct,
                created_at = CURRENT_TIMESTAMP;
        """, (
            row["Ticker"],
            report_date,
            row["Greer Score"],
            row["Above 50%"],
            row["Book%"],
            row["FCF%"],
            row["Margin%"],
            row["Revenue%"],
            row["Income%"],
            row["Shares%"]
        ))
    conn.commit()

# ----------------------------------------------------------
# Convert any numpy types to native Python types
# ----------------------------------------------------------
def convert_numpy_types(row_dict):
    return {
        k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
        for k, v in row_dict.items()
    }

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
parser = argparse.ArgumentParser(description="Greer Value Analyzer from DB")
parser.add_argument("--file", type=str, help="Optional path to CSV file containing tickers (column: 'ticker')")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)
summary_rows = []
tickers = load_tickers(args.file)
print(f"\n✅ Loaded {len(tickers)} tickers\n")

for ticker in tickers:
    try:
        print(f"📊 Processing {ticker}...")
        df_full = load_data_from_db(conn, ticker)
        if df_full.empty:
            print(f"⚠️ No financial data for {ticker} in DB.")
            continue

        df_full.to_csv(f"data/{ticker}_data.csv", index=False)

        for report_date in df_full["dates"].sort_values().unique():
            df_subset = df_full[df_full["dates"] <= report_date]
            result = compute_greer_value_score(ticker, df_subset)
            r = result[ticker]

            row = {
                "Ticker": ticker,
                "Greer Score": round(r["GreerValue"]["score"], 2),
                "Above 50%": r["GreerValue"]["above_50_count"],
                "Book%": round(r["BOOK_VALUE_PER_SHARE"]["pct"], 2) if not np.isnan(r["BOOK_VALUE_PER_SHARE"]["pct"]) else None,
                "FCF%": round(r["FREE_CASH_FLOW"]["pct"], 2) if not np.isnan(r["FREE_CASH_FLOW"]["pct"]) else None,
                "Margin%": round(r["NET_MARGIN"]["pct"], 2) if not np.isnan(r["NET_MARGIN"]["pct"]) else None,
                "Revenue%": round(r["TOTAL_REVENUE"]["pct"], 2) if not np.isnan(r["TOTAL_REVENUE"]["pct"]) else None,
                "Income%": round(r["NET_INCOME"]["pct"], 2) if not np.isnan(r["NET_INCOME"]["pct"]) else None,
                "Shares%": round(r["DILUTED_SHARES_OUTSTANDING"]["pct"], 2) if not np.isnan(r["DILUTED_SHARES_OUTSTANDING"]["pct"]) else None,
            }

            summary_rows.append(row)
            insert_greer_score(conn, convert_numpy_types(row), report_date)
            print(f"✅ Inserted Greer score for {ticker} (as of {report_date})")

    except Exception as e:
        logger.error(f"❌ Error processing {ticker}: {e}")

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("greer_summary.csv", index=False)
print("\n✅ Saved summary to greer_summary.csv")
print("\n📊 Greer Value Summary:")
print(summary_df)
print("🎯 Unique tickers processed:", summary_df["Ticker"].nunique())
