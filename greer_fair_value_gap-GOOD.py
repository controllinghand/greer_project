# greer_fair_value_gap.py
import pandas as pd
import numpy as np
import argparse
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
DB_CONN_STRING = "postgresql://greer_user@localhost:5432/yfinance_db"
engine = create_engine(DB_CONN_STRING)

# ----------------------------------------------------------
# Load price data from the database
# ----------------------------------------------------------
def load_price_data(ticker, start_date=None):
    query = """
        SELECT date, high_price, low_price, close
        FROM prices
        WHERE ticker = :ticker
    """
    params = {"ticker": ticker}
    if start_date:
        query += " AND date >= :start_date"
        params["start_date"] = start_date
    query += " ORDER BY date ASC"

    with engine.connect() as conn:
        df = pd.read_sql(sqlalchemy.text(query), conn, params=params)

    return df if not df.empty else pd.DataFrame()

# ----------------------------------------------------------
# Get the most recent FVG date
# ----------------------------------------------------------
def get_latest_fvg_date(ticker):
    query = "SELECT MAX(date) FROM fair_value_gaps WHERE ticker = :ticker"
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(query), {"ticker": ticker})
        row = result.fetchone()
        return row[0] if row and row[0] else None

# ----------------------------------------------------------
# Detect Fair Value Gaps
# ----------------------------------------------------------
def detect_fvg(df, threshold=0.0, auto=False):
    fvg_data = []
    bars = df.reset_index(drop=True)

    if auto:
        bars['volatility'] = (bars['high_price'] - bars['low_price']) / bars['low_price']
        threshold = bars['volatility'].mean()

    for i in range(2, len(bars)):
        bar_2, bar_1, bar_0 = bars.iloc[i-2], bars.iloc[i-1], bars.iloc[i]

        # Bullish FVG
        if bar_0['low_price'] > bar_2['high_price'] and bar_1['close'] > bar_2['high_price']:
            gap = (bar_0['low_price'] - bar_2['high_price']) / bar_2['high_price']
            if gap > threshold:
                fvg_data.append({
                    'date': bar_0['date'],
                    'is_bullish': True,
                    'low': bar_0['low_price'],
                    'high': bar_2['high_price'],
                    'mitigated': False
                })

        # Bearish FVG
        if bar_0['high_price'] < bar_2['low_price'] and bar_1['close'] < bar_2['low_price']:
            gap = (bar_2['low_price'] - bar_0['high_price']) / bar_0['high_price']
            if gap > threshold:
                fvg_data.append({
                    'date': bar_0['date'],
                    'is_bullish': False,
                    'low': bar_2['low_price'],
                    'high': bar_0['high_price'],
                    'mitigated': False
                })

    return fvg_data

# ----------------------------------------------------------
# Evaluate Mitigation
# ----------------------------------------------------------
def evaluate_mitigation(df, fvg_data):
    for fvg in fvg_data:
        subset = df[df['date'] > fvg['date']]
        if fvg['is_bullish']:
            if any(subset['close'] < fvg['low']):
                fvg['mitigated'] = True
        else:
            if any(subset['close'] > fvg['high']):
                fvg['mitigated'] = True
    return fvg_data

# ----------------------------------------------------------
# Insert FVGs into the database
# ----------------------------------------------------------
def insert_fvgs_to_db(ticker, fvg_data):
    conn = psycopg2.connect("dbname=yfinance_db user=greer_user")
    cur = conn.cursor()
    for fvg in fvg_data:
        direction = 'bullish' if fvg['is_bullish'] else 'bearish'
        gap_min = float(min(fvg['low'], fvg['high']))
        gap_max = float(max(fvg['low'], fvg['high']))
        cur.execute("""
            INSERT INTO fair_value_gaps (ticker, date, direction, gap_min, gap_max, mitigated)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (
            ticker, fvg['date'], direction, gap_min, gap_max, fvg['mitigated']
        ))
    conn.commit()
    cur.close()
    conn.close()

# ----------------------------------------------------------
# Process a single ticker
# ----------------------------------------------------------
def process_ticker(ticker, threshold, auto, full_history=False):
    print(f"\n📊 Processing {ticker}...")

    if not full_history:
        latest_fvg_date = get_latest_fvg_date(ticker)
        start_date = None
        if latest_fvg_date:
            start_date = (latest_fvg_date - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
    else:
        start_date = None

    df = load_price_data(ticker, start_date=start_date)

    if df is None or df.empty or len(df) < 3:
        print(f"⚠️ Not enough data for {ticker}")
        return

    fvg_data = detect_fvg(df, threshold, auto)
    fvg_data = evaluate_mitigation(df, fvg_data)
    insert_fvgs_to_db(ticker, fvg_data)

    print("\n📈 Unmitigated FVGs:")
    for fvg in fvg_data:
        if not fvg['mitigated']:
            direction = 'Bullish' if fvg['is_bullish'] else 'Bearish'
            print(f"{fvg['date']} | {direction} | {fvg['low']} → {fvg['high']}")

# ----------------------------------------------------------
# Load tickers from file or DB
# ----------------------------------------------------------
def load_tickers(args):
    if args.file:
        print(f"📄 Loading tickers from file: {args.file}")
        with open(args.file, "r") as f:
            return [line.strip().upper() for line in f if line.strip()]
    elif args.ticker:
        print(f"🎯 Loading single ticker: {args.ticker.upper()}")
        return [args.ticker.upper()]
    else:
        print("🗃️  Loading tickers from companies table...")
        with engine.begin() as conn:
            df = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)
            return df["ticker"].dropna().str.upper().tolist()

# ----------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greer Fair Value Gap Calculator")
    parser.add_argument("--file", type=str, help="CSV file with tickers")
    parser.add_argument("--ticker", type=str, help="Single ticker to process")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum gap threshold (0.01 = 1%)")
    parser.add_argument("--auto", action="store_true", help="Use auto volatility threshold")
    parser.add_argument("--full", action="store_true", help="Force full history rerun")
    args = parser.parse_args()

    tickers = load_tickers(args)
    print(f"✅ Loaded {len(tickers)} tickers")

    for ticker in tickers:
        process_ticker(ticker, args.threshold, args.auto, args.full)
