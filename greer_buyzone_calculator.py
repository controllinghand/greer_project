# greer_buyzone_calculator.py
import pandas as pd
import numpy as np
import argparse
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
from db import get_engine

# Logging Setup
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "greer_buyzone_calculator.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Shared DB Engine
engine = get_engine()

# Load price data
def load_price_data(ticker: str):
    query = text("""
        SELECT date, close, high_price AS high, low_price AS low
        FROM prices
        WHERE ticker = :ticker
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'ticker': ticker})
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Calculate BuyZone logic (vectorized, aligned with Pine Script)
def calculate_buyzone(df: pd.DataFrame, aroon_length: int = 233):
    window_size = aroon_length + 1  # 234 bars, including current
    result = []

    # Vectorized rolling computations
    df['high_rolling'] = df['high'].rolling(window=window_size).max()
    df['low_rolling'] = df['low'].rolling(window=window_size).min()
    df['aroon_up'] = 100 * (aroon_length - df['high'].rolling(window=window_size).apply(lambda x: len(x) - 1 - np.argmax(x), raw=True)) / aroon_length
    df['aroon_down'] = 100 * (aroon_length - df['low'].rolling(window=window_size).apply(lambda x: len(x) - 1 - np.argmin(x), raw=True)) / aroon_length
    df['midpoint'] = (df['high_rolling'] + df['low_rolling']) / 2

    # Drop rows before full window
    df = df.dropna().copy()

    # State tracking for buy zone
    in_buyzone = False
    for _, row in df.iterrows():
        buyzone_start = False
        buyzone_end = False
        if row['aroon_down'] == 100 and not in_buyzone:
            buyzone_start = True
            in_buyzone = True
        elif row['aroon_up'] == 100 and in_buyzone:
            buyzone_end = True
            in_buyzone = False

        result.append({
            'date': row['date'],
            'close_price': row['close'],
            'high': row['high_rolling'],
            'low': row['low_rolling'],
            'aroon_upper': row['aroon_up'],
            'aroon_lower': row['aroon_down'],
            'midpoint': row['midpoint'],
            'buyzone_start': buyzone_start,
            'buyzone_end': buyzone_end,
            'in_buyzone': in_buyzone,
            'in_sellzone': not in_buyzone
        })

    return pd.DataFrame(result)

# Save results to DB (bulk upsert)
def save_to_db(ticker: str, df: pd.DataFrame):
    if df.empty:
        return
    query = text("""
        INSERT INTO greer_buyzone_daily (
            ticker, date, close_price, high, low,
            aroon_upper, aroon_lower, midpoint,
            buyzone_start, buyzone_end, in_buyzone, in_sellzone
        )
        VALUES (
            :ticker, :date, :close_price, :high, :low,
            :aroon_upper, :aroon_lower, :midpoint,
            :buyzone_start, :buyzone_end, :in_buyzone, :in_sellzone
        )
        ON CONFLICT (ticker, date) DO UPDATE SET
            close_price = EXCLUDED.close_price,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            aroon_upper = EXCLUDED.aroon_upper,
            aroon_lower = EXCLUDED.aroon_lower,
            midpoint = EXCLUDED.midpoint,
            buyzone_start = EXCLUDED.buyzone_start,
            buyzone_end = EXCLUDED.buyzone_end,
            in_buyzone = EXCLUDED.in_buyzone,
            in_sellzone = EXCLUDED.in_sellzone
    """)
    with engine.connect() as conn:
        conn.execute(query, [
            {
                'ticker': ticker,
                'date': row['date'],
                'close_price': row['close_price'],
                'high': row['high'],
                'low': row['low'],
                'aroon_upper': row['aroon_upper'],
                'aroon_lower': row['aroon_lower'],
                'midpoint': row['midpoint'],
                'buyzone_start': row['buyzone_start'],
                'buyzone_end': row['buyzone_end'],
                'in_buyzone': row['in_buyzone'],
                'in_sellzone': row['in_sellzone']
            } for _, row in df.iterrows()
        ])
        conn.commit()

# Process a single ticker
def process_ticker(ticker: str):
    print(f"📊 Processing {ticker}...")
    try:
        df = load_price_data(ticker)
        if len(df) < 250:
            print(f"⚠️ Not enough data for {ticker}")
            logger.info(f"Not enough data for {ticker}")
            return
        result_df = calculate_buyzone(df)
        save_to_db(ticker, result_df)
        print(f"✅ Finished {ticker}")
        logger.info(f"Finished {ticker}")
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
        logger.error(f"Error processing {ticker}: {e}")

# Load tickers from file or DB
def load_tickers_from_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def load_tickers_from_db():
    df = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", engine)
    return df["ticker"].dropna().str.upper().tolist()

# Process tickers in parallel with memory management
def process_tickers(tickers, max_workers=2):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {ticker} in parallel: {e}")
            # Clear memory
            import gc
            gc.collect()

# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Greer BuyZone calculations.")
    parser.add_argument("--tickers", nargs="+", help="List of tickers")
    parser.add_argument("--file", type=str, help="Path to file with tickers")
    parser.add_argument("--max_workers", type=int, default=2, help="Max concurrent workers")
    args = parser.parse_args()

    if args.file:
        tickers = load_tickers_from_file(args.file)
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        print("🗃️ Loading tickers from companies table...")
        tickers = load_tickers_from_db()

    print(f"✅ Loaded {len(tickers)} tickers")
    process_tickers(tickers, args.max_workers)