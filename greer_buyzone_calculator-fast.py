import pandas as pd
import numpy as np
import argparse
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use shared DB access module
from db import get_engine, get_psycopg_connection

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "greer_buyzone_calculator.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# ----------------------------------------------------------
# Shared DB Engine
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Load price data
# ----------------------------------------------------------
def load_price_data(ticker: str):
    query = f"""
        SELECT date, close, high_price AS high, low_price AS low
        FROM prices
        WHERE ticker = '{ticker}'
        ORDER BY date ASC
    """
    df = pd.read_sql(query, engine)
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Optional: Load all prices at once (enable if RAM allows, ~370 MB for 1000 tickers)
def load_all_prices(tickers):
    query = f"""
        SELECT ticker, date, close, high_price AS high, low_price AS low
        FROM prices
        WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
        ORDER BY ticker, date ASC
    """
    df = pd.read_sql(query, engine)
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(inplace=True)
    return df

# ----------------------------------------------------------
# Calculate BuyZone logic (vectorized)
# ----------------------------------------------------------
def calculate_buyzone(df: pd.DataFrame, aroon_length: int = 233):
    window_size = aroon_length + 1  # Matches original window length
    
    # Vectorized rolling computations
    df['high_rolling'] = df['high'].rolling(window=window_size).max()
    df['low_rolling'] = df['low'].rolling(window=window_size).min()
    df['aroon_up'] = 100 * (aroon_length - df['high'].rolling(window=window_size).apply(np.argmax, raw=True)) / aroon_length
    df['aroon_down'] = 100 * (aroon_length - df['low'].rolling(window=window_size).apply(np.argmin, raw=True)) / aroon_length
    df['midpoint'] = (df['high_rolling'] + df['low_rolling']) / 2
    
    # Drop rows before full window
    df = df.dropna().copy()
    
    # Loop only for state (fast, scalar operations)
    in_zone = False
    result = []
    for _, row in df.iterrows():
        buyzone_start = False
        buyzone_end = False
        if row['aroon_down'] == 100 and not in_zone:
            buyzone_start = True
            in_zone = True
        elif row['aroon_up'] == 100 and in_zone:
            buyzone_end = True
            in_zone = False
        
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
            'in_buyzone': in_zone,
            'in_sellzone': not in_zone
        })
    
    return pd.DataFrame(result)

# ----------------------------------------------------------
# Save results to DB (bulk upsert)
# ----------------------------------------------------------
def save_to_db(ticker: str, df: pd.DataFrame):
    try:
        with get_psycopg_connection() as conn:
            with conn.cursor() as cursor:
                # Prepare data as list of tuples
                data = [
                    (
                        ticker,
                        row['date'],
                        row['close_price'],
                        row['high'],
                        row['low'],
                        row['aroon_upper'],
                        row['aroon_lower'],
                        row['midpoint'],
                        row['buyzone_start'],
                        row['buyzone_end'],
                        row['in_buyzone'],
                        row['in_sellzone']
                    ) for _, row in df.iterrows()
                ]
                
                # Bulk upsert
                query = """
                    INSERT INTO greer_buyzone_daily (
                        ticker, date, close_price, high, low,
                        aroon_upper, aroon_lower, midpoint,
                        buyzone_start, buyzone_end, in_buyzone, in_sellzone
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        in_sellzone = EXCLUDED.in_sellzone;
                """
                cursor.executemany(query, data)
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving buyzone for {ticker}: {e}")
        raise  # Re-raise for parallel processing to catch

# ----------------------------------------------------------
# Process a single ticker
# ----------------------------------------------------------
def process_ticker(ticker: str):
    print(f"📊 Processing {ticker}...")
    try:
        df = load_price_data(ticker)
        if len(df) < 250:
            print(f"⚠️ Not enough data for {ticker}")
            logger.error(f"Not enough data for {ticker}")
            return
        result_df = calculate_buyzone(df)
        save_to_db(ticker, result_df)
        print(f"✅ Finished {ticker}")
    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")
        logger.error(f"Error processing {ticker}: {e}")

# ----------------------------------------------------------
# Load tickers from file or DB
# ----------------------------------------------------------
def load_tickers_from_file(file_path):
    with open(file_path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def load_tickers_from_db():
    df = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", engine)
    return df["ticker"].dropna().str.upper().tolist()

# ----------------------------------------------------------
# CLI Entry Point
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Greer BuyZone calculations.")
    parser.add_argument("--tickers", nargs="+", help="List of tickers")
    parser.add_argument("--file", type=str, help="Path to file with tickers")
    args = parser.parse_args()

    if args.file:
        tickers = load_tickers_from_file(args.file)
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        print("🗃️  Loading tickers from companies table...")
        tickers = load_tickers_from_db()

    print(f"✅ Loaded {len(tickers)} tickers")

    # Parallel processing
    max_workers = 4  # Adjust: 2 for Starter (0.5 CPU), 4 for Standard/Pro (1–2 CPU)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_ticker, ticker) for ticker in tickers]
        for future in as_completed(futures):
            try:
                future.result()  # Raises exceptions if any
            except Exception as e:
                print(f"Error in parallel processing: {e}")

    # Optional: Batch load all prices (uncomment to use, ensure ~370 MB RAM free)
    """
    print("🗃️  Loading all prices at once...")
    all_data = load_all_prices(tickers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ticker, group in all_data.groupby('ticker'):
            futures.append(executor.submit(lambda t, g: (process_ticker(t, g), t), ticker, group))
        for future in as_completed(futures):
            try:
                ticker = future.result()[1]
                print(f"✅ Finished {ticker}")
            except Exception as e:
                print(f"Error in parallel processing: {e}")
    """