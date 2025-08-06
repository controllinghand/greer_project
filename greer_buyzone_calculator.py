# greer_buyzone_calculator.py

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import argparse
import os
import logging

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
# Database Config
# ----------------------------------------------------------
SQLALCHEMY_ENGINE = create_engine("postgresql://greer_user:@localhost:5432/yfinance_db")
PSYCOPG2_CONN_STRING = "host=localhost dbname=yfinance_db user=greer_user port=5432"

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
    df = pd.read_sql(query, SQLALCHEMY_ENGINE)
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(inplace=True)
    return df

# ----------------------------------------------------------
# Calculate BuyZone logic
# ----------------------------------------------------------
def calculate_buyzone(df: pd.DataFrame, aroon_length: int = 233):
    result = []
    in_zone = False  # Current state

    for i in range(aroon_length, len(df)):
        window = df.iloc[i - aroon_length:i + 1]
        row = df.iloc[i]
        date = row['date']
        current_price = row['close']
        highs = window['high'].values
        lows = window['low'].values

        bars_since_high = aroon_length - np.argmax(highs)
        bars_since_low = aroon_length - np.argmin(lows)

        aroon_up = 100 * (aroon_length - bars_since_high) / aroon_length
        aroon_down = 100 * (aroon_length - bars_since_low) / aroon_length

        ahigh = np.max(highs)
        alow = np.min(lows)
        midpoint = (ahigh + alow) / 2

        buyzone_start = False
        buyzone_end = False

        # Simplified Logic
        if aroon_down == 100 and not in_zone:
            buyzone_start = True
            in_zone = True
        elif aroon_up == 100 and in_zone:
            buyzone_end = True
            in_zone = False

        result.append({
            'date': date,
            'close_price': current_price,
            'high': ahigh,
            'low': alow,
            'aroon_upper': aroon_up,
            'aroon_lower': aroon_down,
            'midpoint': midpoint,
            'buyzone_start': buyzone_start,
            'buyzone_end': buyzone_end,
            'in_buyzone': in_zone,
            'in_sellzone': not in_zone
        })

    return pd.DataFrame(result)

# ----------------------------------------------------------
# Save results to DB
# ----------------------------------------------------------
def save_to_db(ticker: str, df: pd.DataFrame):
    try:
        conn = psycopg2.connect(PSYCOPG2_CONN_STRING)
        cursor = conn.cursor()

        for _, row in df.iterrows():
            cursor.execute("""
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
            """, (
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
            ))

        conn.commit()
    except Exception as e:
        logger.error(f"Error saving buyzone for {ticker}: {e}")
    finally:
        cursor.close()
        conn.close()

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
    engine = create_engine("postgresql://greer_user@localhost:5432/yfinance_db")
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

    for ticker in tickers:
        process_ticker(ticker)
