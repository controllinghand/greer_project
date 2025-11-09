# fetch_financials.py
import os
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
import logging

# ----------------------------------------------------------
# DB helpers
# ----------------------------------------------------------
from db import get_engine

# ----------------------------------------------------------
# Logging Setup (file only; no console handler)
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers[:] = []  # avoid duplicate handlers in reloads

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(log_dir, "fetch_financials.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

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
    parts = [p.strip().upper() for p in raw.replace(",", " ").split() if p.strip()]
    seen = set()
    out = []
    for p in parts:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _normalize_index(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = df.index.astype(str).str.strip().str.lower()
    return df

# Broader label aliases (with REIT-friendly variants)
LABELS = {
    "total_assets": [
        "total assets",
    ],
    "total_liabilities": [
        "total liabilities", "total liabilities net minority interest", "liabilities"
    ],
    "shares_out": [
        # Balance sheet (preferred)
        "ordinary shares number", "common stock shares outstanding",
        # Fallbacks from income stmt (diluted average)
        "weighted average diluted shares outstanding",
        "diluted average shares",
        "diluted weighted average shares",
        "weighted average shares outstanding diluted",
    ],
    "operating_cash_flow": [
        "operating cash flow",
        "total cash from operating activities",
        "net cash provided by operating activities",
        "net cash from operating activities",
    ],
    "capex": [
        "capital expenditure", "capital expenditures",
        "purchase of property plant and equipment",
    ],
    "free_cash_flow": [
        "free cash flow",  # prefer explicit FCF row if present
    ],
    "revenue": [
        "total revenue", "revenue", "operating revenue",
        "total operating revenue",  # sometimes shows up on REITs
        "total income",             # last-resort, can overinclude, used only if nothing else
    ],
    "net_income": [
        "net income",
        "net income common stockholders",
        "net income available to common shareholders",
        "net income applicable to common shares",
        "net income attributable to common shareholders",
        "net income to common shareholders",
    ],
}

def _fuzzy_get(df: pd.DataFrame | None, labels: list[str], col) -> float:
    """
    Robust lookup:
      1) exact match on normalized labels
      2) fuzzy: all words of any label appear in an index row
    """
    if df is None or df.empty:
        return np.nan
    idx = df.index  # already normalized to lower
    # Exact first
    for lab in labels:
        key = lab.strip().lower()
        if key in idx:
            try:
                return df.loc[key].get(col, np.nan)
            except Exception:
                return np.nan
    # Fuzzy fallback
    for lab in labels:
        words = [w for w in lab.lower().split() if w]
        for row_lab in idx:
            if all(w in row_lab for w in words):
                try:
                    val = df.loc[row_lab].get(col, np.nan)
                except Exception:
                    val = np.nan
                if not (isinstance(val, float) and np.isnan(val)):
                    return val
    return np.nan

def _fetch_statements(stock: yf.Ticker, freq: str):
    """
    Returns normalized (income, balance_sheet, cashflow) for a given freq: "yearly", "quarterly", or "trailing".
    Applies fallbacks to legacy properties if needed.
    """
    income = stock.get_income_stmt(freq=freq)
    balance_sheet = stock.get_balance_sheet(freq=freq)
    cashflow = stock.get_cashflow(freq=freq)

    if income is None or income.empty:
        income = getattr(stock, "income_stmt", pd.DataFrame())
    if balance_sheet is None or balance_sheet.empty:
        balance_sheet = getattr(stock, "balance_sheet", pd.DataFrame())
    if cashflow is None or cashflow.empty:
        cashflow = getattr(stock, "cashflow", pd.DataFrame())

    income = _normalize_index(income)
    balance_sheet = _normalize_index(balance_sheet)
    cashflow = _normalize_index(cashflow)
    return income, balance_sheet, cashflow

def _compute_rows_from_statements(ticker: str, income: pd.DataFrame, balance_sheet: pd.DataFrame, cashflow: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-period rows from statements. Returns empty DF if nothing usable.
    """
    def _cols(df):
        return [] if df is None or df.empty else list(df.columns)

    years = sorted(set(_cols(income)) | set(_cols(balance_sheet)) | set(_cols(cashflow)))
    if not years:
        logger.warning(f"{ticker}: no reporting columns found after fetch")
        return pd.DataFrame()

    data = {
        "BOOK_VALUE_PER_SHARE": [],
        "FREE_CASH_FLOW": [],
        "NET_MARGIN": [],
        "TOTAL_REVENUE": [],
        "NET_INCOME": [],
        "DILUTED_SHARES_OUTSTANDING": [],
        "dates": []
    }

    for year in years:
        year_str = year.strftime("%Y-%m-%d") if isinstance(year, pd.Timestamp) else str(year)

        total_assets = _fuzzy_get(balance_sheet, LABELS["total_assets"], year)
        total_liab   = _fuzzy_get(balance_sheet, LABELS["total_liabilities"], year)

        # shares: first try BS, then diluted on IS
        shares_out = _fuzzy_get(balance_sheet, LABELS["shares_out"][:2], year)
        if (shares_out is None) or (isinstance(shares_out, float) and np.isnan(shares_out)):
            shares_out = _fuzzy_get(income, LABELS["shares_out"][2:], year)

        ocf        = _fuzzy_get(cashflow, LABELS["operating_cash_flow"], year)
        capex      = _fuzzy_get(cashflow, LABELS["capex"], year)
        fcf_direct = _fuzzy_get(cashflow, LABELS["free_cash_flow"], year)  # NEW: prefer Yahoo's explicit FCF if present

        revenue    = _fuzzy_get(income, LABELS["revenue"], year)
        net_income = _fuzzy_get(income, LABELS["net_income"], year)

        # Calculations
        if all(not np.isnan(x) and x != 0 for x in [total_assets, total_liab, shares_out]):
            book_value = (total_assets - total_liab) / shares_out
        else:
            book_value = np.nan

        # Free Cash Flow: prefer Yahoo's explicit "Free Cash Flow" row; else derive OCF + CapEx
        if not np.isnan(fcf_direct):
            free_cash_flow = fcf_direct
        elif all(not np.isnan(x) for x in [ocf, capex]):
            free_cash_flow = ocf + capex
        else:
            free_cash_flow = np.nan

        if all(not np.isnan(x) and revenue not in (0, np.nan) for x in [net_income, revenue]):
            net_margin = (net_income / revenue) * 100.0
        else:
            net_margin = np.nan

        # Skip if literally everything is NaN for this period
        if all(np.isnan(x) for x in [book_value, free_cash_flow, net_margin, revenue, net_income]) and (np.isnan(shares_out) or shares_out == 0):
            continue

        data["BOOK_VALUE_PER_SHARE"].append(book_value)
        data["FREE_CASH_FLOW"].append(free_cash_flow)
        data["NET_MARGIN"].append(net_margin)
        data["TOTAL_REVENUE"].append(revenue)
        data["NET_INCOME"].append(net_income)
        data["DILUTED_SHARES_OUTSTANDING"].append(shares_out)
        data["dates"].append(year_str)

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning(f"{ticker}: no usable rows after computing metrics")
        return pd.DataFrame()

    df["dates"] = pd.to_datetime(df["dates"], errors="coerce")
    df["ticker"] = ticker
    return df.sort_values("dates").reset_index(drop=True)

# Function: Fetch financial data for a ticker using yfinance (robust endpoints + fallbacks)
def fetch_financial_data(ticker: str, retries: int = 3, delay: int = 2) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)

            # 1) Try annual
            income_y, bs_y, cf_y = _fetch_statements(stock, freq="yearly")
            missing = [name for name, df in [
                ("income", income_y), ("balance_sheet", bs_y), ("cashflow", cf_y)
            ] if df is None or df.empty]
            if missing:
                logger.error(f"[{ticker}] Statements missing (yearly): {', '.join(missing)}")

            df = _compute_rows_from_statements(ticker, income_y, bs_y, cf_y)

            # 2) If no annual rows, try TTM (trailing) to at least capture one period
            if df.empty:
                income_t, bs_t, cf_t = _fetch_statements(stock, freq="trailing")
                missing_t = [name for name, df2 in [
                    ("income", income_t), ("balance_sheet", bs_t), ("cashflow", cf_t)
                ] if df2 is None or df2.empty]
                if missing_t:
                    logger.error(f"[{ticker}] Statements missing (trailing): {', '.join(missing_t)}")

                df = _compute_rows_from_statements(ticker, income_t, bs_t, cf_t)

            return df  # may be empty

        except Exception as e:
            logger.error(f"Error fetching data for {ticker} (Attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)

    logger.error(f"Failed to fetch data for {ticker} after {retries} attempt(s)")
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
def process_tickers(tickers, max_workers=10, reload=False, reload_missing=False):
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

                # Filter out existing dates only if NOT reloading
                if not reload and not reload_missing:
                    ticker_existing_dates = pd.to_datetime(existing_dates.get(ticker, []))
                    df["dates"] = pd.to_datetime(df["dates"])
                    new_rows_df = df[~df["dates"].isin(ticker_existing_dates)]
                else:
                    new_rows_df = df  # keep all rows; rely on ON CONFLICT DO UPDATE

                if new_rows_df.empty:
                    print(f"✅ No new data for {ticker}.")
                    logger.info(f"No new data for {ticker}")
                else:
                    new_rows.append(new_rows_df)
                    action = "reload" if reload else ("reload-missing" if reload_missing else "insert")
                    print(f"✅ Collected {len(new_rows_df)} row(s) for {ticker} ({action}).")
                    logger.info(f"Collected {len(new_rows_df)} row(s) for {ticker} ({action})")

            except Exception as e:
                # File-only logging; no console errors
                logger.error(f"Error processing {ticker}: {e}")

    if not new_rows:
        elapsed = time.time() - start_time
        print(f"Total processing time: {elapsed:.2f} seconds")
        logger.info(f"Total processing time: {elapsed:.2f} seconds")
        return

    all_new_rows = pd.concat(new_rows, ignore_index=True)

    # Base INSERT
    base_insert = """
        INSERT INTO financials (
            ticker, report_date, book_value_per_share, free_cash_flow, 
            net_margin, total_revenue, net_income, shares_outstanding
        )
        VALUES (:ticker, :report_date, :book_value_per_share, :free_cash_flow, 
                :net_margin, :total_revenue, :net_income, :shares_outstanding)
    """

    if reload:
        # Overwrite existing values on conflict
        on_conflict = """
            ON CONFLICT (ticker, report_date) DO UPDATE SET
                book_value_per_share = EXCLUDED.book_value_per_share,
                free_cash_flow       = EXCLUDED.free_cash_flow,
                net_margin           = EXCLUDED.net_margin,
                total_revenue        = EXCLUDED.total_revenue,
                net_income           = EXCLUDED.net_income,
                shares_outstanding   = EXCLUDED.shares_outstanding
        """
    elif reload_missing:
        # Only fill NULLs, leave existing non-NULL values as-is
        on_conflict = """
            ON CONFLICT (ticker, report_date) DO UPDATE SET
                book_value_per_share = COALESCE(financials.book_value_per_share, EXCLUDED.book_value_per_share),
                free_cash_flow       = COALESCE(financials.free_cash_flow,       EXCLUDED.free_cash_flow),
                net_margin           = COALESCE(financials.net_margin,           EXCLUDED.net_margin),
                total_revenue        = COALESCE(financials.total_revenue,        EXCLUDED.total_revenue),
                net_income           = COALESCE(financials.net_income,           EXCLUDED.net_income),
                shares_outstanding   = COALESCE(financials.shares_outstanding,   EXCLUDED.shares_outstanding)
        """
    else:
        on_conflict = "ON CONFLICT (ticker, report_date) DO NOTHING"

    query = text(base_insert + on_conflict)

    payload = [
        {
            "ticker": row["ticker"],
            "report_date": row["dates"],
            "book_value_per_share": None if pd.isna(row["BOOK_VALUE_PER_SHARE"]) else float(row["BOOK_VALUE_PER_SHARE"]),
            "free_cash_flow": None if pd.isna(row["FREE_CASH_FLOW"]) else float(row["FREE_CASH_FLOW"]),
            "net_margin": None if pd.isna(row["NET_MARGIN"]) else float(row["NET_MARGIN"]),
            "total_revenue": None if pd.isna(row["TOTAL_REVENUE"]) else float(row["TOTAL_REVENUE"]),
            "net_income": None if pd.isna(row["NET_INCOME"]) else float(row["NET_INCOME"]),
            "shares_outstanding": None if pd.isna(row["DILUTED_SHARES_OUTSTANDING"]) else float(row["DILUTED_SHARES_OUTSTANDING"]),
        }
        for _, row in all_new_rows.iterrows()
    ]

    # Use a transaction block
    with engine.begin() as conn:
        conn.execute(query, payload)

    print(f"✅ Upserted {len(all_new_rows)} rows into financials.")
    logger.info(f"Upserted {len(all_new_rows)} rows into financials")

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
        help='Optional comma/space separated list of tickers (e.g., "AAPL,MSFT TSLA")'
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Max parallel workers (default: 10)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload/overwrite existing rows on (ticker, report_date)."
    )
    parser.add_argument(
        "--reload-missing",
        action="store_true",
        help="Only fill NULL columns for existing (ticker, report_date)."
    )
    args = parser.parse_args()

    if args.reload and args.reload_missing:
        raise SystemExit("Choose either --reload or --reload-missing, not both.")

    explicit_tickers = parse_tickers_arg(args.tickers)
    tickers = load_tickers(args.file, explicit_tickers)

    print(f"✅ Loaded {len(tickers)} tickers")
    logger.info(f"Loaded {len(tickers)} tickers")

    process_tickers(tickers, max_workers=args.workers, reload=args.reload, reload_missing=args.reload_missing)
