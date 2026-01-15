# backfill_financials_sec.py
import os
import time
import json
import argparse
import logging
import math
import decimal
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

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
file_handler = logging.FileHandler(os.path.join(log_dir, "backfill_financials_sec.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# ----------------------------------------------------------
# DB Engine
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# SEC config
# ----------------------------------------------------------
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "YouRockClub (contact: admin@yourockclub.com)")
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKER_CIK_URLS = [
    "https://www.sec.gov/files/company_tickers_exchange.json",
    "https://www.sec.gov/files/company_tickers.json",
]

# Modest politeness delay per SEC guidance
SEC_SLEEP_SECONDS = float(os.getenv("SEC_SLEEP_SECONDS", "0.2"))

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

def sql_safe(v):
    """
    Convert float('nan') / Decimal('NaN') into None so PostgreSQL stores NULL
    (and so your NO-NaN check constraint never trips).
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, decimal.Decimal) and v.is_nan():
        return None
    return v

# ----------------------------------------------------------
# Function: Load tickers from file or companies table, with explicit override list
# ----------------------------------------------------------
def load_tickers(file_path: str | None, explicit_tickers: list[str] | None) -> list[str]:
    """
    Priority:
      1) explicit_tickers (from --tickers)
      2) file_path (CSV with 'ticker' column)
      3) companies table (default)
    """
    if explicit_tickers:
        logger.info(f"Using explicit tickers: {explicit_tickers}")
        return explicit_tickers

    if file_path:
        df = pd.read_csv(file_path)
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

    with engine.connect() as conn:
        tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
    logger.info(f"Loaded {len(tickers)} tickers from companies table")
    return tickers

# ----------------------------------------------------------
# Function: Get existing annual rows per ticker
# ----------------------------------------------------------
def get_existing_counts() -> Dict[str, int]:
    q = text("SELECT ticker, COUNT(*) AS n FROM financials GROUP BY ticker")
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    if df.empty:
        return {}
    return dict(zip(df["ticker"].astype(str), df["n"].astype(int)))

# ----------------------------------------------------------
# Function: Download and cache ticker->CIK map from SEC
# ----------------------------------------------------------
def load_ticker_cik_map(cache_path: str = None) -> Dict[str, str]:
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), "sec_ticker_cik_cache.json")

    # Use cache if fresh-ish (7 days)
    try:
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < 7 * 24 * 3600:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
    except Exception:
        pass

    sess = requests.Session()

    # SEC mapping files are on www.sec.gov, not data.sec.gov
    sec_headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
        "Host": "www.sec.gov",
    }

    for url in TICKER_CIK_URLS:
        try:
            time.sleep(SEC_SLEEP_SECONDS)
            r = sess.get(url, headers=sec_headers, timeout=30)
            r.raise_for_status()
            payload = r.json()

            out: Dict[str, str] = {}

            # company_tickers_exchange.json structure differs from company_tickers.json
            if isinstance(payload, dict) and "data" in payload and "fields" in payload:
                fields = payload["fields"]
                data_rows = payload["data"]
                try:
                    cik_idx = fields.index("cik")
                    tkr_idx = fields.index("ticker")
                except Exception:
                    cik_idx, tkr_idx = 0, 1

                for row in data_rows:
                    try:
                        tkr = str(row[tkr_idx]).upper().strip()
                        cik = str(int(row[cik_idx])).zfill(10)
                        if tkr:
                            out[tkr] = cik
                    except Exception:
                        continue

            elif isinstance(payload, dict):
                # company_tickers.json is dict of numeric keys to dicts like {cik_str, ticker, title}
                for _, row in payload.items():
                    try:
                        tkr = str(row.get("ticker", "")).upper().strip()
                        cik = str(int(row.get("cik_str"))).zfill(10)
                        if tkr:
                            out[tkr] = cik
                    except Exception:
                        continue

            if out:
                try:
                    with open(cache_path, "w") as f:
                        json.dump(out, f)
                except Exception:
                    pass
                return out

        except Exception as e:
            logger.warning(f"Failed to load ticker->CIK map from {url}: {e}")

    logger.error("Could not load SEC ticker->CIK map from any source.")
    return {}

# ----------------------------------------------------------
# Function: Extract annual 'FY' facts for a given XBRL tag
# ----------------------------------------------------------
def _extract_annual_facts(companyfacts: Dict[str, Any], namespace: str, tag: str) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with keys: end, val, fy, fp, form, frame (optional).
    Only includes annual FP=FY entries AND annual filing forms (10-K/20-F/40-F).
    """
    try:
        node = companyfacts["facts"][namespace][tag]
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    units = node.get("units", {})

    for _, arr in units.items():
        if not isinstance(arr, list):
            continue

        for item in arr:
            try:
                fp = str(item.get("fp", "")).upper()
                if fp != "FY":
                    continue

                form = str(item.get("form", "")).upper()
                if form not in ("10-K", "20-F", "40-F"):
                    continue

                end = item.get("end")
                val = item.get("val")
                if end is None or val is None:
                    continue

                out.append({
                    "end": end,
                    "val": val,
                    "fy": item.get("fy"),
                    "fp": fp,
                    "form": form,
                    "frame": item.get("frame"),
                })
            except Exception:
                continue

    # de-dupe by end date, keep the last one encountered
    dedup: Dict[str, Dict[str, Any]] = {}
    for x in out:
        dedup[x["end"]] = x
    out = list(dedup.values())

    try:
        out.sort(key=lambda x: x["end"])
    except Exception:
        pass

    return out

# ----------------------------------------------------------
# Function: Pick the best tag from a list of candidates
# ----------------------------------------------------------
def _pick_series(companyfacts: Dict[str, Any], candidates: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    candidates: [(namespace, tag), ...]
    returns first non-empty annual series
    """
    for ns, tag in candidates:
        s = _extract_annual_facts(companyfacts, ns, tag)
        if s:
            return s
    return []

# ----------------------------------------------------------
# Function: Build annual rows (report_date=end) from SEC companyfacts
# ----------------------------------------------------------
def sec_financial_rows_for_ticker(ticker: str, cik: str, min_years: int = 10) -> pd.DataFrame:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    sess = requests.Session()

    time.sleep(SEC_SLEEP_SECONDS)
    r = sess.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    facts = r.json()

    # ---- Tag candidates (US-GAAP only here; you can extend later) ----
    revenue = _pick_series(facts, [
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
    ])

    net_income = _pick_series(facts, [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
    ])

    shares = _pick_series(facts, [
        ("us-gaap", "CommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    ])

    equity = _pick_series(facts, [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    ])

    ocf = _pick_series(facts, [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
    ])

    capex = _pick_series(facts, [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsToAcquireProductiveAssets"),
    ])

    def to_map(series: List[Dict[str, Any]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for x in series:
            try:
                out[str(x["end"])] = float(x["val"])
            except Exception:
                continue
        return out

    rev_m = to_map(revenue)
    ni_m  = to_map(net_income)
    sh_m  = to_map(shares)
    eq_m  = to_map(equity)
    ocf_m = to_map(ocf)
    cap_m = to_map(capex)

    # ----------------------------------------------------------
    # Anchor annual rows on Revenue FY end-dates (fallback: Net Income)
    # This prevents orphan rows from unioning mismatched end dates.
    # ----------------------------------------------------------
    all_ends = sorted(set(rev_m.keys()))
    if not all_ends:
        all_ends = sorted(set(ni_m.keys()))
    if not all_ends:
        return pd.DataFrame()

    rows = []
    for end in all_ends:
        rev = rev_m.get(end, np.nan)
        ni  = ni_m.get(end, np.nan)
        sh  = sh_m.get(end, np.nan)
        eq  = eq_m.get(end, np.nan)
        oc  = ocf_m.get(end, np.nan)
        cx  = cap_m.get(end, np.nan)

        # If neither revenue nor net income exists for this anchor date, skip
        if np.isnan(rev) and np.isnan(ni):
            continue

        # Book Value Per Share = equity / shares
        if not np.isnan(eq) and not np.isnan(sh) and sh != 0:
            bvps = eq / sh
        else:
            bvps = np.nan

        # Free Cash Flow (handle capex sign safely)
        if not np.isnan(oc) and not np.isnan(cx):
            fcf = oc + cx if cx < 0 else oc - cx
        else:
            fcf = np.nan

        # Net margin as percent
        if not np.isnan(ni) and not np.isnan(rev) and rev != 0:
            nm = (ni / rev) * 100.0
        else:
            nm = np.nan

        # ----------------------------------------------------------
        # PATCH: Drop quarterly-like rows
        # If BOTH FCF and shares are missing, it's almost never a usable annual row.
        # ----------------------------------------------------------
        if np.isnan(fcf) and np.isnan(sh):
            continue

        rows.append({
            "ticker": ticker,
            "report_date": pd.to_datetime(end, errors="coerce"),
            "book_value_per_share": None if np.isnan(bvps) else float(bvps),
            "free_cash_flow": None if np.isnan(fcf) else float(fcf),
            "net_margin": None if np.isnan(nm) else float(nm),
            "total_revenue": None if np.isnan(rev) else float(rev),
            "net_income": None if np.isnan(ni) else float(ni),
            "shares_outstanding": None if np.isnan(sh) else float(sh),
        })

    df = (
        pd.DataFrame(rows)
        .dropna(subset=["report_date"])
        .sort_values("report_date")
        .reset_index(drop=True)
    )

    # Keep last N years (if you only want depth-minimum)
    keep_n = max(min_years, 12)
    if len(df) > keep_n:
        df = df.tail(keep_n).reset_index(drop=True)

    return df

# ----------------------------------------------------------
# Function: Upsert rows into financials (matches your existing schema)
# ----------------------------------------------------------
def upsert_financial_rows(df: pd.DataFrame, reload: bool, reload_missing: bool) -> int:
    if df is None or df.empty:
        return 0

    base_insert = """
        INSERT INTO financials (
            ticker, report_date, book_value_per_share, free_cash_flow,
            net_margin, total_revenue, net_income, shares_outstanding
        )
        VALUES (
            :ticker, :report_date, :book_value_per_share, :free_cash_flow,
            :net_margin, :total_revenue, :net_income, :shares_outstanding
        )
    """

    if reload:
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

    payload = df.to_dict(orient="records")
    payload = [{k: sql_safe(v) for k, v in row.items()} for row in payload]

    with engine.begin() as conn:
        conn.execute(query, payload)

    return len(payload)

# ----------------------------------------------------------
# Function: Backfill worker for one ticker
# ----------------------------------------------------------
def backfill_one_ticker(
    ticker: str,
    cik_map: Dict[str, str],
    existing_counts: Dict[str, int],
    min_years: int,
    reload: bool,
    reload_missing: bool,
) -> Tuple[str, int, str]:
    try:
        # Skip if already deep enough and not reloading
        n = int(existing_counts.get(ticker, 0))
        if (not reload and not reload_missing) and n >= min_years:
            return (ticker, 0, f"skip (already {n} rows)")

        cik = cik_map.get(ticker)
        if not cik:
            return (ticker, 0, "no CIK (non-US or not in SEC map)")

        df = sec_financial_rows_for_ticker(ticker, cik, min_years=min_years)
        if df.empty:
            return (ticker, 0, "SEC returned no usable rows")

        upserted = upsert_financial_rows(df, reload=reload, reload_missing=reload_missing)
        return (ticker, upserted, f"sec upserted {upserted}")

    except Exception as e:
        logger.error(f"{ticker}: backfill failed: {e}")
        return (ticker, 0, f"error: {e}")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="One-time backfill of financials using SEC companyfacts (10+ years)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV file of tickers (expects 'ticker' column)")
    group.add_argument("--tickers", type=str, help='Optional comma/space separated list of tickers')
    parser.add_argument("--min-years", type=int, default=10, help="Minimum annual rows desired per ticker (default: 10)")
    parser.add_argument("--workers", type=int, default=3, help="Max parallel workers (default: 3)")
    parser.add_argument("--reload", action="store_true", help="Reload/overwrite existing rows on (ticker, report_date).")
    parser.add_argument("--reload-missing", action="store_true", help="Only fill NULL columns for existing rows.")
    args = parser.parse_args()

    if args.reload and args.reload_missing:
        raise SystemExit("Choose either --reload or --reload-missing, not both.")

    explicit = parse_tickers_arg(args.tickers)
    tickers = load_tickers(args.file, explicit)

    logger.info(f"Loaded {len(tickers)} tickers")
    existing_counts = get_existing_counts()
    cik_map = load_ticker_cik_map()

    start = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                backfill_one_ticker,
                t,
                cik_map,
                existing_counts,
                args.min_years,
                args.reload,
                args.reload_missing,
            ): t
            for t in tickers
        }

        for fut in as_completed(futs):
            t = futs[fut]
            ticker, upserted, status = fut.result()
            results.append((ticker, upserted, status))
            logger.info(f"{ticker}: {status}")

    elapsed = time.time() - start
    total_upserted = sum(x[1] for x in results)

    logger.info(f"Done. Total upserted rows: {total_upserted}. Elapsed: {elapsed:.2f}s")

    print(f"✅ Done. Total upserted rows: {total_upserted}. Elapsed: {elapsed:.2f}s")
    bad = [x for x in results if x[1] == 0 and ("error" in x[2] or "no CIK" in x[2] or "no usable" in x[2])]
    if bad:
        print("⚠️ Some tickers did not backfill (see log for details). Examples:")
        for row in bad[:15]:
            print(f"  - {row[0]}: {row[2]}")

if __name__ == "__main__":
    main()
