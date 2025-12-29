# fetch_iv_summary.py

import os
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import text
import logging
import threading
from math import log, sqrt, exp, erf

from db import get_engine

# ----------------------------------------------------------
# Logging Setup (file only; no console handler)
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers[:] = []  # avoid duplicate handlers if reloaded

fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(os.path.join(log_dir, "fetch_iv_summary.log"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)

# ----------------------------------------------------------
# Initialize DB Connection (global engine)
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Function: Load tickers from CSV or DB
# ----------------------------------------------------------
def load_tickers(file_path: str | None) -> list[str]:
    if file_path:
        logger.info(f"Loading tickers from file: {file_path}")
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

    logger.info("Loading tickers from companies table…")
    with engine.connect() as conn:
        tickers = pd.read_sql("SELECT ticker FROM companies ORDER BY ticker", conn)["ticker"].tolist()
    tickers = [t.strip().upper() for t in tickers]
    logger.info(f"Loaded {len(tickers)} tickers from companies table")
    return tickers

# ----------------------------------------------------------
# Function: Run a function with timeout
# ----------------------------------------------------------
def run_with_timeout(func, args=(), kwargs=None, timeout_sec=60):
    if kwargs is None:
        kwargs = {}
    result = {}
    exc = {}

    def target():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:
            exc["error"] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_sec} seconds")
    if "error" in exc:
        raise exc["error"]
    return result.get("value")

# ----------------------------------------------------------
# Function: Standard normal CDF (no scipy dependency)
# ----------------------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

# ----------------------------------------------------------
# Function: Black-Scholes delta (call/put)
# Notes:
# - Uses IV from yfinance per strike.
# - r is a configurable constant; good enough for strike ranking.
# - q (dividend yield) omitted initially (set to 0).
# ----------------------------------------------------------
def bs_delta(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float | None:
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None

        d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        if option_type.lower() == "call":
            return exp(-q * T) * norm_cdf(d1)
        else:
            # put delta
            return exp(-q * T) * (norm_cdf(d1) - 1.0)
    except Exception:
        return None

# ----------------------------------------------------------
# Function: Choose an expiry based on DTE window
# ----------------------------------------------------------
def pick_expiry(expiries: list[str], dte_min: int, dte_max: int) -> tuple[str | None, int | None]:
    if not expiries:
        return None, None

    today = pd.Timestamp.utcnow().date()
    candidates = []
    for e in expiries:
        try:
            ed = pd.to_datetime(e).date()
            dte = (ed - today).days
            candidates.append((e, dte))
        except Exception:
            continue

    # Filter by window
    windowed = [(e, d) for (e, d) in candidates if dte_min <= d <= dte_max]
    if windowed:
        # nearest inside the window
        windowed.sort(key=lambda x: x[1])
        return windowed[0][0], windowed[0][1]

    # fallback: nearest future expiry
    future = [(e, d) for (e, d) in candidates if d > 0]
    if not future:
        return expiries[0], None
    future.sort(key=lambda x: x[1])
    return future[0][0], future[0][1]

# ----------------------------------------------------------
# Function: Pick a strike closest to target delta (+0.20 call / -0.20 put)
# Also calculates premium using mid(bid/ask) first, else lastPrice.
# ----------------------------------------------------------
def pick_target_delta_option(
    df: pd.DataFrame,
    option_type: str,
    S: float,
    expiry_ts: pd.Timestamp,
    target_delta: float,
    r: float,
    q: float = 0.0,
    min_oi: int = 0,
    min_vol: int = 0,
    max_spread_pct: float = 0.30
) -> dict | None:
    if df is None or df.empty:
        return None

    # Ensure required columns exist
    for col in ["strike", "impliedVolatility"]:
        if col not in df.columns:
            return None

    # Clean + basic filters
    work = df.dropna(subset=["strike", "impliedVolatility"]).copy()

    # Liquidity filters if columns exist
    if "openInterest" in work.columns and min_oi > 0:
        work = work[work["openInterest"].fillna(0) >= min_oi]
    if "volume" in work.columns and min_vol > 0:
        work = work[work["volume"].fillna(0) >= min_vol]

    if work.empty:
        return None

    # Time to expiration in years (UTC-aware)
    today = pd.Timestamp.now(tz="UTC")

    exp_ts = pd.to_datetime(expiry_ts)
    if exp_ts.tz is None:
        exp_ts = exp_ts.tz_localize("UTC")
    else:
        exp_ts = exp_ts.tz_convert("UTC")

    # Clamp: if we're past expiration moment, still keep a tiny T
    T_days = (exp_ts - today).total_seconds() / 86400.0
    T = max(T_days, 0.001) / 365.0

    # Compute deltas + premiums
    deltas = []
    premiums = []
    spreads_pct = []

    for _, row in work.iterrows():
        K = float(row["strike"])
        sigma = float(row["impliedVolatility"])

        d = bs_delta(option_type=option_type, S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        deltas.append(d)

        bid = row.get("bid", np.nan)
        ask = row.get("ask", np.nan)
        last = row.get("lastPrice", np.nan)

        prem = None
        # Prefer mid if bid/ask available
        if (bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask)
            and not np.isnan(bid) and not np.isnan(ask) and float(ask) >= float(bid)):
            prem = (float(bid) + float(ask)) / 2.0
        elif last is not None and not (pd.isna(last) or np.isnan(last)):
            prem = float(last)

        premiums.append(prem)

        # spread pct vs mid
        sp = None
        if (bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask)
            and not np.isnan(bid) and not np.isnan(ask) and float(ask) >= float(bid)):
            mid = (float(bid) + float(ask)) / 2.0
            if mid > 0:
                sp = (float(ask) - float(bid)) / mid
        spreads_pct.append(sp)

    work["est_delta"] = deltas
    work["premium"] = premiums
    work["spread_pct"] = spreads_pct

    # Remove rows with missing delta or premium
    work = work.dropna(subset=["est_delta", "premium"])
    if work.empty:
        return None

    # Spread filter if available
    work = work[(work["spread_pct"].isna()) | (work["spread_pct"] <= max_spread_pct)]
    if work.empty:
        return None

    # Pick closest to target delta
    work["delta_dist"] = (work["est_delta"] - target_delta).abs()
    best = work.loc[work["delta_dist"].idxmin()]

    return {
        "strike": float(best["strike"]),
        "iv": float(best["impliedVolatility"]),
        "delta": float(best["est_delta"]),
        "premium": float(best["premium"]),
        "bid": float(best["bid"]) if "bid" in best and not pd.isna(best["bid"]) else None,
        "ask": float(best["ask"]) if "ask" in best and not pd.isna(best["ask"]) else None,
        "spread_pct": float(best["spread_pct"]) if not pd.isna(best["spread_pct"]) else None,
        "open_interest": int(best["openInterest"]) if "openInterest" in best and not pd.isna(best["openInterest"]) else None,
        "volume": int(best["volume"]) if "volume" in best and not pd.isna(best["volume"]) else None,
    }

# ----------------------------------------------------------
# Function: Fetch IV summary + ATM + 20Δ Put/Call for Wheel
# ----------------------------------------------------------
def fetch_iv_summary_for_ticker(
    ticker: str,
    max_retries: int = 3,
    initial_backoff: float = 30.0,
    timeout_sec: float = 60.0,
    dte_min: int = 5,
    dte_max: int = 10,
    risk_free_rate: float = 0.05,
    min_oi: int = 0,
    min_vol: int = 0,
    max_spread_pct: float = 0.30
) -> dict | None:
    today = pd.Timestamp.utcnow().date()

    with engine.connect() as conn:
        exists_row = conn.execute(
            text("SELECT 1 FROM iv_summary WHERE ticker = :ticker AND fetch_date = :today"),
            {"ticker": ticker, "today": today}
        ).fetchone()
    if exists_row:
        logger.info(f"{ticker}: already processed for today ({today}) — skipping.")
        return None

    logger.info(f"{ticker}: starting fetch.")
    stock = yf.Ticker(ticker)

    expiries = stock.options
    if not expiries:
        logger.warning(f"{ticker}: no option expirations found")
        return None

    expiry_str, dte = pick_expiry(expiries, dte_min=dte_min, dte_max=dte_max)
    if not expiry_str:
        logger.warning(f"{ticker}: could not select expiry")
        return None

    logger.info(f"{ticker}: selected expiry {expiry_str} (DTE={dte}).")
    attempt = 0
    backoff = initial_backoff

    while attempt < max_retries:
        logger.info(f"{ticker}: attempt {attempt+1}/{max_retries}. Timeout per attempt = {timeout_sec}s.")
        try:
            opt = run_with_timeout(stock.option_chain, args=(expiry_str,), timeout_sec=timeout_sec)
            logger.info(f"{ticker}: option_chain returned successfully on attempt {attempt+1}.")
            break
        except TimeoutError:
            logger.warning(f"{ticker}: fetch timed out on attempt {attempt+1}/{max_retries}; sleeping {backoff}s before retry.")
            time.sleep(backoff)
            attempt += 1
            backoff *= 2
            continue
        except Exception as e:
            err_str = str(e)
            logger.error(f"{ticker}: exception when fetching option chain: {err_str}")
            if ("Too Many Requests" in err_str) or ("Rate limited" in err_str) or ("429" in err_str) or ("YFRateLimitError" in err_str):
                logger.warning(f"{ticker}: detected rate-limit/block on attempt {attempt+1}/{max_retries}. Sleeping {backoff}s before retry.")
                time.sleep(backoff)
                attempt += 1
                backoff *= 2
                continue
            logger.error(f"{ticker}: non-rate-limit/non-timeout error fetching option chain for expiry {expiry_str}: {e}")
            return None
    else:
        logger.error(f"{ticker}: exceeded max retries ({max_retries}) due to timeout or rate-limit; skipping ticker.")
        return None

    calls = opt.calls.copy() if opt and hasattr(opt, "calls") else pd.DataFrame()
    puts = opt.puts.copy() if opt and hasattr(opt, "puts") else pd.DataFrame()

    # Calls-only IV stats (your existing behavior)
    calls_iv = calls.dropna(subset=["impliedVolatility"]) if not calls.empty else pd.DataFrame()
    if calls_iv.empty:
        logger.warning(f"{ticker}: no call option data w/ impliedVolatility for expiry {expiry_str}")
        return None

    stats = calls_iv["impliedVolatility"].describe()
    logger.info(f"{ticker}: impliedVolatility stats computed.")

    # Underlying price (prefer fast_info, fallback to info)
    price = None
    try:
        fi = getattr(stock, "fast_info", None)
        if fi and "last_price" in fi and fi["last_price"] is not None:
            price = float(fi["last_price"])
    except Exception:
        pass

    if price is None:
        try:
            price = stock.info.get("regularMarketPrice", None)
            if price is not None:
                price = float(price)
        except Exception:
            price = None

    if price is None or price <= 0:
        logger.warning(f"{ticker}: could not retrieve underlying price")
        return None

    expiry_dt = pd.to_datetime(expiry_str)

    # Make it UTC-aware
    if expiry_dt.tz is None:
        expiry_dt = expiry_dt.tz_localize("UTC")
    else:
        expiry_dt = expiry_dt.tz_convert("UTC")

    # Set to approx market close (21:00 UTC ~ 4pm ET / 1pm PT)
    expiry_dt = expiry_dt.normalize() + pd.Timedelta(hours=21)

    # ✅ FIX: define expiry_date used in result payload + DB
    expiry_date = expiry_dt.date()

    # ATM call (your existing ATM calc)
    atm_iv = None
    atm_premium = None
    atm_premium_pct = None

    calls_iv["dist"] = (calls_iv["strike"] - price).abs()
    atm_call = calls_iv.loc[calls_iv["dist"].idxmin()]

    atm_iv = float(atm_call["impliedVolatility"])
    last_price = atm_call.get("lastPrice", None)
    if last_price is not None and not (pd.isna(last_price) or np.isnan(last_price)):
        atm_premium = float(last_price)
        atm_premium_pct = atm_premium / price
        logger.info(
            f"{ticker}: ATM strike {atm_call['strike']}, iv_atm={atm_iv:.6f}, "
            f"premium={atm_premium:.2f}, premium_pct={atm_premium_pct:.4f}"
        )
    else:
        logger.warning(f"{ticker}: ATM call lastPrice not available (None or NaN).")

    # 20-delta PUT (CSP candidate)
    put_20 = None
    if not puts.empty:
        put_20 = pick_target_delta_option(
            df=puts,
            option_type="put",
            S=price,
            expiry_ts=expiry_dt,
            target_delta=-0.20,
            r=risk_free_rate,
            q=0.0,
            min_oi=min_oi,
            min_vol=min_vol,
            max_spread_pct=max_spread_pct
        )

    # 20-delta CALL (CC candidate)
    call_20 = None
    if not calls.empty:
        call_20 = pick_target_delta_option(
            df=calls,
            option_type="call",
            S=price,
            expiry_ts=expiry_dt,
            target_delta=0.20,
            r=risk_free_rate,
            q=0.0,
            min_oi=min_oi,
            min_vol=min_vol,
            max_spread_pct=max_spread_pct
        )

    # Premium percentages
    put_20_premium_pct = None
    if put_20 and put_20.get("premium") is not None and put_20.get("strike") is not None and put_20["strike"] > 0:
        # CSP premium as % of cash secured per share ≈ premium / strike
        put_20_premium_pct = float(put_20["premium"]) / float(put_20["strike"])

    call_20_premium_pct = None
    if call_20 and call_20.get("premium") is not None and price > 0:
        # CC premium as % of underlying price
        call_20_premium_pct = float(call_20["premium"]) / float(price)

    if put_20:
        logger.info(
            f"{ticker}: PUT ~-0.20Δ strike={put_20['strike']} delta={put_20['delta']:.3f} "
            f"prem={put_20['premium']:.2f} prem_pct={put_20_premium_pct if put_20_premium_pct else None}"
        )
    else:
        logger.info(f"{ticker}: PUT ~-0.20Δ not found (filters/chain availability).")

    if call_20:
        logger.info(
            f"{ticker}: CALL ~+0.20Δ strike={call_20['strike']} delta={call_20['delta']:.3f} "
            f"prem={call_20['premium']:.2f} prem_pct={call_20_premium_pct if call_20_premium_pct else None}"
        )
    else:
        logger.info(f"{ticker}: CALL ~+0.20Δ not found (filters/chain availability).")

    result = {
        "ticker": ticker,
        "fetch_date": today,
        "expiry": expiry_date,
        "dte": int(dte) if dte is not None else None,
        "underlying_price": float(price),

        "contract_count": int(stats["count"]),
        "iv_mean": float(stats["mean"]),
        "iv_std": float(stats["std"]),
        "iv_min": float(stats["min"]),
        "iv_25": float(stats["25%"]),
        "iv_median": float(stats["50%"]),
        "iv_75": float(stats["75%"]),
        "iv_max": float(stats["max"]),

        "iv_atm": atm_iv,
        "atm_premium": atm_premium,
        "atm_premium_pct": atm_premium_pct,

        "put_20d_strike": float(put_20["strike"]) if put_20 else None,
        "put_20d_iv": float(put_20["iv"]) if put_20 else None,
        "put_20d_premium": float(put_20["premium"]) if put_20 else None,
        "put_20d_premium_pct": float(put_20_premium_pct) if put_20_premium_pct is not None else None,
        "put_20d_delta": float(put_20["delta"]) if put_20 else None,

        "call_20d_strike": float(call_20["strike"]) if call_20 else None,
        "call_20d_iv": float(call_20["iv"]) if call_20 else None,
        "call_20d_premium": float(call_20["premium"]) if call_20 else None,
        "call_20d_premium_pct": float(call_20_premium_pct) if call_20_premium_pct is not None else None,
        "call_20d_delta": float(call_20["delta"]) if call_20 else None,
    }

    logger.info(f"{ticker}: fetched IV summary (wheel-ready) for expiry {expiry_str}.")
    return result

# ----------------------------------------------------------
# Function: Process tickers in parallel
# ----------------------------------------------------------
def process_tickers(
    tickers: list[str],
    max_workers: int = 3,
    delay_between: float = 2.0,
    max_retries: int = 3,
    timeout_sec: float = 60.0,
    dte_min: int = 5,
    dte_max: int = 10,
    risk_free_rate: float = 0.05,
    min_oi: int = 0,
    min_vol: int = 0,
    max_spread_pct: float = 0.30
):
    start_time = time.time()
    summaries = []

    logger.info(
        f"process_tickers: starting {len(tickers)} tickers, workers={max_workers}, delay={delay_between}s, "
        f"dte=[{dte_min},{dte_max}], r={risk_free_rate}, min_oi={min_oi}, min_vol={min_vol}, max_spread_pct={max_spread_pct}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(
                fetch_iv_summary_for_ticker,
                t,
                max_retries,
                30.0,
                timeout_sec,
                dte_min,
                dte_max,
                risk_free_rate,
                min_oi,
                min_vol,
                max_spread_pct
            ): t for t in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                res = future.result()
                if res:
                    summaries.append(res)
                    logger.info(f"{ticker}: summary collected successfully.")
                    logger.info(f"{ticker}: waiting {delay_between}s before next ticker.")
                    time.sleep(delay_between)
                else:
                    logger.info(f"{ticker}: summary returned None (skipped or error). No wait.")
            except Exception:
                # ✅ full traceback (super helpful)
                logger.exception(f"{ticker}: unexpected error in future result")
                logger.info(f"{ticker}: waiting {delay_between}s before next ticker.")
                time.sleep(delay_between)

    elapsed = time.time() - start_time
    logger.info(f"process_tickers: finished in {elapsed:.2f}s; collected {len(summaries)} summaries.")

    if not summaries:
        logger.info("No summaries to insert. Ending process_tickers without DB insert.")
        return

    insert_query = text("""
        INSERT INTO iv_summary (
            ticker, fetch_date, expiry, dte, underlying_price,
            contract_count, iv_mean, iv_std,
            iv_min, iv_25, iv_median, iv_75,
            iv_max, iv_atm, atm_premium, atm_premium_pct,
            put_20d_strike, put_20d_iv, put_20d_premium, put_20d_premium_pct, put_20d_delta,
            call_20d_strike, call_20d_iv, call_20d_premium, call_20d_premium_pct, call_20d_delta
        )
        VALUES (
            :ticker, :fetch_date, :expiry, :dte, :underlying_price,
            :contract_count, :iv_mean, :iv_std,
            :iv_min, :iv_25, :iv_median, :iv_75,
            :iv_max, :iv_atm, :atm_premium, :atm_premium_pct,
            :put_20d_strike, :put_20d_iv, :put_20d_premium, :put_20d_premium_pct, :put_20d_delta,
            :call_20d_strike, :call_20d_iv, :call_20d_premium, :call_20d_premium_pct, :call_20d_delta
        )
        ON CONFLICT (ticker, fetch_date, expiry) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(insert_query, summaries)
        logger.info(f"Inserted {len(summaries)} records into iv_summary table.")

    elapsed2 = time.time() - start_time
    logger.info(f"Completed DB insert in {elapsed2:.2f}s")

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch option-chain IV summary + 20Δ strikes and store in DB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV with tickers (expects 'ticker' column)")
    group.add_argument("--tickers", nargs="+", type=str, help="List of tickers (e.g., TSLA AAPL MSFT)")

    parser.add_argument("--workers", type=int, default=3, help="Max parallel workers (default: 3)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay seconds between each ticker completion")
    parser.add_argument("--retries", type=int, default=3, help="Max retries on timeout/rate limit errors")
    parser.add_argument("--timeout", type=float, default=60.0, help="Timeout seconds for each option-chain fetch")

    parser.add_argument("--batch_size", type=int, default=50, help="Number of tickers per batch")
    parser.add_argument("--pause_between_batches", type=float, default=30.0, help="Pause seconds between batches")

    # Wheel / 20Δ settings
    parser.add_argument("--dte_min", type=int, default=5, help="Min DTE for expiry selection (default: 5)")
    parser.add_argument("--dte_max", type=int, default=10, help="Max DTE for expiry selection (default: 10)")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate used for delta estimation (default: 0.05)")
    parser.add_argument("--min_oi", type=int, default=0, help="Minimum open interest filter (default: 0)")
    parser.add_argument("--min_vol", type=int, default=0, help="Minimum volume filter (default: 0)")
    parser.add_argument("--max_spread_pct", type=float, default=0.30, help="Max bid/ask spread pct vs mid (default: 0.30)")

    args = parser.parse_args()

    if args.tickers:
        tickers_list = [t.strip().upper() for t in args.tickers]
        logger.info(f"Using explicit tickers: {tickers_list}")
    elif args.file:
        tickers_list = load_tickers(args.file)
    else:
        tickers_list = load_tickers(None)

    logger.info(
        f"Starting IV summary fetch for {len(tickers_list)} tickers "
        f"workers={args.workers}, delay={args.delay}s, retries={args.retries}, timeout={args.timeout}s, "
        f"batch_size={args.batch_size}, pause={args.pause_between_batches}s, "
        f"dte=[{args.dte_min},{args.dte_max}], r={args.r}, min_oi={args.min_oi}, min_vol={args.min_vol}, max_spread_pct={args.max_spread_pct}"
    )

    for i in range(0, len(tickers_list), args.batch_size):
        batch = tickers_list[i : i + args.batch_size]
        logger.info(f"Processing batch {i//args.batch_size + 1} (tickers={batch})")

        process_tickers(
            batch,
            max_workers=args.workers,
            delay_between=args.delay,
            max_retries=args.retries,
            timeout_sec=args.timeout,
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            risk_free_rate=args.r,
            min_oi=args.min_oi,
            min_vol=args.min_vol,
            max_spread_pct=args.max_spread_pct
        )

        if i + args.batch_size < len(tickers_list):
            logger.info(f"Batch {i//args.batch_size + 1} done — sleeping {args.pause_between_batches}s before next batch.")
            time.sleep(args.pause_between_batches)

    logger.info("Finished IV summary fetch.")
