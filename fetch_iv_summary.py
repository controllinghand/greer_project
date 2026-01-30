# fetch_iv_summary.py

import os
import time
import random
import re
import argparse
import numpy as np
import pandas as pd
import requests
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
# Tradier Client
# Env Vars:
# - TRADIER_TOKEN (required)
# - TRADIER_ENV   ("live" or "sandbox", default "live")
# ----------------------------------------------------------
class TradierClient:
    # ----------------------------------------------------------
    # Quota handling shared across threads
    # ----------------------------------------------------------
    _quota_lock = threading.Lock()
    _quota_until_epoch = 0.0  # epoch seconds (shared across all instances)
    _quota_re = re.compile(r"quota\s+viol\w*\s*:\s*expires\s*(\d+)", re.IGNORECASE)

    def _maybe_wait_for_quota(self) -> None:
        """
        If we previously saw a quota window, wait until it clears.
        This prevents all worker threads from hammering Tradier while blocked.
        """
        while True:
            with TradierClient._quota_lock:
                now = time.time()
                until = TradierClient._quota_until_epoch
                if until <= now:
                    return
                remaining = until - now

            # Sleep in chunks so we can recover quickly if window is short
            buffer_sec = 0.75
            jitter = random.uniform(0.05, 0.25)
            time.sleep(min(remaining + buffer_sec + jitter, 30.0))

    def _record_quota_from_body(self, body_text: str) -> bool:
        """
        Detects: 'Quota Violation: Expires 1769804340000'
        Records the expiry time globally. Returns True if quota was detected.
        """
        if not body_text:
            return False

        m = TradierClient._quota_re.search(body_text)
        if not m:
            return False

        try:
            ms = int(m.group(1))
            # sanity check for epoch-ms
            if ms < 10_000_000_000:
                return False
            until_epoch = ms / 1000.0
        except Exception:
            return False

        with TradierClient._quota_lock:
            if until_epoch > TradierClient._quota_until_epoch:
                TradierClient._quota_until_epoch = until_epoch

        return True

    def __init__(self, token: str, env: str = "live"):
        self.token = token.strip()
        self.env = (env or "live").strip().lower()

        # Tradier environments (docs):
        # - live:    https://api.tradier.com/v1/...
        # - sandbox: https://sandbox.tradier.com/v1/...
        if self.env == "sandbox":
            self.base_url = "https://sandbox.tradier.com/v1"
        else:
            self.base_url = "https://api.tradier.com/v1"

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            }
        )

    # ----------------------------------------------------------
    # GET helper with retry/backoff (always returns dict)
    # ----------------------------------------------------------
    def get(
        self,
        path: str,
        params: dict | None = None,
        timeout: float = 20.0,
        max_retries: int = 4,
    ) -> dict:
        url = f"{self.base_url}{path}"
        backoff = 1.5

        for attempt in range(1, max_retries + 1):
            # ✅ If we are currently in a known quota window, wait before calling
            self._maybe_wait_for_quota()

            try:
                r = self.session.get(url, params=params, timeout=timeout)

                # Capture snippet once (used by multiple branches)
                body_text = ""
                try:
                    body_text = r.text or ""
                except Exception:
                    body_text = ""

                # ✅ Quota violation (Tradier often returns 400 with this body)
                # Example: "Quota Violation: Expires 1769804340000"
                if r.status_code in (400, 429) and self._record_quota_from_body(body_text):
                    # Don’t burn retries aggressively; just wait and retry
                    # (your outer loop will re-check quota and block)
                    logger.warning(
                        f"Tradier quota violation on {path} params={params} "
                        f"(attempt {attempt}/{max_retries}); waiting for reset"
                    )
                    time.sleep(0.2)
                    continue

                # Rate limit (classic)
                if r.status_code == 429:
                    logger.warning(
                        f"Tradier 429 rate-limit on {path} (attempt {attempt}/{max_retries}); sleeping {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # Retry on 5xx
                if 500 <= r.status_code < 600:
                    logger.warning(
                        f"Tradier {r.status_code} server error on {path} (attempt {attempt}/{max_retries}); sleeping {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                if r.status_code >= 400:
                    snippet = (body_text or "")[:200].replace("\n", " ")
                    logger.warning(f"Tradier HTTP {r.status_code} on {path} params={params} body={snippet}")

                # Hard fail on other non-2xx
                r.raise_for_status()

                # Empty body => retry (transient)
                if not r.content:
                    logger.warning(
                        f"Tradier empty response body on {path} (attempt {attempt}/{max_retries}); sleeping {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # JSON parse with retry
                try:
                    data = r.json()
                except Exception as e:
                    logger.warning(
                        f"Tradier JSON decode error on {path}: {e} (attempt {attempt}/{max_retries}); sleeping {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # Tradier sometimes returns null / non-dict
                if not isinstance(data, dict):
                    logger.warning(f"Tradier unexpected JSON type on {path}: {type(data).__name__}")
                    return {}

                # API fault => retry
                if "fault" in data:
                    logger.warning(
                        f"Tradier fault on {path}: {data.get('fault')} (attempt {attempt}/{max_retries}); sleeping {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                return data

            except requests.RequestException as e:
                err = str(e)
                if attempt < max_retries:
                    logger.warning(
                        f"Tradier request error on {path}: {err} (attempt {attempt}/{max_retries}); sleeping {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise

        raise RuntimeError(f"Tradier GET failed after {max_retries} attempts: {path}")




    # ----------------------------------------------------------
    # Get underlying quote (last/close fallback)
    # ----------------------------------------------------------
    def get_underlying_price(self, symbol: str) -> float | None:
        data = self.get("/markets/quotes", params={"symbols": symbol}, timeout=15.0)
        quotes = data.get("quotes", {})
        q = quotes.get("quote")

        if isinstance(q, list):
            # try to pick the matching symbol
            q_match = None
            for item in q:
                if isinstance(item, dict) and (item.get("symbol") == symbol):
                    q_match = item
                    break
            q = q_match or (q[0] if q else None)

        if not isinstance(q, dict):
            return None

        last = q.get("last")
        close = q.get("close")

        try:
            if last is not None and not pd.isna(last):
                return float(last)
        except Exception:
            pass

        try:
            if close is not None and not pd.isna(close):
                return float(close)
        except Exception:
            pass

        return None


    # ----------------------------------------------------------
    # Get option expirations (returns list[str] of YYYY-MM-DD)
    # ----------------------------------------------------------
    def get_expirations(self, symbol: str, include_all_roots: bool = True) -> list[str]:
        params = {"symbol": symbol}
        if include_all_roots:
            params["includeAllRoots"] = "true"

        data = self.get("/markets/options/expirations", params=params, timeout=20.0)

        # ✅ Critical fix: Tradier can return None (or something unexpected)
        if not isinstance(data, dict):
            return []

        expirations = data.get("expirations")

        # Sometimes expirations itself may be None
        if not isinstance(expirations, dict):
            return []

        dates = expirations.get("date")

        # Can be a single string or a list of strings
        if isinstance(dates, str) and dates:
            return [dates]

        if isinstance(dates, list):
            return [str(x) for x in dates if x]

        return []


    # ----------------------------------------------------------
    # Get option chain for a given expiry (YYYY-MM-DD)
    # Returns tuple(DataFrame calls, DataFrame puts)
    # ----------------------------------------------------------
    def get_option_chain(self, symbol: str, expiration: str, greeks: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        params = {"symbol": symbol, "expiration": expiration}
        if greeks:
            params["greeks"] = "true"

        data = self.get("/markets/options/chains", params=params, timeout=25.0)

        # ✅ Critical fix: Tradier can return None or unexpected
        if not isinstance(data, dict):
            return pd.DataFrame(), pd.DataFrame()

        options = data.get("options")
        if not isinstance(options, dict):
            return pd.DataFrame(), pd.DataFrame()

        opts = options.get("option")
        if isinstance(opts, dict):
            opts = [opts]
        if not isinstance(opts, list):
            return pd.DataFrame(), pd.DataFrame()

        rows = []
        for o in opts:
            if not isinstance(o, dict):
                continue

            opt_type = (o.get("option_type") or "").lower()  # "call" / "put"
            strike = o.get("strike")
            bid = o.get("bid")
            ask = o.get("ask")
            last = o.get("last")
            vol = o.get("volume")
            oi = o.get("open_interest")

            g = o.get("greeks") if isinstance(o.get("greeks"), dict) else {}

            # IV: prefer ORATS smoothed vol, else mid_iv, else bid/ask iv
            iv = g.get("smv_vol")
            if iv is None:
                iv = g.get("mid_iv")
            if iv is None:
                iv = g.get("bid_iv")
            if iv is None:
                iv = g.get("ask_iv")

            delta = g.get("delta")

            rows.append(
                {
                    "option_type": opt_type,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "lastPrice": last,          # match your existing column name
                    "volume": vol,
                    "openInterest": oi,         # match your existing column name
                    "impliedVolatility": iv,    # match your existing column name
                    "delta": delta,             # Tradier-provided delta (best)
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Coerce numeric
        for c in ["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        calls = df[df["option_type"] == "call"].copy()
        puts = df[df["option_type"] == "put"].copy()
        return calls, puts


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
# (kept for parity with your existing architecture)
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
# - Tradier usually provides delta directly; BS is fallback only.
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

    windowed = [(e, d) for (e, d) in candidates if dte_min <= d <= dte_max]
    if windowed:
        windowed.sort(key=lambda x: x[1])
        return windowed[0][0], windowed[0][1]

    future = [(e, d) for (e, d) in candidates if d > 0]
    if not future:
        return expiries[0], None
    future.sort(key=lambda x: x[1])
    return future[0][0], future[0][1]

# ----------------------------------------------------------
# Function: Pick a strike closest to target delta (+0.20 call / -0.20 put)
# - Prefers Tradier "delta" if present; otherwise uses BS-estimated delta.
# - Premium uses mid(bid/ask) first, else lastPrice.
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
    max_spread_pct: float = 0.30,
    ticker: str | None = None
) -> dict | None:
    if df is None or df.empty:
        return None

    if "strike" not in df.columns:
        return None

    # Need IV only if we must BS fallback
    has_tradier_delta = ("delta" in df.columns) and df["delta"].notna().any()

    label = ticker or "UNKNOWN"
    logger.info(f"{label} {option_type.upper()}: using_tradier_delta={has_tradier_delta}")

    if not has_tradier_delta and "impliedVolatility" not in df.columns:
        return None

    work = df.dropna(subset=["strike"]).copy()

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

    T_days = (exp_ts - today).total_seconds() / 86400.0
    T = max(T_days, 0.001) / 365.0

    def f(x):
        try:
            if x is None or pd.isna(x):
                return None
            return float(x)
        except Exception:
            return None

    deltas = []
    premiums = []
    spreads_pct = []

    opt_is_put = option_type.lower() == "put"

    for _, row in work.iterrows():
        K = f(row.get("strike"))
        if K is None:
            deltas.append(None)
            premiums.append(None)
            spreads_pct.append(None)
            continue

        # Prefer Tradier delta if present
        d = f(row.get("delta"))

        # Normalize delta sign just in case
        if d is not None:
            if opt_is_put and d > 0:
                d = -abs(d)
            elif (not opt_is_put) and d < 0:
                d = abs(d)

        # Fallback to BS delta if needed
        if d is None:
            sigma = f(row.get("impliedVolatility"))
            if sigma is None:
                d = None
            else:
                d = bs_delta(option_type=option_type, S=S, K=K, T=T, r=r, sigma=sigma, q=q)

        deltas.append(d)

        bid = f(row.get("bid"))
        ask = f(row.get("ask"))
        last = f(row.get("lastPrice"))

        prem = None
        if bid is not None and ask is not None and ask >= bid:
            prem = (bid + ask) / 2.0
        elif last is not None:
            prem = last
        premiums.append(prem)

        sp = None
        if bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2.0
            if mid and mid > 0:
                sp = (ask - bid) / mid
        spreads_pct.append(sp)

    work["est_delta"] = deltas
    work["premium"] = premiums
    work["spread_pct"] = spreads_pct

    work = work.dropna(subset=["est_delta", "premium"])
    if work.empty:
        return None

    work = work[(work["spread_pct"].isna()) | (work["spread_pct"] <= max_spread_pct)]
    if work.empty:
        return None

    work["delta_dist"] = (work["est_delta"] - target_delta).abs()
    best = work.loc[work["delta_dist"].idxmin()]

    iv_val = f(best.get("impliedVolatility"))

    return {
        "strike": float(best["strike"]),
        "iv": iv_val,
        "delta": float(best["est_delta"]),
        "premium": float(best["premium"]),
        "bid": f(best.get("bid")),
        "ask": f(best.get("ask")),
        "spread_pct": f(best.get("spread_pct")),
        "open_interest": int(best["openInterest"]) if "openInterest" in best and not pd.isna(best["openInterest"]) else None,
        "volume": int(best["volume"]) if "volume" in best and not pd.isna(best["volume"]) else None,
    }


# ----------------------------------------------------------
# Function: Fetch IV summary + ATM + 20Δ Put/Call for Wheel
# Uses Tradier instead of yfinance
# ----------------------------------------------------------
def fetch_iv_summary_for_ticker(
    ticker: str,
    tradier: TradierClient,
    max_retries: int = 3,
    initial_backoff: float = 3.0,
    timeout_sec: float = 60.0,
    dte_min: int = 5,
    dte_max: int = 10,
    risk_free_rate: float = 0.05,
    min_oi: int = 0,
    min_vol: int = 0,
    max_spread_pct: float = 0.30,
    force: bool = False
) -> dict | None:
    today = pd.Timestamp.utcnow().date()

    with engine.connect() as conn:
        exists_row = conn.execute(
            text("SELECT 1 FROM iv_summary WHERE ticker = :ticker AND fetch_date = :today"),
            {"ticker": ticker, "today": today}
        ).fetchone()
    if exists_row and not force:
        logger.info(f"{ticker}: already processed for today ({today}) — skipping.")
        return None

    logger.info(f"{ticker}: starting fetch (Tradier).")

    # ----------------------------------------------------------
    # Get expirations
    # ----------------------------------------------------------
    expiries = []
    attempt = 0
    backoff = initial_backoff
    while attempt < max_retries:
        try:
            expiries = run_with_timeout(tradier.get_expirations, args=(ticker,), timeout_sec=timeout_sec)
            break
        except Exception as e:
            logger.warning(f"{ticker}: expirations error attempt {attempt+1}/{max_retries}: {e}; sleeping {backoff}s")
            time.sleep(backoff)
            attempt += 1
            backoff *= 2

    if not expiries:
        logger.warning(f"{ticker}: no option expirations found")
        return None

    expiry_str, dte = pick_expiry(expiries, dte_min=dte_min, dte_max=dte_max)
    if not expiry_str:
        logger.warning(f"{ticker}: could not select expiry")
        return None

    logger.info(f"{ticker}: selected expiry {expiry_str} (DTE={dte}).")

    # ----------------------------------------------------------
    # Get option chain
    # ----------------------------------------------------------
    attempt = 0
    backoff = initial_backoff
    calls = pd.DataFrame()
    puts = pd.DataFrame()

    while attempt < max_retries:
        logger.info(f"{ticker}: chain attempt {attempt+1}/{max_retries}. Timeout per attempt = {timeout_sec}s.")
        try:
            calls, puts = run_with_timeout(
                tradier.get_option_chain,
                args=(ticker, expiry_str, True),
                timeout_sec=timeout_sec
            )
            logger.info(f"{ticker}: option chain returned successfully on attempt {attempt+1}.")
            break
        except TimeoutError:
            logger.warning(f"{ticker}: chain fetch timed out on attempt {attempt+1}/{max_retries}; sleeping {backoff}s")
            time.sleep(backoff)
            attempt += 1
            backoff *= 2
            continue
        except Exception as e:
            logger.warning(f"{ticker}: chain fetch error attempt {attempt+1}/{max_retries}: {e}; sleeping {backoff}s")
            time.sleep(backoff)
            attempt += 1
            backoff *= 2
            continue
    else:
        logger.error(f"{ticker}: exceeded max retries fetching chain; skipping.")
        return None

    if calls.empty:
        logger.warning(f"{ticker}: calls chain empty for expiry {expiry_str}")
        return None

    # Calls-only IV stats (match your existing behavior)
    calls_iv = calls.dropna(subset=["impliedVolatility"]) if "impliedVolatility" in calls.columns else pd.DataFrame()
    if calls_iv.empty:
        logger.warning(f"{ticker}: no call option data w/ impliedVolatility for expiry {expiry_str}")
        return None

    stats = calls_iv["impliedVolatility"].describe()
    logger.info(f"{ticker}: impliedVolatility stats computed.")

    # ----------------------------------------------------------
    # Underlying price
    # ----------------------------------------------------------
    price = None
    attempt = 0
    backoff = initial_backoff
    while attempt < max_retries:
        try:
            price = run_with_timeout(tradier.get_underlying_price, args=(ticker,), timeout_sec=timeout_sec)
            break
        except Exception as e:
            logger.warning(f"{ticker}: quote error attempt {attempt+1}/{max_retries}: {e}; sleeping {backoff}s")
            time.sleep(backoff)
            attempt += 1
            backoff *= 2

    if price is None or price <= 0:
        logger.warning(f"{ticker}: could not retrieve underlying price")
        return None

    # Expiry timestamp for delta calculations (UTC)
    expiry_dt = pd.to_datetime(expiry_str)
    if expiry_dt.tz is None:
        expiry_dt = expiry_dt.tz_localize("UTC")
    else:
        expiry_dt = expiry_dt.tz_convert("UTC")
    expiry_dt = expiry_dt.normalize() + pd.Timedelta(hours=21)
    expiry_date = expiry_dt.date()

    # ----------------------------------------------------------
    # ATM call
    # ----------------------------------------------------------
    atm_iv = None
    atm_premium = None
    atm_premium_pct = None

    calls_iv = calls_iv.copy()
    calls_iv["dist"] = (calls_iv["strike"] - price).abs()
    atm_call = calls_iv.loc[calls_iv["dist"].idxmin()]

    atm_iv = float(atm_call["impliedVolatility"])
    bid = atm_call.get("bid", None)
    ask = atm_call.get("ask", None)
    last_price = atm_call.get("lastPrice", None)

    # Prefer mid, else last
    if bid is not None and ask is not None and not pd.isna(bid) and not pd.isna(ask) and float(ask) >= float(bid):
        atm_premium = (float(bid) + float(ask)) / 2.0
        atm_premium_pct = atm_premium / price
        logger.info(
            f"{ticker}: ATM strike {atm_call['strike']}, iv_atm={atm_iv:.6f}, "
            f"premium(mid)={atm_premium:.2f}, premium_pct={atm_premium_pct:.4f}"
        )
    elif last_price is not None and not (pd.isna(last_price) or np.isnan(last_price)):
        atm_premium = float(last_price)
        atm_premium_pct = atm_premium / price
        logger.info(
            f"{ticker}: ATM strike {atm_call['strike']}, iv_atm={atm_iv:.6f}, "
            f"premium(last)={atm_premium:.2f}, premium_pct={atm_premium_pct:.4f}"
        )
    else:
        logger.warning(f"{ticker}: ATM premium not available (no bid/ask and lastPrice missing).")

    # ----------------------------------------------------------
    # 20-delta PUT (CSP candidate)
    # ----------------------------------------------------------
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
            max_spread_pct=max_spread_pct,
            ticker=ticker,
        )

    # ----------------------------------------------------------
    # 20-delta CALL (CC candidate)
    # ----------------------------------------------------------
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
            max_spread_pct=max_spread_pct,
            ticker=ticker,
        )

    # Premium percentages
    put_20_premium_pct = None
    if put_20 and put_20.get("premium") is not None and put_20.get("strike") is not None and put_20["strike"] > 0:
        put_20_premium_pct = float(put_20["premium"]) / float(put_20["strike"])

    call_20_premium_pct = None
    if call_20 and call_20.get("premium") is not None and price > 0:
        call_20_premium_pct = float(call_20["premium"]) / float(price)

    if put_20:
        logger.debug(
            f"{ticker}: PUT ~-0.20Δ strike={put_20['strike']} delta={put_20['delta']:.3f} "
            f"prem={put_20['premium']:.2f} prem_pct={put_20_premium_pct}"
        )
    else:
        logger.debug(f"{ticker}: PUT ~-0.20Δ not found (filters/chain availability).")

    if call_20:
        logger.debug(
            f"{ticker}: CALL ~+0.20Δ strike={call_20['strike']} delta={call_20['delta']:.3f} "
            f"prem={call_20['premium']:.2f} prem_pct={call_20_premium_pct}"
        )
    else:
        logger.debug(f"{ticker}: CALL ~+0.20Δ not found (filters/chain availability).")

    logger.info(
        f"{ticker}: ATM strike {atm_call['strike']}, iv_atm={atm_iv:.3f}, "
        f"premium={atm_premium:.2f}, pct={atm_premium_pct:.4f}"
    )


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
        "put_20d_iv": float(put_20["iv"]) if put_20 and put_20.get("iv") is not None else None,
        "put_20d_premium": float(put_20["premium"]) if put_20 else None,
        "put_20d_premium_pct": float(put_20_premium_pct) if put_20_premium_pct is not None else None,
        "put_20d_delta": float(put_20["delta"]) if put_20 else None,

        "call_20d_strike": float(call_20["strike"]) if call_20 else None,
        "call_20d_iv": float(call_20["iv"]) if call_20 and call_20.get("iv") is not None else None,
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
    tradier: TradierClient,
    max_workers: int = 3,
    delay_between: float = 2.0,
    max_retries: int = 3,
    timeout_sec: float = 60.0,
    dte_min: int = 5,
    dte_max: int = 10,
    risk_free_rate: float = 0.05,
    min_oi: int = 0,
    min_vol: int = 0,
    max_spread_pct: float = 0.30,
    force: bool = False
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
                tradier,
                max_retries,
                3.0,            # initial_backoff (smaller than yfinance)
                timeout_sec,
                dte_min,
                dte_max,
                risk_free_rate,
                min_oi,
                min_vol,
                max_spread_pct,
                force
            ): t
            for t in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                res = future.result()

                if res:
                    summaries.append(res)
                    logger.info(f"{ticker}: summary collected successfully.")
                else:
                    logger.info(f"{ticker}: summary returned None (skipped or error).")

            except Exception:
                logger.exception(f"{ticker}: unexpected error in future result")

            # ----------------------------------------------------------
            # Throttle only in single-worker mode
            # (Prevents stacking sleeps when running parallel)
            # ----------------------------------------------------------
            if delay_between > 0 and max_workers == 1:
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
        ON CONFLICT (ticker, fetch_date) DO UPDATE SET
            expiry = EXCLUDED.expiry,
            dte = EXCLUDED.dte,
            underlying_price = EXCLUDED.underlying_price,

            contract_count = EXCLUDED.contract_count,
            iv_mean = EXCLUDED.iv_mean,
            iv_std = EXCLUDED.iv_std,
            iv_min = EXCLUDED.iv_min,
            iv_25 = EXCLUDED.iv_25,
            iv_median = EXCLUDED.iv_median,
            iv_75 = EXCLUDED.iv_75,
            iv_max = EXCLUDED.iv_max,

            iv_atm = EXCLUDED.iv_atm,
            atm_premium = EXCLUDED.atm_premium,
            atm_premium_pct = EXCLUDED.atm_premium_pct,

            put_20d_strike = EXCLUDED.put_20d_strike,
            put_20d_iv = EXCLUDED.put_20d_iv,
            put_20d_premium = EXCLUDED.put_20d_premium,
            put_20d_premium_pct = EXCLUDED.put_20d_premium_pct,
            put_20d_delta = EXCLUDED.put_20d_delta,

            call_20d_strike = EXCLUDED.call_20d_strike,
            call_20d_iv = EXCLUDED.call_20d_iv,
            call_20d_premium = EXCLUDED.call_20d_premium,
            call_20d_premium_pct = EXCLUDED.call_20d_premium_pct,
            call_20d_delta = EXCLUDED.call_20d_delta
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
    parser = argparse.ArgumentParser(description="Fetch option-chain IV summary + 20Δ strikes and store in DB (Tradier)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=str, help="Optional path to CSV with tickers (expects 'ticker' column)")
    group.add_argument("--tickers", nargs="+", type=str, help="List of tickers (e.g., TSLA AAPL MSFT)")

    parser.add_argument("--workers", type=int, default=6, help="Max parallel workers (default: 6)")
    parser.add_argument("--delay", type=float, default=0.0, help="Throttle delay in seconds (only applied when --workers=1). Default: 0")
    parser.add_argument("--retries", type=int, default=3, help="Max retries on timeout/rate limit errors")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout seconds for each fetch step (default: 30)")

    parser.add_argument("--batch_size", type=int, default=200, help="Number of tickers per batch (default: 200)")
    parser.add_argument("--pause_between_batches", type=float, default=2.0, help="Pause seconds between batches (default: 2)")

    # Wheel / 20Δ settings
    parser.add_argument("--dte_min", type=int, default=5, help="Min DTE for expiry selection (default: 5)")
    parser.add_argument("--dte_max", type=int, default=10, help="Max DTE for expiry selection (default: 10)")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate used for BS fallback delta (default: 0.05)")
    parser.add_argument("--min_oi", type=int, default=0, help="Minimum open interest filter (default: 0)")
    parser.add_argument("--min_vol", type=int, default=0, help="Minimum volume filter (default: 0)")
    parser.add_argument("--max_spread_pct", type=float, default=0.30, help="Max bid/ask spread pct vs mid (default: 0.30)")

    parser.add_argument("--force",action="store_true",help="Re-fetch and overwrite today's IV snapshot (testing mode)"
)


    args = parser.parse_args()

    # ----------------------------------------------------------
    # Tradier init
    # ----------------------------------------------------------
    token = os.getenv("TRADIER_TOKEN", "").strip()
    env = os.getenv("TRADIER_ENV", "live").strip()

    if not token:
        raise SystemExit("Missing TRADIER_TOKEN env var. Example: export TRADIER_TOKEN='YOUR_TOKEN_HERE'")

    tradier = TradierClient(token=token, env=env)
    logger.info(f"Tradier initialized (env={env}).")

    # ----------------------------------------------------------
    # Tickers
    # ----------------------------------------------------------
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
            tradier=tradier,
            max_workers=args.workers,
            delay_between=args.delay,
            max_retries=args.retries,
            timeout_sec=args.timeout,
            dte_min=args.dte_min,
            dte_max=args.dte_max,
            risk_free_rate=args.r,
            min_oi=args.min_oi,
            min_vol=args.min_vol,
            max_spread_pct=args.max_spread_pct,
            force=args.force
        )

        if i + args.batch_size < len(tickers_list):
            logger.info(f"Batch {i//args.batch_size + 1} done — sleeping {args.pause_between_batches}s before next batch.")
            time.sleep(args.pause_between_batches)

    logger.info("Finished IV summary fetch.")
