# greer_fair_value_calculator.py
# ----------------------------------------------------------
# Computes Greer Fair Value (per-share) daily and stores results
# in greer_fair_value_daily.
#
# This version separates growth sources by model:
#   - DCF: base = FCF/share, growth = FCF/share growth
#   - Graham: base = EPS, growth = EPS growth (in PERCENT inside formula)
#
# Growth estimation:
#   - Methods: auto (default), reg (log-linear), cagr
#   - Caps: growth_cap (default 0.15) and growth_floor (default -0.10)
#   - Manual overrides: --growth_rate_fcf, --growth_rate_eps
#
# DB writes:
#   - growth_rate_fcf, growth_rate_eps
#   - growth_method_fcf, growth_method_eps
#   - legacy growth_rate continues to store FCF growth for backward compat
#
# Upsert key: (ticker, date)
# ----------------------------------------------------------

import argparse
import logging
import os
import math
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

# ----------------------------------------------------------
# DB helpers (project-standard)
# ----------------------------------------------------------
from db import get_engine, get_psycopg_connection  # noqa: F401

# ----------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "greer_fair_value_calculator.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Type normalization
# ----------------------------------------------------------
def to_float(x) -> Optional[float]:
    """Coerce Decimal/str/int/float/NaN to clean float or None."""
    if x is None:
        return None
    try:
        if isinstance(x, Decimal):
            try:
                if x.is_nan():  # type: ignore[attr-defined]
                    return None
            except Exception:
                pass
            return float(x)
        if isinstance(x, (int, float)):
            if isinstance(x, float) and math.isnan(x):
                return None
            return float(x)
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (InvalidOperation, ValueError, TypeError):
        return None


# ----------------------------------------------------------
# Load tickers (DB default, or --file / --tickers)
# ----------------------------------------------------------
def load_tickers(engine, tickers_arg: Optional[str] = None, file_arg: Optional[str] = None) -> List[str]:
    if tickers_arg:
        return [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    if file_arg:
        df = pd.read_csv(file_arg)
        col = next((c for c in df.columns if c.lower() in ("ticker", "tickers", "symbol")), None)
        if not col:
            raise ValueError("CSV must contain a column named ticker/tickers/symbol")
        return df[col].dropna().astype(str).str.upper().unique().tolist()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker FROM companies WHERE COALESCE(delisted,false) = false")).fetchall()
    return [r[0].upper() for r in rows]


# ----------------------------------------------------------
# Financial helpers (FY-only, latest <= as_of)
# ----------------------------------------------------------
def get_latest_eps_and_fcfps(engine, ticker: str, as_of: date) -> Tuple[Optional[float], Optional[float]]:
    """
    EPS = net_income / shares_outstanding
    FCF/share = free_cash_flow / shares_outstanding
    """
    q = text("""
        WITH f AS (
          SELECT net_income, free_cash_flow, shares_outstanding
          FROM financials
          WHERE ticker = :t AND report_date <= :d
          ORDER BY report_date DESC
          LIMIT 1
        )
        SELECT
          CASE WHEN shares_outstanding IS NOT NULL AND shares_outstanding <> 0
               THEN net_income::numeric / shares_outstanding::numeric
               ELSE NULL END AS eps,
          CASE WHEN shares_outstanding IS NOT NULL AND shares_outstanding <> 0
               THEN free_cash_flow::numeric / shares_outstanding::numeric
               ELSE NULL END AS fcfps
        FROM f
    """)
    with engine.connect() as conn:
        row = conn.execute(q, {"t": ticker, "d": as_of}).fetchone()
    if not row:
        return None, None
    return to_float(row.eps), to_float(row.fcfps)


def get_fcfps_history_fy(engine, ticker: str, max_points: int, as_of: date) -> pd.DataFrame:
    q = text("""
        SELECT report_date::date AS report_date,
               CASE WHEN shares_outstanding IS NOT NULL AND shares_outstanding <> 0
                    THEN free_cash_flow::numeric / shares_outstanding::numeric
                    ELSE NULL END AS fcfps
        FROM financials
        WHERE ticker = :t AND report_date <= :d
        ORDER BY report_date DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"t": ticker, "d": as_of, "lim": max_points})
    if df.empty:
        return df
    df["fcfps"] = df["fcfps"].apply(to_float)
    df = df.dropna(subset=["fcfps"]).copy()
    return df.iloc[::-1].reset_index(drop=True)  # oldest -> newest


def get_eps_history_fy(engine, ticker: str, max_points: int, as_of: date) -> pd.DataFrame:
    q = text("""
        SELECT report_date::date AS report_date,
               CASE WHEN shares_outstanding IS NOT NULL AND shares_outstanding <> 0
                    THEN net_income::numeric / shares_outstanding::numeric
                    ELSE NULL END AS eps
        FROM financials
        WHERE ticker = :t AND report_date <= :d
        ORDER BY report_date DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"t": ticker, "d": as_of, "lim": max_points})
    if df.empty:
        return df
    df["eps"] = df["eps"].apply(to_float)
    df = df.dropna(subset=["eps"]).copy()
    return df.iloc[::-1].reset_index(drop=True)  # oldest -> newest


# ----------------------------------------------------------
# Growth estimators
# ----------------------------------------------------------
def compute_cagr_from_series(s: pd.Series) -> Optional[float]:
    """CAGR from first->last value; requires >=2 positive points."""
    if s is None or len(s) < 2:
        return None
    vals = s.dropna().astype(float)
    if len(vals) < 2:
        return None
    start = float(vals.iloc[0])
    end = float(vals.iloc[-1])
    n = len(vals) - 1
    if start <= 0 or end <= 0 or n <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0


def compute_log_reg_growth(vals: pd.Series) -> Optional[float]:
    """
    Regress ln(values) on time index to estimate constant growth:
      ln(v_t) = a + b*t  => g = exp(b) - 1
    Uses only strictly-positive points. Requires >= 3 positives.
    """
    if vals is None:
        return None
    v = pd.to_numeric(vals, errors="coerce").dropna().astype(float)
    v = v[v > 0]  # <- critical: logs require positive values
    n = len(v)
    if n < 3:
        return None

    y = v.values
    t = np.arange(n, dtype=float)

    y_log = np.log(y)  # safe now
    t_mean = t.mean()
    y_mean = y_log.mean()
    den = np.sum((t - t_mean) ** 2)
    if den == 0:
        return None
    b = np.sum((t - t_mean) * (y_log - y_mean)) / den
    return float(np.exp(b) - 1.0)


def choose_growth(
    vals: pd.Series,
    method: str,
    cap: float,
    floor: float,
    manual: Optional[float]
) -> Tuple[Optional[float], str, int]:
    """
    Returns (g_used, method_used, points_used).
    - manual overrides everything
    - 'reg' uses log-reg on positive points (>=3 needed)
    - 'cagr' uses CAGR on positive points (>=2 endpoints)
    - 'auto' tries reg then falls back to CAGR
    """
    if manual is not None:
        return apply_caps(to_float(manual), cap, floor), "manual", len(vals or [])

    if vals is None or len(vals) == 0:
        return None, method, 0

    series = pd.to_numeric(vals, errors="coerce").dropna().astype(float)
    pos_series = series[series > 0]  # useful for both methods

    if method == "cagr":
        g = compute_cagr_from_series(pos_series)
        return apply_caps(g, cap, floor), "cagr", len(pos_series)

    if method == "reg":
        g = compute_log_reg_growth(series)  # it filters positives internally
        return apply_caps(g, cap, floor), "reg", len(series[series > 0])

    # auto: prefer reg; if unavailable, fall back to CAGR
    g = compute_log_reg_growth(series)
    if g is not None:
        return apply_caps(g, cap, floor), "reg", len(series[series > 0])

    g = compute_cagr_from_series(pos_series)
    used = "cagr" if g is not None else "auto"
    return apply_caps(g, cap, floor), used, len(pos_series)


def apply_caps(g: Optional[float], cap: float, floor: float) -> Optional[float]:
    if g is None:
        return None
    return max(floor, min(cap, g))


# ----------------------------------------------------------
# Valuation models
# ----------------------------------------------------------
def ben_graham_value(eps: Optional[float], g_eps: Optional[float], Y: Optional[float]) -> Optional[float]:
    """
    V = EPS * (8.5 + 2*g_percent) * (4.4 / Y)
    g_eps is a decimal rate (e.g., 0.10), converted to percent within.
    """
    eps_f = to_float(eps)
    g_f = to_float(g_eps)
    Y_f = to_float(Y)
    if eps_f is None or eps_f <= 0:
        return None
    if g_f is None or g_f < 0:
        return None
    if Y_f is None or Y_f <= 0:
        return None
    return eps_f * (8.5 + 2.0 * (g_f * 100.0)) * (4.4 / Y_f)


def dcf_value_from_fcfps(fcfps: Optional[float], g_fcf: Optional[float], r: Optional[float], n: int, terminal_g: Optional[float]) -> Optional[float]:
    """
    Simple per-share DCF on FCF:
      - Grow n years at g_fcf discounted at r
      - Terminal value with Gordon Growth at terminal_g
    """
    f = to_float(fcfps)
    g = to_float(g_fcf)
    r_ = to_float(r)
    tg = to_float(terminal_g)
    if f is None or f <= 0: return None
    if g is None: return None
    if r_ is None or tg is None: return None
    if n is None or n <= 0: return None
    if r_ <= tg: return None
    pv = 0.0
    cf = f
    for i in range(1, n + 1):
        cf *= (1.0 + g)
        pv += cf / ((1.0 + r_) ** i)
    terminal_cf = cf * (1.0 + tg)
    terminal_val = terminal_cf / (r_ - tg)
    pv += terminal_val / ((1.0 + r_) ** n)
    return pv


# ----------------------------------------------------------
# Prices
# ----------------------------------------------------------
def get_price_range(engine, ticker: str, start: date, end: date) -> pd.DataFrame:
    q = text("""
        SELECT date::date AS d, close::numeric AS close
        FROM prices
        WHERE ticker = :t AND date >= :s AND date <= :e
        ORDER BY date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"t": ticker, "s": start, "e": end})
    if not df.empty:
        df["close"] = df["close"].apply(to_float)
    return df


# ----------------------------------------------------------
# Upsert
# ----------------------------------------------------------
def upsert_gfv_row(engine, row: dict) -> None:
    q = text("""
        INSERT INTO greer_fair_value_daily
        (date, ticker, close_price, eps, fcf_per_share,
         cagr_years, growth_rate,                      -- legacy (stores FCF growth)
         growth_rate_fcf, growth_rate_eps,
         growth_method_fcf, growth_method_eps,
         discount_rate, terminal_growth, graham_yield_Y,
         graham_value, dcf_value, gfv_price, gfv_status)
        VALUES
        (:date, :ticker, :close_price, :eps, :fcfps,
         :cagr_years, :growth_rate,                    -- legacy
         :growth_rate_fcf, :growth_rate_eps,
         :growth_method_fcf, :growth_method_eps,
         :discount_rate, :terminal_growth, :graham_yield_Y,
         :graham_value, :dcf_value, :gfv_price, :gfv_status)
        ON CONFLICT (ticker, date) DO UPDATE SET
          close_price        = EXCLUDED.close_price,
          eps                = EXCLUDED.eps,
          fcf_per_share      = EXCLUDED.fcf_per_share,
          cagr_years         = EXCLUDED.cagr_years,
          growth_rate        = EXCLUDED.growth_rate,           -- legacy stays in sync with FCF growth
          growth_rate_fcf    = EXCLUDED.growth_rate_fcf,
          growth_rate_eps    = EXCLUDED.growth_rate_eps,
          growth_method_fcf  = EXCLUDED.growth_method_fcf,
          growth_method_eps  = EXCLUDED.growth_method_eps,
          discount_rate      = EXCLUDED.discount_rate,
          terminal_growth    = EXCLUDED.terminal_growth,
          graham_yield_Y     = EXCLUDED.graham_yield_Y,
          graham_value       = EXCLUDED.graham_value,
          dcf_value          = EXCLUDED.dcf_value,
          gfv_price          = EXCLUDED.gfv_price,
          gfv_status         = EXCLUDED.gfv_status,
          updated_at         = now()
    """)
    with engine.begin() as conn:
        conn.execute(q, row)


# ----------------------------------------------------------
# Display status
# ----------------------------------------------------------
def pick_status_and_display(price: Optional[float], dcfv: Optional[float], gv: Optional[float]) -> Tuple[Optional[float], str]:
    p = to_float(price); d = to_float(dcfv); g = to_float(gv)
    if p is None or p <= 0: return None, "red"
    candidates = [v for v in (d, g) if v is not None and v > 0]
    if not candidates: return None, "red"
    minv = min(candidates); maxv = max(candidates)
    if p <= minv: return minv, "gold"
    elif p <= maxv: return maxv, "green"
    else: return maxv, "red"


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def run(args) -> None:
    engine = get_engine()
    tickers = load_tickers(engine, args.tickers, args.file)

    # Date range
    if args.full:
        start = pd.to_datetime(args.start).date() if args.start else (date.today() - timedelta(days=1825))
    else:
        start = date.today() - timedelta(days=args.days)
    end = date.today()

    # Model params (floats with safe fallbacks)
    Y_model = to_float(args.graham_Y) or 4.4
    r_model = to_float(args.discount_rate) or 0.10
    tg_model = to_float(args.terminal_growth) or 0.03
    n_years = int(args.dcf_years) if args.dcf_years else 5

    for t in tickers:
        try:
            logger.info(f"Computing GFV for {t} {start}..{end}")
            prices = get_price_range(engine, t, start, end)
            if prices.empty:
                logger.info(f"No prices for {t} in range; skipping")
                continue

            for _, pr in prices.iterrows():
                d = pr["d"]
                px = to_float(pr["close"])

                # Latest fundamentals
                eps_latest, fcfps_latest = get_latest_eps_and_fcfps(engine, t, d)

                # FY histories (oldest -> newest)
                fcf_df = get_fcfps_history_fy(engine, t, args.cagr_years, d)
                eps_df = get_eps_history_fy(engine, t, args.cagr_years, d)

                # Choose growths independently
                g_fcf, used_fcf, pts_fcf = choose_growth(
                    fcf_df["fcfps"] if not fcf_df.empty else pd.Series(dtype=float),
                    method=args.growth_method_fcf,
                    cap=args.growth_cap_fcf,
                    floor=args.growth_floor_fcf,
                    manual=args.growth_rate_fcf
                )
                g_eps, used_eps, pts_eps = choose_growth(
                    eps_df["eps"] if not eps_df.empty else pd.Series(dtype=float),
                    method=args.growth_method_eps,
                    cap=args.growth_cap_eps,
                    floor=args.growth_floor_eps,
                    manual=args.growth_rate_eps
                )

                logger.debug(f"{t} {d} FCF growth={g_fcf} ({used_fcf},{pts_fcf}pts)  EPS growth={g_eps} ({used_eps},{pts_eps}pts)")

                # Valuations
                graham = ben_graham_value(eps_latest, g_eps, Y_model)
                dcfv = dcf_value_from_fcfps(fcfps_latest, g_fcf, r_model, n_years, tg_model)

                # Display
                display_val, status = pick_status_and_display(px, dcfv, graham)

                # Persist (legacy growth_rate mirrors FCF growth)
                row = {
                    "date": d,
                    "ticker": t,
                    "close_price": px,
                    "eps": eps_latest,
                    "fcfps": fcfps_latest,
                    "cagr_years": int(args.cagr_years),
                    "growth_rate": g_fcf,            # legacy column: keep as FCF growth
                    "growth_rate_fcf": g_fcf,
                    "growth_rate_eps": g_eps,
                    "growth_method_fcf": used_fcf,
                    "growth_method_eps": used_eps,
                    "discount_rate": r_model,
                    "terminal_growth": tg_model,
                    "graham_yield_Y": Y_model,
                    "graham_value": None if graham is None else float(graham),
                    "dcf_value": None if dcfv is None else float(dcfv),
                    "gfv_price": None if display_val is None else float(display_val),
                    "gfv_status": status,
                }
                upsert_gfv_row(engine, row)

        except Exception as e:
            logger.exception(f"Error computing GFV for {t}: {e}")


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compute Greer Fair Value (daily)")

    # Selection
    p.add_argument("--tickers", type=str, help="Comma-separated list of tickers")
    p.add_argument("--file", type=str, help="CSV with a 'ticker' column")

    # Range
    p.add_argument("--full", action="store_true", help="Recompute full history")
    p.add_argument("--start", type=str, help="YYYY-MM-DD when using --full")
    p.add_argument("--days", type=int, default=30, help="Incremental window, default 30 days")

    # Growth (FCF for DCF)
    p.add_argument("--cagr_years", type=int, default=5, help="Max FY points to use for growth estimation")
    p.add_argument("--growth_method_fcf", type=str, default="auto", choices=["auto", "reg", "cagr"],
                   help="Growth estimator for DCF (FCF/share): auto (default), reg, or cagr")
    p.add_argument("--growth_cap_fcf", type=float, default=0.15, help="Upper cap for FCF growth (e.g., 0.15)")
    p.add_argument("--growth_floor_fcf", type=float, default=-0.10, help="Lower cap for FCF growth (e.g., -0.10)")
    p.add_argument("--growth_rate_fcf", type=float, default=None,
                   help="Manual FCF growth (e.g., 0.10). Overrides estimator if provided")

    # Growth (EPS for Graham)
    p.add_argument("--growth_method_eps", type=str, default="auto", choices=["auto", "reg", "cagr"],
                   help="Growth estimator for Graham (EPS): auto (default), reg, or cagr")
    p.add_argument("--growth_cap_eps", type=float, default=0.20, help="Upper cap for EPS growth (e.g., 0.20)")
    p.add_argument("--growth_floor_eps", type=float, default=0.0,  # <-- was -0.10
                   help="Lower cap for EPS growth (e.g., 0.00 to disallow negative growth in Graham)")
    p.add_argument("--growth_rate_eps", type=float, default=None,
                   help="Manual EPS growth (e.g., 0.12). Overrides estimator if provided")

    # DCF / Graham params
    p.add_argument("--discount_rate", type=float, default=0.10)
    p.add_argument("--dcf_years", type=int, default=5)
    p.add_argument("--terminal_growth", type=float, default=0.03)
    p.add_argument("--graham_Y", type=float, default=4.4, help="AAA bond yield; 4.4 neutralizes multiplier")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
    print("✅ Greer Fair Value calculation complete.")
