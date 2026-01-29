# test_yfinance_earnings_days_v2.py

import sys
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _to_date(x: Any) -> Optional[date]:
    if x is None:
        return None

    try:
        # pandas Timestamp
        if hasattr(x, "to_pydatetime"):
            return x.to_pydatetime().date()

        # python datetime
        if isinstance(x, datetime):
            return x.date()

        # unix timestamp (seconds)
        if isinstance(x, (int, float)) and x > 10_000:
            return datetime.fromtimestamp(int(x), tz=timezone.utc).date()

        # string
        if isinstance(x, str) and len(x) >= 10:
            return datetime.fromisoformat(x[:10]).date()

    except Exception:
        return None

    return None


def _days_to(d: Optional[date], today: date) -> Optional[int]:
    if d is None:
        return None
    return (d - today).days


def _pick_upcoming(dates: List[Optional[date]], today: date) -> Optional[date]:
    clean = sorted({d for d in dates if d is not None and d >= today})
    return clean[0] if clean else None


def _fmt(d: Optional[date]) -> str:
    return "N/A" if d is None else d.isoformat()


# ----------------------------------------------------------
# Pullers
# ----------------------------------------------------------
def _from_get_earnings_dates(t: yf.Ticker, today: date) -> Optional[Tuple[date, date, str]]:
    """
    Newer yfinance API. Often returns a DataFrame indexed by earnings datetime.
    """
    if not hasattr(t, "get_earnings_dates"):
        return None

    try:
        df = t.get_earnings_dates(limit=16)
        if isinstance(df, pd.DataFrame) and not df.empty:
            idx = [_to_date(x) for x in df.index]
            next_day = _pick_upcoming(idx, today)
            if next_day:
                return (next_day, next_day, "get_earnings_dates(index)")
    except Exception:
        return None

    return None


def _from_earnings_dates_attr(t: yf.Ticker, today: date) -> Optional[Tuple[date, date, str]]:
    """
    Older attribute; sometimes works.
    """
    try:
        df = t.earnings_dates
        if isinstance(df, pd.DataFrame) and not df.empty:
            idx = [_to_date(x) for x in df.index]
            next_day = _pick_upcoming(idx, today)
            if next_day:
                return (next_day, next_day, "earnings_dates(index)")
    except Exception:
        return None

    return None


def _from_calendar(t: yf.Ticker, today: date) -> Optional[Tuple[date, date, str]]:
    """
    Calendar sometimes includes a window.
    """
    try:
        cal = t.calendar

        earnings_val = None
        if isinstance(cal, dict):
            earnings_val = cal.get("Earnings Date")
        elif isinstance(cal, pd.DataFrame):
            # try common shapes
            if "Earnings Date" in cal.columns and len(cal) > 0:
                earnings_val = cal["Earnings Date"].iloc[0]
            elif "Earnings Date" in cal.index:
                v = cal.loc["Earnings Date"]
                earnings_val = v.iloc[0] if hasattr(v, "iloc") else v

        if isinstance(earnings_val, (list, tuple)) and len(earnings_val) >= 1:
            d0 = _to_date(earnings_val[0])
            d1 = _to_date(earnings_val[1]) if len(earnings_val) > 1 else d0
            if d0 and d0 >= today:
                return (d0, d1 or d0, "calendar(Earnings Date window)")
        else:
            d0 = _to_date(earnings_val)
            if d0 and d0 >= today:
                return (d0, d0, "calendar(Earnings Date)")
    except Exception:
        return None

    return None


def _from_info_timestamps(t: yf.Ticker, today: date) -> Optional[Tuple[date, date, str]]:
    """
    Yahoo 'info' often has unix timestamps for earnings.
    """
    try:
        info = t.info or {}
        start = _to_date(info.get("earningsTimestampStart"))
        end = _to_date(info.get("earningsTimestampEnd"))
        single = _to_date(info.get("earningsTimestamp"))

        # Prefer window if present
        if start and start >= today:
            return (start, end or start, "info(earningsTimestampStart/End)")
        if single and single >= today:
            return (single, single, "info(earningsTimestamp)")
    except Exception:
        return None

    return None


# ----------------------------------------------------------
# Core
# ----------------------------------------------------------
def fetch_next_earnings(ticker: str) -> Dict[str, Any]:
    today = date.today()
    t = yf.Ticker(ticker)

    # Try in order (most likely to work first)
    attempts = [
        _from_get_earnings_dates,
        _from_info_timestamps,
        _from_earnings_dates_attr,
        _from_calendar,
    ]

    for fn in attempts:
        got = fn(t, today)
        if got:
            start_d, end_d, source = got
            return {
                "ticker": ticker.upper(),
                "source": source,
                "start_date": start_d,
                "end_date": end_d,
                "days_to_start": _days_to(start_d, today),
                "days_to_end": _days_to(end_d, today),
            }

    return {
        "ticker": ticker.upper(),
        "source": None,
        "start_date": None,
        "end_date": None,
        "days_to_start": None,
        "days_to_end": None,
    }


def main(argv: List[str]) -> int:
    tickers = [x.strip().upper() for x in argv[1:] if x.strip()]
    if not tickers:
        tickers = ["AAPL", "SMCI", "NVDA", "CHTR"]

    rows = [fetch_next_earnings(tk) for tk in tickers]
    df = pd.DataFrame(rows, columns=["ticker", "source", "start_date", "end_date", "days_to_start", "days_to_end"])

    # Make output deterministic + readable
    df["start_date"] = df["start_date"].apply(_fmt)
    df["end_date"] = df["end_date"].apply(_fmt)

    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
