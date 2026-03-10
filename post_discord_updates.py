# ----------------------------------------------------------
# post_discord_updates.py
# Posts the YR Fund Comparison leaderboard to Discord
# ----------------------------------------------------------

from pathlib import Path
from dotenv import load_dotenv
import os
import requests
import pandas as pd
from datetime import date
from sqlalchemy import text

from db import get_engine

# ----------------------------------------------------------
# Load environment variables from project .env
# ----------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


# ----------------------------------------------------------
# Load portfolios
# ----------------------------------------------------------
def load_portfolios() -> pd.DataFrame:
    sql = """
        SELECT portfolio_id, code, name, start_date
        FROM portfolios
        ORDER BY code
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


# ----------------------------------------------------------
# Get first NAV on/after Jan 1 AND last NAV on/before as-of
# ----------------------------------------------------------
def load_ytd_start_end(portfolio_id: int, y0: date, asof: date) -> dict | None:
    sql = """
        WITH start_row AS (
            SELECT nav_date, nav
            FROM portfolio_nav_daily
            WHERE portfolio_id = :pid
              AND nav_date >= :y0
            ORDER BY nav_date ASC
            LIMIT 1
        ),
        end_row AS (
            SELECT nav_date, nav
            FROM portfolio_nav_daily
            WHERE portfolio_id = :pid
              AND nav_date <= :asof
            ORDER BY nav_date DESC
            LIMIT 1
        )
        SELECT
            (SELECT nav_date FROM start_row) AS start_date,
            (SELECT nav FROM start_row)     AS start_nav,
            (SELECT nav_date FROM end_row)  AS end_date,
            (SELECT nav FROM end_row)       AS end_nav
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            text(sql),
            {"pid": portfolio_id, "y0": y0, "asof": asof},
        ).mappings().first()

    if not row or row["start_nav"] is None or row["end_nav"] is None:
        return None

    return dict(row)


BASELINE_CODES = {"SPY-26", "QQQ-26", "GLD-26", "BTC-26"}


def fund_group(code: str, bench_code: str) -> str:
    c = (code or "").upper()
    base = c.split("-")[0]

    if c == (bench_code or "").upper():
        return "Benchmark"

    if base.endswith("I"):
        return "Income"
    if base.endswith("G"):
        return "Growth"

    return "Other"


def table_type(code: str, name: str, bench_code: str) -> str:
    c = (code or "").upper()
    n = (name or "").lower()

    if c in BASELINE_CODES or "baseline" in n:
        return "Baseline"

    return fund_group(c, bench_code)

# ----------------------------------------------------------
# Format return with green/red arrows
# ----------------------------------------------------------
def fmt_return(val: float) -> str:

    if val is None or pd.isna(val):
        return ""

    pct = val * 100

    if pct > 0:
        return f"🟢 ↑ {pct:+.2f}%"
    elif pct < 0:
        return f"🔴 ↓ {pct:+.2f}%"
    else:
        return f"⚪ 0.00%"

# ----------------------------------------------------------
# Build comparison table
# ----------------------------------------------------------
def compute_comparison(
    pmeta: pd.DataFrame,
    year: int,
    asof: date,
    selected_codes: list[str],
    bench_code: str,
) -> pd.DataFrame:
    y0 = date(year, 1, 1)
    out = []

    for _, r in pmeta[pmeta["code"].isin(selected_codes)].iterrows():
        pid = int(r["portfolio_id"])
        code = str(r["code"])
        name = str(r.get("name") or "")

        se = load_ytd_start_end(pid, y0, asof)
        if not se:
            continue

        start_nav = float(se["start_nav"])
        end_nav = float(se["end_nav"])
        pnl = end_nav - start_nav
        ret = (end_nav / start_nav - 1.0) if start_nav else None

        out.append(
            {
                "Type": table_type(code, name, bench_code),
                "Code": code,
                "Name": name,
                "Start NAV": start_nav,
                "End NAV": end_nav,
                "P&L": pnl,
                "YTD Return": ret,
                "Start Date Used": se["start_date"],
                "End Date Used": se["end_date"],
            }
        )

    df = pd.DataFrame(out)
    if df.empty:
        return df

    df = df.sort_values("YTD Return", ascending=False).reset_index(drop=True)
    return df

# ----------------------------------------------------------
# Format return with green/red icon
# ----------------------------------------------------------
def fmt_return_compact(val: float) -> str:

    if val is None or pd.isna(val):
        return ""

    pct = val * 100

    if pct > 0:
        icon = "🟢"
    elif pct < 0:
        icon = "🔴"
    else:
        icon = "⚪"

    return f"{icon} {pct:+.2f}%"

# ----------------------------------------------------------
# Build Discord message
# ----------------------------------------------------------
def build_message(df: pd.DataFrame, bench_code: str, asof: date) -> str:
    bench = df[df["Code"] == bench_code]
    bench_ret = None

    if not bench.empty and pd.notna(bench.iloc[0]["YTD Return"]):
        bench_ret = float(bench.iloc[0]["YTD Return"])

    lines = []
    lines.append("🏁 **YOU ROCK FUND RACE**")
    lines.append(f"**As of:** {asof.strftime('%b %-d, %Y')}")
    lines.append("")
    lines.append("```text")

    for idx, row in df.iterrows():
        rank = idx + 1
        code = str(row["Code"])
        ret = fmt_return_compact(row["YTD Return"])

        if code == bench_code:
            lines.append(" -------------------------")
            lines.append(f"    {code:<8} {ret}")
            lines.append(" -------------------------")
        else:
            lines.append(f"{rank:>2}. {code:<8} {ret}")

    lines.append("```")

    return "\n".join(lines)


# ----------------------------------------------------------
# Post to Discord
# ----------------------------------------------------------
def post_to_discord(message: str) -> None:
    if not WEBHOOK_URL:
        print("[ERROR] DISCORD_WEBHOOK_URL is not set")
        return

    try:
        response = requests.post(
            WEBHOOK_URL,
            json={"content": message},
            timeout=30,
        )

        if response.status_code == 204:
            print("[INFO] Discord post successful")
        else:
            print(f"[ERROR] Discord post failed: {response.status_code}")
            print(response.text)

    except Exception as e:
        print("[ERROR] Exception sending Discord message")
        print(str(e))


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def post_fund_comparison() -> None:
    print("[INFO] Loading portfolios")
    pmeta = load_portfolios()

    if pmeta.empty:
        print("[WARNING] No portfolios found")
        return

    available = pmeta["code"].astype(str).tolist()

    selected_codes = [
        c for c in [
            "YRVI-26",
            "YRSI-26",
            "YROG-26",
            "YR3G-26",
            "YRQI-26",
            "QQQ-26",
            "SPY-26",
            "BTC-26",
            "GLD-26",
        ] if c in available
    ]

    bench_code = "QQQ-26" if "QQQ-26" in available else available[0]
    if bench_code not in selected_codes:
        selected_codes.append(bench_code)

    year = date.today().year
    asof = date.today()

    print("[INFO] Computing comparison")
    df = compute_comparison(
        pmeta=pmeta,
        year=year,
        asof=asof,
        selected_codes=selected_codes,
        bench_code=bench_code,
    )

    if df.empty:
        print("[WARNING] No NAV data found")
        return

    print(f"[INFO] Retrieved {len(df)} funds")

    message = build_message(df, bench_code=bench_code, asof=asof)

    print("[INFO] Posting to Discord")
    post_to_discord(message)


if __name__ == "__main__":
    post_fund_comparison()