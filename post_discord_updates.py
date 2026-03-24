# ----------------------------------------------------------
# post_discord_updates.py
# Posts You Rock fund updates to Discord
# - fund comparison
# - fund leaderboard
# - weekly performance
# - recent public fund trades only
# - growth fund trades to their own fund channels
# ----------------------------------------------------------

from pathlib import Path
from dotenv import load_dotenv
import os
import requests
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text

from db import get_engine


# ----------------------------------------------------------
# Load environment variables from project .env
# ----------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

WEBHOOK_FUND_COMPARISON = os.getenv("DISCORD_WEBHOOK_FUND_COMPARISON")
WEBHOOK_FUND_LEADERBOARD = os.getenv("DISCORD_WEBHOOK_FUND_LEADERBOARD")
WEBHOOK_WEEKLY_PERFORMANCE = os.getenv("DISCORD_WEBHOOK_WEEKLY_PERFORMANCE")
WEBHOOK_FUND_TRADES = os.getenv("DISCORD_WEBHOOK_FUND_TRADES")

WEBHOOK_YR3G_TRADES = os.getenv("DISCORD_WEBHOOK_YR3G_TRADES")
WEBHOOK_YROG_TRADES = os.getenv("DISCORD_WEBHOOK_YROG_TRADES")
WEBHOOK_YRVG_TRADES = os.getenv("DISCORD_WEBHOOK_YRVG_TRADES")
WEBHOOK_YRRG_TRADES = os.getenv("DISCORD_WEBHOOK_YRRG_TRADES")

WEBHOOK_GOI_WEEKLY = os.getenv("DISCORD_WEBHOOK_GOI_WEEKLY")

# ----------------------------------------------------------
# Check whether a trade was already posted to a Discord channel
# ----------------------------------------------------------
def was_trade_posted(event_id: int, channel_key: str) -> bool:
    sql = """
        SELECT 1
        FROM discord_trade_posts
        WHERE event_id = :event_id
          AND channel_key = :channel_key
        LIMIT 1
    """
    with get_engine().connect() as conn:
        row = conn.execute(
            text(sql),
            {"event_id": int(event_id), "channel_key": channel_key},
        ).first()

    return row is not None


# ----------------------------------------------------------
# Record a successful Discord trade post
# ----------------------------------------------------------
def record_trade_post(
    event_id: int,
    channel_key: str,
    discord_status_code: int | None = None,
    discord_response_text: str | None = None,
) -> None:
    sql = """
        INSERT INTO discord_trade_posts (
            event_id,
            channel_key,
            discord_status_code,
            discord_response_text
        )
        VALUES (
            :event_id,
            :channel_key,
            :discord_status_code,
            :discord_response_text
        )
        ON CONFLICT (event_id, channel_key) DO NOTHING
    """
    with get_engine().begin() as conn:
        conn.execute(
            text(sql),
            {
                "event_id": int(event_id),
                "channel_key": channel_key,
                "discord_status_code": discord_status_code,
                "discord_response_text": discord_response_text,
            },
        )
        
# ----------------------------------------------------------
# Load latest and prior-week Greer Opportunity Index rows
# ----------------------------------------------------------
def load_goi_weekly_rows(asof: date) -> tuple[dict | None, dict | None]:
    sql = """
        WITH latest_row AS (
            SELECT
                date,
                buyzone_count,
                total_tickers,
                buyzone_pct
            FROM buyzone_breadth
            WHERE date <= :asof
            ORDER BY date DESC
            LIMIT 1
        ),
        prior_row AS (
            SELECT
                date,
                buyzone_count,
                total_tickers,
                buyzone_pct
            FROM buyzone_breadth
            WHERE date <= :prior_asof
            ORDER BY date DESC
            LIMIT 1
        )
        SELECT
            'latest' AS row_type,
            date,
            buyzone_count,
            total_tickers,
            buyzone_pct
        FROM latest_row

        UNION ALL

        SELECT
            'prior' AS row_type,
            date,
            buyzone_count,
            total_tickers,
            buyzone_pct
        FROM prior_row
    """

    prior_asof = asof - timedelta(days=7)

    with get_engine().connect() as conn:
        df = pd.read_sql(
            text(sql),
            conn,
            params={"asof": asof, "prior_asof": prior_asof},
            parse_dates=["date"],
        )

    if df.empty:
        return None, None

    latest_row = df[df["row_type"] == "latest"]
    prior_row = df[df["row_type"] == "prior"]

    latest = latest_row.iloc[0].to_dict() if not latest_row.empty else None
    prior = prior_row.iloc[0].to_dict() if not prior_row.empty else None

    return latest, prior


# ----------------------------------------------------------
# Determine GOI zone
# ----------------------------------------------------------
def get_goi_zone(pct: float) -> str:
    if pct < 10:
        return "Extreme Greed"
    if pct < 14:
        return "Low Opportunity"
    if pct < 46:
        return "Normal Range"
    if pct < 66:
        return "Elevated Opportunity"
    return "Extreme Opportunity"


# ----------------------------------------------------------
# GOI interpretation text
# ----------------------------------------------------------
def get_goi_interpretation(zone: str) -> str:
    if zone == "Extreme Opportunity":
        return "Broad opportunity is very high. Historically this has aligned with panic-style conditions."
    if zone == "Elevated Opportunity":
        return "Opportunity is expanding across the market. Conditions are becoming more attractive."
    if zone == "Normal Range":
        return "Market is in a typical environment. Selectivity matters most here."
    if zone == "Low Opportunity":
        return "Market is relatively strong. Good opportunities are more limited."
    return "Market is overheated. Opportunity is scarce and risk is elevated."


# ----------------------------------------------------------
# Build weekly GOI Discord message
# ----------------------------------------------------------
def build_goi_weekly_message(latest: dict, prior: dict | None) -> str:
    current_pct = float(latest["buyzone_pct"])
    current_count = int(latest["buyzone_count"])
    total_tickers = int(latest["total_tickers"])
    latest_date = pd.to_datetime(latest["date"]).strftime("%b %-d, %Y")

    zone = get_goi_zone(current_pct)
    interpretation = get_goi_interpretation(zone)

    change_line = "N/A"
    trend_line = "N/A"

    if prior is not None:
        prior_pct = float(prior["buyzone_pct"])
        delta = round(current_pct - prior_pct, 1)

        if delta > 0:
            trend_line = "Rising 📈"
        elif delta < 0:
            trend_line = "Falling 📉"
        else:
            trend_line = "Flat ➡️"

        change_line = f"{delta:+.1f} pts"

    lines = []
    lines.append("🎯 **Greer Opportunity Index — Weekly Update**")
    lines.append(f"**As of:** {latest_date}")
    lines.append("")
    lines.append(f"Current: **{current_pct:.1f}%**")
    lines.append(f"Zone: **{zone}**")
    lines.append(f"Companies in BuyZone: **{current_count:,} / {total_tickers:,}**")
    lines.append(f"Weekly Change: **{change_line}**")
    lines.append(f"Trend: **{trend_line}**")
    lines.append("")
    lines.append(f"💡 **What it means:** {interpretation}")

    return "\n".join(lines)


# ----------------------------------------------------------
# Post weekly GOI update on Mondays only
# ----------------------------------------------------------
def post_weekly_goi_update(asof: date) -> None:
    if asof.weekday() != 0:
        print("[INFO] Skipping GOI weekly update - not Monday")
        return

    print("[INFO] Loading Greer Opportunity Index weekly data")
    latest, prior = load_goi_weekly_rows(asof)

    if latest is None:
        print("[WARNING] No GOI data found")
        return

    message = build_goi_weekly_message(latest, prior)
    post_to_discord(message, WEBHOOK_GOI_WEEKLY, "GOI Weekly Update")

# ----------------------------------------------------------
# Growth fund trade webhooks
# Each fund channel receives only its own trades
# ----------------------------------------------------------
GROWTH_TRADE_WEBHOOKS = {
    "YR3G-26": WEBHOOK_YR3G_TRADES,
    "YROG-26": WEBHOOK_YROG_TRADES,
    "YRRG-26": WEBHOOK_YRRG_TRADES,  # 🔥 NEW
    "YRVG-26": WEBHOOK_YRVG_TRADES,
}


# ----------------------------------------------------------
# Load portfolios
# ----------------------------------------------------------
def load_portfolios() -> pd.DataFrame:
    sql = """
        SELECT
            portfolio_id,
            code,
            name,
            start_date,
            is_public
        FROM portfolios
        ORDER BY code
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


# ----------------------------------------------------------
# Get selected public fund codes
# ----------------------------------------------------------
def get_selected_codes(pmeta: pd.DataFrame) -> list[str]:
    if pmeta.empty or "is_public" not in pmeta.columns:
        return []

    public_df = pmeta[pmeta["is_public"] == True].copy()
    return public_df["code"].astype(str).tolist()


# ----------------------------------------------------------
# Load recent public fund trades
# ----------------------------------------------------------
def load_recent_fund_trades(asof: date, days_back: int = 1) -> pd.DataFrame:
    sql = """
        SELECT
            pe.event_id,
            pe.event_time,
            pe.event_type,
            pe.ticker,
            pe.quantity,
            pe.price,
            pe.fees,
            pe.option_type,
            pe.strike,
            pe.expiry,
            pe.cash_delta,
            pe.notes,
            p.code AS portfolio_code,
            p.name AS portfolio_name,

            gts.signal_id,
            gts.signal_type AS growth_signal_type,
            gts.trigger_gain_pct,
            gts.sell_pct,
            gts.shares_before,
            gts.shares_to_sell,
            gts.expected_shares_after,
            gts.market_price AS growth_market_price
        FROM portfolio_events pe
        JOIN portfolios p
          ON p.portfolio_id = pe.portfolio_id
        LEFT JOIN growth_trade_signals gts
          ON gts.execution_event_id = pe.event_id
        WHERE pe.event_time >= :start_ts
          AND pe.event_time < :end_ts
          AND p.is_public = TRUE
        ORDER BY pe.event_time DESC, pe.event_id DESC
    """

    start_ts = pd.Timestamp(asof) - pd.Timedelta(days=days_back)
    end_ts = pd.Timestamp(asof) + pd.Timedelta(days=1)

    with get_engine().connect() as conn:
        return pd.read_sql(
            text(sql),
            conn,
            params={"start_ts": start_ts, "end_ts": end_ts},
        )


# ----------------------------------------------------------
# Format money
# ----------------------------------------------------------
def fmt_money(val) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"${float(val):,.2f}"


# ----------------------------------------------------------
# Format quantity
# ----------------------------------------------------------
def fmt_quantity(val) -> str:
    if val is None or pd.isna(val):
        return "N/A"

    val = float(val)
    if val.is_integer():
        return f"{int(val):,}"

    return f"{val:,.2f}"


# ----------------------------------------------------------
# Format short timestamp
# ----------------------------------------------------------
def fmt_event_time(val) -> str:
    if val is None or pd.isna(val):
        return "N/A"

    ts = pd.to_datetime(val)
    return ts.strftime("%b %-d, %Y %I:%M %p")


# ----------------------------------------------------------
# Format short date
# ----------------------------------------------------------
def fmt_short_date(val) -> str:
    if val is None or pd.isna(val):
        return "N/A"

    dt = pd.to_datetime(val)
    return dt.strftime("%b %-d, %Y")


# ----------------------------------------------------------
# Build human-friendly trade title
# ----------------------------------------------------------
def get_trade_title(row: pd.Series) -> tuple[str, str]:
    event_type = str(row.get("event_type") or "").strip().lower()
    option_type = str(row.get("option_type") or "").strip().lower()

    if option_type == "call":
        if "sell" in event_type:
            return "📈", "Sold Covered Call"
        if "buy" in event_type:
            return "📘", "Bought Call Option"

    if option_type == "put":
        if "sell" in event_type:
            return "📉", "Sold Cash-Secured Put"
        if "buy" in event_type:
            return "📙", "Bought Put Option"

    if "buy" in event_type:
        return "🟢", "Bought Shares"

    if "sell" in event_type:
        return "🔴", "Sold Shares"

    return "📌", str(row.get("event_type") or "Fund Trade").title()


# ----------------------------------------------------------
# Build one fund trade Discord message
# ----------------------------------------------------------
def build_fund_trade_message(row: pd.Series) -> str:
    icon, title = get_trade_title(row)

    portfolio_code = str(row.get("portfolio_code") or "UNKNOWN")
    ticker = str(row.get("ticker") or "N/A")
    quantity = row.get("quantity")
    price = row.get("price")
    strike = row.get("strike")
    expiry = row.get("expiry")
    cash_delta = row.get("cash_delta")
    notes = row.get("notes")
    option_type = str(row.get("option_type") or "").strip().lower()
    event_time = row.get("event_time")

    lines = []
    lines.append(f"{icon} **{portfolio_code}**")
    lines.append("")
    lines.append(f"**{title}**")
    lines.append(f"Ticker: **{ticker}**")

    if quantity is not None and not pd.isna(quantity):
        if option_type in {"call", "put"}:
            contracts = float(quantity)
            shares_covered = contracts * 100

            if contracts.is_integer():
                lines.append(f"Contracts: **{int(contracts):,}**")
            else:
                lines.append(f"Contracts: **{contracts:,.2f}**")

            if shares_covered.is_integer():
                lines.append(f"Shares Covered: **{int(shares_covered):,}**")
            else:
                lines.append(f"Shares Covered: **{shares_covered:,.2f}**")
        else:
            lines.append(f"Shares: **{fmt_quantity(quantity)}**")

    if price is not None and not pd.isna(price):
        if option_type in {"call", "put"}:
            lines.append(f"Premium: **{fmt_money(price)}**")
        else:
            lines.append(f"Price: **{fmt_money(price)}**")

    if strike is not None and not pd.isna(strike):
        lines.append(f"Strike: **{fmt_money(strike)}**")

    if expiry is not None and not pd.isna(expiry):
        lines.append(f"Expiry: **{fmt_short_date(expiry)}**")

    if cash_delta is not None and not pd.isna(cash_delta):
        cash_val = float(cash_delta)
        event_type_lower = str(row.get("event_type") or "").lower()

        if option_type in {"call", "put"}:
            if "sell" in event_type_lower and cash_val > 0:
                lines.append(f"Premium Collected: **{fmt_money(cash_val)}**")
            elif "buy" in event_type_lower and cash_val < 0:
                lines.append(f"Premium Paid: **{fmt_money(abs(cash_val))}**")
            else:
                if cash_val > 0:
                    lines.append(f"Cash In: **{fmt_money(cash_val)}**")
                elif cash_val < 0:
                    lines.append(f"Cash Out: **{fmt_money(abs(cash_val))}**")
        else:
            if cash_val > 0:
                lines.append(f"Cash In: **{fmt_money(cash_val)}**")
            elif cash_val < 0:
                lines.append(f"Cash Out: **{fmt_money(abs(cash_val))}**")

    lines.append(f"Trade Time: **{fmt_event_time(event_time)}**")

    if notes and str(notes).strip():
        lines.append(f"Notes: {str(notes).strip()}")

    return "\n".join(lines)

# ----------------------------------------------------------
# Build one auto-growth trade Discord message
# ----------------------------------------------------------
def build_growth_trade_message(row: pd.Series) -> str:
    portfolio_code = str(row.get("portfolio_code") or "UNKNOWN")
    ticker = str(row.get("ticker") or "N/A")
    signal_type = str(row.get("growth_signal_type") or "").strip().upper()
    trigger_gain_pct = row.get("trigger_gain_pct")
    sell_pct = row.get("sell_pct")
    shares_before = row.get("shares_before")
    shares_to_sell = row.get("shares_to_sell")
    expected_shares_after = row.get("expected_shares_after")
    market_price = row.get("growth_market_price") or row.get("price")
    notes = str(row.get("notes") or "").strip()
    event_time = row.get("event_time")

    lines = []

    if signal_type == "STOP_LOSS":
        lines.append(f"🚨 **{portfolio_code} Stop Loss Trigger Executed**")
        lines.append(f"Ticker: **{ticker}**")
        lines.append(f"Current Price: **{fmt_money(market_price)}**")
        lines.append("Rule: Sell **100%** below **-10%**")
        lines.append("Action: **SELL ALL SHARES**")

        if shares_to_sell is not None and not pd.isna(shares_to_sell):
            lines.append(f"Shares to Sell: **{fmt_quantity(shares_to_sell)}**")

    else:
        lines.append(f"📈 **{portfolio_code} Profit Trigger Executed**")
        lines.append(f"Ticker: **{ticker}**")

        if trigger_gain_pct is not None and not pd.isna(trigger_gain_pct):
            lines.append(f"Gain Level Hit: **+{float(trigger_gain_pct):.0f}%**")

        lines.append(f"Current Price: **{fmt_money(market_price)}**")

        if sell_pct is not None and not pd.isna(sell_pct):
            lines.append(f"Action: Sell **{float(sell_pct):.0f}%** of current position")

        if shares_before is not None and not pd.isna(shares_before):
            lines.append(f"Shares Before: **{fmt_quantity(shares_before)}**")

        if shares_to_sell is not None and not pd.isna(shares_to_sell):
            lines.append(f"Shares to Sell: **{fmt_quantity(shares_to_sell)}**")

        if expected_shares_after is not None and not pd.isna(expected_shares_after):
            lines.append(f"Expected Shares After: **{fmt_quantity(expected_shares_after)}**")

    lines.append("✅ Recorded automatically in portfolio_events")
    lines.append(f"Trade Time: **{fmt_event_time(event_time)}**")

    if notes:
        lines.append(f"Notes: {notes}")

    return "\n".join(lines)
    
# ----------------------------------------------------------
# Post recent fund trades
# - all public trades -> fund-trades channel
# - each growth fund trade -> only its own channel
# ----------------------------------------------------------
def post_recent_fund_trades(asof: date) -> None:
    print("[INFO] Loading recent public fund trades")
    trades_df = load_recent_fund_trades(asof=asof, days_back=7)

    if trades_df.empty:
        print("[INFO] No recent public fund trades found")
        return

    print(f"[INFO] Retrieved {len(trades_df)} recent public fund trades")

    # ----------------------------------------------------------
    # Post all public trades to the main fund-trades channel
    # ----------------------------------------------------------
    for _, row in trades_df.iterrows():
        event_id = row.get("event_id")
        if event_id is None or pd.isna(event_id):
            continue

        if was_trade_posted(int(event_id), "fund-trades"):
            print(f"[INFO] Skipping event_id {int(event_id)} for fund-trades (already posted)")
            continue

        if row.get("signal_id") is not None and not pd.isna(row.get("signal_id")):
            message = build_growth_trade_message(row)
        else:
            message = build_fund_trade_message(row)

        success, status_code, response_text = post_to_discord(
            message,
            WEBHOOK_FUND_TRADES,
            "Fund Trades",
        )

        if success:
            record_trade_post(
                event_id=int(event_id),
                channel_key="fund-trades",
                discord_status_code=status_code,
                discord_response_text=response_text,
            )

    # ----------------------------------------------------------
    # Post each growth fund only to its own channel
    # ----------------------------------------------------------
    trades_df["portfolio_code"] = trades_df["portfolio_code"].astype(str).str.upper()

    for fund_code, webhook_url in GROWTH_TRADE_WEBHOOKS.items():
        if not webhook_url:
            print(f"[WARNING] {fund_code} trade webhook is not set")
            continue

        fund_df = trades_df[trades_df["portfolio_code"] == fund_code].copy()

        if fund_df.empty:
            print(f"[INFO] No recent trades found for {fund_code}")
            continue

        print(f"[INFO] Posting {len(fund_df)} recent trade(s) to {fund_code}")

        for _, row in fund_df.iterrows():
            event_id = row.get("event_id")
            if event_id is None or pd.isna(event_id):
                continue

            if was_trade_posted(int(event_id), fund_code):
                print(f"[INFO] Skipping event_id {int(event_id)} for {fund_code} (already posted)")
                continue

            if row.get("signal_id") is not None and not pd.isna(row.get("signal_id")):
                message = build_growth_trade_message(row)
            else:
                message = build_fund_trade_message(row)

            success, status_code, response_text = post_to_discord(
                message,
                webhook_url,
                f"{fund_code} Trades",
            )

            if success:
                record_trade_post(
                    event_id=int(event_id),
                    channel_key=fund_code,
                    discord_status_code=status_code,
                    discord_response_text=response_text,
                )


# ----------------------------------------------------------
# Get first NAV on/after period start AND last NAV on/before as-of
# ----------------------------------------------------------
def load_period_start_end(portfolio_id: int, start_date: date, asof: date) -> dict | None:
    sql = """
        WITH start_row AS (
            SELECT nav_date, nav
            FROM portfolio_nav_daily
            WHERE portfolio_id = :pid
              AND nav_date >= :start_date
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
            {"pid": portfolio_id, "start_date": start_date, "asof": asof},
        ).mappings().first()

    if not row or row["start_nav"] is None or row["end_nav"] is None:
        return None

    return dict(row)


BASELINE_CODES = {"SPY-26", "QQQ-26", "GLD-26", "BTC-26"}


# ----------------------------------------------------------
# Classify fund type
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# Classify table type
# ----------------------------------------------------------
def table_type(code: str, name: str, bench_code: str) -> str:
    c = (code or "").upper()
    n = (name or "").lower()

    if c in BASELINE_CODES or "baseline" in n:
        return "Baseline"

    return fund_group(c, bench_code)


# ----------------------------------------------------------
# Format compact return with icon
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
# Format plain return
# ----------------------------------------------------------
def fmt_return_plain(val: float) -> str:
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val * 100:+.2f}%"


# ----------------------------------------------------------
# Compute return table for a given start date
# ----------------------------------------------------------
def compute_returns_for_period(
    pmeta: pd.DataFrame,
    start_date: date,
    asof: date,
    selected_codes: list[str],
    bench_code: str,
) -> pd.DataFrame:
    out = []

    for _, r in pmeta[pmeta["code"].isin(selected_codes)].iterrows():
        pid = int(r["portfolio_id"])
        code = str(r["code"])
        name = str(r.get("name") or "")

        se = load_period_start_end(pid, start_date, asof)
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
                "Return": ret,
                "Start Date Used": se["start_date"],
                "End Date Used": se["end_date"],
            }
        )

    df = pd.DataFrame(out)
    if df.empty:
        return df

    df = df.sort_values("Return", ascending=False).reset_index(drop=True)
    return df


# ----------------------------------------------------------
# Compute YTD comparison
# ----------------------------------------------------------
def compute_comparison(
    pmeta: pd.DataFrame,
    year: int,
    asof: date,
    selected_codes: list[str],
    bench_code: str,
) -> pd.DataFrame:
    y0 = date(year, 1, 1)
    df = compute_returns_for_period(
        pmeta=pmeta,
        start_date=y0,
        asof=asof,
        selected_codes=selected_codes,
        bench_code=bench_code,
    )

    if not df.empty:
        df = df.rename(columns={"Return": "YTD Return"})

    return df


# ----------------------------------------------------------
# Build fund comparison message
# ----------------------------------------------------------
def build_fund_comparison_message(df: pd.DataFrame, bench_code: str, asof: date) -> str:
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
# Build daily leaderboard message
# ----------------------------------------------------------
def build_fund_leaderboard_message(df: pd.DataFrame, asof: date) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    lines = []
    lines.append("🏆 **YOU ROCK FUND LEADERBOARD**")
    lines.append(f"**As of:** {asof.strftime('%b %-d, %Y')}")
    lines.append("")

    for idx, row in df.iterrows():
        rank = idx + 1
        code = str(row["Code"])
        ret = fmt_return_plain(row["YTD Return"])
        prefix = medals.get(rank, f"{rank}.")

        lines.append(f"{prefix} **{code}** {ret}")

    return "\n".join(lines)

# ----------------------------------------------------------
# Build weekly narrative sections for scorecard
# ----------------------------------------------------------
def build_weekly_narrative(df: pd.DataFrame, asof: date) -> tuple[list[str], list[str], list[str]]:
    worked: list[str] = []
    didnt: list[str] = []
    focus: list[str] = []

    if df.empty:
        return worked, didnt, focus

    income_df = df[df["Type"] == "Income"].copy()
    growth_df = df[df["Type"] == "Growth"].copy()
    baseline_df = df[df["Type"] == "Baseline"].copy()

    income_avg = income_df["Return"].mean() if not income_df.empty else None
    growth_avg = growth_df["Return"].mean() if not growth_df.empty else None
    baseline_avg = baseline_df["Return"].mean() if not baseline_df.empty else None

    best = df.iloc[0]
    worst = df.iloc[-1]

    # ----------------------------------------------------------
    # What Worked
    # ----------------------------------------------------------
    if income_avg is not None and income_avg > 0:
        worked.append("Income strategies held up well and benefited from premium capture.")

    if growth_avg is not None and baseline_avg is not None and growth_avg > baseline_avg:
        worked.append("Growth portfolios outperformed the baseline group on a relative basis.")

    if pd.notna(best["Return"]) and float(best["Return"]) > 0:
        worked.append(f"{best['Code']} led the week and helped anchor overall performance.")

    # ----------------------------------------------------------
    # What Didn't
    # ----------------------------------------------------------
    if growth_avg is not None and growth_avg < 0:
        didnt.append("Growth exposure struggled as market conditions remained uneven.")

    if baseline_avg is not None and baseline_avg < 0:
        didnt.append("Broader market weakness created a tougher backdrop for risk assets.")

    if pd.notna(worst["Return"]) and float(worst["Return"]) < 0:
        didnt.append(f"{worst['Code']} was the weakest performer this week and weighed on results.")

    # ----------------------------------------------------------
    # Next Week Focus
    # ----------------------------------------------------------
    focus.append("Continue monitoring the highest-quality setups across both income and growth strategies.")

    latest_goi, prior_goi = load_goi_weekly_rows(asof)
    if latest_goi is not None:
        current_pct = float(latest_goi["buyzone_pct"])
        zone = get_goi_zone(current_pct)
        focus.append(f"GOI is currently **{current_pct:.1f}%** ({zone}), which will help guide overall opportunity levels.")

        if prior_goi is not None:
            prior_pct = float(prior_goi["buyzone_pct"])
            delta = round(current_pct - prior_pct, 1)

            if delta > 0:
                focus.append("GOI is rising, so market-wide opportunity may be expanding.")
            elif delta < 0:
                focus.append("GOI is falling, so selectivity and patience remain important.")
            else:
                focus.append("GOI is flat week-over-week, so discipline remains the priority.")

    if not growth_df.empty and not income_df.empty:
        focus.append("Use YRVI positioning to help identify the strongest equity candidates for YRVG.")

    return worked, didnt, focus

# ----------------------------------------------------------
# Build weekly performance message
# ----------------------------------------------------------
def build_weekly_performance_message(df: pd.DataFrame, asof: date) -> str:
    lines = []
    lines.append("📊 **WEEKLY FUND PERFORMANCE**")
    lines.append(f"**As of:** {asof.strftime('%b %-d, %Y')}")
    lines.append("")
    lines.append("```text")

    for _, row in df.iterrows():
        code = str(row["Code"])
        ret = fmt_return_compact(row["Return"])
        lines.append(f"{code:<8} {ret}")

    lines.append("```")

    if not df.empty:
        best = df.iloc[0]
        worst = df.iloc[-1]

        lines.append("")
        lines.append(f"🏆 Best This Week: **{best['Code']}** {fmt_return_plain(best['Return'])}")
        lines.append(f"📉 Worst This Week: **{worst['Code']}** {fmt_return_plain(worst['Return'])}")

        worked, didnt, focus = build_weekly_narrative(df, asof)

        if worked:
            lines.append("")
            lines.append("## 🧠 What Worked")
            for item in worked:
                lines.append(f"* {item}")

        if didnt:
            lines.append("")
            lines.append("## ⚠️ What Didn’t")
            for item in didnt:
                lines.append(f"* {item}")

        if focus:
            lines.append("")
            lines.append("## 🔄 Next Week Focus")
            for item in focus:
                lines.append(f"* {item}")

    return "\n".join(lines)


# ----------------------------------------------------------
# Post to Discord
# ----------------------------------------------------------
def post_to_discord(message: str, webhook_url: str | None, label: str) -> tuple[bool, int | None, str | None]:
    if not webhook_url:
        print(f"[WARNING] {label} webhook is not set")
        return False, None, "Webhook not set"

    try:
        response = requests.post(
            webhook_url,
            json={"content": message},
            timeout=30,
        )

        if response.status_code == 204:
            print(f"[INFO] {label} Discord post successful")
            return True, response.status_code, None

        print(f"[ERROR] {label} Discord post failed: {response.status_code}")
        print(response.text)
        return False, response.status_code, response.text

    except Exception as e:
        print(f"[ERROR] Exception sending {label} Discord message")
        print(str(e))
        return False, None, str(e)


# ----------------------------------------------------------
# Main runner
# ----------------------------------------------------------
def post_all_discord_updates() -> None:
    print("[INFO] Loading portfolios")
    pmeta = load_portfolios()

    if pmeta.empty:
        print("[WARNING] No portfolios found")
        return

    selected_codes = get_selected_codes(pmeta)

    if not selected_codes:
        print("[WARNING] No public portfolios selected")
        return

    available = pmeta["code"].astype(str).tolist()

    bench_code = "QQQ-26" if "QQQ-26" in selected_codes else selected_codes[0]
    if bench_code not in selected_codes and bench_code in available:
        selected_codes.append(bench_code)

    asof = date.today()
    year = asof.year

    print("[INFO] Computing YTD comparison")
    comparison_df = compute_comparison(
        pmeta=pmeta,
        year=year,
        asof=asof,
        selected_codes=selected_codes,
        bench_code=bench_code,
    )

    if comparison_df.empty:
        print("[WARNING] No YTD NAV data found")
        return

    weekly_start = asof - timedelta(days=7)

    print("[INFO] Computing weekly performance")
    weekly_df = compute_returns_for_period(
        pmeta=pmeta,
        start_date=weekly_start,
        asof=asof,
        selected_codes=selected_codes,
        bench_code=bench_code,
    )

    comparison_msg = build_fund_comparison_message(comparison_df, bench_code, asof)
    leaderboard_msg = build_fund_leaderboard_message(comparison_df, asof)

    post_to_discord(comparison_msg, WEBHOOK_FUND_COMPARISON, "Fund Comparison")
    post_to_discord(leaderboard_msg, WEBHOOK_FUND_LEADERBOARD, "Fund Leaderboard")
    post_recent_fund_trades(asof)
    post_weekly_goi_update(asof)

    # ----------------------------------------------------------
    # Optional: only post weekly on Fridays
    # ----------------------------------------------------------
    if asof.weekday() == 4 and not weekly_df.empty:
        weekly_msg = build_weekly_performance_message(weekly_df, asof)
        post_to_discord(weekly_msg, WEBHOOK_WEEKLY_PERFORMANCE, "Weekly Performance")


if __name__ == "__main__":
    post_all_discord_updates()