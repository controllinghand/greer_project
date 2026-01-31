# Wk_IV_Targets.py
import os
import streamlit as st
import pandas as pd
from sqlalchemy import text
from db import get_engine
from datetime import date, timedelta

# ----------------------------------------------------------
# Admin flag (set on Render)
# - public service:  YRC_ADMIN=0 (or unset)
# - admin service:   YRC_ADMIN=1
# ----------------------------------------------------------
IS_ADMIN = os.getenv("YRC_ADMIN", "0") == "1"

st.set_page_config(page_title="Weekly IV Targets", layout="wide")

# ----------------------------------------------------------
# Convert numeric star rating into a pretty "⭐⭐⭐" string
# ----------------------------------------------------------
def stars_display(x) -> str:
    try:
        if pd.isna(x):
            return ""
        n = int(x)
        if n <= 0:
            return ""
        return "⭐" * n
    except Exception:
        return ""

# ----------------------------------------------------------
# Wheel safety flags (keep rows but warn / optionally hide)
# ----------------------------------------------------------
def add_wheel_flags(df: pd.DataFrame, wheel_mode: str) -> pd.DataFrame:
    out = df.copy()

    # Ensure numeric
    out["underlying_price"] = pd.to_numeric(out["underlying_price"], errors="coerce")
    out["put_20d_strike"] = pd.to_numeric(out["put_20d_strike"], errors="coerce")
    out["call_20d_strike"] = pd.to_numeric(out["call_20d_strike"], errors="coerce")
    out["put_20d_delta"] = pd.to_numeric(out["put_20d_delta"], errors="coerce")
    out["call_20d_delta"] = pd.to_numeric(out["call_20d_delta"], errors="coerce")

    # Basic ITM checks (relative to underlying)
    out["put_itm_flag"] = (
        out["put_20d_strike"].notna()
        & out["underlying_price"].notna()
        & (out["put_20d_strike"] > out["underlying_price"])
    )
    out["call_itm_flag"] = (
        out["call_20d_strike"].notna()
        & out["underlying_price"].notna()
        & (out["call_20d_strike"] < out["underlying_price"])
    )

    # “Delta mismatch” heuristic: ~20Δ should not be deep ITM.
    out["put_delta_abs"] = out["put_20d_delta"].abs()
    out["call_delta_abs"] = out["call_20d_delta"].abs()

    out["delta_mismatch_flag"] = False
    out.loc[out["put_delta_abs"].notna(), "delta_mismatch_flag"] |= (out["put_delta_abs"] > 0.35)
    out.loc[out["call_delta_abs"].notna(), "delta_mismatch_flag"] |= (out["call_delta_abs"] > 0.35)

    # Which side matters depends on wheel mode
    if wheel_mode.startswith("CASH"):
        bad_itm = out["put_itm_flag"]
        bad_delta = out["put_delta_abs"] > 0.35
    else:
        bad_itm = out["call_itm_flag"]
        bad_delta = out["call_delta_abs"] > 0.35

    # Build reason text
    reasons = []
    for i in range(len(out)):
        r = []

        if bool(out.iloc[i]["put_itm_flag"]):
            r.append("PUT strike > underlying (PUT ITM)")
        if bool(out.iloc[i]["call_itm_flag"]):
            r.append("CALL strike < underlying (CALL ITM)")

        if wheel_mode.startswith("CASH"):
            if pd.notna(out.iloc[i]["put_delta_abs"]) and float(out.iloc[i]["put_delta_abs"]) > 0.35:
                r.append(f"PUT |Δ| too high ({out.iloc[i]['put_delta_abs']:.2f})")
        else:
            if pd.notna(out.iloc[i]["call_delta_abs"]) and float(out.iloc[i]["call_delta_abs"]) > 0.35:
                r.append(f"CALL |Δ| too high ({out.iloc[i]['call_delta_abs']:.2f})")

        reasons.append(" · ".join(r) if r else "OK")

    out["wheel_reason"] = reasons

    # Wheel flag (✅ / ⚠️ / ❌)
    out["wheel_flag"] = "✅"

    red = (bad_itm.fillna(False)) | (bad_delta.fillna(False))
    out.loc[red, "wheel_flag"] = "❌"

    yellow = (~red) & (
        out["delta_mismatch_flag"].fillna(False)
        | out["put_itm_flag"].fillna(False)
        | out["call_itm_flag"].fillna(False)
    )
    out.loc[yellow, "wheel_flag"] = "⚠️"

    out["wheel_fit"] = out["wheel_flag"].map({"✅": "Wheel-ready", "⚠️": "Caution", "❌": "Avoid/Review"})

    return out

# ----------------------------------------------------------
# Load Weekly Targets
# - latest fetch_date per ticker
# - within that fetch_date, prefer nearest expiry (smallest dte then expiry)
# Adds: earnings_date, days_to_earnings (from latest_company_earnings)
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_weekly_targets(iv_min_atm: float, market_cap_min: float, min_star_rating: int) -> pd.DataFrame:
    engine = get_engine()
    query = text("""
    WITH latest_price AS (
      SELECT
        p.ticker,
        p.close AS latest_price
      FROM prices p
      JOIN (
        SELECT ticker, MAX(date) AS max_date
        FROM prices
        GROUP BY ticker
      ) mp
        ON mp.ticker = p.ticker
       AND mp.max_date = p.date
    ),
    latest_shares AS (
      SELECT
        f.ticker,
        f.shares_outstanding
      FROM financials f
      JOIN (
        SELECT ticker, MAX(report_date) AS max_report
        FROM financials
        GROUP BY ticker
      ) mf
        ON mf.ticker = f.ticker
       AND mf.max_report = f.report_date
    ),
    mc AS (
      SELECT
        ls.ticker,
        (ls.shares_outstanding * lp.latest_price) AS market_cap
      FROM latest_shares ls
      JOIN latest_price lp ON ls.ticker = lp.ticker
    ),
    latest_fetch AS (
      SELECT
        ticker,
        MAX(fetch_date) AS max_fetch_date
      FROM iv_summary
      GROUP BY ticker
    ),
    recent_iv AS (
      SELECT DISTINCT ON (ivs.ticker)
        ivs.*
      FROM iv_summary ivs
      JOIN latest_fetch lf
        ON lf.ticker = ivs.ticker
       AND lf.max_fetch_date = ivs.fetch_date
      ORDER BY ivs.ticker, ivs.dte ASC NULLS LAST, ivs.expiry ASC
    )
    SELECT
      r.ticker,
      COALESCE(c.greer_star_rating, 0) AS greer_star_rating,
      mc.market_cap,
      lp.latest_price AS latest_price,

      r.fetch_date,
      r.expiry,
      r.dte,
      r.contract_count,
      r.iv_atm,
      r.iv_median,
      r.atm_premium,
      r.atm_premium_pct,
      r.underlying_price,

      r.put_20d_strike,
      r.put_20d_iv,
      r.put_20d_premium,
      r.put_20d_premium_pct,
      r.put_20d_delta,

      r.call_20d_strike,
      r.call_20d_iv,
      r.call_20d_premium,
      r.call_20d_premium_pct,
      r.call_20d_delta,

      e.earnings_date,
      e.days_to_earnings

    FROM recent_iv r
    JOIN mc ON r.ticker = mc.ticker
    JOIN latest_price lp ON r.ticker = lp.ticker
    LEFT JOIN companies c ON c.ticker = r.ticker
    LEFT JOIN latest_company_earnings e ON e.ticker = r.ticker
    WHERE
      mc.market_cap >= :market_cap_min
      AND r.iv_atm >= :iv_min_atm
      AND COALESCE(c.greer_star_rating, 0) >= :min_star_rating
    ORDER BY
      r.iv_atm DESC,
      mc.market_cap DESC
    """)
    df = pd.read_sql(
        query,
        engine,
        params={
            "market_cap_min": market_cap_min,
            "iv_min_atm": iv_min_atm,
            "min_star_rating": min_star_rating
        }
    )
    return df

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    st.title("📋 Weekly IV Targets (20Δ Wheel Targets)")

    st.markdown(
        """
        **Weekend plan:** Use this page on Sat/Sun to prep your Monday orders for options expiring Friday.

        **Included:**  
        - **~20Δ Put** (Cash-Secured Put candidate)  
        - **~20Δ Call** (Covered Call candidate)  
        - Premium % filters to target **~1%/week**
        """
    )

    # ----------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------
    top1, top2, top3, top4 = st.columns([1.1, 1.2, 1.1, 1.2])
    with top1:
        iv_min_atm = st.number_input(
            "Minimum implied volatility (ATM) (decimal form)",
            value=0.40,
            step=0.05,
            format="%.2f"
        )
    with top2:
        market_cap_min = st.number_input(
            "Minimum market cap",
            value=10_000_000_000,
            step=1_000_000_000,
            format="%d"
        )
    with top3:
        min_star_rating = st.slider("Min ⭐ rating", 0, 5, 0)
    with top4:
        expiry_days = st.slider("Max expiry days from today", 1, 14, 7)

    colA, colB, colC, colD, colE = st.columns([1.35, 1.15, 1.2, 1.2, 1.2])
    with colA:
        wheel_mode = st.selectbox(
            "Wheel Mode (what you're preparing for Monday)",
            ["CASH (Sell CSP)", "SHARES (Sell CC)"]
        )
    with colB:
        min_target_premium_pct = st.number_input(
            "Min target premium % (0.01 = 1%)",
            min_value=0.0,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
    with colC:
        sort_by = st.selectbox("Sort by", ["Action premium %", "IV ATM", "Stars", "Market Cap", "Ticker"])
    with colD:
        earnings_days_hide = st.slider("Hide upcoming earnings within X days (0 = off)", 0, 30, 7)
    with colE:
        earnings_recent_hide = st.slider("Hide companies that already reported within the last X days (default 0 = show all)",0, 30, 0)



    # ----------------------------------------------------------
    # Load data (✅ must happen BEFORE df[...] usage)
    # ----------------------------------------------------------
    df = load_weekly_targets(
        iv_min_atm=iv_min_atm,
        market_cap_min=market_cap_min,
        min_star_rating=min_star_rating
    )

    if df.empty:
        st.info("No candidates found under market cap / IV / star filters.")
        return

    # ----------------------------------------------------------
    # Add star display + clean types
    # ----------------------------------------------------------
    df["stars"] = df["greer_star_rating"].apply(stars_display)

    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    df["fetch_date"] = pd.to_datetime(df["fetch_date"], errors="coerce").dt.date
    df["contract_count"] = pd.to_numeric(df["contract_count"], errors="coerce").astype("Int64")

    df["latest_price"] = pd.to_numeric(df["latest_price"], errors="coerce").round(2)
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce").round(2)
    df["iv_atm"] = pd.to_numeric(df["iv_atm"], errors="coerce").round(3)
    df["iv_median"] = pd.to_numeric(df["iv_median"], errors="coerce").round(3)

    # Earnings columns
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.date
    df["days_to_earnings"] = pd.to_numeric(df["days_to_earnings"], errors="coerce").astype("Int64")
    dte = pd.to_numeric(df["days_to_earnings"], errors="coerce")


    # Simple earnings risk flag
    def earnings_flag(d) -> str:
        """
        Interpret days_to_earnings:
          - negative => earnings likely already happened
          - small negative => recently reported
          - small positive => upcoming soon
          - None => unknown
        """
        try:
            if pd.isna(d):
                return "❓"
            v = int(d)

            # Recently reported
            if v < 0:
                if v >= -3:
                    return "🟣"  # just reported (last 3 days)
                return "⚫"      # reported (older)

            # Upcoming
            if v == 0:
                return "🚨"      # today
            if v <= 3:
                return "🟥"      # very soon
            if v <= 7:
                return "🟧"      # soon
            return "✅"
        except Exception:
            return "❓"


    df["earnings_flag"] = df["days_to_earnings"].apply(earnings_flag)

    # NEW: recent earnings helpers
    recent_window = int(earnings_recent_hide)
    df["earnings_recent_flag"] = (
        (recent_window > 0)
        & dte.notna()
        & (dte < 0)
        & (dte >= -recent_window)
    )

    upcoming_window = int(earnings_days_hide)
    df["earnings_upcoming_flag"] = (
        (upcoming_window > 0)
        & dte.notna()
        & (dte >= 0)
        & (dte <= upcoming_window)
    )

    # Market cap formatting
    df["market_cap_raw"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df["market_cap"] = df["market_cap_raw"].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "")

    # Format helpers
    def fmt_money_local(x):
        return f"${float(x):.2f}" if pd.notnull(x) else ""

    def fmt_pct_local(x):
        return f"{float(x)*100:.2f}%" if pd.notnull(x) else ""

    df["put_20d_premium_fmt"] = df["put_20d_premium"].apply(fmt_money_local)
    df["put_20d_premium_pct_fmt"] = df["put_20d_premium_pct"].apply(fmt_pct_local)

    df["call_20d_premium_fmt"] = df["call_20d_premium"].apply(fmt_money_local)
    df["call_20d_premium_pct_fmt"] = df["call_20d_premium_pct"].apply(fmt_pct_local)

    # ----------------------------------------------------------
    # Filters
    # ----------------------------------------------------------
    df = df[df["contract_count"].fillna(0) >= 10]

    today = date.today()
    max_allowed = today + timedelta(days=expiry_days)
    df = df[df["expiry"] <= max_allowed]

    # ✅ Recompute after filtering df
    earn_dte = pd.to_numeric(df["days_to_earnings"], errors="coerce")

    # Hide upcoming earnings within X days (0 = off)
    # Hide ONLY if 0 <= dte <= window (negative = already reported = always show)
    if int(earnings_days_hide) > 0:
        window = int(earnings_days_hide)
        df = df[dte.isna() | (dte < 0) | (dte > window)]

    # Hide recent earnings reported within X days (0 = off)
    if int(earnings_recent_hide) > 0:
        window = int(earnings_recent_hide)
        # hide if -window <= dte < 0
        df = df[dte.isna() | (dte >= 0) | (dte < -window)]


    if df.empty:
        st.info("Nothing left after earnings filters. Try widening the windows.")
        return


    # ----------------------------------------------------------
    # Wheel mode action columns
    # ----------------------------------------------------------
    if wheel_mode.startswith("CASH"):
        df["action_strategy"] = "CSP"
        df["action_type"] = "put"
        df["action_strike"] = df["put_20d_strike"]
        df["action_delta"] = df["put_20d_delta"]
        df["action_premium"] = df["put_20d_premium"]
        df["action_premium_pct"] = df["put_20d_premium_pct"]
        df["action_premium_fmt"] = df["put_20d_premium_fmt"]
        df["action_premium_pct_fmt"] = df["put_20d_premium_pct_fmt"]
    else:
        df["action_strategy"] = "CC"
        df["action_type"] = "call"
        df["action_strike"] = df["call_20d_strike"]
        df["action_delta"] = df["call_20d_delta"]
        df["action_premium"] = df["call_20d_premium"]
        df["action_premium_pct"] = df["call_20d_premium_pct"]
        df["action_premium_fmt"] = df["call_20d_premium_fmt"]
        df["action_premium_pct_fmt"] = df["call_20d_premium_pct_fmt"]

    # Require action fields
    df = df[df["action_premium_pct"].notna() & df["action_strike"].notna() & df["action_premium"].notna()]

    # Premium target filter
    df = df[df["action_premium_pct"] >= float(min_target_premium_pct)]

    if df.empty:
        st.info("No tickers match all criteria after premium filter.")
        return

    # Optional ticker search filter
    search = st.text_input("Search ticker (optional):").upper().strip()
    if search:
        df = df[df["ticker"].str.contains(search)]

    if df.empty:
        st.info("No tickers match your search.")
        return

    # ----------------------------------------------------------
    # Wheel flags + optional hide
    # ----------------------------------------------------------
    df = add_wheel_flags(df, wheel_mode=wheel_mode)

    f1, f2 = st.columns([1.1, 1.1])
    with f1:
        hide_red = st.checkbox("Hide ❌ Red flags (recommended)", value=True)
    with f2:
        hide_caution = st.checkbox("Hide ⚠️ Caution (recommended)", value=True)
    

    if hide_red:
        df = df[df["wheel_flag"] != "❌"]
    if hide_caution:
        df = df[df["wheel_flag"] != "⚠️"]

    if df.empty:
        st.info("Nothing left after hiding flags/earnings risk. Try unchecking one of the filters.")
        return

    # Sorting
    if sort_by == "Action premium %":
        df = df.sort_values(by="action_premium_pct", ascending=False, na_position="last")
    elif sort_by == "IV ATM":
        df = df.sort_values(by="iv_atm", ascending=False, na_position="last")
    elif sort_by == "Stars":
        df = df.sort_values(by="greer_star_rating", ascending=False, na_position="last")
    elif sort_by == "Market Cap":
        df = df.sort_values(by="market_cap_raw", ascending=False, na_position="last")
    else:
        df = df.sort_values(by="ticker", ascending=True)

    st.subheader(f"🧮 Found {len(df)} targets (Wheel Mode: {wheel_mode})")

    # ----------------------------------------------------------
    # Table columns (keep wheel columns on the far right)
    # ----------------------------------------------------------
    columns = [
        # Core identity / filters
        "ticker",
        "stars",
        "latest_price",
        "market_cap",
        "iv_atm",
        "iv_median",
        "expiry",
        "dte",
        "contract_count",

        # Put side
        "put_20d_strike",
        "put_20d_delta",
        "put_20d_premium_fmt",
        "put_20d_premium_pct_fmt",

        # Call side
        "call_20d_strike",
        "call_20d_delta",
        "call_20d_premium_fmt",
        "call_20d_premium_pct_fmt",

        # “Action” side (depends on Wheel Mode)
        "action_strategy",
        "action_strike",
        "action_delta",
        "action_premium_fmt",
        "action_premium_pct_fmt",

        # Other flags
        "put_itm_flag",
        "call_itm_flag",

        # Earnings (NEW)
        "earnings_flag",
        "earnings_date",
        "days_to_earnings",
        "earnings_recent_flag",
        "earnings_upcoming_flag",

        # Wheel info (far right)
        "wheel_flag",
        "wheel_fit",
        "wheel_reason",
    ]

    st.dataframe(df[columns], hide_index=True, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="weekly_iv_targets_filtered.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
