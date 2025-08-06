# ----------------------------------------------------------
# Company Detail Page for Greer Financial Toolkit  (v2.0.2)
# ----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

BADGE_HEIGHT = 92  # px for uniform badge boxes

# ----------------------------------------------------------
# GLOBAL CSS – hide overlay buttons & fix badge size
# ----------------------------------------------------------
st.markdown(
    f"""
    <style>
    div.stButton > button {{
        position: relative;
        top: -{BADGE_HEIGHT}px;
        width: 100%;
        height: {BADGE_HEIGHT}px;
        opacity: 0;
        border: none;
        background: transparent;
        cursor: pointer;
        padding: 0 !important;
        margin: 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# DATABASE CONNECTION
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection():
    return create_engine("postgresql://greer_user:@localhost:5432/yfinance_db")

# ----------------------------------------------------------
# SNAPSHOT FETCH
# ----------------------------------------------------------
def get_latest_snapshot(ticker: str, engine):
    return pd.read_sql(
        "SELECT * FROM latest_company_snapshot WHERE ticker = %(t)s LIMIT 1;",
        engine,
        params={"t": ticker},
    )

# ----------------------------------------------------------
# BADGE RENDERER
# ----------------------------------------------------------
def render_badge(label: str, html_value: str, background: str, label_color: str = "black"):
    st.markdown(
        f"""
        <div style='text-align:center;width:100%;height:{BADGE_HEIGHT}px;border-radius:8px;background:{background};display:flex;flex-direction:column;justify-content:center;'>
            <span style='font-size:14px;font-weight:600;color:{label_color};'>{label}</span>
            {html_value}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------
# CLASSIFIERS
# ----------------------------------------------------------
def classify_greer(score: float | None, above_50: int | None):
    if pd.notnull(above_50) and above_50 == 6:
        return "Exceptional", "#D4AF37", "black"
    if pd.notnull(score) and score >= 50:
        return "Strong", "#4CAF50", "white"
    if pd.notnull(score):
        return "Weak", "#F44336", "white"
    return "—", "#E0E0E0", "black"


def classify_yield(yield_score: int | None):
    if yield_score is None:
        return "—", "#E0E0E0", "black"
    if yield_score == 4:
        return "Exceptional Value", "#D4AF37", "black"
    if yield_score == 3:
        return "Undervalued", "#4CAF50", "white"
    if yield_score in (1, 2):
        return "Fairly Valued", "#2196F3", "white"
    return "Overvalued", "#F44336", "white"

# ==========================================================
# DETAIL PAGES
# ==========================================================
def render_gv_details(ticker: str, engine):
    """Greer Value radar + badge."""
    df = pd.read_sql(
        """
        SELECT *
        FROM   greer_scores
        WHERE  ticker = %(t)s
        ORDER  BY report_date DESC
        LIMIT  1;
        """,
        engine,
        params={"t": ticker},
    )
    if df.empty:
        st.warning("No Greer Value data for this ticker.")
        return

    greer_score = df["greer_score"].iloc[0]
    above_50    = df["above_50_count"].iloc[0]
    grade, grade_color, txt_color = classify_greer(greer_score, above_50)

    st.markdown(
        f"""
        <div style='text-align:center;margin:20px 0;'>
            <div style='font-size:32px;font-weight:bold;'>
                Greer Value Score: <span style='color:{grade_color};'>{greer_score:.2f}%</span>
            </div>
            <div style='margin-top:8px;'>
                <span style='background-color:{grade_color};color:{txt_color};padding:6px 16px;border-radius:20px;font-size:20px;font-weight:bold;'>
                    {grade}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = ["Book", "FCF", "Margin", "Revenue", "Income", "Shares"]
    values = [
        df["book_pct"].iloc[0] or 0,
        df["fcf_pct"].iloc[0] or 0,
        df["margin_pct"].iloc[0] or 0,
        df["revenue_pct"].iloc[0] or 0,
        df["income_pct"].iloc[0] or 0,
        df["shares_pct"].iloc[0] or 0,
    ]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(f"{ticker.upper()} – Greer Component Radar", y=1.08)
    st.pyplot(fig)


def render_yield_details(ticker: str, engine):
    """Yield snapshot, badge, and TVPCT vs TVAVG chart."""
    snap = pd.read_sql(
        """
        SELECT *
        FROM   greer_yields_daily
        WHERE  ticker = %(t)s
        ORDER  BY date DESC
        LIMIT  1;
        """,
        engine,
        params={"t": ticker},
    )
    if snap.empty:
        st.warning("No yield data found for this ticker.")
        return

    row = snap.iloc[0]
    st.markdown("### 📊 Greer Value Yield Snapshot")
    st.markdown(
        f"""
        - **EPS Yield:** {row['eps_yield']:.2f}% vs Avg {row['avg_eps_yield']:.2f}%  
        - **FCF Yield:** {row['fcf_yield']:.2f}% vs Avg {row['avg_fcf_yield']:.2f}%  
        - **Revenue Yield:** {row['revenue_yield']:.2f}% vs Avg {row['avg_revenue_yield']:.2f}%  
        - **Book Yield:** {row['book_yield']:.2f}% vs Avg {row['avg_book_yield']:.2f}%  
        - **Total Yield:** {row['tvpct']:.2f}% vs TVAVG {row['tvavg']:.2f}%  
        - **Score:** {int(row['score'])}/4
        """
    )

    score_int = int(row["score"])
    grade, bg, txt = classify_yield(score_int)
    st.markdown(
        f"""
        <div style='text-align:center;margin-top:10px;'>
            <span style='background-color:{bg};color:{txt};padding:6px 16px;border-radius:20px;font-size:18px;font-weight:bold;'>
                Yield Score: {score_int}/4 – {grade}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hist = pd.read_sql(
        """
        SELECT date, tvpct, tvavg
        FROM   greer_yields_daily
        WHERE  ticker = %(t)s
        ORDER  BY date;
        """,
        engine,
        params={"t": ticker},
    )
    if not hist.empty:
        st.markdown("### 📈 Yield Trend: TVPCT vs TVAVG")
        fig, ax = plt.subplots()
        ax.plot(hist["date"], hist["tvpct"], label="TVPCT (Current)", linewidth=2)
        ax.plot(hist["date"], hist["tvavg"], label="TVAVG (Average)", linestyle="--")
        ax.set_ylabel("Yield %")
        ax.set_title("Valuation Yield Over Time")
        ax.legend()
        st.pyplot(fig)

    with st.expander("ℹ️ Yield grade legend"):
        st.markdown(
            """
            - **Exceptional Value**: All four yields above historical average  
            - **Undervalued**: 3 of 4 above average  
            - **Fairly Valued**: 1–2 above average  
            - **Overvalued**: 0 above average
            """
        )


# ----------------------------------------------------------
# BuyZone Details
# ----------------------------------------------------------
def render_buyzone_details(ticker: str, engine):
    """Close-price line with a green/gray BuyZone ribbon (last 180 trading days)."""
    import matplotlib.pyplot as plt

    # 180-day history
    df = pd.read_sql(
        '''
        SELECT date, close_price, in_buyzone
        FROM   greer_buyzone_daily
        WHERE  ticker = %(t)s
          AND  date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER  BY date;
        ''',
        engine,
        params={'t': ticker},
        parse_dates=['date'],
    )
    if df.empty:
        st.warning("No BuyZone data for the last 180 days.")
        return

    df.sort_values("date", inplace=True)
    mask = df["in_buyzone"].astype(bool).values

    # -------- figure --------
    fig, (ax_price, ax_ribbon) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), gridspec_kw={"height_ratios": [4, 0.4]}
    )

    # price line
    ax_price.plot(df["date"], df["close_price"], color="black", lw=1)
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{ticker.upper()} – BuyZone Map (180 trading days)")

    # ribbon
    ax_ribbon.fill_between(df["date"], 0, 1, where=mask,  color="#4CAF50", alpha=0.7)
    ax_ribbon.fill_between(df["date"], 0, 1, where=~mask, color="#E0E0E0", alpha=0.7)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.set_xlabel("Date")

    st.pyplot(fig)

    # quick stats
    total = len(df)
    days_bz = mask.sum()
    pct = 100 * days_bz / total
    st.markdown(f"**Days in BuyZone:** {days_bz} / {total} ({pct:.1f} %)")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "The green ribbon shows every day the Greer model flagged the stock as inside its BuyZone. "
            "Gray blocks are neutral days. The price line above provides context."
        )


def render_fvg_details(ticker: str, engine):
    """
    Price + ribbon for Fair-Value Gaps (last 180 days),
    and a data table underneath.
    """
    import matplotlib.pyplot as plt

    # 1️⃣ Pull 180-day gap history + closing prices
    gaps = pd.read_sql(
        '''
        SELECT date, direction, gap_min, gap_max, mitigated
        FROM   fair_value_gaps
        WHERE  ticker = %(t)s
          AND  date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER  BY date;
        ''', engine, params={'t': ticker}, parse_dates=['date']
    )
    price = pd.read_sql(
        '''
        SELECT date, close_price
        FROM   greer_buyzone_daily        -- already daily prices
        WHERE  ticker = %(t)s
          AND  date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER  BY date;
        ''', engine, params={'t': ticker}, parse_dates=['date']
    )

    if gaps.empty:
        st.info("No Fair-Value Gap events in the last 180 days.")
        return

    # 2️⃣ Stats
    open_bull = gaps[(gaps["direction"] == "bullish") & (~gaps["mitigated"])]
    open_bear = gaps[(gaps["direction"] == "bearish") & (~gaps["mitigated"])]
    st.markdown(
        f"**Open bullish gaps:** {len(open_bull)} &nbsp;|&nbsp; "
        f"**Open bearish gaps:** {len(open_bear)} &nbsp;|&nbsp; "
        f"**Avg days open:** {int(gaps.assign(days=(pd.Timestamp('now')-gaps.date).dt.days).loc[lambda d: d.mitigated==False,'days'].mean() or 0)}"
    )

    # 3️⃣ Price + ribbon plot
    fig, (ax_price, ax_ribbon) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5),
        gridspec_kw={"height_ratios": [4, 0.6]}
    )
    # price line
    ax_price.plot(price['date'], price['close_price'], lw=1, color='black')
    ax_price.set_ylabel("Close")

    # create same-length boolean arrays
    date_index = price['date']
    bull_open  = date_index.isin(open_bull['date'])
    bear_open  = date_index.isin(open_bear['date'])
    mitigated  = date_index.isin(gaps.query("mitigated").date)

    ax_ribbon.fill_between(date_index, 0, 1, where=bull_open, color="#4CAF50", alpha=.7)
    ax_ribbon.fill_between(date_index, 0, 1, where=bear_open, color="#F44336", alpha=.7)
    ax_ribbon.fill_between(date_index, 0, 1, where=mitigated, color="#BDBDBD", alpha=.7)
    ax_ribbon.set_yticks([]); ax_ribbon.set_ylim(0,1); ax_ribbon.set_xlabel("Date")
    ax_price.set_title(f"{ticker.upper()} – Fair-Value Gaps (180 days)")
    st.pyplot(fig)

    # 4️⃣ Detail table (Streamlit’s native dataframe is sortable)
    gaps['days_open'] = (pd.Timestamp('now') - gaps['date']).dt.days.where(~gaps['mitigated'])
    st.markdown("### Gap detail")
    st.dataframe(
        gaps[['date','direction','gap_min','gap_max','mitigated','days_open']]
        .sort_values('date', ascending=False)
    )


# ==========================================================
# MAIN APP
# ==========================================================
def main():
    st.set_page_config(page_title="Company Snapshot", layout="wide")
    engine = get_connection()

    # --- ticker input -------------------------------------------------------
    ticker = st.sidebar.text_input("Ticker", value="AAPL").upper().strip()
    if not ticker:
        st.stop()

    snap = get_latest_snapshot(ticker, engine)
    if snap.empty:
        st.error(f"Ticker '{ticker}' not found.")
        st.stop()

    row = snap.iloc[0]
    col_gv, col_yield, col_bz, col_fvg = st.columns(4)

    # -----------------------------------------------------------------------
    # Greer Value badge
    gv_score = row["greer_value_score"] if pd.notnull(row["greer_value_score"]) else None
    grade, gv_bg, gv_txt = classify_greer(gv_score, row.get("above_50_count"))
    gv_html = "—" if gv_score is None else (
        f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{gv_txt};'>{gv_score:.2f}%</span>"
        "<br>"
        f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{gv_txt};'>{grade}</span>"
    )
    with col_gv:
        render_badge("Greer Value", gv_html, gv_bg, label_color=gv_txt)
        if st.button(" ", key="badge_gv"):
            st.session_state["view"] = "GV"

    # -----------------------------------------------------------------------
    # Yield badge
    ys_score = int(row["greer_yield_score"]) if pd.notnull(row["greer_yield_score"]) else None
    y_grade, y_bg, y_txt = classify_yield(ys_score)
    y_html = "—" if ys_score is None else (
        f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{y_txt};'>{ys_score}/4</span>"
        "<br>"
        f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{y_txt};'>{y_grade}</span>"
    )
    with col_yield:
        render_badge("Yield Score", y_html, y_bg, label_color=y_txt)
        if st.button(" ", key="badge_yield"):
            st.session_state["view"] = "GY"

    # -----------------------------------------------------------------------
    # Buy-Zone badge  (status + date)
    in_bz  = bool(row["buyzone_flag"]) if pd.notnull(row["buyzone_flag"]) else False
    bz_bg  = "#4CAF50" if in_bz else "#E0E0E0"
    bz_txt = "white"   if in_bz else "black"

    if in_bz:
        dt = row.get("bz_start_date")
        sub = f"since {pd.to_datetime(dt).strftime('%Y-%m-%d')}" if pd.notnull(dt) else ""
        main_line = "Triggered"
    else:
        dt = row.get("bz_end_date")
        sub = f"left {pd.to_datetime(dt).strftime('%Y-%m-%d')}" if pd.notnull(dt) else ""
        main_line = "No Signal"

    bz_html = (
        f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{bz_txt};'>{main_line}</span>"
        "<br>"
        f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{bz_txt};'>{sub}</span>"
    )
    with col_bz:
        render_badge("BuyZone", bz_html, bz_bg, label_color=bz_txt)
        if st.button(" ", key="badge_bz"):
            st.session_state["view"] = "BZ"

    # -----------------------------------------------------------------------
    # Fair-Value Gap badge (white text)
    fvg_dir  = row["fvg_last_direction"] if pd.notnull(row["fvg_last_direction"]) else None
    fvg_date = row["fvg_last_date"].strftime("%Y-%m-%d") if pd.notnull(row["fvg_last_date"]) else "—"
    fvg_bg   = "#4CAF50" if fvg_dir == "bullish" else "#F44336" if fvg_dir == "bearish" else "#90CAF9"
    fvg_txt  = "white"

    fvg_html = (
        f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{fvg_txt};'>"
        f"{fvg_dir.capitalize() if fvg_dir else 'No Gap'}</span>"
        "<br>"
        f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{fvg_txt};'>last {fvg_date}</span>"
    )
    with col_fvg:
        render_badge("Fair Value Gap", fvg_html, fvg_bg, label_color=fvg_txt)
        if st.button(" ", key="badge_fvg"):
            st.session_state["view"] = "FVG"


    # -----------------------------------------------------------------------
    # Route to detail sections
    st.markdown("---")
    view = st.session_state.get("view", "GV")
    if view == "GV":
        render_gv_details(ticker, engine)
    elif view == "GY":
        render_yield_details(ticker, engine)
    elif view == "BZ":
        render_buyzone_details(ticker, engine)
    elif view == "FVG":
        render_fvg_details(ticker, engine)


if __name__ == "__main__":
    main()
