# ----------------------------------------------------------
# Quick DB connection test (DEBUG ONLY)
# ----------------------------------------------------------
from db import get_engine
import pandas as pd
import streamlit as st

try:
    engine = get_engine()
    df_test = pd.read_sql("SELECT ticker FROM companies LIMIT 5;", engine)
    st.success("✅ DB Connection Successful")
    st.write("Sample tickers from DB:", df_test["ticker"].tolist())
except Exception as e:
    st.error(f"❌ DB Connection Failed: {e}")

# Home.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from db import get_engine  # ✅ Centralized DB connection

# ----------------------------------------------------------
# Page Configuration and Global CSS
# ----------------------------------------------------------
st.set_page_config(page_title="Greer Value Search", layout="wide")

st.markdown("""
<style>
    .greer-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        font-size: 20px;
        height: 2.5rem;
        text-align: center;
    }
    .badge-container {
        position: relative;
        height: 92px;
        width: 100%;
        overflow: visible;
        padding: 0;
        margin: 0;
    }
</style>
<style>
    button[kind="secondary"] {
        color: #1976d2 !important;
        background: none !important;
        border: none !important;
        padding: 0 !important;
        font-size: 14px;
        text-decoration: underline;
        cursor: pointer;
    }
</style>

""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Database Connection (Render-Compatible)
# ----------------------------------------------------------

@st.cache_data
def fetch_first_trade_date(ticker: str):
    engine = get_engine()
    df = pd.read_sql(
        "SELECT MIN(date) AS first_date FROM prices WHERE ticker = %(t)s;",
        engine,
        params={"t": ticker},
    )
    if df.empty or pd.isna(df.first_date.iloc[0]):
        return None
    return pd.to_datetime(df.first_date.iloc[0]).date()

# ----------------------------------------------------------
# Helper Functions: Classifiers and Badge Renderer
# ----------------------------------------------------------
def classify_greer(score, above_50):
    if pd.notnull(above_50) and above_50 == 6:
        return "Exceptional", "#D4AF37", "black"
    if pd.notnull(score) and score >= 50:
        return "Strong", "#4CAF50", "white"
    if pd.notnull(score):
        return "Weak", "#F44336", "white"
    return "—", "#E0E0E0", "black"

def classify_yield(yield_score):
    if yield_score is None:
        return "—", "#E0E0E0", "black"
    if yield_score == 4:
        return "Exceptional Value", "#D4AF37", "black"
    if yield_score == 3:
        return "Undervalued", "#4CAF50", "white"
    if yield_score in (1, 2):
        return "Fairly Valued", "#2196F3", "white"
    return "Overvalued", "#F44336", "white"

def render_badge(label, html_value, background, label_color="black"):
    st.markdown(
        f"""
        <div style='text-align:center;width:100%;height:92px;border-radius:8px;background:{background};display:flex;flex-direction:column;justify-content:center;'>
            <span style='font-size:14px;font-weight:600;color:{label_color};'>{label}</span>
            {html_value}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_latest_snapshot(ticker, engine):
    return pd.read_sql(
        "SELECT * FROM latest_company_snapshot WHERE ticker = %(t)s LIMIT 1;",
        engine,
        params={"t": ticker}
    )

# ----------------------------------------------------------
# Detail Renderers
# ----------------------------------------------------------
def render_gv_details(ticker, engine):
    df = pd.read_sql(
        """
        SELECT *
        FROM   greer_scores
        WHERE  ticker = %(t)s
        ORDER  BY report_date DESC
        LIMIT  1;
        """,
        engine,
        params={"t": ticker}
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

def render_yield_details(ticker, engine):
    snap = pd.read_sql(
        """
        SELECT *
        FROM   greer_yields_daily
        WHERE  ticker = %(t)s
        ORDER  BY date DESC
        LIMIT  1;
        """,
        engine,
        params={"t": ticker}
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
        unsafe_allow_html=True
    )

    hist = pd.read_sql(
        """
        SELECT date, tvpct, tvavg
        FROM   greer_yields_daily
        WHERE  ticker = %(t)s
        ORDER  BY date;
        """,
        engine,
        params={"t": ticker}
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

def render_buyzone_details(ticker, engine):
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

    fig, (ax_price, ax_ribbon) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), gridspec_kw={"height_ratios": [4, 0.4]}
    )
    ax_price.plot(df["date"], df["close_price"], color="black", lw=1)
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{ticker.upper()} – BuyZone Map (180 trading days)")

    ax_ribbon.fill_between(df["date"], 0, 1, where=mask,  color="#4CAF50", alpha=0.7)
    ax_ribbon.fill_between(df["date"], 0, 1, where=~mask, color="#E0E0E0", alpha=0.7)
    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.set_xlabel("Date")
    st.pyplot(fig)

    total = len(df)
    days_bz = mask.sum()
    pct = 100 * days_bz / total
    st.markdown(f"**Days in BuyZone:** {days_bz} / {total} ({pct:.1f} %)")

    with st.expander("ℹ️ How to read this chart"):
        st.markdown(
            "The green ribbon shows every day the Greer model flagged the stock as inside its BuyZone. "
            "Gray blocks are neutral days. The price line above provides context."
        )

def render_fvg_details(ticker, engine):
    gaps = pd.read_sql(
        '''
        SELECT date, direction, gap_min, gap_max, mitigated
        FROM   fair_value_gaps
        WHERE  ticker = %(t)s
          AND  date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER  BY date;
        ''', engine, params={'t': ticker}, parse_dates=['date'])
    price = pd.read_sql(
        '''
        SELECT date, close_price
        FROM   greer_buyzone_daily        
        WHERE  ticker = %(t)s
          AND  date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER  BY date;
        ''', engine, params={'t': ticker}, parse_dates=['date'])

    if gaps.empty:
        st.info("No Fair-Value Gap events in the last 180 days.")
        return

    open_bull = gaps[(gaps["direction"] == "bullish") & (~gaps["mitigated"])]
    open_bear = gaps[(gaps["direction"] == "bearish") & (~gaps["mitigated"])]
    avg_days = int(
        gaps.assign(days=(pd.Timestamp('now')-gaps.date).dt.days)
             .loc[lambda d: d.mitigated==False, 'days']
             .mean() or 0
    )
    st.markdown(
        f"**Open bullish gaps:** {len(open_bull)} &nbsp;|&nbsp; "
        f"**Open bearish gaps:** {len(open_bear)} &nbsp;|&nbsp; "
        f"**Avg days open:** {avg_days}"
    )

    fig, (ax_price, ax_ribbon) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), gridspec_kw={"height_ratios": [4, 0.6]}
    )
    ax_price.plot(price['date'], price['close_price'], lw=1, color='black')
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{ticker.upper()} – Fair-Value Gaps (180 days)")

    date_index = price['date']
    bull_open  = date_index.isin(open_bull['date'])
    bear_open  = date_index.isin(open_bear['date'])
    mitigated  = date_index.isin(gaps.query("mitigated").date)

    ax_ribbon.fill_between(date_index, 0, 1, where=bull_open, color="#4CAF50", alpha=.7)
    ax_ribbon.fill_between(date_index, 0, 1, where=bear_open, color="#F44336", alpha=.7)
    ax_ribbon.fill_between(date_index, 0, 1, where=mitigated, color="#BDBDBD", alpha=.7)
    ax_ribbon.set_yticks([]); ax_ribbon.set_ylim(0,1); ax_ribbon.set_xlabel("Date")
    st.pyplot(fig)

    gaps['days_open'] = (pd.Timestamp('now') - gaps['date']).dt.days.where(~gaps['mitigated'])
    st.markdown("### Gap detail")
    st.dataframe(
        gaps[['date','direction','gap_min','gap_max','mitigated','days_open']]
        .sort_values('date', ascending=False)
    )

# ----------------------------------------------------------
# Main UI (Greer Value Search)
# ----------------------------------------------------------
st.markdown('<div class="greer-header">Greer Value Search</div>', unsafe_allow_html=True)
ticker = st.text_input(
    "Search",
    placeholder="Enter a Ticker Symbol (e.g., AAPL)",
    key="ticker_input",
    label_visibility="collapsed"
)

# Start rendering if a ticker is entered
if ticker:
    ticker = ticker.upper().strip()
    engine = get_engine()
    first_trade = fetch_first_trade_date(ticker)
    snap = get_latest_snapshot(ticker, engine)

    if snap.empty:
        st.error(f"Ticker '{ticker}' not found.")
    else:
        row = snap.iloc[0]
        col_gv, col_yield, col_bz, col_fvg = st.columns(4)

        # -----------------------------
        # Greer Value badge
        # -----------------------------
        gv_score = row.get("greer_value_score")
        grade, gv_bg, gv_txt = classify_greer(gv_score, row.get("above_50_count"))
        gv_html = "—" if gv_score is None else (
            f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{gv_txt};'>{gv_score:.2f}%</span><br>"
            f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{gv_txt};'>{grade}</span>"
        )
        with col_gv:
            st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
            render_badge("Greer Value", gv_html, gv_bg, label_color=gv_txt)
            if st.button("🔍 Show Details", key="gv_details"):
                st.session_state["view"] = "GV"
            st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # Yield badge
        # -----------------------------
        ys_score = row.get("greer_yield_score")
        ys_score = int(ys_score) if pd.notnull(ys_score) else None
        y_grade, y_bg, y_txt = classify_yield(ys_score)
        y_html = "—" if ys_score is None else (
            f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{y_txt};'>{ys_score}/4</span><br>"
            f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{y_txt};'>{y_grade}</span>"
        )
        with col_yield:
            st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
            render_badge("Yield Score", y_html, y_bg, label_color=y_txt)
            if st.button("🔍 Show Details", key="gy_details"):
                st.session_state["view"] = "GY"
            st.markdown("</div>", unsafe_allow_html=True)

            if ys_score is None:
                cutoff = date.today().replace(month=12, day=31, year=date.today().year - 1)
                if first_trade and first_trade > cutoff:
                    st.info(f"📈 First traded on {first_trade} — too new for historical yields.")
                else:
                    st.warning("⛔ No historical yield data available for this ticker.")

        # -----------------------------
        # BuyZone badge
        # -----------------------------
        in_bz = bool(row.get("buyzone_flag", False))
        bz_bg = "#4CAF50" if in_bz else "#E0E0E0"
        bz_txt = "white" if in_bz else "black"
        if in_bz:
            sub = f"since {pd.to_datetime(row.get('bz_start_date')).strftime('%Y-%m-%d')}" if row.get('bz_start_date') else ""
            main_line = "Triggered"
        else:
            sub = f"left {pd.to_datetime(row.get('bz_end_date')).strftime('%Y-%m-%d')}" if row.get('bz_end_date') else ""
            main_line = "No Signal"
        bz_html = (
            f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{bz_txt};'>{main_line}</span><br>"
            f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{bz_txt};'>{sub}</span>"
        )
        with col_bz:
            st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
            render_badge("BuyZone", bz_html, bz_bg, label_color=bz_txt)
            if st.button("🔍 Show Details", key="bz_details"):
                st.session_state["view"] = "BZ"
            st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # FVG badge
        # -----------------------------
        fvg_dir = row.get("fvg_last_direction")
        fvg_date = pd.to_datetime(row.get("fvg_last_date")).strftime('%Y-%m-%d') if row.get('fvg_last_date') else "—"
        fvg_bg = "#4CAF50" if fvg_dir == "bullish" else "#F44336" if fvg_dir == "bearish" else "#90CAF9"
        fvg_txt = "white"
        fvg_html = (
            f"<span style='font-size:20px;font-weight:700;line-height:1.1;color:{fvg_txt};'>{(fvg_dir or 'No Gap').capitalize()}</span><br>"
            f"<span style='font-size:12px;font-weight:600;line-height:1.1;color:{fvg_txt};'>last {fvg_date}</span>"
        )
        with col_fvg:
            st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
            render_badge("Fair Value Gap", fvg_html, fvg_bg, label_color=fvg_txt)
            if st.button("🔍 Show Details", key="fvg_details"):
                st.session_state["view"] = "FVG"
            st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------
        # View logic from query param or session
        # -----------------------------
        query_view = st.query_params.get("view")
        if query_view:
            st.session_state["view"] = query_view

        view = st.session_state.get("view")
        if view:
            st.markdown("---")
            if view == "GV":
                render_gv_details(ticker, engine)
            elif view == "GY":
                render_yield_details(ticker, engine)
            elif view == "BZ":
                render_buyzone_details(ticker, engine)
            elif view == "FVG":
                render_fvg_details(ticker, engine)

