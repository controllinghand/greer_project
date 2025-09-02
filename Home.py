# Home.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from html import escape  
from datetime import date
from sqlalchemy import text
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

    /* Shared card look */
    .company-card, .metric-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        background: #fafafa;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }
    .company-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .company-meta {
        font-size: 14px;
        margin: 2px 0;
    }

    /* Metric card content */
    .metric-card {
        display: flex;
        flex-direction: column;
        gap: 6px;
        min-height: 120px; /* keeps all cards same height */
        justify-content: center;
        text-align: center;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        opacity: .9;
    }
    .metric-main {
        font-size: 20px;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 2px;
    }
    .metric-sub {
        font-size: 12px;
        font-weight: 600;
        opacity: .95;
        margin-top: 2px;
    }

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
# Database Queries (cached)
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_first_trade_date(ticker: str, _cache_buster=None):
    engine = get_engine()
    df = pd.read_sql(
        "SELECT MIN(date) AS first_date FROM prices WHERE ticker = %(t)s;",
        engine,
        params={"t": ticker},
    )
    if df.empty or pd.isna(df.first_date.iloc[0]):
        return None
    return pd.to_datetime(df.first_date.iloc[0]).date()

@st.cache_data(ttl=300)
def get_latest_snapshot(ticker: str, _cache_buster=None):
    engine = get_engine()
    return pd.read_sql(
        "SELECT * FROM latest_company_snapshot WHERE ticker = %(t)s LIMIT 1;",
        engine,
        params={"t": ticker}
    )

@st.cache_data(ttl=600)
def get_company_info(ticker: str):
    engine = get_engine()
    df = pd.read_sql(
        """
        SELECT ticker, name, sector, industry, exchange, delisted, delisted_date
        FROM companies
        WHERE ticker = %(t)s
        LIMIT 1;
        """,
        engine,
        params={"t": ticker}
    )
    return df

@st.cache_data(ttl=300)
def get_latest_gfv(ticker: str, _cache_buster=None):
    """
    Tries greer_fair_value_latest view first; falls back to DISTINCT ON if the view isn't present.
    Returns a 1-row DataFrame or empty DataFrame.
    """
    engine = get_engine()
    try:
        df = pd.read_sql(
            """
            SELECT ticker, date, close_price, gfv_price, gfv_status,
                   dcf_value, graham_value,
                   growth_rate_fcf, growth_rate_eps,
                   growth_rate_fcf_raw, growth_rate_eps_raw,      -- NEW
                   discount_rate, terminal_growth, graham_yield_Y, fcf_per_share, eps
            FROM greer_fair_value_latest
            WHERE ticker = %(t)s
            LIMIT 1;
            """,
            engine,
            params={"t": ticker},
        )
        if not df.empty:
            return df
    except Exception:
        pass  # view might not exist or may not yet expose _raw columns

    # Fallback: pull latest from daily
    df = pd.read_sql(
        """
        SELECT DISTINCT ON (ticker)
               ticker, date, close_price, gfv_price, gfv_status,
               dcf_value, graham_value,
               growth_rate_fcf, growth_rate_eps,
               growth_rate_fcf_raw, growth_rate_eps_raw,        -- NEW
               discount_rate, terminal_growth, graham_yield_Y, fcf_per_share, eps
        FROM greer_fair_value_daily
        WHERE ticker = %(t)s
        ORDER BY ticker, date DESC
        LIMIT 1;
        """,
        engine,
        params={"t": ticker},
    )
    return df

# ----------------------------------------------------------
# Helper Functions: Classifiers and Card Renderers
# ----------------------------------------------------------
def classify_greer(score, above_50):
    if pd.notnull(above_50) and int(above_50) == 6:
        return "Exceptional", "#D4AF37", "black"
    if pd.notnull(score) and float(score) >= 50:
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

def render_company_card(ticker: str):
    info = get_company_info(ticker)
    if info.empty:
        return
    r = info.iloc[0]
    name = r["name"] if pd.notnull(r["name"]) else "—"
    sector = r["sector"] if pd.notnull(r["sector"]) else "—"
    industry = r["industry"] if pd.notnull(r["industry"]) else "—"
    exchange = r["exchange"] if pd.notnull(r["exchange"]) else "—"
    delisted = bool(r["delisted"]) if pd.notnull(r["delisted"]) else False
    delisted_line = ""
    if delisted and pd.notnull(r["delisted_date"]):
        dd = pd.to_datetime(r["delisted_date"]).date()
        delisted_line = f"<div class='company-meta'><b>Delisted:</b> {dd}</div>"

    st.markdown(
        f"""
        <div class="company-card" style="color: rgb(49, 51, 63);">
            <div class="company-title">{ticker} — {name}</div>
            <div class="company-meta"><b>Exchange:</b> {exchange}</div>
            <div class="company-meta"><b>Sector:</b> {sector}</div>
            <div class="company-meta"><b>Industry:</b> {industry}</div>
            {delisted_line}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_metric_card(label: str, main_html: str, sub_html: str, bg: str, fg: str):
    st.markdown(
        f"""
        <div class="metric-card" style="background:{bg}; color:{fg};">
            <div class="metric-label">{label}</div>
            <div class="metric-main">{main_html}</div>
            <div class="metric-sub">{sub_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )



def render_gfv_badge(gfv_row: pd.Series):
    # Pull values
    today_price = gfv_row.get("close_price")
    gfv         = gfv_row.get("gfv_price")
    status      = (gfv_row.get("gfv_status") or "").lower()

    dcf_val     = gfv_row.get("dcf_value")
    graham_val  = gfv_row.get("graham_value")
    g_fcf       = gfv_row.get("growth_rate_fcf")
    g_eps       = gfv_row.get("growth_rate_eps")
    # NEW: raw (pre-cap) fields
    g_fcf_raw   = gfv_row.get("growth_rate_fcf_raw")
    g_eps_raw   = gfv_row.get("growth_rate_eps_raw")

    r           = gfv_row.get("discount_rate")
    tg          = gfv_row.get("terminal_growth")
    Y           = gfv_row.get("graham_yield_y")

    badge_color = {"gold": "#D4AF37", "green": "#22c55e", "red": "#ef4444"}.get(status, "#9ca3af")

    # helper near money()/pct()
    def yield_pp_str(y):  # y is in percent points, e.g., 4.4
        return f"{float(y):.1f}%" if is_num(y) else "—"

    def is_num(x):
        return (x is not None) and isinstance(x, (int, float, np.floating)) and not pd.isna(x)

    def money(x):
        return f"${float(x):,.2f}" if is_num(x) else "—"

    def pct(x):
        return f"{float(x)*100:.1f}%" if is_num(x) else "—"

    def cap_note(used, raw):
        """Return e.g. ' (capped from 22.3%)' only if raw is present and differs."""
        if is_num(used) and is_num(raw) and abs(float(used) - float(raw)) > 1e-9:
            return f" (capped from {pct(raw)})"
        return ""

    # AAA Y display (it’s a yield, not money)
    Y_str = f"{float(Y):.2f}%" if is_num(Y) else "—"

    # --- Tooltip builder ---
    fcfps = gfv_row.get("fcf_per_share")
    eps = gfv_row.get("eps")

    # Reasons for unavailability
    reason_dcf = ""
    if dcf_val is None:
        if isinstance(fcfps, (int, float, np.floating)) and not pd.isna(fcfps) and fcfps <= 0:
            reason_dcf = " (unavailable: negative FCF/share)"
        else:
            reason_dcf = " (unavailable)"

    reason_graham = ""
    if graham_val is None:
        if isinstance(eps, (int, float, np.floating)) and not pd.isna(eps) and eps <= 0:
            reason_graham = " (unavailable: negative EPS)"
        else:
            reason_graham = " (unavailable)"

    # Growth raw vs capped
    g_fcf_raw = gfv_row.get("growth_rate_fcf_raw")
    g_eps_raw = gfv_row.get("growth_rate_eps_raw")

    def growth_str(capped, raw):
        if not is_num(capped):
            return "—"
        if is_num(raw) and round(float(raw), 4) != round(float(capped), 4):
            return f"{pct(capped)} (capped from {pct(raw)})"
        return pct(capped)

    g_fcf_str = growth_str(g_fcf, g_fcf_raw)
    g_eps_str = growth_str(g_eps, g_eps_raw)

    # Add header if incomplete
    header_line = "⚠️ Incomplete signal: one or both models unavailable." if status == "gray" else None

    tooltip_lines = []
    if header_line:
        tooltip_lines.append(header_line)
    tooltip_lines += [
        f"Today’s Price: {money(today_price)}",
        f"GFV Status: {status or '—'}",
        "",
        f"DCF FV: {money(dcf_val)}{reason_dcf}  |  FCF/Share: {money(fcfps)}  | FCF growth: {g_fcf_str}  |  r: {pct(r)}  |  terminal g: {pct(tg)}",
        f"Graham FV: {money(graham_val)}{reason_graham}  |  EPS: {money(eps)} | EPS growth: {g_eps_str}  |  AAA Y: {yield_pp_str(Y)}",
    ]

    from html import escape  # keep local import to avoid top-level duplication
    tooltip_attr = escape("\n".join(tooltip_lines), quote=True).replace("\n", "&#10;")

    st.markdown(f"""
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:10px;">
      <div class="metric-card" style="background:#111; color:#fff; min-height:auto; padding:8px 12px;">
        <div class="metric-label" style="opacity:.7;">Today’s Price</div>
        <div class="metric-main" style="font-size:18px;">{money(today_price)}</div>
      </div>

      <span title="{tooltip_attr}">
        <div class="metric-card" style="background:{badge_color}; color:#111; min-height:auto; padding:8px 12px; cursor:help;">
          <div class="metric-label" style="opacity:.9; display:flex; align-items:center; gap:6px;">
            Greer Fair Value <span style="font-weight:700;">ⓘ</span>
          </div>
          <div class="metric-main" style="font-size:18px;">{money(gfv)}</div>
        </div>
      </span>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------
# Detail Renderers
# ----------------------------------------------------------
def render_gv_details(ticker, engine):
    # Fetch latest Greer Value Score
    df = pd.read_sql(
        """
        SELECT *
        FROM greer_scores
        WHERE ticker = %(t)s
        ORDER BY report_date DESC
        LIMIT 1;
        """,
        engine,
        params={"t": ticker}
    )
    if df.empty:
        st.warning("No Greer Value data for this ticker.")
        return

    # Display latest score and grade
    greer_score = df["greer_score"].iloc[0]
    above_50 = df["above_50_count"].iloc[0]
    grade, grade_color, txt_color = classify_greer(greer_score, above_50)

    st.markdown(
        f"""
        <h3 style='margin:20px 0 10px;'>📊 {ticker.upper()} – Greer Value Snapshot</h3>
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

    # Small radar chart in top-left corner
    col1, col2 = st.columns([1, 3])  # Adjusted column widths: 1 for radar, 3 for empty space
    with col1:  # Moved radar chart to col1 (left side)
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

        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=1, color="#1976D2")
        ax.fill(angles, values, alpha=0.25, color="#1976D2")
        ax.set_ylim(0, 100)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{ticker.upper()} – Component Radar", fontsize=10, y=1.08)
        st.pyplot(fig)

    # Fetch 180 days data for price and score chart
    hist = pd.read_sql(
        """
        SELECT g.date, g.greer_score, g.above_50_count, p.close
        FROM greer_scores_daily g
        JOIN prices p ON g.ticker = p.ticker AND g.date = p.date
        WHERE g.ticker = %(t)s
          AND g.date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY g.date;
        """,
        engine,
        params={'t': ticker},
        parse_dates=['date']
    )
    if hist.empty:
        st.warning("No Greer Value data for the last 180 days. Falling back to greer_scores.")
        # Fallback to greer_scores
        hist = pd.read_sql(
            """
            SELECT g.report_date AS date, g.greer_score, g.above_50_count, p.close
            FROM greer_scores g
            JOIN prices p ON g.ticker = p.ticker AND g.report_date = p.date
            WHERE g.ticker = %(t)s
              AND g.report_date >= CURRENT_DATE - INTERVAL '180 days'
            ORDER BY g.report_date;
            """,
            engine,
            params={'t': ticker},
            parse_dates=['date']
        )
        if hist.empty:
            st.warning("No Greer Value data available for the last 180 days.")
            return

    # Price and score ribbon chart
    st.markdown(f"<h3 style='margin:20px 0 10px;'>📈 {ticker.upper()} – Greer Value Score Map (180 trading days)</h3>", unsafe_allow_html=True)
    fig, (ax_price, ax_ribbon) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), gridspec_kw={"height_ratios": [4, 0.4]}
    )
    ax_price.plot(hist["date"], hist["close"], color="black", lw=1)
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{ticker.upper()} – Greer Value Score Map (180 trading days)")

    # Create masks for each score category
    exceptional_mask = (hist["above_50_count"] == 6)
    strong_mask = (hist["greer_score"] >= 50) & (hist["above_50_count"] != 6)
    weak_mask = (hist["greer_score"] < 50) & (hist["greer_score"].notnull())
    none_mask = hist["greer_score"].isnull()

    # Apply colors based on classify_greer
    ax_ribbon.fill_between(hist["date"], 0, 1, where=exceptional_mask, color="#D4AF37", alpha=0.7, label="Exceptional")
    ax_ribbon.fill_between(hist["date"], 0, 1, where=strong_mask, color="#4CAF50", alpha=0.7, label="Strong")
    ax_ribbon.fill_between(hist["date"], 0, 1, where=weak_mask, color="#F44336", alpha=0.7, label="Weak")
    ax_ribbon.fill_between(hist["date"], 0, 1, where=none_mask, color="#E0E0E0", alpha=0.7, label="None")

    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.set_xlabel("Date")
    st.pyplot(fig)

    # Score distribution stats
    total = len(hist)
    counts = {
        "Exceptional": exceptional_mask.sum(),
        "Strong": strong_mask.sum(),
        "Weak": weak_mask.sum(),
        "None": none_mask.sum()
    }
    stats_lines = []
    for grade, count in counts.items():
        pct = (count / total * 100) if total > 0 else 0
        stats_lines.append(f"**{grade}:** {count} days ({pct:.1f}%)")
    st.markdown("**Greer Score Distribution (last 180 days):** " + " | ".join(stats_lines))

    with st.expander("ℹ️ Greer Value grade legend"):
        st.markdown(
            """
            - **Exceptional**: Above 50% in all six components (Above_50_count = 6)  
            - **Strong**: Greer Score ≥ 50%  
            - **Weak**: Greer Score < 50%  
            - **None**: No score available
            """
        )
# Replace the existing render_yield_details function with this updated version

def render_yield_details(ticker, engine):
    snap = pd.read_sql(
        """
        SELECT *
        FROM greer_yields_daily
        WHERE ticker = %(t)s
        ORDER BY date DESC
        LIMIT 1;
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

    # Fetch 180 days data for chart
    df = pd.read_sql(
        """
        SELECT y.date, y.score, p.close
        FROM greer_yields_daily y
        JOIN prices p ON y.ticker = p.ticker AND y.date = p.date
        WHERE y.ticker = %(t)s
          AND y.date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY y.date;
        """,
        engine,
        params={'t': ticker},
        parse_dates=['date']
    )
    if df.empty:
        st.warning("No yield data for the last 180 days.")
        return

    df['score'] = df['score'].astype(int)

    fig, (ax_price, ax_ribbon) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5), gridspec_kw={"height_ratios": [4, 0.4]}
    )
    ax_price.plot(df["date"], df["close"], color="black", lw=1)
    ax_price.set_ylabel("Close")
    ax_price.set_title(f"{ticker.upper()} – Yield Score Map (180 trading days)")

    colors = {
        4: "#D4AF37",  # gold
        3: "#4CAF50",  # green
        2: "#2196F3",  # blue
        1: "#2196F3",  # blue
        0: "#F44336",  # red
    }
    for score_val, color in colors.items():
        mask = df["score"] == score_val
        ax_ribbon.fill_between(df["date"], 0, 1, where=mask, color=color, alpha=0.7)

    ax_ribbon.set_yticks([])
    ax_ribbon.set_ylim(0, 1)
    ax_ribbon.set_xlabel("Date")
    st.pyplot(fig)

    # Yield score distribution stats
    total = len(df)
    counts = df['score'].value_counts().sort_index(ascending=False)
    stats_lines = []
    for sc in range(4, -1, -1):
        days = counts.get(sc, 0)
        pct = (days / total * 100) if total > 0 else 0
        stats_lines.append(f"**{sc}:** {days} days ({pct:.1f}%)")
    st.markdown("**Yield Score Distribution (last 180 days):** " + " | ".join(stats_lines))

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
        """
        SELECT date, close_price, in_buyzone
        FROM greer_buyzone_daily
        WHERE ticker = %(t)s
          AND date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY date;
        """,
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

    ax_ribbon.fill_between(df["date"], 0, 1, where=mask, color="#4CAF50", alpha=0.7)
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
        """
        SELECT date, direction, gap_min, gap_max, mitigated
        FROM fair_value_gaps
        WHERE ticker = %(t)s
          AND date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY date;
        """,
        engine,
        params={'t': ticker},
        parse_dates=['date']
    )
    price = pd.read_sql(
        """
        SELECT date, close_price
        FROM greer_buyzone_daily
        WHERE ticker = %(t)s
          AND date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY date;
        """,
        engine,
        params={'t': ticker},
        parse_dates=['date']
    )

    if gaps.empty:
        st.info("No Fair-Value Gap events in the last 180 days.")
        return

    open_bull = gaps[(gaps["direction"] == "bullish") & (~gaps["mitigated"])]
    open_bear = gaps[(gaps["direction"] == "bearish") & (~gaps["mitigated"])]
    unmitigated_gaps = gaps[gaps['mitigated'] == False]
    if unmitigated_gaps.empty:
        avg_days = 0
    else:
        avg_days = int(
            unmitigated_gaps
            .assign(days=(pd.Timestamp('now') - unmitigated_gaps['date']).dt.days)
            ['days']
            .mean()
            or 0
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
    bull_open = date_index.isin(open_bull['date'])
    bear_open = date_index.isin(open_bear['date'])
    mitigated = date_index.isin(gaps[gaps["mitigated"]]['date'])

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
    placeholder="Enter ticker (e.g., AAPL)",
    key="ticker_input",
    label_visibility="collapsed"
)

# Start rendering if a ticker is entered
if ticker:
    ticker = ticker.upper().strip()
    engine = get_engine()


    first_trade = fetch_first_trade_date(ticker)
    snap = get_latest_snapshot(ticker)

    if snap.empty:
        cols = st.columns([2, 1, 1, 1, 1])
        with cols[0]:
            render_company_card(ticker)

        st.error(f"Ticker '{ticker}' not found in latest snapshot.")

        # Offer to add the company (links to pages/add_company.py and pre-fills the ticker)
        # Prefer st.page_link (Streamlit 1.33+) — it works even if page file is renamed or ordered.
        # Pass ticker via query param so the Add page pre-fills it
        st.session_state["pending_add_ticker"] = ticker
        st.query_params["ticker"] = ticker
        st.page_link("pages/add_company.py", label="Would you like to add this company? Click here.", icon="➕", use_container_width=True)

        # Also expose a button that switches directly (for older Streamlit versions that support switch_page)
        #col_a, col_b = st.columns([1, 3])
        #with col_a:
        #    if st.button("Add this company", key="add_company_btn", use_container_width=True):
        #        # Store for cross-page prefill
        #        st.session_state["pending_add_ticker"] = ticker
        #        # Pass ticker via query param so the Add page pre-fills it
        #        st.query_params["ticker"] = ticker
        #        try:
        #            st.switch_page("pages/add_company.py")
        #        except Exception:
        #            # Fallback: show a link if switch_page isn't available
        #            st.info("Open the **Add Company** page from the left sidebar. Ticker pre-filled.")
    
    else:
        row = snap.iloc[0]

        # --- Five columns: Company card + 4 metric cards ---
        col_company, col_gv, col_yield, col_bz, col_fvg = st.columns([2, 1, 1, 1, 1])

        # Company info
        with col_company:
            render_company_card(ticker)

            # GFV pill under the company card
            gfv_df = get_latest_gfv(ticker)
            if not gfv_df.empty:
                render_gfv_badge(gfv_df.iloc[0])
            else:
                st.info("No Greer Fair Value yet for this ticker.")


        # Greer Value card
        gv_score = row.get("greer_value_score")
        grade, gv_bg, gv_txt = classify_greer(gv_score, row.get("above_50_count"))
        gv_main = "—" if gv_score is None else f"{gv_score:.2f}%"
        gv_sub = grade if gv_score is not None else "—"
        with col_gv:
            render_metric_card("Greer Value", gv_main, gv_sub, gv_bg, gv_txt)
            if st.button("🔍 Show Details", key="gv_details"):
                st.session_state["view"] = "GV"

        # Yield card
        ys_score = row.get("greer_yield_score")
        ys_score = int(ys_score) if pd.notnull(ys_score) else None
        y_grade, y_bg, y_txt = classify_yield(ys_score)
        y_main = "—" if ys_score is None else f"{ys_score}/4"
        y_sub = y_grade if ys_score is not None else "—"
        with col_yield:
            render_metric_card("Yield Score", y_main, y_sub, y_bg, y_txt)
            if st.button("🔍 Show Details", key="gy_details"):
                st.session_state["view"] = "GY"
            if ys_score is None:
                cutoff = date.today().replace(month=12, day=31, year=date.today().year - 1)
                if first_trade and first_trade > cutoff:
                    st.info(f"📈 First traded on {first_trade} — too new for historical yields.")
                else:
                    st.warning("⛔ No historical yield data available for this ticker.")

        # BuyZone card
        in_bz = bool(row.get("buyzone_flag", False))
        bz_bg = "#4CAF50" if in_bz else "#E0E0E0"
        bz_txt = "white" if in_bz else "black"
        if in_bz:
            bz_main = "Triggered"
            bz_sub = f"since {pd.to_datetime(row.get('bz_start_date')).strftime('%Y-%m-%d')}" if row.get('bz_start_date') else ""
        else:
            bz_main = "No Signal"
            bz_sub = f"left {pd.to_datetime(row.get('bz_end_date')).strftime('%Y-%m-%d')}" if row.get('bz_end_date') else ""
        with col_bz:
            render_metric_card("BuyZone", bz_main, bz_sub, bz_bg, bz_txt)
            if st.button("🔍 Show Details", key="bz_details"):
                st.session_state["view"] = "BZ"

        # FVG card
        fvg_dir = row.get("fvg_last_direction")
        fvg_date = pd.to_datetime(row.get("fvg_last_date")).strftime('%Y-%m-%d') if row.get('fvg_last_date') else "—"
        fvg_bg = "#4CAF50" if fvg_dir == "bullish" else "#F44336" if fvg_dir == "bearish" else "#90CAF9"
        fvg_txt = "white"
        fvg_main = (fvg_dir or "No Gap").capitalize()
        fvg_sub = f"last {fvg_date}"
        with col_fvg:
            render_metric_card("Fair Value Gap", fvg_main, fvg_sub, fvg_bg, fvg_txt)
            if st.button("🔍 Show Details", key="fvg_details"):
                st.session_state["view"] = "FVG"

        # View logic from query param or session
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
