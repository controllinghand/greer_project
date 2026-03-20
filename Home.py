# Home.py
import streamlit as st

# ----------------------------------------------------------
# HOME UI
# - This module is imported by pages/0_Home.py
# - DO NOT call st.set_page_config() here
# - DO NOT run navigation at import-time
# ----------------------------------------------------------
def render_home():
    import textwrap
    from datetime import date

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sqlalchemy import text

    from db import get_engine
    from market_cycle_utils import (
        classify_phase_with_confidence,
        phase_badge_inline,
        phase_confidence_note,
        phase_interpretation_text,
    )

    # ----------------------------------------------------------
    # Page CSS
    # ----------------------------------------------------------
    st.markdown("""
    <style>
        .greer-header {
            font-size: 2.4rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.50rem;
        }

        .greer-subheader {
            text-align: center;
            font-size: 0.95rem;
            color: #666;
            margin-top: 0;
            margin-bottom: 1.25rem;
        }

        .stTextInput > div > div > input {
            font-size: 20px;
            height: 2.6rem;
            text-align: center;
        }

        .company-card, .metric-card, .summary-banner {
            border: 1px solid #e0e0e0;
            border-radius: 14px;
            background: #fafafa;
            padding: 12px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,.04);
        }

        .company-card {
            min-height: 245px;
        }

        .company-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .company-meta {
            font-size: 14px;
            margin: 2px 0;
        }

        .metric-card {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-height: 110px;
            justify-content: center;
            text-align: center;
        }

        .metric-label {
            font-size: 13px;
            font-weight: 700;
            opacity: .92;
        }

        .metric-main {
            font-size: 22px;
            font-weight: 800;
            line-height: 1.05;
            margin-top: 2px;
        }

        .metric-sub {
            font-size: 12px;
            font-weight: 600;
            opacity: .95;
            margin-top: 2px;
        }

        .summary-banner {
            background: #f6f8fb;
            margin-top: 14px;
            margin-bottom: 14px;
            font-size: 15px;
            line-height: 1.45;
        }

        .summary-banner b {
            color: #1f2937;
        }

        .card-section-title {
            font-size: 0.90rem;
            font-weight: 700;
            color: #4b5563;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        div[data-baseweb="tab-list"] {
            gap: 8px;
        }

        button[data-baseweb="tab"] {
            border-radius: 10px !important;
            padding: 8px 14px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def index_color(score):
        if score is None or pd.isnull(score):
            return "#9CA3AF"
        score = float(score)
        if score >= 75:
            return "#4CAF50"   # strong
        if score >= 55:
            return "#5BC0DE"   # middle
        return "#F44336"       # weak
    def first_non_null(*values):
        for v in values:
            if pd.notnull(v):
                return v
        return None

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def safe_round(value, digits=2):
        return round(float(value), digits) if pd.notnull(value) else None

    def safe_pct_to_fair_value(current_price, gfv_price):
        if pd.isnull(current_price) or pd.isnull(gfv_price):
            return None

        current_price = float(current_price)
        gfv_price = float(gfv_price)

        if current_price == 0:
            return None

        return round(((gfv_price / current_price) - 1.0) * 100.0, 1)

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

    def phase_label_with_icon(phase_name: str) -> str:
        p = str(phase_name).strip().upper()
        if p == "EXPANSION":
            return "🟢 Expansion"
        if p == "EUPHORIA":
            return "🟠 Euphoria"
        if p == "RECOVERY":
            return "🔵 Recovery"
        if p == "CONTRACTION":
            return "🔴 Contraction"
        return "⚪ Unknown"

    def transition_risk_label(confidence: float) -> str:
        c = float(confidence)
        if c < 0.35:
            return "⚠ High Shift Risk"
        if c < 0.60:
            return "👀 Watch Transition"
        return "✅ Stable Trend"

    # ----------------------------------------------------------
    # Company-cycle scoring helpers
    # ----------------------------------------------------------
    def compute_health_pct(gv_score, above_50_count) -> float:
        gv = float(gv_score) if pd.notnull(gv_score) else 0.0
        a50 = float(above_50_count) if pd.notnull(above_50_count) else 0.0
        a50_pct = (a50 / 6.0) * 100.0
        score = (gv * 0.85) + (a50_pct * 0.15)
        return round(clamp(score, 0.0, 100.0), 1)

    def compute_buyzone_score(buyzone_flag) -> float:
        return 25.0 if bool(buyzone_flag) else 75.0

    def compute_fvg_score(fvg_last_direction) -> float:
        s = str(fvg_last_direction).strip().lower() if pd.notnull(fvg_last_direction) else ""
        if s in ["bullish", "up", "green"]:
            return 100.0
        if s in ["bearish", "down", "red"]:
            return 0.0
        return 50.0

    def compute_direction_pct(buyzone_flag, fvg_last_direction, sector_direction_pct) -> float:
        buyzone_score = compute_buyzone_score(buyzone_flag)
        fvg_score = compute_fvg_score(fvg_last_direction)
        sector_score = float(sector_direction_pct) if pd.notnull(sector_direction_pct) else 50.0

        score = (
            (buyzone_score * 0.35) +
            (fvg_score * 0.35) +
            (sector_score * 0.30)
        )
        return round(clamp(score, 0.0, 100.0), 1)

    def compute_gfv_score(gfv_status) -> float:
        s = str(gfv_status).strip().lower() if pd.notnull(gfv_status) else "gray"
        if s == "gold":
            return 100.0
        if s == "green":
            return 75.0
        if s == "gray":
            return 50.0
        return 0.0

    def compute_opportunity_pct(greer_yield_score, gfv_status) -> float:
        ys = float(greer_yield_score) if pd.notnull(greer_yield_score) else 0.0
        ys_pct = (ys / 4.0) * 100.0
        gfv_pct = compute_gfv_score(gfv_status)

        score = (ys_pct * 0.5) + (gfv_pct * 0.5)
        return round(clamp(score, 0.0, 100.0), 1)

    def compute_company_index(health_pct, direction_pct, opportunity_pct) -> float:
        score = (float(health_pct) + float(direction_pct) + float(opportunity_pct)) / 3.0
        return round(clamp(score, 0.0, 100.0), 2)

    def compute_company_buyzone_proxy(direction_pct: float) -> float:
        return round(clamp(100.0 - float(direction_pct), 0.0, 100.0), 2)

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
            SELECT ticker, name, sector, industry, exchange, delisted, delisted_date, greer_star_rating
            FROM companies
            WHERE ticker = %(t)s
            LIMIT 1;
            """,
            engine,
            params={"t": ticker}
        )
        return df

    @st.cache_data(ttl=600)
    def get_star_transition_dates(ticker: str):
        engine = get_engine()
        df = pd.read_sql(
            """
            WITH tracking AS (
              SELECT MIN(snapshot_date)::date AS tracking_start_date
              FROM public.company_snapshot
              WHERE greer_star_rating IS NOT NULL
            ),
            s AS (
              SELECT
                cs.snapshot_date::date AS d,
                cs.greer_star_rating AS star,
                LAG(cs.greer_star_rating)  OVER (ORDER BY cs.snapshot_date) AS prev_star,
                LEAD(cs.greer_star_rating) OVER (ORDER BY cs.snapshot_date) AS next_star
              FROM public.company_snapshot cs
              WHERE cs.ticker = %(t)s
                AND cs.greer_star_rating IS NOT NULL
            ),
            enters AS (
              SELECT d AS entered_3star_date
              FROM s
              WHERE star >= 3 AND (prev_star < 3 OR prev_star IS NULL)
            ),
            last_enter AS (
              SELECT MAX(entered_3star_date) AS entered_3star_date
              FROM enters
            ),
            exits AS (
              SELECT d AS exited_3star_date
              FROM s
              WHERE star < 3 AND prev_star >= 3
            ),
            last_exit AS (
              SELECT MIN(e.exited_3star_date) AS exited_3star_date
              FROM exits e
              WHERE e.exited_3star_date > (SELECT entered_3star_date FROM last_enter)
            ),
            last_day AS (
              SELECT MAX(d) AS last_day_3star
              FROM s
              WHERE star >= 3
                AND next_star < 3
                AND d >= (SELECT entered_3star_date FROM last_enter)
            )
            SELECT
              (SELECT entered_3star_date FROM last_enter) AS entered_3star_date,
              (SELECT exited_3star_date FROM last_exit)   AS exited_3star_date,
              CASE
                WHEN (SELECT entered_3star_date FROM last_enter) IS NULL THEN NULL
                WHEN (SELECT exited_3star_date FROM last_exit) IS NOT NULL
                  THEN ((SELECT last_day_3star FROM last_day) - (SELECT entered_3star_date FROM last_enter) + 1)
                ELSE NULL
              END AS days_in_3star,
              (SELECT tracking_start_date FROM tracking) AS tracking_start_date
            ;
            """,
            engine,
            params={"t": ticker},
        )
        return df

    @st.cache_data(ttl=300)
    def get_latest_gfv(ticker: str, _cache_buster=None):
        engine = get_engine()
        try:
            df = pd.read_sql(
                """
                SELECT ticker, date, close_price, gfv_price, gfv_status,
                       dcf_value, graham_value,
                       growth_rate_fcf, growth_rate_eps,
                       growth_rate_fcf_raw, growth_rate_eps_raw,
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
            pass

        df = pd.read_sql(
            """
            SELECT DISTINCT ON (ticker)
                   ticker, date, close_price, gfv_price, gfv_status,
                   dcf_value, graham_value,
                   growth_rate_fcf, growth_rate_eps,
                   growth_rate_fcf_raw, growth_rate_eps_raw,
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

    @st.cache_data(ttl=600)
    def get_latest_sector_context(sector: str):
        engine = get_engine()
        df = pd.read_sql(
            """
            SELECT
              summary_date,
              sector,
              total_companies,
              buyzone_pct,
              greer_market_index
            FROM sector_summary_daily
            WHERE summary_date = (
                SELECT MAX(summary_date) FROM sector_summary_daily
            )
              AND sector = %(sector)s
            LIMIT 1;
            """,
            engine,
            params={"sector": sector},
        )
        return df

    # ----------------------------------------------------------
    # Bundle builder
    # ----------------------------------------------------------
    @st.cache_data(ttl=300)
    def build_home_bundle(ticker: str):
        snap = get_latest_snapshot(ticker)
        info = get_company_info(ticker)
        gfv_df = get_latest_gfv(ticker)
        first_trade = fetch_first_trade_date(ticker)
        trans = get_star_transition_dates(ticker)

        bundle = {
            "ticker": ticker,
            "found": not snap.empty,
            "snapshot": snap,
            "info": info,
            "gfv": gfv_df,
            "first_trade": first_trade,
            "transition": trans,
            "sector_context": pd.DataFrame(),
            "company_cycle": {},
            "sector_cycle": {},
        }

        if snap.empty:
            return bundle

        row = snap.iloc[0]
        info_row = info.iloc[0] if not info.empty else None
        sector = first_non_null(
            info_row.get("sector") if info_row is not None else None,
            row.get("sector"),
        )
        sector_df = get_latest_sector_context(sector) if pd.notnull(sector) else pd.DataFrame()
        bundle["sector_context"] = sector_df

        sector_direction_pct = 50.0
        sector_gmi = None
        sector_phase = None
        sector_conf = None

        if not sector_df.empty:
            srow = sector_df.iloc[0]
            sector_buyzone_pct = float(srow["buyzone_pct"]) if pd.notnull(srow["buyzone_pct"]) else 50.0
            sector_direction_pct = round(100.0 - sector_buyzone_pct, 1)
            sector_gmi = safe_round(srow.get("greer_market_index"), 2)

            gv_score = row.get("greer_value_score")
            above_50_count = row.get("above_50_count")
            ys_score = row.get("greer_yield_score")

            gfv_status = None
            if not gfv_df.empty:
                gfv_status = gfv_df.iloc[0].get("gfv_status")

            health_pct = compute_health_pct(gv_score, above_50_count)
            opportunity_pct = compute_opportunity_pct(ys_score, gfv_status)
            company_direction_pct = compute_direction_pct(
                row.get("buyzone_flag"),
                row.get("fvg_last_direction"),
                sector_direction_pct,
            )

            sector_opportunity_pct = opportunity_pct

            sector_phase, sector_conf = classify_phase_with_confidence(
                health_pct,
                sector_buyzone_pct,
                sector_opportunity_pct,
            )

        gfv_status = None
        gfv_price = None
        current_price = row.get("current_price")

        if not gfv_df.empty:
            gfv_row = gfv_df.iloc[0]
            gfv_status = gfv_row.get("gfv_status")
            gfv_price = gfv_row.get("gfv_price")

        health_pct = compute_health_pct(row.get("greer_value_score"), row.get("above_50_count"))
        direction_pct = compute_direction_pct(
            row.get("buyzone_flag"),
            row.get("fvg_last_direction"),
            sector_direction_pct,
        )
        opportunity_pct = compute_opportunity_pct(row.get("greer_yield_score"), gfv_status)
        company_index = compute_company_index(health_pct, direction_pct, opportunity_pct)

        company_phase, company_conf = classify_phase_with_confidence(
            health_pct,
            compute_company_buyzone_proxy(direction_pct),
            opportunity_pct,
        )

        bundle["company_cycle"] = {
            "health_pct": health_pct,
            "direction_pct": direction_pct,
            "opportunity_pct": opportunity_pct,
            "greer_company_index": company_index,
            "phase": company_phase,
            "confidence": round(company_conf, 4),
            "transition_risk": transition_risk_label(company_conf),
        }

        bundle["sector_cycle"] = {
            "sector_direction_pct": sector_direction_pct,
            "sector_phase": sector_phase,
            "sector_confidence": round(sector_conf, 4) if sector_conf is not None else None,
            "sector_greer_market_index": sector_gmi,
        }

        return bundle

    # ----------------------------------------------------------
    # Card renderers
    # ----------------------------------------------------------
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

    def render_company_card(bundle: dict):
        ticker = bundle["ticker"]
        info = bundle["info"]
        trans = bundle["transition"]

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

        stars = int(r.get("greer_star_rating", 0) or 0)
        star_html = ""
        if stars > 0:
            star_icons = "★" * stars + "☆" * (3 - stars)
            star_html = (
                f"<div style='font-size:1.3rem; color:#D4AF37; margin-top:6px;'>"
                f"{star_icons} &nbsp; {stars} Gold Star{'s' if stars>1 else ''}"
                f"</div>"
            )

        became_line = ""
        if not trans.empty:
            entered = trans.loc[0, "entered_3star_date"]
            exited = trans.loc[0, "exited_3star_date"]
            days = trans.loc[0, "days_in_3star"]
            tracking_start = trans.loc[0, "tracking_start_date"]

            entered_d = pd.to_datetime(entered).date() if pd.notnull(entered) else None
            exited_d = pd.to_datetime(exited).date() if pd.notnull(exited) else None

            if entered_d is None and stars >= 3 and pd.notnull(tracking_start):
                entered_d = pd.to_datetime(tracking_start).date()

            if entered_d:
                latest_d = pd.read_sql(
                    "SELECT MAX(snapshot_date)::date AS d FROM public.company_snapshot WHERE ticker=%(t)s;",
                    get_engine(),
                    params={"t": ticker},
                )
                asof = (
                    pd.to_datetime(latest_d.loc[0, "d"]).date()
                    if not latest_d.empty and pd.notnull(latest_d.loc[0, "d"])
                    else date.today()
                )

                if exited_d:
                    days_in = int(days) if pd.notnull(days) else (exited_d - entered_d).days + 1
                    became_line = (
                        f"<div class='company-meta'><b>3⭐ Entered:</b> {entered_d} "
                        f"&nbsp;|&nbsp; <b>Days @ 3⭐:</b> {days_in} "
                        f"&nbsp;|&nbsp; <b>Status:</b> Exited {exited_d}</div>"
                    )
                else:
                    days_so_far = (asof - entered_d).days + 1
                    became_line = (
                        f"<div class='company-meta'><b>3⭐ Entered:</b> {entered_d} "
                        f"&nbsp;|&nbsp; <b>Days @ 3⭐:</b> {days_so_far} "
                        f"&nbsp;|&nbsp; <b>Status:</b> Active</div>"
                    )

        html = textwrap.dedent(f"""\
        <div class="company-card" style="color: rgb(49, 51, 63);">
          <div class="card-section-title">Company</div>
          <div class="company-title">{ticker} — {name}</div>
          <div class="company-meta"><b>Exchange:</b> {exchange}</div>
          <div class="company-meta"><b>Sector:</b> {sector}</div>
          <div class="company-meta"><b>Industry:</b> {industry}</div>
          {delisted_line}
          {star_html}
          {became_line}
        </div>
        """)

        st.markdown(html, unsafe_allow_html=True)

    def render_gfv_card(gfv_row: pd.Series):
        gfv = gfv_row.get("gfv_price")
        current_price = gfv_row.get("close_price")
        gfv_status = str(gfv_row.get("gfv_status") or "gray").lower()
        upside_pct = safe_pct_to_fair_value(current_price, gfv)

        bg = {
            "gold": "#D4AF37",
            "green": "#4CAF50",
            "red": "#F44336",
            "gray": "#9CA3AF",
        }.get(gfv_status, "#9CA3AF")

        fg = "black" if gfv_status == "gold" else "white"
        main = "—" if pd.isnull(gfv) else f"${float(gfv):,.2f}"

        if upside_pct is None:
            sub = f"{gfv_status.title()}"
        else:
            sign = "+" if upside_pct >= 0 else ""
            sub = f"{gfv_status.title()} • {sign}{upside_pct:.1f}% vs price"

        render_metric_card("Greer Fair Value", main, sub, bg, fg)

    def render_price_card(current_price, snapshot_date):
        price_main = "—" if pd.isnull(current_price) else f"${float(current_price):,.2f}"
        price_sub = f"as of {snapshot_date}" if snapshot_date else "—"
        render_metric_card("Price", price_main, price_sub, "#111111", "white")

    def render_company_cycle_card(company_cycle: dict):
        if not company_cycle:
            render_metric_card("Company Cycle", "—", "No data", "#E0E0E0", "black")
            return

        phase = phase_label_with_icon(company_cycle["phase"])
        conf = round(float(company_cycle["confidence"]) * 100)
        risk = company_cycle["transition_risk"]
        main = f"{company_cycle['greer_company_index']:.1f}"
        sub = f"{phase} • {conf}% • {risk}"

        def company_index_color(score):
            if score >= 75:
                return "#4CAF50"   # strong
            elif score >= 55:
                return "#5BC0DE"   # neutral/transition
            else:
                return "#F44336"   # weak
        bg = company_index_color(company_cycle["greer_company_index"])
        render_metric_card("Greer Company Index", main, sub, bg, "white")

    def render_sector_backdrop_card(sector_cycle: dict):
        if not sector_cycle:
            render_metric_card("Greer Sector Index", "—", "No data", "#E0E0E0", "black")
            return

        phase = phase_label_with_icon(sector_cycle.get("sector_phase"))
        gmi = sector_cycle.get("sector_greer_market_index")

        bg = index_color(gmi)
        fg = "white"

        main = "—" if gmi is None or pd.isnull(gmi) else f"{float(gmi):.1f}"

        if gmi is None or pd.isnull(gmi):
            sub = f"{phase}"
        else:
            if float(gmi) >= 75:
                backdrop = "Strong Backdrop"
            elif float(gmi) >= 55:
                backdrop = "Supportive Backdrop"
            else:
                backdrop = "Weak Backdrop"

            sub = f"{phase} • {backdrop}"

        render_metric_card("Greer Sector Index", main, sub, bg, fg)

    def render_signal_summary(bundle: dict):
        snap = bundle["snapshot"].iloc[0]
        company_cycle = bundle["company_cycle"]
        sector_cycle = bundle["sector_cycle"]

        gv_score = snap.get("greer_value_score")
        ys_score = snap.get("greer_yield_score")
        in_bz = bool(snap.get("buyzone_flag", False))
        fvg_dir = str(snap.get("fvg_last_direction") or "").lower()

        quality_text = "strong fundamentals" if pd.notnull(gv_score) and float(gv_score) >= 50 else "weaker fundamentals"
        value_text = "attractive valuation" if pd.notnull(ys_score) and float(ys_score) >= 3 else "mixed valuation"
        bz_text = "BuyZone is active" if in_bz else "BuyZone is inactive"

        if fvg_dir == "bullish":
            fvg_text = "bullish FVG support is present"
        elif fvg_dir == "bearish":
            fvg_text = "bearish FVG pressure remains"
        else:
            fvg_text = "no strong FVG signal is present"

        company_phase = phase_label_with_icon(company_cycle.get("phase"))
        sector_phase = phase_label_with_icon(sector_cycle.get("sector_phase"))

        st.markdown(
            f"""
            <div class="summary-banner">
                <b>Signal Summary:</b>
                {bundle['ticker']} is currently in <b>{company_phase}</b> with
                <b>{round(float(company_cycle.get('confidence', 0)) * 100)}% confidence</b>.
                The stock shows <b>{quality_text}</b>, <b>{value_text}</b>, and
                <b>{bz_text}</b>. On the technical side, <b>{fvg_text}</b>.
                Sector backdrop is <b>{sector_phase}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

    def render_snapshot_header(bundle: dict):
        snap = bundle["snapshot"].iloc[0]
        info = bundle["info"]
        gfv_df = bundle["gfv"]
        company_cycle = bundle["company_cycle"]
        sector_cycle = bundle["sector_cycle"]

        info_row = info.iloc[0] if not info.empty else None
        gfv_row = gfv_df.iloc[0] if not gfv_df.empty else None

        snapshot_date_raw = first_non_null(
            gfv_row.get("date") if gfv_row is not None else None,
            snap.get("snapshot_date"),
            snap.get("fvg_last_date"),
        )
        snapshot_date = pd.to_datetime(snapshot_date_raw).date() if pd.notnull(snapshot_date_raw) else None

        sector = first_non_null(
            info_row.get("sector") if info_row is not None else None,
            snap.get("sector"),
        )
        sector = sector if pd.notnull(sector) else "—"

        parts = [
            f"Snapshot: {snapshot_date}" if snapshot_date else "Snapshot: —",
            f"Sector: {sector}",
        ]

        if company_cycle:
            parts.append(f"Company: {phase_label_with_icon(company_cycle.get('phase'))}")
        if sector_cycle and sector_cycle.get("sector_phase"):
            parts.append(f"Sector Phase: {phase_label_with_icon(sector_cycle.get('sector_phase'))}")

        st.markdown(
            f"<div class='greer-subheader'>{' &nbsp;•&nbsp; '.join(parts)}</div>",
            unsafe_allow_html=True
        )

    # ----------------------------------------------------------
    # Existing detail renderers
    # - Kept from your current page so we can plug them into tabs
    # ----------------------------------------------------------
    def render_gv_details(ticker, engine):
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

        col1, col2 = st.columns([1, 3])
        with col1:
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

        st.markdown(
            f"<h3 style='margin:20px 0 10px;'>📈 {ticker.upper()} – Greer Value Score Map (180 trading days)</h3>",
            unsafe_allow_html=True
        )
        fig, (ax_price, ax_ribbon) = plt.subplots(
            2, 1, sharex=True, figsize=(9, 5), gridspec_kw={"height_ratios": [4, 0.4]}
        )
        ax_price.plot(hist["date"], hist["close"], color="black", lw=1)
        ax_price.set_ylabel("Close")
        ax_price.set_title(f"{ticker.upper()} – Greer Value Score Map (180 trading days)")

        exceptional_mask = (hist["above_50_count"] == 6)
        strong_mask = (hist["greer_score"] >= 50) & (hist["above_50_count"] != 6)
        weak_mask = (hist["greer_score"] < 50) & (hist["greer_score"].notnull())
        none_mask = hist["greer_score"].isnull()

        ax_ribbon.fill_between(hist["date"], 0, 1, where=exceptional_mask, color="#D4AF37", alpha=0.7)
        ax_ribbon.fill_between(hist["date"], 0, 1, where=strong_mask, color="#4CAF50", alpha=0.7)
        ax_ribbon.fill_between(hist["date"], 0, 1, where=weak_mask, color="#F44336", alpha=0.7)
        ax_ribbon.fill_between(hist["date"], 0, 1, where=none_mask, color="#E0E0E0", alpha=0.7)

        ax_ribbon.set_yticks([])
        ax_ribbon.set_ylim(0, 1)
        ax_ribbon.set_xlabel("Date")
        st.pyplot(fig)

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
            4: "#D4AF37",
            3: "#4CAF50",
            2: "#2196F3",
            1: "#2196F3",
            0: "#F44336",
        }
        for score_val, color in colors.items():
            mask = df["score"] == score_val
            ax_ribbon.fill_between(df["date"], 0, 1, where=mask, color=color, alpha=0.7)

        ax_ribbon.set_yticks([])
        ax_ribbon.set_ylim(0, 1)
        ax_ribbon.set_xlabel("Date")
        st.pyplot(fig)

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
        ax_ribbon.set_yticks([])
        ax_ribbon.set_ylim(0, 1)
        ax_ribbon.set_xlabel("Date")
        st.pyplot(fig)

    # ----------------------------------------------------------
    # Search Header
    # ----------------------------------------------------------
    st.markdown('<div class="greer-header">Greer Company Snapshot</div>', unsafe_allow_html=True)

    # Read URL param once
    qp = st.query_params
    ticker_param = qp.get("ticker")

    if isinstance(ticker_param, (list, tuple)):
        url_ticker = ticker_param[0]
    elif isinstance(ticker_param, str):
        url_ticker = ticker_param
    else:
        url_ticker = ""

    # Initialize session state only once
    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = url_ticker.upper().strip() if url_ticker else ""

    # Input is controlled by session state only
    user_ticker = st.text_input(
        "Search",
        placeholder="Enter ticker (e.g., AAPL)",
        key="ticker_input",
        label_visibility="collapsed",
    )

    ticker = st.session_state["ticker_input"].strip().upper()

    if not ticker:
        return

    # Keep URL synced, but do not use it to keep resetting the widget
    st.query_params["ticker"] = ticker

    bundle = build_home_bundle(ticker)

    if not bundle["found"]:
        cols = st.columns([2, 1, 1, 1, 1])
        with cols[0]:
            if not bundle["info"].empty:
                render_company_card(bundle)

        st.error(f"Ticker '{ticker}' not found in latest snapshot.")
        st.session_state["pending_add_ticker"] = ticker

        from app_nav import build_pages
        _, PAGE_MAP = build_pages()

        st.page_link(
            PAGE_MAP["Add Company"],
            label="Would you like to add this company? Click here.",
            icon="➕",
            use_container_width=True
        )
        return

    # ----------------------------------------------------------
    # Top Snapshot Layout
    # ----------------------------------------------------------
    render_snapshot_header(bundle)

    snap = bundle["snapshot"].iloc[0]
    gfv_df = bundle["gfv"]
    company_cycle = bundle["company_cycle"]
    sector_cycle = bundle["sector_cycle"]

    top_left, top_right = st.columns([2.2, 4.0])

    with top_left:
        render_company_card(bundle)

    with top_right:
        row1 = st.columns(4)
        row2 = st.columns(4)

        gfv_row = gfv_df.iloc[0] if not gfv_df.empty else None

        current_price = first_non_null(
            gfv_row.get("close_price") if gfv_row is not None else None,
            snap.get("current_price"),
        )

        snapshot_date_raw = first_non_null(
            gfv_row.get("date") if gfv_row is not None else None,
            snap.get("snapshot_date"),
            snap.get("fvg_last_date"),
        )
        snapshot_date = pd.to_datetime(snapshot_date_raw).date() if pd.notnull(snapshot_date_raw) else None
        
        with row1[0]:
            render_price_card(current_price, snapshot_date)

        with row1[1]:
            if not gfv_df.empty:
                render_gfv_card(gfv_df.iloc[0])
            else:
                render_metric_card("Greer Fair Value", "—", "No data", "#9CA3AF", "white")

        gv_score = snap.get("greer_value_score")
        gv_grade, gv_bg, gv_txt = classify_greer(gv_score, snap.get("above_50_count"))
        gv_main = "—" if pd.isnull(gv_score) else f"{float(gv_score):.2f}%"
        gv_sub = gv_grade if pd.notnull(gv_score) else "—"
        with row1[2]:
            render_metric_card("Greer Value", gv_main, gv_sub, gv_bg, gv_txt)

        ys_score = snap.get("greer_yield_score")
        ys_score = int(ys_score) if pd.notnull(ys_score) else None
        y_grade, y_bg, y_txt = classify_yield(ys_score)
        y_main = "—" if ys_score is None else f"{ys_score}/4"
        y_sub = y_grade if ys_score is not None else "—"
        with row1[3]:
            render_metric_card("Yield Score", y_main, y_sub, y_bg, y_txt)

        in_bz = bool(snap.get("buyzone_flag", False))
        bz_bg = "#4CAF50" if in_bz else "#E0E0E0"
        bz_txt = "white" if in_bz else "black"
        if in_bz:
            bz_main = "Triggered"
            bz_sub = f"since {pd.to_datetime(snap.get('bz_start_date')).strftime('%Y-%m-%d')}" if snap.get("bz_start_date") else ""
        else:
            bz_main = "No Signal"
            bz_sub = f"left {pd.to_datetime(snap.get('bz_end_date')).strftime('%Y-%m-%d')}" if snap.get("bz_end_date") else ""
        with row2[0]:
            render_metric_card("BuyZone", bz_main, bz_sub, bz_bg, bz_txt)

        fvg_dir = snap.get("fvg_last_direction")
        fvg_date = pd.to_datetime(snap.get("fvg_last_date")).strftime('%Y-%m-%d') if snap.get("fvg_last_date") else "—"
        fvg_bg = "#4CAF50" if fvg_dir == "bullish" else "#F44336" if fvg_dir == "bearish" else "#90CAF9"
        fvg_main = (fvg_dir or "No Gap").capitalize()
        fvg_sub = f"last {fvg_date}"
        with row2[1]:
            render_metric_card("Fair Value Gap", fvg_main, fvg_sub, fvg_bg, "white")

        with row2[2]:
            render_company_cycle_card(company_cycle)

        with row2[3]:
            render_sector_backdrop_card(sector_cycle)

    render_signal_summary(bundle)

    # ----------------------------------------------------------
    # Detail Tabs
    # ----------------------------------------------------------
    engine = get_engine()

    tab_overview, tab_valuation, tab_technicals, tab_cycle = st.tabs([
        "Overview",
        "Valuation",
        "Technicals",
        "Cycle",
    ])

    with tab_overview:
        st.subheader("Overview")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Greer Company Index",
                f"{company_cycle.get('greer_company_index', 0):.2f}" if company_cycle else "—",
                help="Average of company health, direction, and opportunity."
            )

        with c2:
            conf_pct = round(float(company_cycle.get("confidence", 0)) * 100) if company_cycle else None
            st.metric(
                "Company Confidence",
                f"{conf_pct}%" if conf_pct is not None else "—",
            )

        with c3:
            st.metric(
                "Transition Risk",
                company_cycle.get("transition_risk", "—") if company_cycle else "—",
            )

        st.markdown("---")
        st.markdown("### Current Phase")
        if company_cycle:
            phase_badge_inline(company_cycle["phase"], company_cycle["confidence"], label_prefix="Company Phase")
            st.caption(phase_interpretation_text(company_cycle["phase"], company_cycle["confidence"]))
            st.caption(phase_confidence_note(company_cycle["confidence"]))

        st.markdown("---")
        st.markdown("### At-a-glance Notes")
        notes = []

        if pd.notnull(gv_score):
            notes.append(f"- **Greer Value:** {float(gv_score):.1f}% ({gv_grade})")
        if ys_score is not None:
            notes.append(f"- **Yield Score:** {ys_score}/4 ({y_grade})")
        if not gfv_df.empty:
            gfv_row = gfv_df.iloc[0]
            upside_pct = safe_pct_to_fair_value(gfv_row.get("close_price"), gfv_row.get("gfv_price"))
            if upside_pct is not None:
                sign = "+" if upside_pct >= 0 else ""
                notes.append(f"- **GFV Upside:** {sign}{upside_pct:.1f}%")
        notes.append(f"- **BuyZone:** {'Triggered' if in_bz else 'Not active'}")
        notes.append(f"- **FVG:** {(fvg_dir or 'None').capitalize()}")

        st.markdown("\n".join(notes))

    with tab_valuation:
        st.subheader("Valuation")
        render_gv_details(ticker, engine)
        st.markdown("---")
        render_yield_details(ticker, engine)

    with tab_technicals:
        st.subheader("Technicals")
        render_buyzone_details(ticker, engine)
        st.markdown("---")
        render_fvg_details(ticker, engine)

    with tab_cycle:
        st.subheader("Company & Sector Cycle")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Company Cycle")
            if company_cycle:
                st.write(
                    {
                        "Ticker": ticker,
                        "Health %": company_cycle.get("health_pct"),
                        "Direction %": company_cycle.get("direction_pct"),
                        "Opportunity %": company_cycle.get("opportunity_pct"),
                        "Greer Company Index": company_cycle.get("greer_company_index"),
                        "Phase": company_cycle.get("phase"),
                        "Confidence %": round(float(company_cycle.get("confidence", 0)) * 100, 1),
                        "Transition Risk": company_cycle.get("transition_risk"),
                    }
                )

        with c2:
            st.markdown("### Sector Backdrop")
            if sector_cycle:
                st.write(
                    {
                        "Sector": snap.get("sector"),
                        "Sector Direction %": sector_cycle.get("sector_direction_pct"),
                        "Sector Phase": sector_cycle.get("sector_phase"),
                        "Sector Confidence %": round(float(sector_cycle.get("sector_confidence", 0)) * 100, 1) if sector_cycle.get("sector_confidence") is not None else None,
                        "Sector Greer Market Index": sector_cycle.get("sector_greer_market_index"),
                    }
                )


# ----------------------------------------------------------
# App entrypoint
# - Only run navigation when this file is executed directly:
#   streamlit run Home.py
# - When imported (by pages/0_Home.py), nothing executes.
# ----------------------------------------------------------
def main():
    from app_nav import build_pages
    all_pages, _PAGE_MAP = build_pages()
    st.navigation(all_pages, position="sidebar").run()

if __name__ == "__main__":
    main()