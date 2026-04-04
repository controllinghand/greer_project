# Home.py
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
from value_utils import get_value_level, value_level_label, value_level_short

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

    from db import get_engine
    from market_cycle_utils import (
        classify_phase_with_confidence,
        phase_badge_inline,
        phase_confidence_note,
        phase_interpretation_text,
    )
    from prediction_utils import calculate_prediction_score

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
    # Load current prediction inputs for one ticker
    # ----------------------------------------------------------
    @st.cache_data(ttl=300)
    def load_company_prediction_input(ticker: str):
        engine = get_engine()

        query = """
            WITH latest_market AS (
                SELECT
                    date,
                    buyzone_pct,
                    CASE
                        WHEN buyzone_pct >= 66 THEN 'EXTREME_OPPORTUNITY'
                        WHEN buyzone_pct >= 46 THEN 'ELEVATED_OPPORTUNITY'
                        WHEN buyzone_pct >= 14 THEN 'NORMAL'
                        WHEN buyzone_pct >= 10 THEN 'LOW_OPPORTUNITY'
                        ELSE 'EXTREME_GREED'
                    END AS goi_zone
                FROM buyzone_breadth
                ORDER BY date DESC
                LIMIT 1
            ),
            company_phase_history AS (
                SELECT
                    g.ticker,
                    g.date,
                    g.phase,
                    g.confidence,
                    LAG(g.phase) OVER (PARTITION BY g.ticker ORDER BY g.date) AS prior_phase,
                    ROW_NUMBER() OVER (PARTITION BY g.ticker ORDER BY g.date DESC) AS rn
                FROM greer_company_index_daily g
                WHERE g.ticker = %(t)s
            ),
            latest_company_phase AS (
                SELECT
                    ticker,
                    date AS snapshot_date,
                    phase,
                    prior_phase,
                    confidence
                FROM company_phase_history
                WHERE rn = 1
            )
            SELECT
                ds.ticker,
                ds.name,
                ds.sector,
                ds.industry,
                ds.current_price,
                ds.greer_star_rating,
                ds.greer_value_score,
                ds.greer_yield_score,
                ds.buyzone_flag,
                ds.gfv_price,
                ds.gfv_status,
                lcp.snapshot_date,
                lcp.phase,
                lcp.prior_phase,
                lcp.confidence,
                lm.buyzone_pct AS market_buyzone_pct,
                lm.goi_zone
            FROM dashboard_snapshot ds
            JOIN latest_company_phase lcp
              ON lcp.ticker = ds.ticker
            CROSS JOIN latest_market lm
            WHERE ds.ticker = %(t)s
            LIMIT 1;
        """

        return pd.read_sql(query, engine, params={"t": ticker})
    # ----------------------------------------------------------
    # Render company prediction section
    # ----------------------------------------------------------
    def render_company_prediction_section(ticker: str):
        pred_df = load_company_prediction_input(ticker)

        if pred_df.empty:
            st.info("No prediction data available for this company.")
            return

        row = pred_df.iloc[0]
        score = calculate_prediction_score(row)

        prediction_score = float(score["prediction_score"])
        raw_bucket = score["score_bucket"]
        calibration_bucket = score["calibration_bucket"]
        signal_tier = score["signal_tier"]
        signal_horizon = score["signal_horizon"]
        expected_win_rate = score["expected_win_rate_trend"]
        expected_return = score["expected_return_trend"]
        setup_label = score["setup_label"]

        current_goi_zone = row.get("goi_zone", "UNKNOWN")
        current_goi_label = str(current_goi_zone).replace("_", " ").title()

        # ----------------------------------------------------------
        # Signal color / explanation / guidance
        # ----------------------------------------------------------
        signal_color_map = {
            "Optimal": "#1565C0",            # blue
            "High Opportunity": "#2E7D32",   # green
            "Over-Filtered": "#EF6C00",      # orange
            "Watchlist": "#9E9E9E",          # gray
        }
        score_color = signal_color_map.get(signal_tier, "#9E9E9E")

        signal_explanations = {
            "Optimal": "Historically the best balance of return and win rate. This is the strongest calibrated trend bucket.",
            "High Opportunity": "Strong setup with attractive upside, but slightly noisier than the optimal bucket.",
            "Over-Filtered": "Very selective setup, but historical testing suggests too much filtering reduces opportunity.",
            "Watchlist": "No strong statistical edge right now. Better treated as a monitor-only setup.",
        }
        signal_explanation = signal_explanations.get(
            signal_tier,
            "No strong statistical edge right now."
        )

        if signal_tier == "Optimal":
            positioning_guidance = "✅ Best used as a 120–180 day trend position. Patience matters more than tight risk management."
        elif signal_tier == "High Opportunity":
            positioning_guidance = "👍 Attractive setup with strong historical edge. Best treated as a medium-term trend hold."
        elif signal_tier == "Over-Filtered":
            positioning_guidance = "⚠️ Selective setup, but research suggests this bucket can miss opportunity. Use with caution."
        else:
            positioning_guidance = "👀 No strong edge — monitor for a better setup before acting."

        st.markdown("### 🔮 Prediction")

        st.caption(
            "Probability-based trend outlook using phase, transitions, GOI regime, BuyZone, confidence, and fundamentals."
        )

        st.warning("""
This is a medium-term trend system, not a quick trade.

- Best historical results came from holding 120–180 trading days
- Many winners experience ~10% drawdowns before moving higher
- Tight stop losses can exit strong future winners too early
""")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Prediction Score", f"{prediction_score:.1f}")
            st.caption(f"Raw Bucket: {raw_bucket}")

            st.markdown(
                f"""
                <div style="margin-top: 0.25rem; font-weight: 600; color: {score_color};">
                    {signal_tier}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            if expected_win_rate is not None:
                st.metric("Expected Trend Win Rate", f"{expected_win_rate * 100:.1f}%")
                st.caption(f"Calibration Bucket: {calibration_bucket}")
            else:
                st.metric("Expected Trend Win Rate", "N/A")
                st.caption("Calibration Bucket: —")

        with c3:
            if expected_return is not None:
                st.metric("Expected Trend Return", f"{expected_return * 100:.1f}%")
                st.caption(signal_horizon)
            else:
                st.metric("Expected Trend Return", "N/A")
                st.caption(signal_horizon)

        st.markdown(
            f"""
            <div class="summary-banner">
                <b>Setup:</b> {setup_label} <br>
                <b>GOI:</b> {current_goi_label} <br>
                <b>Phase Transition:</b> {(row.get("prior_phase") or "Unknown").title()} → {(row.get("phase") or "Unknown").title()} <br>
                <b>Signal Horizon:</b> {signal_horizon}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"🔍 {signal_explanation}")

        st.info(positioning_guidance)

        st.markdown("### Historical Research Snapshot")
        research_df = pd.DataFrame([
            {"Metric": "Optimal bucket", "Value": "110+"},
            {"Metric": "Best holding period", "Value": "120–180 trading days"},
            {"Metric": "Typical winner drawdown", "Value": "~10% before trend develops"},
            {"Metric": "180d historical profile", "Value": "~73% win rate / ~20% average return"},
        ])
        st.dataframe(research_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------------
    # Load Greer Company Index history
    # ----------------------------------------------------------
    @st.cache_data(ttl=300)
    def load_company_index_history(ticker: str):
        engine = get_engine()

        query = """
            SELECT
                date,
                greer_company_index,
                phase,
                health_pct,
                direction_pct,
                opportunity_pct
            FROM greer_company_index_daily
            WHERE ticker = %(t)s
            ORDER BY date;
        """

        df = pd.read_sql(
            query,
            engine,
            params={"t": ticker},
            parse_dates=["date"],
        )

        return df
    
    # ----------------------------------------------------------
    # Build continuous phase bands for background shading
    # - Merges very short runs to reduce barcode striping
    # ----------------------------------------------------------
    def build_phase_bands(df: pd.DataFrame, min_run_days: int = 14) -> list[dict]:
        if df.empty or "phase" not in df.columns or "date" not in df.columns:
            return []

        phase_colors = {
            "RECOVERY": "rgba(91,192,222,0.10)",
            "EXPANSION": "rgba(92,184,92,0.10)",
            "EUPHORIA": "rgba(240,173,78,0.10)",
            "CONTRACTION": "rgba(217,83,79,0.10)",
        }

        work = (
            df[["date", "phase"]]
            .dropna(subset=["date", "phase"])
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )

        if work.empty:
            return []

        # ------------------------------------------
        # First pass: build raw contiguous runs
        # ------------------------------------------
        runs = []
        start_idx = 0
        current_phase = work.loc[0, "phase"]

        for i in range(1, len(work)):
            phase = work.loc[i, "phase"]
            if phase != current_phase:
                runs.append(
                    {
                        "phase": current_phase,
                        "start_idx": start_idx,
                        "end_idx": i - 1,
                        "x0": work.loc[start_idx, "date"],
                        "x1": work.loc[i - 1, "date"],
                    }
                )
                current_phase = phase
                start_idx = i

        runs.append(
            {
                "phase": current_phase,
                "start_idx": start_idx,
                "end_idx": len(work) - 1,
                "x0": work.loc[start_idx, "date"],
                "x1": work.loc[len(work) - 1, "date"],
            }
        )

        # ------------------------------------------
        # Add run lengths
        # ------------------------------------------
        for run in runs:
            run["run_days"] = max((run["x1"] - run["x0"]).days + 1, 1)

        # ------------------------------------------
        # Second pass: merge short runs
        # Rule:
        # - if a run is shorter than min_run_days
        # - merge it into the longer adjacent run
        # - if both neighbors have same phase, merge into that phase
        # ------------------------------------------
        changed = True
        while changed and len(runs) > 1:
            changed = False
            new_runs = []
            i = 0

            while i < len(runs):
                run = runs[i]

                if run["run_days"] >= min_run_days:
                    new_runs.append(run)
                    i += 1
                    continue

                prev_run = new_runs[-1] if new_runs else None
                next_run = runs[i + 1] if i + 1 < len(runs) else None

                # If both neighbors exist and match, absorb short run into that phase
                if prev_run and next_run and prev_run["phase"] == next_run["phase"]:
                    prev_run["x1"] = next_run["x1"]
                    prev_run["end_idx"] = next_run["end_idx"]
                    prev_run["run_days"] = max((prev_run["x1"] - prev_run["x0"]).days + 1, 1)
                    i += 2
                    changed = True
                    continue

                # Otherwise merge into longer neighbor
                if prev_run and next_run:
                    if prev_run["run_days"] >= next_run["run_days"]:
                        prev_run["x1"] = run["x1"]
                        prev_run["end_idx"] = run["end_idx"]
                        prev_run["run_days"] = max((prev_run["x1"] - prev_run["x0"]).days + 1, 1)
                    else:
                        next_run["x0"] = run["x0"]
                        next_run["start_idx"] = run["start_idx"]
                        next_run["run_days"] = max((next_run["x1"] - next_run["x0"]).days + 1, 1)
                    i += 1
                    changed = True
                    continue

                # Only previous neighbor exists
                if prev_run and not next_run:
                    prev_run["x1"] = run["x1"]
                    prev_run["end_idx"] = run["end_idx"]
                    prev_run["run_days"] = max((prev_run["x1"] - prev_run["x0"]).days + 1, 1)
                    i += 1
                    changed = True
                    continue

                # Only next neighbor exists
                if next_run and not prev_run:
                    next_run["x0"] = run["x0"]
                    next_run["start_idx"] = run["start_idx"]
                    next_run["run_days"] = max((next_run["x1"] - next_run["x0"]).days + 1, 1)
                    i += 1
                    changed = True
                    continue

                new_runs.append(run)
                i += 1

            runs = new_runs

        # ------------------------------------------
        # Final output for plotly vrect bands
        # ------------------------------------------
        bands = []
        for run in runs:
            bands.append(
                {
                    "phase": run["phase"],
                    "x0": run["x0"],
                    "x1": run["x1"],
                    "color": phase_colors.get(run["phase"], "rgba(160,160,160,0.08)"),
                }
            )

        return bands

    # ----------------------------------------------------------
    # Load price history for company index chart
    # ----------------------------------------------------------
    @st.cache_data(ttl=300)
    def load_price_history(ticker: str):
        engine = get_engine()

        query = """
            SELECT
                date,
                close
            FROM prices
            WHERE ticker = %(t)s
            ORDER BY date;
        """

        df = pd.read_sql(
            query,
            engine,
            params={"t": ticker},
            parse_dates=["date"],
        )

        return df

    # ----------------------------------------------------------
    # Coverage score for historical company index quality
    # ----------------------------------------------------------
    def add_history_coverage_flags(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        out = df.copy()

        out["has_health"] = out["health_pct"].notna() & (out["health_pct"] > 0)
        out["has_direction"] = out["direction_pct"].notna()
        out["has_opportunity"] = out["opportunity_pct"].notna()

        out["coverage_count"] = (
            out["has_health"].astype(int)
            + out["has_direction"].astype(int)
            + out["has_opportunity"].astype(int)
        )

        out["is_full_coverage"] = out["coverage_count"] >= 3
        out["is_partial_coverage"] = out["coverage_count"] >= 2

        return out


    def gv_bucket_label(score, above_50):
        if pd.notnull(above_50) and int(above_50) == 6:
            return "Gold"
        if pd.notnull(score) and float(score) >= 50:
            return "Green"
        if pd.notnull(score):
            return "Red"
        return "—"


    def ys_bucket_label(yield_score):
        if yield_score is None:
            return "—"
        if int(yield_score) == 4:
            return "Gold"
        if int(yield_score) >= 2:
            return "Green"
        return "Red"

    def clamp(x: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, x))

    def first_non_null(*values):
        for v in values:
            if pd.notnull(v):
                return v
        return None

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
    # Bar / Gauge UI helpers
    # ----------------------------------------------------------
    def dial_bucket(value: float) -> str:
        x = float(value)
        if x <= 24:
            return "weak"
        if x <= 49:
            return "mixed"
        if 49 < x < 51:
            return "neutral"
        if x <= 74:
            return "strong"
        return "very_strong"

    def dial_label(bucket: str) -> str:
        return {
            "weak": "Weak 🔴",
            "mixed": "Mixed 🟡",
            "neutral": "Neutral ⚪",
            "strong": "Strong 🟢",
            "very_strong": "Very Strong 🟩",
        }.get(bucket, "Mixed 🟡")

    def bar_color(value: float) -> str:
        v = float(value)
        if v <= 24:
            return "#D9534F"
        if v <= 49:
            return "#F0AD4E"
        if v < 51:
            return "#DDDDDD"
        if v <= 74:
            return "#A6D96A"
        return "#5CB85C"

    def render_top_score_bar(
        title: str,
        value: float,
        subtitle: str = "",
        fill_color: str | None = None,
        height: int = 78,
    ):
        v = clamp(float(value), 0.0, 100.0)
        color = fill_color if fill_color is not None else bar_color(v)
        bucket = dial_bucket(v)

        html = f"""
        <div style="margin-bottom:10px; font-family: sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="font-weight:700; font-size:15px; color:#111827;">{title}</div>
                <div style="font-weight:700; font-size:15px; color:#111827;">{v:.1f}%</div>
            </div>

            <div style="width:100%; background:#E5E7EB; border-radius:999px; height:16px; overflow:hidden; border:1px solid rgba(0,0,0,0.06);">
                <div style="width:{v}%; background:{color}; height:100%; border-radius:999px;"></div>
            </div>

            <div style="margin-top:5px; font-size:12px; color:#6B7280; line-height:1.3;">
                {subtitle} <span style="opacity:0.5;">•</span> <span style="font-weight:600;">{dial_label(bucket)}</span>
            </div>
        </div>
        """
        components.html(html, height=height)

    def render_horizontal_score_bar(title: str, value: float, subtitle: str = ""):
        v = clamp(float(value), 0.0, 100.0)
        color = bar_color(v)
        bucket = dial_bucket(v)

        html = f"""
        <div style="margin-bottom:14px; font-family: sans-serif;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="font-weight:700; font-size:15px;">{title}</div>
                <div style="font-weight:700; font-size:15px;">{v:.1f}%</div>
            </div>

            <div style="width:100%; background:#E5E7EB; border-radius:999px; height:14px; overflow:hidden; border:1px solid rgba(0,0,0,0.06);">
                <div style="width:{v}%; background:{color}; height:100%; border-radius:999px;"></div>
            </div>

            <div style="margin-top:5px; font-size:12px; color:#6B7280;">
                {subtitle} <span style="opacity:0.5;">•</span> <span style="font-weight:600;">{dial_label(bucket)}</span>
            </div>
        </div>
        """
        components.html(html, height=80)

    def render_status_pill(label: str, value: str, bg: str, fg: str = "white"):
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:8px 12px;
                border-radius:999px;
                background:{bg};
                color:{fg};
                font-size:13px;
                font-weight:700;
                margin-right:8px;
                margin-bottom:8px;
            ">
                {label}: {value}
            </div>
            """,
            unsafe_allow_html=True
        )

    def render_semicircle_gauge(title: str, value: float, subtitle: str, chart_key: str):
        v = clamp(float(value), 0.0, 100.0)
        b = dial_bucket(v)

        fig = go.Figure(
            go.Indicator(
                mode="gauge",
                value=v,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999999"},
                    "bar": {"color": "#777777", "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 24], "color": "#D9534F"},
                        {"range": [25, 49], "color": "#F0AD4E"},
                        {"range": [50, 50], "color": "#DDDDDD"},
                        {"range": [51, 74], "color": "#A6D96A"},
                        {"range": [75, 100], "color": "#5CB85C"},
                    ],
                    "threshold": {
                        "line": {"color": "#666666", "width": 6},
                        "thickness": 0.8,
                        "value": v,
                    },
                },
            )
        )

        fig.add_annotation(
            text=f"{v:.1f}%",
            x=0.5,
            y=0.22,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=28, color="#222222"),
            xanchor="center",
            yanchor="middle",
        )

        fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
            font=dict(color="#222222"),
        )

        st.subheader(title)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
            key=chart_key,
        )
        st.caption(f"Status: **{dial_label(b)}**")

        if subtitle:
            st.markdown(
                f"""
                <div style="
                    margin-top: -2px;
                    font-size: 12px;
                    line-height: 1.35;
                    color: #6B7280;
                ">
                    {subtitle}
                </div>
                """,
                unsafe_allow_html=True,
            )

    def render_cycle_strip(current_phase: str):
        phase = str(current_phase).strip().upper()

        phases = ["RECOVERY", "EXPANSION", "EUPHORIA", "CONTRACTION"]

        colors = {
            "RECOVERY": "#5BC0DE",
            "EXPANSION": "#5CB85C",
            "EUPHORIA": "#F0AD4E",
            "CONTRACTION": "#D9534F",
        }

        icons = {
            "RECOVERY": "🩹",
            "EXPANSION": "📈",
            "EUPHORIA": "🔥",
            "CONTRACTION": "📉",
        }

        if phase not in phases:
            phase = phases[0]

        st.markdown(
            """
            <style>
              .cycle-wrap{
                display:flex;
                align-items:center;
                gap:10px;
                flex-wrap:wrap;
                margin-top:6px;
                margin-bottom:6px;
              }
              .cycle-step{
                display:flex;
                align-items:center;
                gap:8px;
                padding:8px 12px;
                border-radius:999px;
                border:1px solid rgba(0,0,0,0.08);
                background: rgba(255,255,255,0.65);
                font-size:12px;
                font-weight:700;
                letter-spacing:0.2px;
                color:#1F2937;
              }
              .cycle-step .dot{
                width:10px; height:10px;
                border-radius:50%;
                display:inline-block;
              }
              .cycle-arrow{
                font-size:18px;
                opacity:0.55;
                user-select:none;
              }
              .cycle-active{
                box-shadow: 0 6px 18px rgba(0,0,0,0.10);
                border: 2px solid rgba(0,0,0,0.08);
                transform: translateY(-1px);
              }
              .cycle-sub{
                margin-top:2px;
                opacity:0.75;
                font-size:12px;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        parts = ['<div class="cycle-wrap">']
        for i, p in enumerate(phases):
            is_active = (p == phase)
            dot = colors.get(p, "#999999")
            icon = icons.get(p, "•")
            cls = "cycle-step cycle-active" if is_active else "cycle-step"

            parts.append(
                f"""
                <div class="{cls}">
                  <span class="dot" style="background:{dot};"></span>
                  <span>{icon} {p.title()}</span>
                </div>
                """
            )

            if i < len(phases) - 1:
                parts.append('<div class="cycle-arrow">→</div>')

        parts.append("</div>")
        parts.append(
            f"""
            <div class="cycle-sub">
              You are here: <b>{phase.title()}</b>
            </div>
            """
        )

        st.markdown("".join(parts), unsafe_allow_html=True)

    def render_mini_metric(label: str, value: str):
        st.markdown(
            f"""
            <div style="padding:6px 4px;">
                <div style="font-size:12px; color:#6B7280; margin-bottom:2px;">
                    {label}
                </div>
                <div style="font-size:20px; font-weight:600; color:#111827;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------------------------------------------
    # Score helpers
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

    def yield_score_to_pct(score) -> float:
        if pd.isnull(score):
            return 0.0
        return round((float(score) / 4.0) * 100.0, 1)

    def buyzone_to_pct(flag) -> float:
        return 100.0 if bool(flag) else 0.0

    def normalize_gfv_upside(upside_pct, cap=30.0) -> float:
        if upside_pct is None or pd.isnull(upside_pct):
            return 50.0
        x = max(-cap, min(cap, float(upside_pct)))
        return round(((x + cap) / (2.0 * cap)) * 100.0, 1)

    def fvg_status_color(direction: str) -> str:
        s = str(direction or "").strip().lower()
        if s == "bullish":
            return "#4CAF50"
        if s == "bearish":
            return "#F44336"
        return "#90A4AE"

    # ----------------------------------------------------------
    # Database Queries (cached)
    # ----------------------------------------------------------
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
        return pd.read_sql(
            """
            SELECT ticker, name, sector, industry, exchange, delisted, delisted_date, greer_star_rating
            FROM companies
            WHERE ticker = %(t)s
            LIMIT 1;
            """,
            engine,
            params={"t": ticker}
        )

    @st.cache_data(ttl=600)
    def get_level_transition_dates(ticker: str):
        engine = get_engine()
        return pd.read_sql(
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

        return pd.read_sql(
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

    @st.cache_data(ttl=600)
    def get_latest_sector_context(sector: str):
        engine = get_engine()
        return pd.read_sql(
            """
            SELECT
              summary_date,
              sector,
              total_companies,

              gv_gold, gv_green, gv_red,
              ys_gold, ys_green, ys_red,
              gfv_gold, gfv_green, gfv_red, gfv_gray,

              buyzone_count,
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

    # ----------------------------------------------------------
    # Bundle builder
    # ----------------------------------------------------------
    @st.cache_data(ttl=300)
    def build_home_bundle(ticker: str):
        snap = get_latest_snapshot(ticker)
        info = get_company_info(ticker)
        gfv_df = get_latest_gfv(ticker)
        trans = get_level_transition_dates(ticker)

        bundle = {
            "ticker": ticker,
            "found": not snap.empty,
            "snapshot": snap,
            "info": info,
            "gfv": gfv_df,
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
        sector_health_pct = None
        sector_opportunity_pct = None
        sector_gmi = None
        sector_phase = None
        sector_conf = None

        if not sector_df.empty:
            srow = sector_df.iloc[0]
            total = int(srow["total_companies"]) if pd.notnull(srow["total_companies"]) else 0

            sector_buyzone_pct = float(srow["buyzone_pct"]) if pd.notnull(srow["buyzone_pct"]) else 50.0
            sector_direction_pct = round(100.0 - sector_buyzone_pct, 1)
            sector_gmi = srow.get("greer_market_index")

            gv_bullish_pct = (
                ((int(srow["gv_green"]) + int(srow["gv_gold"])) / total) * 100.0
                if total else 0.0
            )

            ys_bullish_pct = (
                ((int(srow["ys_green"]) + int(srow["ys_gold"])) / total) * 100.0
                if total else 0.0
            )

            gfv_bullish_pct = (
                ((int(srow["gfv_green"]) + int(srow["gfv_gold"])) / total) * 100.0
                if total else 0.0
            )

            sector_health_pct = round(gv_bullish_pct, 1)
            sector_opportunity_pct = round((ys_bullish_pct + gfv_bullish_pct) / 2.0, 1)

            sector_phase, sector_conf = classify_phase_with_confidence(
                sector_health_pct,
                sector_buyzone_pct,
                sector_opportunity_pct,
            )

        gfv_status = None
        if not gfv_df.empty:
            gfv_row = gfv_df.iloc[0]
            gfv_status = gfv_row.get("gfv_status")

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
            "sector_health_pct": sector_health_pct,
            "sector_direction_pct": sector_direction_pct,
            "sector_opportunity_pct": sector_opportunity_pct,
            "sector_phase": sector_phase,
            "sector_confidence": round(sector_conf, 4) if sector_conf is not None else None,
            "sector_greer_market_index": sector_gmi,
        }

        return bundle

    # ----------------------------------------------------------
    # Card / Header renderers
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
        level = get_value_level(stars)
        level_label = value_level_label(level)

        level_html = ""
        if level > 0:
            level_html = (
                f"<div style='font-size:1.05rem; font-weight:700; margin-top:6px;'>"
                f"<b>Value Signal:</b> {level_label}"
                f"</div>"
            )

        level_status_line = ""
        if not trans.empty:
            entered = trans.loc[0, "entered_3star_date"]
            exited = trans.loc[0, "exited_3star_date"]
            days = trans.loc[0, "days_in_3star"]
            tracking_start = trans.loc[0, "tracking_start_date"]

            entered_d = pd.to_datetime(entered).date() if pd.notnull(entered) else None
            exited_d = pd.to_datetime(exited).date() if pd.notnull(exited) else None

            if entered_d is None and level == 3 and pd.notnull(tracking_start):
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
                    level_status_line = (
                        f"<div class='company-meta'><b>{value_level_short(level)} Entered:</b> {entered_d} "
                        f"&nbsp;|&nbsp; <b>Days @ {value_level_short(level)}:</b> {days_in} "
                        f"&nbsp;|&nbsp; <b>Status:</b> Exited {exited_d}</div>"
                    )
                else:
                    days_so_far = (asof - entered_d).days + 1
                    level_status_line = (
                        f"<div class='company-meta'><b>{value_level_short(level)} Entered:</b> {entered_d} "
                        f"&nbsp;|&nbsp; <b>Days @ {value_level_short(level)}:</b> {days_so_far} "
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
          {level_html}
          {level_status_line}
        </div>
        """)
        st.markdown(html, unsafe_allow_html=True)

    def render_signal_summary(bundle: dict):
        snap = bundle["snapshot"].iloc[0]
        company_cycle = bundle["company_cycle"]
        sector_cycle = bundle["sector_cycle"]

        gv_score_local = snap.get("greer_value_score")
        ys_score_local = snap.get("greer_yield_score")
        in_bz_local = bool(snap.get("buyzone_flag", False))
        fvg_dir_local = str(snap.get("fvg_last_direction") or "").lower()

        quality_text = "strong fundamentals" if pd.notnull(gv_score_local) and float(gv_score_local) >= 50 else "weaker fundamentals"
        value_text = "attractive valuation" if pd.notnull(ys_score_local) and float(ys_score_local) >= 3 else "mixed valuation"
        bz_text = "BuyZone is active" if in_bz_local else "BuyZone is inactive"

        if fvg_dir_local == "bullish":
            fvg_text = "bullish FVG support is present"
        elif fvg_dir_local == "bearish":
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
    # Detail renderers
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
        snap_df = pd.read_sql(
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
        if snap_df.empty:
            st.warning("No yield data found for this ticker.")
            return

        row = snap_df.iloc[0]
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

    qp = st.query_params
    ticker_param = qp.get("ticker")

    if isinstance(ticker_param, (list, tuple)):
        url_ticker = ticker_param[0]
    elif isinstance(ticker_param, str):
        url_ticker = ticker_param
    else:
        url_ticker = ""

    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = url_ticker.upper().strip() if url_ticker else ""

    st.text_input(
        "Search",
        placeholder="Enter ticker (e.g., AAPL)",
        key="ticker_input",
        label_visibility="collapsed",
    )

    ticker = st.session_state["ticker_input"].strip().upper()
    if not ticker:
        return

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

    gv_score = snap.get("greer_value_score")
    gv_grade, gv_bg, gv_txt = classify_greer(gv_score, snap.get("above_50_count"))

    ys_score_raw = snap.get("greer_yield_score")
    ys_score = int(ys_score_raw) if pd.notnull(ys_score_raw) else None
    y_grade, y_bg, y_txt = classify_yield(ys_score)

    in_bz = bool(snap.get("buyzone_flag", False))
    fvg_dir = snap.get("fvg_last_direction")
    fvg_date = pd.to_datetime(snap.get("fvg_last_date")).strftime('%Y-%m-%d') if snap.get("fvg_last_date") else "—"

    top_left, top_mid, top_right = st.columns([2.2, 2.4, 1.4])

    with top_left:
        render_company_card(bundle)
        price_c1, price_c2 = st.columns(2)
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

        gfv_price = gfv_row.get("gfv_price") if gfv_row is not None else None
        gfv_status = str(gfv_row.get("gfv_status") or "gray").lower() if gfv_row is not None else "gray"
        gfv_upside_pct = safe_pct_to_fair_value(current_price, gfv_price)

        with price_c1:
            price_main = "—" if pd.isnull(current_price) else f"${float(current_price):,.2f}"
            price_sub = f"as of {snapshot_date}" if snapshot_date else "—"
            render_metric_card("Price", price_main, price_sub, "#111111", "white")

        with price_c2:
            if gfv_row is not None:
                gfv_main = "—" if pd.isnull(gfv_price) else f"${float(gfv_price):,.2f}"
                if gfv_upside_pct is None:
                    gfv_sub = f"{gfv_status.title()}"
                else:
                    sign = "+" if gfv_upside_pct >= 0 else ""
                    gfv_sub = f"{gfv_status.title()} • {sign}{gfv_upside_pct:.1f}% vs price"

                gfv_bg = {
                    "gold": "linear-gradient(90deg, #D4AF37, #FFD700)",
                    "green": "#4CAF50",
                    "red": "#F44336",
                    "gray": "#9CA3AF",
                }.get(gfv_status, "#9CA3AF")

                gfv_fg = "black" if gfv_status == "gold" else "white"
                render_metric_card("Greer Fair Value", gfv_main, gfv_sub, gfv_bg, gfv_fg)
            else:
                render_metric_card("Greer Fair Value", "—", "No data", "#9CA3AF", "white")
    with top_mid:
        gv_bucket = gv_bucket_label(gv_score, snap.get("above_50_count"))
        gv_color = {
            "Gold": "linear-gradient(90deg, #D4AF37, #FFD700)",
            "Green": "#5CB85C",
            "Red": "#D9534F",
        }.get(gv_bucket, "#9CA3AF")

        render_top_score_bar(
            "Greer Value",
            float(gv_score) if pd.notnull(gv_score) else 0.0,
            f"{gv_bucket} • {gv_grade}" if pd.notnull(gv_score) else "No data",
            fill_color=gv_color,
        )

        ys_bucket = ys_bucket_label(ys_score)
        ys_color = {
            "Gold": "linear-gradient(90deg, #D4AF37, #FFD700)",
            "Green": "#5CB85C",
            "Red": "#D9534F",
        }.get(ys_bucket, "#9CA3AF")

        render_top_score_bar(
            "Yield Score",
            yield_score_to_pct(ys_score),
            f"{ys_score}/4 • {ys_bucket} • {y_grade}" if ys_score is not None else "No data",
            fill_color=ys_color,
        )

        render_top_score_bar(
            "Greer Company Index",
            company_cycle.get("greer_company_index", 0.0),
            f"{phase_label_with_icon(company_cycle.get('phase'))} • {round(float(company_cycle.get('confidence', 0)) * 100)}% confidence" if company_cycle else "No data",
        )

        sector_gmi = sector_cycle.get("sector_greer_market_index") if sector_cycle else None
        sector_conf = sector_cycle.get("sector_confidence") if sector_cycle else None

        sector_subtitle = "No data"
        if sector_cycle and sector_gmi is not None and pd.notnull(sector_gmi):
            conf_pct = round(float(sector_conf or 0) * 100)

            sector_subtitle = (
                f"{phase_label_with_icon(sector_cycle.get('sector_phase'))} "
                f"• {conf_pct}% confidence"
            )

        render_top_score_bar(
            "Greer Sector Index",
            float(sector_gmi) if sector_gmi is not None and pd.notnull(sector_gmi) else 0.0,
            sector_subtitle,
        )
    with top_right:
        st.markdown("**Status**")

        company_phase_text = phase_label_with_icon(company_cycle.get("phase")) if company_cycle else "—"
        sector_phase_text = phase_label_with_icon(sector_cycle.get("sector_phase")) if sector_cycle else "—"

        transition_text = company_cycle.get("transition_risk", "") if company_cycle else ""
        transition_map = {
            "✅ Stable Trend": "Stable",
            "👀 Watch Transition": "Watch",
            "⚠ High Shift Risk": "Shift Risk",
        }
        transition_short = transition_map.get(transition_text, "")

        company_phase_text = phase_label_with_icon(company_cycle.get("phase")) if company_cycle else "—"

        render_status_pill(
            "Company Phase",
            company_phase_text,
            "#5BC0DE" if company_cycle and str(company_cycle.get("phase")).upper() == "RECOVERY" else
            "#5CB85C" if company_cycle and str(company_cycle.get("phase")).upper() == "EXPANSION" else
            "#F0AD4E" if company_cycle and str(company_cycle.get("phase")).upper() == "EUPHORIA" else
            "#D9534F" if company_cycle and str(company_cycle.get("phase")).upper() == "CONTRACTION" else
            "#90A4AE",
            "white",
        )

        render_status_pill(
            "Sector Phase",
            sector_phase_text,
            "#5BC0DE" if sector_cycle and str(sector_cycle.get("sector_phase")).upper() == "RECOVERY" else
            "#5CB85C" if sector_cycle and str(sector_cycle.get("sector_phase")).upper() == "EXPANSION" else
            "#F0AD4E" if sector_cycle and str(sector_cycle.get("sector_phase")).upper() == "EUPHORIA" else
            "#D9534F" if sector_cycle and str(sector_cycle.get("sector_phase")).upper() == "CONTRACTION" else
            "#90A4AE",
            "white",
        )

        render_status_pill(
            "BuyZone",
            "Triggered" if in_bz else "No Signal",
            "#4CAF50" if in_bz else "#9E9E9E",
            "white",
        )

        render_status_pill(
            "FVG",
            (fvg_dir or "No Gap").capitalize(),
            fvg_status_color(fvg_dir),
            "white",
        )


    render_signal_summary(bundle)

    # ----------------------------------------------------------
    # Detail Tabs
    # ----------------------------------------------------------
    engine = get_engine()

    tab_overview, tab_prediction, tab_valuation, tab_technicals, tab_cycle, tab_history = st.tabs([
        "Overview",
        "Prediction 🔮",
        "Valuation",
        "Technicals",
        "Cycle",
        "History 📈",
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
                "Phase Stability",
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

    with tab_prediction:
        render_company_prediction_section(ticker)
    
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

        strip_c1, strip_c2 = st.columns(2)

        with strip_c1:
            st.markdown("### Company Phase")
            if company_cycle:
                render_cycle_strip(company_cycle.get("phase"))

        with strip_c2:
            st.markdown("### Sector Phase")
            if sector_cycle and sector_cycle.get("sector_phase"):
                render_cycle_strip(sector_cycle.get("sector_phase"))

        st.divider()

        g1, g2 = st.columns(2)

        with g1:
            if company_cycle:
                render_semicircle_gauge(
                    "🏢 Greer Company Index",
                    company_cycle.get("greer_company_index", 0.0),
                    "Overall company cycle score based on Health, Direction, and Opportunity.",
                    chart_key=f"{ticker}_company_index_gauge",
                )

        with g2:
            if sector_cycle and sector_cycle.get("sector_greer_market_index") is not None:
                render_semicircle_gauge(
                    "🏭 Greer Sector Index",
                    sector_cycle.get("sector_greer_market_index", 0.0),
                    "Overall sector backdrop score for this company’s sector.",
                    chart_key=f"{ticker}_sector_index_gauge",
                )

        st.divider()
        st.markdown("### Cycle Components")

        comp_col, sect_col = st.columns(2)

        with comp_col:
            st.markdown("#### Company")
            if company_cycle:
                render_horizontal_score_bar(
                    "🟢 Health",
                    company_cycle.get("health_pct", 0.0),
                    "Fundamental quality",
                )

                render_horizontal_score_bar(
                    "📉 Direction",
                    company_cycle.get("direction_pct", 0.0),
                    "Technical + sector backdrop",
                )

                render_horizontal_score_bar(
                    "💰 Opportunity",
                    company_cycle.get("opportunity_pct", 0.0),
                    "Valuation opportunity",
                )

        with sect_col:
            st.markdown("#### Sector")
            if sector_cycle:
                render_horizontal_score_bar(
                    "🟢 Health",
                    sector_cycle.get("sector_health_pct", 0.0) or 0.0,
                    "Sector fundamental breadth",
                )

                render_horizontal_score_bar(
                    "📉 Direction",
                    sector_cycle.get("sector_direction_pct", 0.0) or 0.0,
                    "100 - sector BuyZone %",
                )

                render_horizontal_score_bar(
                    "💰 Opportunity",
                    sector_cycle.get("sector_opportunity_pct", 0.0) or 0.0,
                    "Sector valuation breadth",
                )


        st.divider()

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            render_mini_metric(
                "Company Phase",
                phase_label_with_icon(company_cycle.get("phase")) if company_cycle else "—"
            )

        with s2:
            conf = round(float(company_cycle.get("confidence", 0)) * 100) if company_cycle else None
            render_mini_metric(
                "Company Confidence",
                f"{conf}%" if conf is not None else "—"
            )

        with s3:
            render_mini_metric(
                "Transition Risk",
                company_cycle.get("transition_risk", "—") if company_cycle else "—"
            )

        with s4:
            render_mini_metric(
                "Sector Phase",
                phase_label_with_icon(sector_cycle.get("sector_phase")) if sector_cycle else "—"
            )

    with tab_history:
        st.subheader("📈 Greer Company Index History")

        df_hist = load_company_index_history(ticker)

        if df_hist.empty:
            st.warning("No historical data available.")
        else:
            df_hist = add_history_coverage_flags(df_hist).copy()
            df_hist = df_hist.sort_values("date").reset_index(drop=True)

            range_col1, range_col2 = st.columns([1.2, 2.8])

            with range_col1:
                history_range = st.radio(
                    "Range",
                    ["3Y", "5Y", "All"],
                    index=1,  # 👈 this makes 5Y the default
                    horizontal=True,
                    key=f"{ticker}_history_range",
                )

            with range_col2:
                include_partial = st.checkbox(
                    "Include partial-history periods",
                    value=False,
                    key=f"{ticker}_include_partial_history",
                    help="When off, history starts only once all major components are available.",
                )

            latest_date = df_hist["date"].max()

            if history_range == "3Y":
                cutoff = latest_date - pd.DateOffset(years=3)
                df_plot = df_hist[df_hist["date"] >= cutoff].copy()
            elif history_range == "5Y":
                cutoff = latest_date - pd.DateOffset(years=5)
                df_plot = df_hist[df_hist["date"] >= cutoff].copy()
            else:
                df_plot = df_hist.copy()

            if include_partial:
                df_plot = df_plot[df_plot["is_partial_coverage"]].copy()
            else:
                df_plot = df_plot[df_plot["is_full_coverage"]].copy()

            price_hist = load_price_history(ticker)

            if not price_hist.empty:
                df_plot = df_plot.merge(price_hist, on="date", how="left")

            if df_plot.empty:
                st.info("No historical rows match the selected coverage/range settings.")
            else:
                df_plot["company_index_50dma"] = (
                    df_plot["greer_company_index"]
                    .rolling(50, min_periods=10)
                    .mean()
                )

                phase_bands = build_phase_bands(df_plot, min_run_days=14)

                fig = go.Figure()

                # ------------------------------------------
                # Background regime bands
                # ------------------------------------------
                for band in phase_bands:
                    fig.add_vrect(
                        x0=band["x0"],
                        x1=band["x1"],
                        fillcolor=band["color"],
                        line_width=0,
                        layer="below",
                    )

                # ------------------------------------------
                # Price line on secondary axis
                # ------------------------------------------
                if "close" in df_plot.columns and df_plot["close"].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=df_plot["date"],
                            y=df_plot["close"],
                            mode="lines",
                            name="Price",
                            line=dict(width=2, color="black"),
                            opacity=0.5,
                            yaxis="y2",
                            hovertemplate=(
                                "<b>%{x|%Y-%m-%d}</b><br>"
                                "Price: $%{y:,.2f}<extra></extra>"
                            ),
                        )
                    )

                # ------------------------------------------
                # Raw company index line
                # ------------------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["date"],
                        y=df_plot["greer_company_index"],
                        mode="lines",
                        name="Company Index",
                        line=dict(width=1.5, color="rgba(72,99,255,0.55)"),
                        hovertemplate=(
                            "<b>%{x|%Y-%m-%d}</b><br>"
                            "Company Index: %{y:.2f}<br>"
                            "Phase: %{customdata[0]}<extra></extra>"
                        ),
                        customdata=df_plot[["phase"]].values,
                    )
                )

                # ------------------------------------------
                # 50-day average as primary signal
                # ------------------------------------------
                fig.add_trace(
                    go.Scatter(
                        x=df_plot["date"],
                        y=df_plot["company_index_50dma"],
                        mode="lines",
                        name="50-Day Avg",
                        line=dict(width=2.5, dash="dash", color="#FF5A36"),
                        hovertemplate=(
                            "<b>%{x|%Y-%m-%d}</b><br>"
                            "50-Day Avg: %{y:.2f}<extra></extra>"
                        ),
                    )
                )

                # ------------------------------------------
                # High / Low markers
                # ------------------------------------------
                high_row = df_plot.loc[df_plot["greer_company_index"].idxmax()]
                low_row = df_plot.loc[df_plot["greer_company_index"].idxmin()]

                fig.add_trace(
                    go.Scatter(
                        x=[high_row["date"]],
                        y=[high_row["greer_company_index"]],
                        mode="markers+text",
                        name="High",
                        text=["High"],
                        textposition="top center",
                        marker=dict(size=9, symbol="diamond"),
                        showlegend=False,
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=[low_row["date"]],
                        y=[low_row["greer_company_index"]],
                        mode="markers+text",
                        name="Low",
                        text=["Low"],
                        textposition="bottom center",
                        marker=dict(size=9, symbol="diamond"),
                        showlegend=False,
                    )
                )
                # ------------------------------------------
                # Today marker (current position)
                # ------------------------------------------
                latest_row = df_plot.iloc[-1]

                # Keep the dot
                fig.add_trace(
                    go.Scatter(
                        x=[latest_row["date"]],
                        y=[latest_row["greer_company_index"]],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="black",
                            symbol="circle"
                        ),
                        showlegend=False,
                    )
                )

                # Add annotation to the RIGHT
                fig.add_annotation(
                    x=latest_row["date"],
                    y=latest_row["greer_company_index"],
                    text=f"{latest_row['phase']} • {latest_row['greer_company_index']:.0f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=50,
                    ay=0,
                    font=dict(
                        size=12,
                        color="#333"
                    )
                )

                fig.add_vline(
                    x=latest_row["date"],
                    line_dash="dot",
                    line_color="black",
                    opacity=0.4
                )

                fig.update_layout(
                    height=500,
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis=dict(title="Date"),
                    yaxis=dict(
                        title="Greer Company Index",
                        range=[0, max(100, float(df_plot["greer_company_index"].max()) + 5)],
                    ),
                    yaxis2=dict(
                        title="Price",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🔍 Key Levels")
                st.markdown(
                    f"""
                    - **High:** {high_row['greer_company_index']:.2f} on {high_row['date'].date()} ({high_row['phase']})
                    - **Low:** {low_row['greer_company_index']:.2f} on {low_row['date'].date()} ({low_row['phase']})
                    - **Rows Shown:** {len(df_plot):,}
                    """
                )

                st.markdown("### 🧠 Component Breakdown")

                fig2 = go.Figure()

                fig2.add_trace(
                    go.Scatter(
                        x=df_plot["date"],
                        y=df_plot["health_pct"],
                        name="Health",
                        mode="lines",
                        line=dict(width=2),
                    )
                )

                fig2.add_trace(
                    go.Scatter(
                        x=df_plot["date"],
                        y=df_plot["direction_pct"],
                        name="Direction",
                        mode="lines",
                        line=dict(width=2),
                    )
                )

                fig2.add_trace(
                    go.Scatter(
                        x=df_plot["date"],
                        y=df_plot["opportunity_pct"],
                        name="Opportunity",
                        mode="lines",
                        line=dict(width=2),
                    )
                )

                fig2.update_layout(
                    height=360,
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="Component %",
                    xaxis_title="Date",
                    yaxis=dict(range=[0, 100]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )

                st.plotly_chart(fig2, use_container_width=True)


# ----------------------------------------------------------
# App entrypoint
# ----------------------------------------------------------
def main():
    from app_nav import build_pages
    all_pages, _PAGE_MAP = build_pages()
    st.navigation(all_pages, position="sidebar").run()


if __name__ == "__main__":
    main()