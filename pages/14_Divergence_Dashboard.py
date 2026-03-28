# pages/14_Divergence_Dashboard.py

import streamlit as st
import pandas as pd
from sqlalchemy import text

from db import get_engine


# ----------------------------------------------------------
# Style helpers
# ----------------------------------------------------------
def price_vs_peak_color(val) -> str:
    if pd.isna(val):
        return ""
    if val <= -50:
        return "background-color: rgba(33, 150, 243, 0.20);"
    if val <= -35:
        return "background-color: rgba(76, 175, 80, 0.18);"
    if val <= -20:
        return "background-color: rgba(255, 193, 7, 0.16);"
    return ""


# ----------------------------------------------------------
# Style helpers
# ----------------------------------------------------------
def trend_color(val) -> str:
    if val == "Rising":
        return "background-color: rgba(76, 175, 80, 0.18);"
    if val == "Flat":
        return "background-color: rgba(255, 193, 7, 0.16);"
    if val == "Falling":
        return "background-color: rgba(244, 67, 54, 0.18);"
    return ""


# ----------------------------------------------------------
# Phase color
# ----------------------------------------------------------
def phase_color(val) -> str:
    if val == "EXPANSION":
        return "background-color: rgba(76, 175, 80, 0.18);"
    if val == "RECOVERY":
        return "background-color: rgba(33, 150, 243, 0.18);"
    if val == "EUPHORIA":
        return "background-color: rgba(255, 193, 7, 0.18);"
    if val == "CONTRACTION":
        return "background-color: rgba(244, 67, 54, 0.18);"
    return ""


# ----------------------------------------------------------
# Data loader
# ----------------------------------------------------------
@st.cache_data(ttl=300)
def load_divergence_dashboard() -> pd.DataFrame:
    engine = get_engine()

    query = text("""
        SELECT *
        FROM divergence_snapshot
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM divergence_snapshot
        )
        ORDER BY
            CASE divergence_bucket
                WHEN 'High Conviction Pullback' THEN 1
                WHEN 'Deep Value Divergence' THEN 2
                WHEN 'Watchlist Divergence' THEN 3
                ELSE 99
            END,
            greer_company_index DESC,
            price_vs_peak_pct ASC
    """)

    return pd.read_sql(query, engine)


# ----------------------------------------------------------
# Table renderer
# ----------------------------------------------------------
def render_bucket_table(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"### {title}")

    if df.empty:
        st.caption("No matches.")
        return

    display_cols = [
        "ticker",
        "name",
        "sector",
        "current_price",
        "peak_52w_price",
        "price_vs_peak_pct",
        "greer_company_index",
        "company_index_50dma",
        "index_vs_50dma",
        "index_trend",
        "phase",
        "confidence_pct",
        "health_pct",
        "direction_pct",
        "opportunity_pct",
        "greer_value_score",
        "greer_yield_score",
        "buyzone_flag",
        "fvg_last_direction",
    ]

    styled = (
        df[display_cols]
        .style
        .map(price_vs_peak_color, subset=["price_vs_peak_pct"])
        .map(trend_color, subset=["index_trend"])
        .map(phase_color, subset=["phase"])
        .format(
            {
                "current_price": "${:,.2f}",
                "peak_52w_price": "${:,.2f}",
                "price_vs_peak_pct": "{:.2f}%",
                "greer_company_index": "{:.2f}",
                "company_index_50dma": "{:.2f}",
                "index_vs_50dma": "{:.2f}",
                "confidence_pct": "{:.0f}%",
                "health_pct": "{:.2f}",
                "direction_pct": "{:.2f}",
                "opportunity_pct": "{:.2f}",
                "greer_value_score": "{:.2f}",
            }
        )
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# Page
# ----------------------------------------------------------
def render_page() -> None:
    st.title("Divergence Dashboard")
    st.caption("High-conviction companies whose price is still below recent highs.")

    df = load_divergence_dashboard()

    if df.empty:
        st.warning("No divergence candidates found.")
        return

    st.markdown(
        """
        <style>
        .bucket-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            background: #fafafa;
            padding: 12px 16px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        min_index = st.slider("Min Company Index", 0, 100, 70, 1)

    with c2:
        max_price_vs_peak = st.slider("Max Price vs Peak %", -90, 0, -15, 1)

    with c3:
        min_confidence = st.slider("Min Confidence %", 0, 100, 0, 5)

    with c4:
        sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Sector", sectors, index=0)

    filtered = df[
        (df["greer_company_index"] >= min_index)
        & (df["price_vs_peak_pct"] <= max_price_vs_peak)
        & (df["confidence_pct"] >= min_confidence)
    ].copy()

    if selected_sector != "All":
        filtered = filtered[filtered["sector"] == selected_sector].copy()

    if filtered.empty:
        st.info("No companies match the current filters.")
        return

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Candidates", len(filtered))
    with m2:
        st.metric("Avg Company Index", f"{filtered['greer_company_index'].mean():.1f}")
    with m3:
        st.metric("Avg Price vs Peak", f"{filtered['price_vs_peak_pct'].mean():.1f}%")

    bucket_order = [
        "High Conviction Pullback",
        "Deep Value Divergence",
        "Watchlist Divergence",
    ]

    for bucket in bucket_order:
        bucket_df = filtered[filtered["divergence_bucket"] == bucket].copy()

        if bucket == "High Conviction Pullback":
            st.markdown(
                """
                <div class="bucket-card">
                    <b>🟢 High Conviction Pullback</b><br>
                    Best-looking setups: strong Company Index, discounted price, and index trend not falling apart.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif bucket == "Deep Value Divergence":
            st.markdown(
                """
                <div class="bucket-card">
                    <b>🔵 Deep Value Divergence</b><br>
                    Bigger drawdowns with still-elevated conviction. Higher upside, higher risk.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif bucket == "Watchlist Divergence":
            st.markdown(
                """
                <div class="bucket-card">
                    <b>🟡 Watchlist Divergence</b><br>
                    Interesting misalignment between conviction and price, but not as clean as Tier 1.
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_bucket_table(bucket_df, f"{bucket} ({len(bucket_df)})")

    st.markdown("### Quick Links")
    link_df = filtered[
        ["ticker", "name", "divergence_bucket", "greer_company_index", "price_vs_peak_pct", "phase"]
    ].copy()

    for _, row in link_df.iterrows():
        st.page_link(
            "pages/0_Home.py",
            label=(
                f"{row['ticker']} — {row['name']} | "
                f"{row['divergence_bucket']} | "
                f"Index {row['greer_company_index']:.1f} | "
                f"{row['price_vs_peak_pct']:.1f}% vs peak | "
                f"{row['phase']}"
            ),
            use_container_width=True,
        )


render_page()