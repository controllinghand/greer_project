# 12_Fund_Compare_YTD.py

import streamlit as st
import pandas as pd
import altair as alt
from sqlalchemy import text
from datetime import date

from db import get_engine

st.set_page_config(page_title="Fund Comparison (YTD)", layout="wide")

# ----------------------------------------------------------
# UI Style Overrides — Multiselect chips (force non-red)
# ----------------------------------------------------------
st.markdown(
    """
    <style>
    /* Base chip */
    span[data-baseweb="tag"] {
        background-color: #E3F4F4 !important;
        color: #0F4C5C !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* Hover state */
    span[data-baseweb="tag"]:hover {
        background-color: #D6E8FA !important;
        color: #1F4E79 !important;
    }

    /* Active / focused state */
    span[data-baseweb="tag"]:focus,
    span[data-baseweb="tag"]:active {
        background-color: #D6E8FA !important;
        color: #1F4E79 !important;
    }

    /* Remove (X) icon */
    span[data-baseweb="tag"] svg {
        fill: #1F4E79 !important;
    }

    /* Defensive: override any red accent fallback */
    div[role="listbox"] span {
        background-color: #E6F0FA !important;
        color: #1F4E79 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# Formatters
# ----------------------------------------------------------
def fmt_money(x):
    try:
        if x is None or pd.isna(x):
            return ""
        return f"${float(x):,.2f}"
    except Exception:
        return ""

def fmt_pct(x):
    try:
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "—"

# ----------------------------------------------------------
# Load portfolios
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
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
# (Uses nav_date and nav from portfolio_nav_daily)
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
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

# ----------------------------------------------------------
# Load NAV series for chart
# ----------------------------------------------------------
@st.cache_data(ttl=3600)
def load_nav_series(portfolio_id: int, d0: date, d1: date) -> pd.DataFrame:
    sql = """
        SELECT nav_date::date AS nav_date, nav
        FROM portfolio_nav_daily
        WHERE portfolio_id = :pid
          AND nav_date >= :d0
          AND nav_date <= :d1
          AND nav IS NOT NULL
        ORDER BY nav_date
    """
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params={"pid": portfolio_id, "d0": d0, "d1": d1})

# ----------------------------------------------------------
# Chart helpers
# ----------------------------------------------------------
def fund_group(code: str, bench_code: str) -> str:
    c = (code or "").upper()
    if c.startswith(bench_code.upper().split("-")[0]):  # e.g., YRQ from YRQ-26
        return "Benchmark"
    if c.endswith("I"):  # YRI, YRSI
        return "Income"
    if c.endswith("G"):  # YROG, YR3G
        return "Growth"
    return "Other"

def build_nav_chart_altair(
    pmeta: pd.DataFrame,
    year: int,
    asof: date,
    selected_codes: list[str],
    bench_code: str,
    show_benchmark: bool = True,
    chart_mode: str = "Indexed",  # "Indexed" or "Raw"
):
    y0 = date(year, 1, 1)
    code_to_pid = dict(zip(pmeta["code"].astype(str), pmeta["portfolio_id"].astype(int)))

    frames = []
    for code in selected_codes:
        pid = code_to_pid.get(code)
        if not pid:
            continue

        navdf = load_nav_series(pid, y0, asof)
        if navdf.empty:
            continue

        navdf = navdf.dropna(subset=["nav"]).copy()
        if navdf.empty:
            continue

        navdf["Code"] = code
        navdf["Group"] = fund_group(code, bench_code)

        base = float(navdf.iloc[0]["nav"])
        navdf["NAV_Raw"] = navdf["nav"].astype(float)

        if chart_mode == "Indexed":
            if base == 0:
                continue
            navdf["NAV_Value"] = (navdf["NAV_Raw"] / base) * 100.0
        else:
            navdf["NAV_Value"] = navdf["NAV_Raw"]

        frames.append(navdf[["nav_date", "Code", "Group", "NAV_Value"]])

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # Optionally hide benchmark
    if not show_benchmark:
        df = df[df["Group"] != "Benchmark"].copy()
        if df.empty:
            return None

    y_title = "NAV (Indexed, start = 100)" if chart_mode == "Indexed" else "NAV ($)"

    # Color by GROUP to match your chip meaning (Income blue, Growth green, Benchmark gray)
    color_scale = alt.Scale(
        domain=["Income", "Growth", "Benchmark", "Other"],
        range=["#1F4E79", "#1E6B3A", "#2B2B2B", "#6B7280"],
    )

    # Base line chart (one line per Code, colored by Group)
    lines = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("nav_date:T", title="Date"),
            y=alt.Y("NAV_Value:Q", title=y_title),
            color=alt.Color("Group:N", scale=color_scale, legend=alt.Legend(title="Type")),
            detail="Code:N",
            tooltip=[
                alt.Tooltip("Code:N", title="Fund"),
                alt.Tooltip("Group:N", title="Type"),
                alt.Tooltip("nav_date:T", title="Date"),
                alt.Tooltip("NAV_Value:Q", title=y_title, format=".2f"),
            ],
        )
    )

    # Make benchmark thicker/darker (overlay layer)
    bench_layer = alt.Chart(df[df["Group"] == "Benchmark"]).mark_line(strokeWidth=4).encode(
        x="nav_date:T",
        y="NAV_Value:Q",
        color=alt.value("#2B2B2B"),
        detail="Code:N",
        tooltip=[
            alt.Tooltip("Code:N", title="Benchmark"),
            alt.Tooltip("nav_date:T", title="Date"),
            alt.Tooltip("NAV_Value:Q", title=y_title, format=".2f"),
        ],
    )

    # ---- End-of-line labels (YTD %) ----
    # Compute last point per Code
    last_points = (
        df.sort_values("nav_date")
          .groupby(["Code", "Group"], as_index=False)
          .tail(1)
          .copy()
    )

    # Compute YTD% from first point per Code
    first_points = (
        df.sort_values("nav_date")
          .groupby(["Code"], as_index=False)
          .head(1)[["Code", "NAV_Value"]]
          .rename(columns={"NAV_Value": "first_val"})
    )

    last_points = last_points.merge(first_points, on="Code", how="left")
    # For Indexed mode: first_val is ~100; for Raw: it's starting NAV
    last_points["ytd_pct"] = (last_points["NAV_Value"] / last_points["first_val"] - 1.0) * 100.0
    last_points["label"] = last_points.apply(
        lambda r: f"{r['Code']}  {r['ytd_pct']:+.2f}%",
        axis=1
    )

    labels = (
        alt.Chart(last_points)
        .mark_text(align="left", dx=6, fontSize=12)
        .encode(
            x="nav_date:T",
            y="NAV_Value:Q",
            text="label:N",
            color=alt.Color("Group:N", scale=color_scale, legend=None),
        )
    )

    # Final chart
    if show_benchmark and not df[df["Group"] == "Benchmark"].empty:
        chart = (lines + bench_layer + labels).properties(height=380).interactive()
    else:
        chart = (lines + labels).properties(height=380).interactive()

    return chart

# ----------------------------------------------------------
# Build YTD NAV chart (Altair)
# - Indexed to 100
# - Benchmark line thicker/darker
# ----------------------------------------------------------
def build_nav_index_chart_altair(
    pmeta: pd.DataFrame,
    year: int,
    asof: date,
    selected_codes: list[str],
    bench_code: str,
):
    y0 = date(year, 1, 1)

    code_to_pid = dict(zip(pmeta["code"].astype(str), pmeta["portfolio_id"].astype(int)))

    frames = []
    for code in selected_codes:
        pid = code_to_pid.get(code)
        if not pid:
            continue

        navdf = load_nav_series(pid, y0, asof)
        if navdf.empty:
            continue

        navdf = navdf.dropna(subset=["nav"]).copy()
        if navdf.empty:
            continue

        base = float(navdf.iloc[0]["nav"])
        if base == 0:
            continue

        navdf["Code"] = code
        navdf["NAV_Index"] = (navdf["nav"].astype(float) / base) * 100.0
        frames.append(navdf[["nav_date", "Code", "NAV_Index"]])

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # Split benchmark vs others
    df_funds = df[df["Code"] != bench_code]
    df_bench = df[df["Code"] == bench_code]

    base = alt.Chart(df_funds).encode(
        x=alt.X("nav_date:T", title="Date"),
        y=alt.Y("NAV_Index:Q", title="NAV (Indexed, start = 100)"),
        tooltip=[
            alt.Tooltip("Code:N", title="Fund"),
            alt.Tooltip("nav_date:T", title="Date"),
            alt.Tooltip("NAV_Index:Q", format=".2f"),
        ],
    )

    funds_layer = base.mark_line().encode(
        color=alt.Color(
            "Code:N",
            legend=alt.Legend(title="Funds"),
            scale=alt.Scale(scheme="blues"),
        )
    )

    bench_layer = (
        alt.Chart(df_bench)
        .mark_line(strokeWidth=4)
        .encode(
            x="nav_date:T",
            y="NAV_Index:Q",
            color=alt.value("#2B2B2B"),  # dark gray
            tooltip=[
                alt.Tooltip("Code:N", title="Benchmark"),
                alt.Tooltip("nav_date:T", title="Date"),
                alt.Tooltip("NAV_Index:Q", format=".2f"),
            ],
        )
    )

    chart = (
        (funds_layer + bench_layer)
        .properties(height=360)
        .interactive()
    )

    return chart


# ----------------------------------------------------------
# Compute table for selected funds
# ----------------------------------------------------------
def compute_comparison(pmeta: pd.DataFrame, year: int, asof: date, selected_codes: list[str]) -> pd.DataFrame:
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
# Styling vs benchmark
# ----------------------------------------------------------
def style_table(df: pd.DataFrame, bench_code: str):
    if df.empty:
        return df.style

    bench = df[df["Code"] == bench_code]
    bench_ret = None
    if not bench.empty and pd.notna(bench.iloc[0].get("YTD Return", None)):
        bench_ret = float(bench.iloc[0]["YTD Return"])

    def row_style(row):
        code = row.get("Code", "")
        alpha = row.get("Alpha vs Benchmark", None)

        styles = [""] * len(row.index)

        if code == bench_code:
            return styles

        if bench_ret is None or alpha is None or pd.isna(alpha):
            return styles

        green = "background-color: rgba(0, 200, 0, 0.18);"
        red = "background-color: rgba(255, 0, 0, 0.18);"

        alpha_style = green if alpha > 0 else red if alpha < 0 else ""

        for col in ["YTD Return", "Alpha vs Benchmark"]:
            if col in row.index:
                idx = list(row.index).index(col)
                styles[idx] = alpha_style

        return styles

    sty = df.style.apply(row_style, axis=1)

    sty = sty.format(
        {
            "Start NAV": fmt_money,
            "End NAV": fmt_money,
            "P&L": fmt_money,
            "YTD Return": fmt_pct,
            "Alpha vs Benchmark": fmt_pct,
        }
    )

    return sty

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    st.title("📊 Fund Comparison – YTD Returns")

    pmeta = load_portfolios()
    if pmeta.empty:
        st.error("No portfolios found in the database.")
        return

    available = pmeta["code"].astype(str).tolist()

    default_codes = ["YRI-26", "YRSI", "YROG", "YR3G", "YRQ-26", "YRQI-26"]
    selected_default = [c for c in default_codes if c in available] or available[:5]

    years = list(range(date.today().year, date.today().year - 6, -1))

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        year = st.selectbox("Year", years, index=0)
    with c2:
        asof = st.date_input("As-of date", value=date.today())
    with c3:
        bench_code = st.selectbox(
            "Benchmark",
            options=available,
            index=available.index("YRQ-26") if "YRQ-26" in available else 0,
        )

    selected_codes = st.multiselect("Funds to compare", options=available, default=selected_default)

    if not selected_codes:
        st.warning("Select at least one fund.")
        return

    # Ensure benchmark is included for table alpha AND chart overlay
    if bench_code not in selected_codes:
        st.info(f"Benchmark **{bench_code}** was added to the comparison so Alpha can be calculated.")
        selected_codes = list(dict.fromkeys(selected_codes + [bench_code]))

    # ----------------------------------------------------------
    # Table
    # ----------------------------------------------------------
    df = compute_comparison(pmeta, year=year, asof=asof, selected_codes=selected_codes)
    if df.empty:
        st.info("No NAV data found for the selected funds in this window.")
        return

    # alpha
    bench = df[df["Code"] == bench_code]
    bench_ret = None
    if not bench.empty and pd.notna(bench.iloc[0].get("YTD Return", None)):
        bench_ret = float(bench.iloc[0]["YTD Return"])

    df["Alpha vs Benchmark"] = (df["YTD Return"] - bench_ret) if bench_ret is not None else None

    cols = [
        "Code",
        "Name",
        "Start NAV",
        "End NAV",
        "P&L",
        "YTD Return",
        "Alpha vs Benchmark",
        "Start Date Used",
        "End Date Used",
    ]
    df = df[cols]

    st.caption("Green = outperforming benchmark. Red = underperforming benchmark.")
    st.dataframe(style_table(df, bench_code=bench_code), use_container_width=True)

    # quick callouts
    if bench_ret is not None:
        c1, c2 = st.columns(2)

        outperform = df[df["Alpha vs Benchmark"] > 0].sort_values("Alpha vs Benchmark", ascending=False)
        underperform = df[df["Alpha vs Benchmark"] < 0].sort_values("Alpha vs Benchmark", ascending=True)

        with c1:
            st.subheader("✅ Outperforming")
            if outperform.empty:
                st.write("None")
            else:
                for _, r in outperform.iterrows():
                    st.write(f"**{r['Code']}** — {fmt_pct(r['YTD Return'])} (α {fmt_pct(r['Alpha vs Benchmark'])})")

        with c2:
            st.subheader("⚠️ Underperforming")
            if underperform.empty:
                st.write("None")
            else:
                for _, r in underperform.iterrows():
                    st.write(f"**{r['Code']}** — {fmt_pct(r['YTD Return'])} (α {fmt_pct(r['Alpha vs Benchmark'])})")

    # ----------------------------------------------------------
    # NAV Chart Controls
    # ----------------------------------------------------------
    st.divider()
    st.subheader("📈 NAV Path (YTD)")

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        chart_mode = st.radio("Chart Mode", ["Indexed", "Raw"], index=0, horizontal=True)
    with cc2:
        show_benchmark = st.checkbox("Show Benchmark Overlay", value=True)

    st.caption(
        "Indexed mode starts every fund at 100 on its first trading day of the year. "
        "Income funds are blue, Growth funds are green, and the Benchmark is thick/dark gray."
    )

    chart = build_nav_chart_altair(
        pmeta=pmeta,
        year=year,
        asof=asof,
        selected_codes=selected_codes,
        bench_code=bench_code,
        show_benchmark=show_benchmark,
        chart_mode=chart_mode,
    )

    if chart is None:
        st.info("No NAV series found for the selected funds in this window.")
    else:
        st.altair_chart(chart, use_container_width=True)



if __name__ == "__main__":
    main()

