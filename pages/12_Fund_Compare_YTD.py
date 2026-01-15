# 12_Fund_Compare_YTD.py

import streamlit as st
import pandas as pd
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
# Build YTD NAV chart (indexed to 100) for selected funds
# ----------------------------------------------------------
def build_nav_index_chart(pmeta: pd.DataFrame, year: int, asof: date, selected_codes: list[str]) -> pd.DataFrame:
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
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)

    # Pivot for st.line_chart
    chart_df = (
        all_df.pivot(index="nav_date", columns="Code", values="NAV_Index")
        .sort_index()
    )

    return chart_df

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

    default_codes = ["YRI-26", "YRSI", "YROG", "YR3G", "YRQ-26"]
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
    # NAV Line Chart (Indexed)
    # ----------------------------------------------------------
    st.subheader("📈 NAV Path (YTD) — Indexed to 100")
    st.caption("All funds start at 100 on the first available trading day. This makes performance comparable across funds.")

    chart_df = build_nav_index_chart(pmeta, year=year, asof=asof, selected_codes=selected_codes) 
    if chart_df.empty:
        st.info("No NAV series found for the selected funds in this window.")
    else:
        st.line_chart(chart_df)

    st.divider()

if __name__ == "__main__":
    main()

