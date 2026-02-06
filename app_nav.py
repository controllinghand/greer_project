# app_nav.py
import os
import streamlit as st

def is_admin() -> bool:
    return os.getenv("YRC_ADMIN", "0") == "1"

def build_pages():
    pages = {
        "Dashboards": [
            st.Page("pages/0_Home.py", title="Greer Value Search", icon="🔎", default=True),
            st.Page("pages/1_Dashboard.py", title="Dashboard", icon="📊"),
            st.Page("pages/2_Dashboard-mini.py", title="Dashboard Mini", icon="🧩"),
        ],
        "Portfolios": [
            st.Page("pages/12_Fund_Compare_YTD.py", title="Fund Compare (YTD)", icon="🆚"),

            # ---- Income Funds ----
            st.Page("pages/_spacer.py", title="— Income Funds —", url_path="spacer-income"),
            st.Page("pages/11_YRQI-26.py", title="  YRQI-26", icon="⚡"),
            st.Page("pages/11_YRVI-26.py", title="  YRVI-26", icon="📣"),
            st.Page("pages/11_YRSI-26.py", title="  YRSI-26", icon="💵"),

            # ---- Growth Funds ----
            st.Page("pages/_spacer.py", title="— Growth Funds —", url_path="spacer-growth"),
            st.Page("pages/12_YR3G-26.py", title="  YR3G-26", icon="⭐"),
            st.Page("pages/12_YROG-26.py", title="  YROG-26", icon="🪨"),

            # ---- Baseline ----
            st.Page("pages/_spacer.py", title="— Baselines —", url_path="spacer-baseline"),
            st.Page("pages/12_SPY-26.py", title="  SPY-26", icon="📐"),
            st.Page("pages/12_QQQ-26.py", title="  QQQ-26", icon="📐"),
        ],
        "Tools": [
            st.Page("pages/10_Opportunities_IV.py", title="Opportunities (IV)", icon="🎯"),
            st.Page("pages/11_WK_IV_Targets.py", title="Weekly IV Targets", icon="🧲"),
            st.Page("pages/6_Backtesting.py", title="Backtesting", icon="🧪"),
            st.Page("pages/add_company.py", title="Add Company", icon="➕"),
            st.Page("pages/8_all_stars.py", title="All Stars", icon="⭐"),
            st.Page("pages/9_all_stars_cards.py", title="All Stars Cards", icon="🪪"),
        ],
    }

    if is_admin():
        pages["Admin"] = [
            st.Page("admin/13_Admin_Ledger.py", title="Admin Ledger", icon="🔒"),
            st.Page("admin/12_ROTH-26.py", title="ROTH-26 (Private)", icon="📣"),
        ]

    # Optional: page map for st.page_link
    page_map = {
        "Home": pages["Dashboards"][0],
        "Add Company": next(p for p in pages["Tools"] if p.title == "Add Company"),
    }

    return pages, page_map
