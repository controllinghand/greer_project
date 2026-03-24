# 0_Fund_Build_Guide.py

import streamlit as st


# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.title("🗺️ Fund Build Guide")
st.caption("How each YouRock tool maps to each fund build process.")


# ----------------------------------------------------------
# Helper function to render a fund card
# ----------------------------------------------------------
def render_fund_card(
    fund_name: str,
    fund_type: str,
    build_tool: str,
    description: str,
    process: list[str],
    key_notes: list[str],
):
    st.markdown(f"## {fund_name}")
    st.markdown(f"**Type:** {fund_type}")
    st.markdown(f"**Built From:** {build_tool}")
    st.markdown(description)

    st.markdown("**Build Process**")
    for step in process:
        st.markdown(f"- {step}")

    st.markdown("**Key Notes**")
    for note in key_notes:
        st.markdown(f"- {note}")

    st.divider()


# ----------------------------------------------------------
# Intro section
# ----------------------------------------------------------
st.markdown(
    """
This page explains which tools are used to build each YouRock fund.

Use this as the quick reference for the system:

- **Opportunities** builds **YROG-26**
- **Weekly IV Targets** builds **YRVI-26**, **YRSI-26**, and **YRVG-26**
- **All 3-Stars** builds **YR3G-26**
"""
)

st.info(
    "Think of this page as the map, and the builder pages as the engines."
)


# ----------------------------------------------------------
# Builder summary table
# ----------------------------------------------------------
st.markdown("## Builder Map")

builder_rows = [
    {
        "Tool": "🎯 Opportunities",
        "Primary Use": "Growth fund builder",
        "Builds": "YROG-26",
        "Style": "Opportunity-driven stock selection",
    },
    {
        "Tool": "🧲 Weekly IV Targets",
        "Primary Use": "Weekly income and hybrid builder",
        "Builds": "YRVI-26, YRSI-26, YRVG-26",
        "Style": "IV-driven weekly targets",
    },
    {
        "Tool": "⭐ All 3-Stars",
        "Primary Use": "Quality growth builder",
        "Builds": "YR3G-26",
        "Style": "Best-in-class company selection",
    },
]

st.dataframe(builder_rows, use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# Fund build details
# ----------------------------------------------------------
st.markdown("## Fund-by-Fund Guide")

render_fund_card(
    fund_name="🪨 YROG-26",
    fund_type="Growth Fund",
    build_tool="🎯 Opportunities",
    description=(
        "YROG is built from companies showing attractive opportunity characteristics. "
        "This page is designed to surface candidates with strong value, yield, buyzone, "
        "and fair value support."
    ),
    process=[
        "Open **Opportunities**",
        "Review the highest-quality names using Greer Company Index, phase, and stars",
        "Focus on companies with strong opportunity plus acceptable direction and health",
        "Select stock positions for the YROG portfolio",
        "Enter the trades into the portfolio system",
    ],
    key_notes=[
        "Best for opportunity-driven growth ideas",
        "Now improved with Greer Company Index, Health, Direction, Opportunity, and Phase",
        "Useful when building or refreshing YROG positions",
    ],
)

render_fund_card(
    fund_name="📣 YRVI-26",
    fund_type="Income Fund",
    build_tool="🧲 Weekly IV Targets",
    description=(
        "YRVI is built from weekly IV-based option opportunities. "
        "This is the primary income engine focused on attractive weekly premium generation."
    ),
    process=[
        "Open **Weekly IV Targets**",
        "Review weekly candidates that meet IV and trade quality rules",
        "Choose the best cash-secured put or covered call setups",
        "Build the week’s plan for YRVI",
        "Enter and track trades through the normal weekly process",
    ],
    key_notes=[
        "Primary weekly income builder",
        "Built around IV-driven targets",
        "Typically used for recurring weekly option selection",
    ],
)

render_fund_card(
    fund_name="💵 YRSI-26",
    fund_type="Income Fund",
    build_tool="🧲 Weekly IV Targets",
    description=(
        "YRSI uses the Weekly IV Targets process but limits selections to companies with a star rating, "
        "creating a more selective and quality-focused income strategy."
    ),
    process=[
        "Open **Weekly IV Targets**",
        "Review the week’s option candidates",
        "Select setups that fit the YRSI plan",
        "Build the week’s trade list",
        "Track entries and results through the portfolio workflow",
    ],
    key_notes=[
        "Uses the same weekly builder engine as YRVI",
        "Selection may differ based on fund rules or preferences",
        "Supports systematic weekly income generation",
    ],
)

render_fund_card(
    fund_name="🌪️ YRVG-26",
    fund_type="Hybrid / Volatility Growth Fund",
    build_tool="🧲 Weekly IV Targets",
    description=(
        "YRVG uses the Weekly IV Targets page as a sourcing engine, but instead of building "
        "an income portfolio, it uses those ideas to build a growth-oriented stock portfolio."
    ),
    process=[
        "Open **Weekly IV Targets**",
        "Review the strongest weekly opportunity names",
        "Select stock candidates sourced from the same IV-driven list",
        "Buy shares for YRVG rather than selling options",
        "Track the positions inside the YRVG portfolio",
    ],
    key_notes=[
        "Built from the same idea engine as YRVI and YRSI",
        "Execution style is growth, not income",
        "Useful for converting high-IV opportunity names into stock positions",
    ],
)

render_fund_card(
    fund_name="⭐ YR3G-26",
    fund_type="Quality Growth Fund",
    build_tool="⭐ All 3-Stars",
    description=(
        "YR3G is built from the highest-quality companies in the system. "
        "The All 3-Stars page acts as the source list for elite growth candidates."
    ),
    process=[
        "Open **All 3-Stars**",
        "Review companies currently holding 3-star status",
        "Evaluate names for fit, diversification, and conviction",
        "Select stock positions for YR3G",
        "Track entries and exits in the portfolio system",
    ],
    key_notes=[
        "Best-in-class quality builder",
        "Focused on companies already meeting the 3-star threshold",
        "Simple and clean builder for long-term quality growth",
    ],
)


# ----------------------------------------------------------
# Supporting pages section
# ----------------------------------------------------------
st.markdown("## Supporting Pages")

st.markdown(
    """
These pages support the build process, but they are **not primary builder tools**:

- **🧪 Backtesting** → Research and testing
- **⭐ All 3-Stars Alumni** → Historical research and idea mining
- **➕ Add Company** → Utility / maintenance tool
"""
)


# ----------------------------------------------------------
# Recommended workflow
# ----------------------------------------------------------
st.markdown("## Recommended Workflow")

st.markdown(
    """
### Weekly workflow
1. Start with the builder page for the relevant fund
2. Review candidates and apply the fund’s rules
3. Choose the final names or trades
4. Enter positions into the portfolio system
5. Track results on the fund page

### Tool-to-fund shortcut
- **YROG** → Opportunities
- **YRVI** → Weekly IV Targets
- **YRSI** → Weekly IV Targets
- **YRVG** → Weekly IV Targets
- **YR3G** → All 3-Stars
"""
)


# ----------------------------------------------------------
# Closing note
# ----------------------------------------------------------
st.success(
    "Use this page as the system map. The builder pages are where selections happen."
)