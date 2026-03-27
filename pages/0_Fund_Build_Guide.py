# 0_Fund_Build_Guide.py

import streamlit as st


# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.title("🗺️ Fund Build Guide")
st.caption("How each model maps to each YouRock fund build process.")


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
This page explains which **models** are used to build each YouRock fund.

Use this as the system map:

- **⭐ 3-Stars Model** → **⭐ YR3G-26**
- **↔️ Divergence Model** → **↔️ YRDG-26**
- **🪨 Opportunity Model** → **🪨 YROG-26**
- **🔮 Prediction Model** → **🔮 YRPG-26**
- **🔄 Recovery Model** → **🔄 YRRG-26**
- **🌪️ Volatility Model** → **📣 YRVI-26**, **💵 YRSI-26**, **🌪️ YRVG-26**
"""
)

st.info("Think of this page as the map. The model pages are the engines.")


# ----------------------------------------------------------
# Builder summary table
# ----------------------------------------------------------
st.markdown("## Builder Map")

builder_rows = [
    {
        "Tool": "⭐ 3-Stars Model",
        "Primary Use": "Quality growth builder",
        "Builds": "⭐ YR3G-26",
        "Style": "Best-in-class company selection",
    },
    {
        "Tool": "↔️ Divergence Model",
        "Primary Use": "Divergence growth builder",
        "Builds": "↔️ YRDG-26",
        "Style": "Strong company index + discounted from peak",
    },
    {
        "Tool": "🪨 Opportunity Model",
        "Primary Use": "Opportunity growth builder",
        "Builds": "🪨 YROG-26",
        "Style": "Value + yield + buyzone opportunity",
    },
    {
        "Tool": "🔮 Prediction Model",
        "Primary Use": "Probability-based growth builder",
        "Builds": "🔮 YRPG-26",
        "Style": "60-day forward probability model",
    },
    {
        "Tool": "🔄 Recovery Model",
        "Primary Use": "Cycle recovery builder",
        "Builds": "🔄 YRRG-26",
        "Style": "Recovery phase + improving direction",
    },
    {
        "Tool": "🌪️ Volatility Model",
        "Primary Use": "Income + hybrid builder",
        "Builds": "📣 YRVI-26, 💵 YRSI-26, 🌪️ YRVG-26",
        "Style": "IV-driven opportunity engine",
    },
]

st.dataframe(builder_rows, use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# Fund build details
# ----------------------------------------------------------
st.markdown("## Fund-by-Fund Guide")


# ⭐ YR3G
render_fund_card(
    fund_name="⭐ YR3G-26",
    fund_type="Quality Growth Fund",
    build_tool="⭐ 3-Stars Model",
    description=(
        "YR3G is built from the highest-quality companies in the system. "
        "This model surfaces companies that meet the strongest fundamental thresholds."
    ),
    process=[
        "Open **3-Stars Model**",
        "Review companies currently holding 3-star status",
        "Evaluate names for diversification and conviction",
        "Select stock positions for the YR3G portfolio",
        "Enter trades into the portfolio system",
    ],
    key_notes=[
        "Best-in-class quality builder",
        "Simple and clean long-term strategy",
        "Focus on proven company strength",
    ],
)


# ↔️ YRDG
render_fund_card(
    fund_name="↔️ YRDG-26",
    fund_type="Divergence Growth Fund",
    build_tool="↔️ Divergence Model",
    description=(
        "YRDG focuses on companies with strong fundamentals that are still trading "
        "below recent highs. It captures the gap between company strength and price."
    ),
    process=[
        "Open **Divergence Model**",
        "Identify strong company index + discounted-from-peak setups",
        "Confirm phase and confidence support",
        "Select positions for YRDG",
        "Enter trades into the portfolio system",
    ],
    key_notes=[
        "Designed for divergence setups",
        "Strong company, lagging price",
        "Captures early recovery opportunities",
    ],
)


# 🪨 YROG
render_fund_card(
    fund_name="🪨 YROG-26",
    fund_type="Opportunity Growth Fund",
    build_tool="🪨 Opportunity Model",
    description=(
        "YROG is built from companies with strong opportunity characteristics across "
        "value, yield, buyzone, and fair value signals."
    ),
    process=[
        "Open **Opportunity Model**",
        "Review high opportunity + strong company index names",
        "Confirm acceptable direction and health",
        "Select positions for YROG",
        "Enter trades into the portfolio system",
    ],
    key_notes=[
        "Opportunity-driven builder",
        "Balanced across value + momentum",
        "Core growth fund engine",
    ],
)


# 🔮 YRPG
render_fund_card(
    fund_name="🔮 YRPG-26",
    fund_type="Prediction Growth Fund",
    build_tool="🔮 Prediction Model",
    description=(
        "YRPG is built from a probability model that estimates whether a stock "
        "will be higher in ~60 trading days."
    ),
    process=[
        "Open **Prediction Model**",
        "Review highest conviction prediction setups",
        "Focus on strongest buckets and regime alignment",
        "Select positions for YRPG",
        "Enter trades into the portfolio system",
    ],
    key_notes=[
        "Probability-based strategy",
        "Uses forward-looking return tendencies",
        "Not linear — strongest buckets matter most",
    ],
)


# 🔄 YRRG
render_fund_card(
    fund_name="🔄 YRRG-26",
    fund_type="Recovery Growth Fund",
    build_tool="🔄 Recovery Model",
    description=(
        "YRRG focuses on companies in the Recovery phase of the market cycle, "
        "capturing early trend reversals."
    ),
    process=[
        "Open **Recovery Model**",
        "Filter for Recovery phase companies",
        "Focus on improving direction + strong opportunity",
        "Select positions for YRRG",
        "Enter trades into the portfolio system",
    ],
    key_notes=[
        "Early trend reversal strategy",
        "Cycle-based timing edge",
        "Complements YROG",
    ],
)


# 🌪️ YRVG
render_fund_card(
    fund_name="🌪️ YRVG-26",
    fund_type="Volatility Growth Fund",
    build_tool="🌪️ Volatility Model",
    description=(
        "YRVG converts volatility-driven opportunities into a stock portfolio, "
        "instead of using options."
    ),
    process=[
        "Open **Volatility Model**",
        "Review high IV opportunity names",
        "Select strong candidates",
        "Buy shares for YRVG",
        "Track positions in the portfolio",
    ],
    key_notes=[
        "Hybrid strategy",
        "Uses IV engine for stock selection",
        "Growth version of income models",
    ],
)


# 📣 YRVI
render_fund_card(
    fund_name="📣 YRVI-26",
    fund_type="Income Fund",
    build_tool="🌪️ Volatility Model",
    description=(
        "YRVI is the primary income fund using weekly option strategies "
        "based on volatility opportunities."
    ),
    process=[
        "Open **Volatility Model**",
        "Review weekly setups",
        "Select CSP or CC trades",
        "Execute weekly trades",
        "Track results",
    ],
    key_notes=[
        "Primary income engine",
        "Weekly execution",
        "IV-driven strategy",
    ],
)


# 💵 YRSI
render_fund_card(
    fund_name="💵 YRSI-26",
    fund_type="Income Fund",
    build_tool="🌪️ Volatility Model",
    description=(
        "YRSI is a more selective version of YRVI, focusing on higher-quality names."
    ),
    process=[
        "Open **Volatility Model**",
        "Review weekly setups",
        "Filter for higher quality trades",
        "Execute trades",
        "Track results",
    ],
    key_notes=[
        "Selective income strategy",
        "Quality-focused",
        "Same engine as YRVI",
    ],
)


# ----------------------------------------------------------
# Supporting pages
# ----------------------------------------------------------
st.markdown("## Supporting Pages")

st.markdown(
    """
These pages support the system but are not primary builders:

- 🧪 Backtesting → Research
- ⭐ 3-Stars Alumni → Historical ideas
- ➕ Add Company → Utility
"""
)


# ----------------------------------------------------------
# Workflow
# ----------------------------------------------------------
st.markdown("## Recommended Workflow")

st.markdown(
    """
### Weekly workflow
1. Open the model page
2. Review candidates
3. Select positions
4. Enter trades
5. Track performance

### Quick mapping
- ⭐ YR3G → 3-Stars Model
- ↔️ YRDG → Divergence Model
- 🪨 YROG → Opportunity Model
- 🔮 YRPG → Prediction Model
- 🔄 YRRG → Recovery Model
- 🌪️ YRVG → Volatility Model
- 📣 YRVI → Volatility Model
- 💵 YRSI → Volatility Model
"""
)


# ----------------------------------------------------------
# Closing
# ----------------------------------------------------------
st.success("Use this page as your system map. The models are your engines.")