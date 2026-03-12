"""
app.py
------
Entry point for the AI Company Intelligence Engine.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the Python path so all modules resolve correctly
# when running via `streamlit run app.py` from any working directory.
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from analytics.company_scoring import score_companies
from config import LAYOUT, PAGE_ICON, PAGE_TITLE
from ui.controls import render_sidebar_controls
from ui.dashboard import (
    render_company_deep_analysis,
    render_company_ranking,
    render_global_analytics,
    render_visualisations,
)
from utils.helpers import filter_dataframe

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load & score data (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="🧠 Training models and scoring companies…")
def load_scored_data() -> "pd.DataFrame":  # noqa: F821
    return score_companies()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    # Header
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown(
        "> **AI-powered analytics platform** that predicts growth potential, "
        "hiring expansion, innovation capability, and market risk for companies "
        "across industries."
    )
    st.markdown("---")

    # Load full scored dataset
    full_df = load_scored_data()

    # Sidebar controls
    filters = render_sidebar_controls()

    # Apply filters
    df = filter_dataframe(
        full_df,
        industries=filters["industries"],
        min_employees=filters["min_employees"],
        max_employees=filters["max_employees"],
        min_market_growth=filters["min_market_growth"],
        max_market_growth=filters["max_market_growth"],
        competition_levels=filters["competition_levels"],
    )

    if df.empty:
        st.warning("⚠️ No companies match the current filters. Please adjust the sidebar.")
        return

    # Dashboard sections
    render_global_analytics(df)
    st.markdown("---")
    render_company_ranking(df)
    st.markdown("---")
    render_company_deep_analysis(df)
    st.markdown("---")
    render_visualisations(df)

    # Footer
    st.markdown("---")
    st.caption(
        "AI Company Intelligence Engine · Built with Streamlit, Scikit-learn & Plotly · "
        "Data is synthetic and for demonstration purposes only."
    )


if __name__ == "__main__":
    main()
