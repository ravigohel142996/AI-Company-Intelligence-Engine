"""
ui/controls.py
---------------
Streamlit sidebar filter widgets for the AI Company Intelligence Engine.
Returns a dict of filter values consumed by utils.helpers.filter_dataframe().
"""

from __future__ import annotations

import streamlit as st

from config import COMPETITION_LEVELS, INDUSTRIES


def render_sidebar_controls() -> dict:
    """Render all sidebar controls and return a dict of filter values."""
    st.sidebar.title("🎛️ Analysis Controls")
    st.sidebar.markdown("---")

    # --- Industry filter ---
    st.sidebar.subheader("🏭 Industry Filter")
    selected_industries = st.sidebar.multiselect(
        "Select Industries",
        options=INDUSTRIES,
        default=INDUSTRIES,
        key="industry_filter",
    )

    st.sidebar.markdown("---")

    # --- Company size ---
    st.sidebar.subheader("👥 Company Size (Employees)")
    size_range = st.sidebar.slider(
        "Employee Count Range",
        min_value=0,
        max_value=1_500,
        value=(0, 1_500),
        step=50,
        key="size_filter",
    )

    st.sidebar.markdown("---")

    # --- Market growth ---
    st.sidebar.subheader("📈 Market Growth (%)")
    growth_range = st.sidebar.slider(
        "Annual Growth Range",
        min_value=-5,
        max_value=35,
        value=(-5, 35),
        step=1,
        key="growth_filter",
    )

    st.sidebar.markdown("---")

    # --- Competition level ---
    st.sidebar.subheader("⚔️ Competition Level")
    selected_competition = st.sidebar.multiselect(
        "Competition Levels",
        options=COMPETITION_LEVELS,
        default=COMPETITION_LEVELS,
        key="competition_filter",
    )

    st.sidebar.markdown("---")

    # --- Analyze button ---
    analyze_clicked = st.sidebar.button(
        "🔍 Analyze Companies",
        use_container_width=True,
        type="primary",
        key="analyze_btn",
    )

    return {
        "industries": selected_industries,
        "min_employees": size_range[0],
        "max_employees": size_range[1],
        "min_market_growth": float(growth_range[0]),
        "max_market_growth": float(growth_range[1]),
        "competition_levels": selected_competition,
        "analyze_clicked": analyze_clicked,
    }
