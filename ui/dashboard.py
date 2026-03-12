"""
ui/dashboard.py
----------------
Streamlit dashboard sections for the AI Company Intelligence Engine.
Each function renders one logical section of the UI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.market_analysis import (
    global_summary,
    industry_summary,
    risk_distribution,
    top_companies,
)
from config import TOP_N_COMPANIES
from ui.charts import (
    company_radar_chart,
    company_ranking_chart,
    growth_prediction_chart,
    industry_comparison_chart,
    risk_distribution_chart,
    score_distribution_chart,
)
from utils.helpers import format_currency, label_risk, pct


# ---------------------------------------------------------------------------
# Section 1 — Global KPI cards
# ---------------------------------------------------------------------------

def render_global_analytics(df: pd.DataFrame) -> None:
    """Render platform-wide summary KPI cards."""
    st.header("🌐 Global Company Analytics")
    summary = global_summary(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        label="📈 Avg Growth Score",
        value=f"{summary['avg_growth_score']:.1f}",
        help="Average growth potential score across all filtered companies (0–100)",
    )
    col2.metric(
        label="💡 Avg Innovation Index",
        value=f"{summary['avg_innovation_score']:.1f}",
        help="Average innovation capability score (0–100)",
    )
    col3.metric(
        label="⚠️ Avg Risk Score",
        value=f"{summary['avg_risk_score']:.1f}",
        help="Average market risk score — lower is safer (0–100)",
    )
    col4.metric(
        label="👥 Avg Hiring Prob.",
        value=pct(summary["avg_hiring_probability"]),
        help="Average probability of hiring expansion",
    )
    col5.metric(
        label="🏅 Avg Intelligence Index",
        value=f"{summary['avg_intelligence_index']:.1f}",
        help="Composite intelligence index (0–100)",
    )
    st.caption(f"Analysing **{summary['total_companies']}** companies")


# ---------------------------------------------------------------------------
# Section 2 — Company Ranking
# ---------------------------------------------------------------------------

def render_company_ranking(df: pd.DataFrame) -> None:
    """Render top-N company ranking table and bar chart."""
    st.header(f"🏆 Company Ranking — Top {TOP_N_COMPANIES}")

    top_df = top_companies(df, n=TOP_N_COMPANIES)

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(company_ranking_chart(top_df), use_container_width=True)
    with col_table:
        display = top_df[
            ["company_name", "industry", "intelligence_index", "growth_score", "risk_score"]
        ].rename(
            columns={
                "company_name": "Company",
                "industry": "Industry",
                "intelligence_index": "Index",
                "growth_score": "Growth",
                "risk_score": "Risk",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section 3 — Company Deep Analysis
# ---------------------------------------------------------------------------

def render_company_deep_analysis(df: pd.DataFrame) -> None:
    """Render interactive single-company analysis view."""
    st.header("🔬 Company Deep Analysis")

    company_names = sorted(df["company_name"].tolist())
    selected_name = st.selectbox(
        "Select a company to analyse:",
        options=company_names,
        key="company_selector",
    )

    company = df[df["company_name"] == selected_name].iloc[0]

    # --- Summary cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📈 Growth Score", f"{company['growth_score']:.1f} / 100")
    c2.metric("💡 Innovation Score", f"{company['innovation_score']:.1f} / 100")
    c3.metric("⚠️ Risk Level", label_risk(company["risk_score"]))
    c4.metric("👥 Hiring Probability", pct(company["hiring_probability"]))

    st.markdown("---")

    col_radar, col_info = st.columns([2, 1])
    with col_radar:
        st.plotly_chart(company_radar_chart(company), use_container_width=True)

    with col_info:
        st.subheader("📋 Company Profile")
        st.write(f"**Industry:** {company['industry']}")
        st.write(f"**Funding Stage:** {company['funding_stage']}")
        st.write(f"**Employees:** {company['employee_count']:,}")
        st.write(f"**Revenue:** {format_currency(company['revenue'])}")
        st.write(f"**R&D Spend:** {company['rd_spending']:.1f}%")
        st.write(f"**Market Growth:** {company['market_growth']:.1f}%")
        st.write(f"**Profit Margin:** {company['profit_margin']:.1f}%")
        st.write(f"**Competition Level:** {company['competition_level']}")
        st.write(f"**Product Launches/yr:** {company['product_launch_frequency']}")
        st.markdown("---")
        st.metric("🏅 Intelligence Index", f"{company['intelligence_index']:.1f} / 100")


# ---------------------------------------------------------------------------
# Section 4 — Visualisations
# ---------------------------------------------------------------------------

def render_visualisations(df: pd.DataFrame) -> None:
    """Render the full suite of Plotly analytical charts."""
    st.header("📊 Market Visualisations")

    # Row 1 — Growth & Ranking
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(growth_prediction_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(
            score_distribution_chart(df, "intelligence_index"),
            use_container_width=True,
        )

    # Row 2 — Risk & Industry
    col3, col4 = st.columns(2)
    with col3:
        risk_df = risk_distribution(df)
        st.plotly_chart(risk_distribution_chart(risk_df), use_container_width=True)
    with col4:
        ind_df = industry_summary(df)
        st.plotly_chart(industry_comparison_chart(ind_df), use_container_width=True)
