"""
ui/charts.py
-------------
Plotly chart builders for the AI Company Intelligence Engine dashboard.
All functions accept a DataFrame (or aggregated summary) and return a
plotly Figure object that can be rendered with st.plotly_chart().
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_COLOUR_SCALE = "Viridis"
_PALETTE = px.colors.qualitative.Plotly


# ---------------------------------------------------------------------------
# 1. Growth Prediction Chart (scatter: revenue vs growth_score)
# ---------------------------------------------------------------------------

def growth_prediction_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of revenue vs growth score, coloured by industry."""
    fig = px.scatter(
        df,
        x="revenue",
        y="growth_score",
        color="industry",
        size="employee_count",
        hover_name="company_name",
        hover_data={
            "funding_stage": True,
            "market_growth": ":.1f",
            "revenue": ":.2f",
            "growth_score": ":.1f",
            "employee_count": True,
        },
        title="📈 Growth Score vs Revenue",
        labels={
            "revenue": "Revenue ($M)",
            "growth_score": "Growth Score (0–100)",
            "industry": "Industry",
        },
        template="plotly_dark",
        color_discrete_sequence=_PALETTE,
    )
    fig.update_layout(
        legend=dict(title="Industry", orientation="v"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Company Ranking Bar Chart (top N by intelligence index)
# ---------------------------------------------------------------------------

def company_ranking_chart(top_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top companies by intelligence index."""
    fig = px.bar(
        top_df.sort_values("intelligence_index"),
        x="intelligence_index",
        y="company_name",
        color="intelligence_index",
        color_continuous_scale=_COLOUR_SCALE,
        hover_data={
            "industry": True,
            "growth_score": ":.1f",
            "innovation_score": ":.1f",
            "risk_score": ":.1f",
        },
        title="🏆 Top Companies by Intelligence Index",
        labels={
            "intelligence_index": "Intelligence Index",
            "company_name": "Company",
        },
        orientation="h",
        template="plotly_dark",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Risk Distribution Pie Chart
# ---------------------------------------------------------------------------

def risk_distribution_chart(risk_df: pd.DataFrame) -> go.Figure:
    """Pie chart showing proportion of companies in each risk tier."""
    colours = {"Low Risk": "#2ECC71", "Medium Risk": "#F39C12", "High Risk": "#E74C3C"}
    fig = px.pie(
        risk_df,
        names="risk_tier",
        values="count",
        color="risk_tier",
        color_discrete_map=colours,
        title="⚠️ Risk Distribution Across Companies",
        template="plotly_dark",
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ---------------------------------------------------------------------------
# 4. Industry Comparison Radar / Bar Chart
# ---------------------------------------------------------------------------

def industry_comparison_chart(industry_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing key scores across industries."""
    metrics = ["growth_score", "innovation_score", "risk_score"]
    labels = ["Growth Score", "Innovation Score", "Risk Score"]
    colours = ["#3498DB", "#9B59B6", "#E74C3C"]

    fig = go.Figure()
    for metric, label, colour in zip(metrics, labels, colours):
        fig.add_trace(
            go.Bar(
                name=label,
                x=industry_df["industry"],
                y=industry_df[metric],
                marker_color=colour,
            )
        )

    fig.update_layout(
        title="🏭 Industry Comparison — Key Scores",
        xaxis_title="Industry",
        yaxis_title="Score (0–100)",
        barmode="group",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Company Deep-Analysis Radar Chart
# ---------------------------------------------------------------------------

def company_radar_chart(company: pd.Series) -> go.Figure:
    """Radar chart of a single company's scores."""
    categories = [
        "Growth",
        "Innovation",
        "Hiring ×100",
        "Low Risk",
        "Intelligence",
    ]
    values = [
        company["growth_score"],
        company["innovation_score"],
        company["hiring_probability"] * 100,
        100 - company["risk_score"],   # invert: low risk = good
        company["intelligence_index"],
    ]
    # Close the polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=company["company_name"],
            line_color="#3498DB",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"🔍 {company['company_name']} — Score Profile",
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Score Distribution Histogram
# ---------------------------------------------------------------------------

def score_distribution_chart(df: pd.DataFrame, score_col: str = "intelligence_index") -> go.Figure:
    """Histogram of score distribution across all companies."""
    label_map = {
        "intelligence_index": "Intelligence Index",
        "growth_score": "Growth Score",
        "innovation_score": "Innovation Score",
        "risk_score": "Risk Score",
    }
    label = label_map.get(score_col, score_col)
    fig = px.histogram(
        df,
        x=score_col,
        nbins=20,
        color_discrete_sequence=["#3498DB"],
        title=f"📊 Distribution of {label}",
        labels={score_col: label},
        template="plotly_dark",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig
