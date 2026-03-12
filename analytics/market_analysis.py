"""
analytics/market_analysis.py
------------------------------
Aggregates company scores at the market / industry level for dashboard summaries.
"""

from __future__ import annotations

import pandas as pd


def global_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return platform-wide average scores."""
    return {
        "avg_growth_score": round(df["growth_score"].mean(), 2),
        "avg_innovation_score": round(df["innovation_score"].mean(), 2),
        "avg_risk_score": round(df["risk_score"].mean(), 2),
        "avg_hiring_probability": round(df["hiring_probability"].mean(), 4),
        "avg_intelligence_index": round(df["intelligence_index"].mean(), 2),
        "total_companies": int(len(df)),
    }


def industry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-industry average scores sorted by intelligence index."""
    agg = (
        df.groupby("industry")[
            [
                "growth_score",
                "innovation_score",
                "risk_score",
                "hiring_probability",
                "intelligence_index",
                "revenue",
                "employee_count",
            ]
        ]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("intelligence_index", ascending=False)
    )
    return agg


def top_companies(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return the top *n* companies ranked by intelligence index."""
    cols = [
        "company_name",
        "industry",
        "intelligence_index",
        "growth_score",
        "innovation_score",
        "risk_score",
        "hiring_probability",
        "funding_stage",
        "revenue",
        "employee_count",
    ]
    return (
        df[cols]
        .sort_values("intelligence_index", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Bin companies into risk tiers and return counts."""
    bins = [0, 33, 66, 100]
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    df = df.copy()
    df["risk_tier"] = pd.cut(df["risk_score"], bins=bins, labels=labels)
    return df.groupby("risk_tier", observed=True).size().reset_index(name="count")


def score_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return min/mean/max for each score column — useful for chart axes."""
    score_cols = [
        "growth_score",
        "innovation_score",
        "risk_score",
        "hiring_probability",
        "intelligence_index",
    ]
    return df[score_cols].agg(["min", "mean", "max"]).round(4)
