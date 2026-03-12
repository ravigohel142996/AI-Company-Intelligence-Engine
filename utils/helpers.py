"""
utils/helpers.py
-----------------
Shared utility functions used across the platform.
"""

from __future__ import annotations

import pandas as pd


def label_risk(score: float) -> str:
    """Convert a numeric risk score (0–100) to a human-readable label."""
    if score < 33:
        return "🟢 Low"
    if score < 66:
        return "🟡 Medium"
    return "🔴 High"


def label_score(score: float) -> str:
    """Convert a generic 0–100 score to Low / Medium / High."""
    if score < 33:
        return "Low"
    if score < 66:
        return "Medium"
    return "High"


def format_currency(value: float, unit: str = "M") -> str:
    """Format a numeric value as currency string (default: millions)."""
    return f"${value:,.2f}{unit}"


def pct(value: float) -> str:
    """Format a 0–1 probability as a percentage string."""
    return f"{value * 100:.1f}%"


def filter_dataframe(
    df: pd.DataFrame,
    industries: list[str] | None = None,
    min_employees: int = 0,
    max_employees: int = 50_000,
    min_market_growth: float = -5.0,
    max_market_growth: float = 35.0,
    competition_levels: list[str] | None = None,
) -> pd.DataFrame:
    """Apply sidebar filter criteria to the main DataFrame."""
    mask = pd.Series(True, index=df.index)

    if industries:
        mask &= df["industry"].isin(industries)

    mask &= df["employee_count"].between(min_employees, max_employees)
    mask &= df["market_growth"].between(min_market_growth, max_market_growth)

    if competition_levels:
        mask &= df["competition_level"].isin(competition_levels)

    return df[mask].reset_index(drop=True)
