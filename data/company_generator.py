"""
data/company_generator.py
--------------------------
Generates synthetic company profiles used throughout the platform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COMPETITION_LEVELS,
    FUNDING_STAGES,
    INDUSTRIES,
    NUM_COMPANIES,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Company name components
# ---------------------------------------------------------------------------
_PREFIXES = [
    "Alpha", "Beta", "Apex", "Nova", "Zenith", "Prime", "Edge", "Core",
    "Vantage", "Vertex", "Nexus", "Orbit", "Pinnacle", "Clarity", "Fusion",
    "Insight", "Quantum", "Nimbus", "Aether", "Catalyst", "Meridian", "Axiom",
    "Synergy", "Luminary", "Stratum",
]
_SUFFIXES = [
    "Tech", "Solutions", "Systems", "Labs", "Ventures", "Dynamics", "Analytics",
    "Intelligence", "Works", "Group", "Capital", "Innovations", "Network",
    "Digital", "AI", "Platform", "Advisors", "Research", "Strategies", "Partners",
]


def _generate_company_names(n: int, rng: np.random.Generator) -> list[str]:
    """Create *n* unique synthetic company names."""
    names: list[str] = []
    seen: set[str] = set()
    while len(names) < n:
        prefix = rng.choice(_PREFIXES)
        suffix = rng.choice(_SUFFIXES)
        name = f"{prefix} {suffix}"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def generate_companies(
    n: int = NUM_COMPANIES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a DataFrame of *n* synthetic company profiles.

    Parameters
    ----------
    n:
        Number of companies to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: company_name, industry, employee_count, revenue,
        funding_stage, rd_spending, market_growth, competition_level,
        product_launch_frequency, profit_margin.
    """
    rng = np.random.default_rng(seed)

    industries = rng.choice(INDUSTRIES, size=n)
    funding_stages = rng.choice(FUNDING_STAGES, size=n)
    competition_levels = rng.choice(COMPETITION_LEVELS, size=n)

    # Employee count varies by funding stage
    funding_idx = np.array([FUNDING_STAGES.index(f) for f in funding_stages])
    base_employees = 10 + funding_idx * 120
    employee_count = (
        base_employees + rng.integers(0, 200, size=n)
    ).clip(5, 50_000)

    # Revenue in millions — scales with employees + noise
    revenue_base = employee_count * rng.uniform(0.08, 0.35, size=n)
    revenue = np.round(revenue_base + rng.uniform(0, 20, size=n), 2)

    # R&D spending as % of revenue (0–40 %)
    rd_spending = np.round(rng.uniform(1, 40, size=n), 2)

    # Market growth % per year (−5 % to +35 %)
    market_growth = np.round(rng.uniform(-5, 35, size=n), 2)

    # Product launches per year
    product_launch_frequency = rng.integers(1, 20, size=n)

    # Profit margin % (−15 % to +45 %)
    profit_margin = np.round(rng.uniform(-15, 45, size=n), 2)

    df = pd.DataFrame(
        {
            "company_name": _generate_company_names(n, rng),
            "industry": industries,
            "employee_count": employee_count,
            "revenue": revenue,
            "funding_stage": funding_stages,
            "rd_spending": rd_spending,
            "market_growth": market_growth,
            "competition_level": competition_levels,
            "product_launch_frequency": product_launch_frequency,
            "profit_margin": profit_margin,
        }
    )
    return df
