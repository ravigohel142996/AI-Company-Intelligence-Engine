"""
data/industry_dataset.py
-------------------------
Reference data about industries used for benchmarking and visualisations.
"""

from __future__ import annotations

import pandas as pd

# Industry benchmark data (avg market growth %, avg R&D spend %, risk multiplier)
INDUSTRY_BENCHMARKS: dict[str, dict[str, float]] = {
    "Technology": {
        "avg_market_growth": 18.0,
        "avg_rd_spending": 22.0,
        "risk_multiplier": 1.1,
        "innovation_weight": 1.3,
    },
    "Healthcare": {
        "avg_market_growth": 12.0,
        "avg_rd_spending": 18.0,
        "risk_multiplier": 0.9,
        "innovation_weight": 1.2,
    },
    "Finance": {
        "avg_market_growth": 8.0,
        "avg_rd_spending": 8.0,
        "risk_multiplier": 1.2,
        "innovation_weight": 0.9,
    },
    "Retail": {
        "avg_market_growth": 6.0,
        "avg_rd_spending": 4.0,
        "risk_multiplier": 1.0,
        "innovation_weight": 0.8,
    },
    "Manufacturing": {
        "avg_market_growth": 5.0,
        "avg_rd_spending": 6.0,
        "risk_multiplier": 0.95,
        "innovation_weight": 0.85,
    },
    "Energy": {
        "avg_market_growth": 4.0,
        "avg_rd_spending": 5.0,
        "risk_multiplier": 1.3,
        "innovation_weight": 0.75,
    },
    "Telecommunications": {
        "avg_market_growth": 7.0,
        "avg_rd_spending": 10.0,
        "risk_multiplier": 1.05,
        "innovation_weight": 0.95,
    },
    "Education": {
        "avg_market_growth": 9.0,
        "avg_rd_spending": 5.0,
        "risk_multiplier": 0.7,
        "innovation_weight": 1.0,
    },
    "Real Estate": {
        "avg_market_growth": 6.0,
        "avg_rd_spending": 2.0,
        "risk_multiplier": 1.1,
        "innovation_weight": 0.7,
    },
    "Transportation": {
        "avg_market_growth": 7.0,
        "avg_rd_spending": 7.0,
        "risk_multiplier": 1.0,
        "innovation_weight": 0.85,
    },
}


def get_industry_benchmarks() -> pd.DataFrame:
    """Return industry benchmarks as a tidy DataFrame."""
    records = [
        {"industry": industry, **metrics}
        for industry, metrics in INDUSTRY_BENCHMARKS.items()
    ]
    return pd.DataFrame(records)


def get_benchmark(industry: str, metric: str, default: float = 0.0) -> float:
    """Retrieve a single benchmark value for a given industry and metric."""
    return INDUSTRY_BENCHMARKS.get(industry, {}).get(metric, default)
