"""
config.py
---------
Global configuration constants for the AI Company Intelligence Engine.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------
NUM_COMPANIES: int = 100          # total synthetic companies to generate
RANDOM_SEED: int = 42             # reproducibility seed

# ---------------------------------------------------------------------------
# Industry catalogue
# ---------------------------------------------------------------------------
INDUSTRIES: list[str] = [
    "Technology",
    "Healthcare",
    "Finance",
    "Retail",
    "Manufacturing",
    "Energy",
    "Telecommunications",
    "Education",
    "Real Estate",
    "Transportation",
]

# ---------------------------------------------------------------------------
# Funding stages (encoded as ordered integers in model features)
# ---------------------------------------------------------------------------
FUNDING_STAGES: list[str] = [
    "Bootstrapped",
    "Pre-Seed",
    "Seed",
    "Series A",
    "Series B",
    "Series C",
    "Series D+",
    "Public",
]
FUNDING_STAGE_ENCODING: dict[str, int] = {
    stage: idx for idx, stage in enumerate(FUNDING_STAGES)
}

# ---------------------------------------------------------------------------
# Competition levels (encoded as ordered integers)
# ---------------------------------------------------------------------------
COMPETITION_LEVELS: list[str] = ["Low", "Medium", "High"]
COMPETITION_ENCODING: dict[str, int] = {
    lvl: idx for idx, lvl in enumerate(COMPETITION_LEVELS)
}

# ---------------------------------------------------------------------------
# Score display thresholds (used for colour coding in the UI)
# ---------------------------------------------------------------------------
SCORE_THRESHOLDS: dict[str, dict[str, float]] = {
    "growth": {"low": 33.0, "high": 66.0},
    "innovation": {"low": 33.0, "high": 66.0},
    "risk": {"low": 33.0, "high": 66.0},
    "hiring": {"low": 0.33, "high": 0.66},
}

# ---------------------------------------------------------------------------
# UI settings
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "AI Company Intelligence Engine"
PAGE_ICON: str = "🏢"
LAYOUT: str = "wide"
TOP_N_COMPANIES: int = 15          # companies shown in ranking chart
