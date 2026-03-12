"""
analytics/company_scoring.py
------------------------------
Combines individual ML model outputs into a single Company Intelligence Index
and enriches the company DataFrame with all derived scores.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from data.company_generator import generate_companies
from models.growth_predictor import GrowthPredictor
from models.hiring_predictor import HiringPredictor
from models.innovation_model import InnovationModel
from models.risk_model import RiskModel


# Weights for the combined intelligence index
_WEIGHTS = {
    "growth_score": 0.35,
    "innovation_score": 0.30,
    "hiring_probability": 0.20,   # scaled ×100 before weighting
    "risk_score": -0.15,           # higher risk *lowers* the index
}


def _compute_intelligence_index(row: pd.Series) -> float:
    """Compute a 0–100 Company Intelligence Index from scored columns."""
    idx = (
        _WEIGHTS["growth_score"] * row["growth_score"]
        + _WEIGHTS["innovation_score"] * row["innovation_score"]
        + _WEIGHTS["hiring_probability"] * row["hiring_probability"] * 100
        + _WEIGHTS["risk_score"] * row["risk_score"]
    )
    return float(np.clip(idx, 0, 100))


def score_companies(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train all models on *df*, run predictions, and return enriched DataFrame.

    If *df* is None, a fresh synthetic dataset is generated.

    Returns
    -------
    pd.DataFrame
        Original columns plus: growth_score, innovation_score,
        hiring_probability, risk_score, intelligence_index.
    """
    if df is None:
        df = generate_companies()

    result = df.copy()

    # --- Growth Model ---
    growth_model = GrowthPredictor()
    growth_model.fit(result)
    result["growth_score"] = np.round(growth_model.predict(result), 2)

    # --- Innovation Model ---
    innovation_model = InnovationModel()
    innovation_model.fit(result)
    result["innovation_score"] = np.round(innovation_model.predict(result), 2)

    # --- Hiring Model (needs growth_score) ---
    hiring_model = HiringPredictor()
    hiring_model.fit(result)
    result["hiring_probability"] = np.round(hiring_model.predict_proba(result), 4)

    # --- Risk Model ---
    risk_model = RiskModel()
    risk_model.fit(result)
    result["risk_score"] = np.round(risk_model.predict(result), 2)

    # --- Combined Intelligence Index ---
    result["intelligence_index"] = result.apply(
        _compute_intelligence_index, axis=1
    ).round(2)

    return result
