"""
models/growth_predictor.py
---------------------------
Trains and applies the Growth Potential Score model.

Features : revenue, market_growth, rd_spending, employee_count
Target   : growth_score  (0–100, synthesised from features + noise)
Algorithm: GradientBoostingRegressor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from config import RANDOM_SEED


# ---------------------------------------------------------------------------
# Target synthesis helper
# ---------------------------------------------------------------------------

def _synthesise_growth_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Create a pseudo-ground-truth growth score from raw features."""
    score = (
        0.35 * (df["market_growth"].clip(-5, 35) + 5) / 40 * 100
        + 0.30 * df["revenue"].clip(0, 500) / 500 * 100
        + 0.20 * df["rd_spending"] / 40 * 100
        + 0.15 * np.log1p(df["employee_count"]) / np.log1p(50_000) * 100
    )
    noise = rng.normal(0, 3, size=len(df))
    return (score + noise).clip(0, 100)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GrowthPredictor:
    """Sklearn-based pipeline for predicting company growth scores."""

    FEATURES = ["revenue", "market_growth", "rd_spending", "employee_count"]

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self._seed = seed
        self._pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "GrowthPredictor":
        """Fit the model on *df* using synthesised targets."""
        rng = np.random.default_rng(self._seed)
        X = df[self.FEATURES].copy()
        y = _synthesise_growth_score(df, rng)
        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return growth scores in [0, 100] for each row of *df*."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        X = df[self.FEATURES].copy()
        scores = self._pipeline.predict(X)
        return scores.clip(0, 100)
