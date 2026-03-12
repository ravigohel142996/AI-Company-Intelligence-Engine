"""
models/hiring_predictor.py
---------------------------
Trains and applies the Hiring Expansion Probability model.

Features : revenue, growth_score, funding_stage (encoded), employee_count
Target   : hiring_probability  (0–1, synthesised)
Algorithm: RandomForestClassifier (binary) → probability output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from config import FUNDING_STAGE_ENCODING, RANDOM_SEED


# ---------------------------------------------------------------------------
# Target synthesis helper
# ---------------------------------------------------------------------------

def _synthesise_hiring_label(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Binary label: 1 = company likely expanding headcount, 0 = stable/contracting."""
    prob = (
        0.35 * df["growth_score"] / 100
        + 0.30 * df["funding_stage_enc"] / (len(FUNDING_STAGE_ENCODING) - 1)
        + 0.20 * df["revenue"].clip(0, 500) / 500
        + 0.15 * np.log1p(df["employee_count"]) / np.log1p(50_000)
    )
    noise = rng.normal(0, 0.08, size=len(df))
    return ((prob + noise) > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HiringPredictor:
    """Sklearn pipeline predicting hiring expansion probability."""

    FEATURES = ["revenue", "growth_score", "funding_stage_enc", "employee_count"]

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self._seed = seed
        self._pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        random_state=seed,
                    ),
                ),
            ]
        )
        self._is_fitted = False

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add encoded funding stage column."""
        out = df.copy()
        out["funding_stage_enc"] = out["funding_stage"].map(FUNDING_STAGE_ENCODING).fillna(0)
        return out

    def fit(self, df: pd.DataFrame) -> "HiringPredictor":
        """Fit on *df* which must already contain *growth_score*."""
        encoded = self._encode_features(df)
        rng = np.random.default_rng(self._seed)
        X = encoded[self.FEATURES]
        y = _synthesise_hiring_label(encoded, rng)
        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of hiring expansion in [0, 1] for each row."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        encoded = self._encode_features(df)
        X = encoded[self.FEATURES]
        return self._pipeline.predict_proba(X)[:, 1]
