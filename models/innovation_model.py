"""
models/innovation_model.py
---------------------------
Trains and applies the Innovation Capability Index model.

Features : rd_spending, product_launch_frequency, industry (encoded),
           competition_level (encoded)
Target   : innovation_score  (0–100, synthesised)
Algorithm: GradientBoostingRegressor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from config import COMPETITION_ENCODING, INDUSTRIES, RANDOM_SEED
from data.industry_dataset import get_benchmark

# Industry innovation weight lookup
_INNOVATION_WEIGHT: dict[str, float] = {
    industry: get_benchmark(industry, "innovation_weight", 1.0)
    for industry in INDUSTRIES
}


# ---------------------------------------------------------------------------
# Target synthesis helper
# ---------------------------------------------------------------------------

def _synthesise_innovation_score(
    df: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    ind_weight = df["industry"].map(_INNOVATION_WEIGHT).fillna(1.0).to_numpy()
    score = (
        0.40 * df["rd_spending"] / 40 * 100
        + 0.35 * df["product_launch_frequency"].clip(1, 20) / 20 * 100
        + 0.25 * (1 - df["competition_enc"] / (len(COMPETITION_ENCODING) - 1)) * 100
    ) * ind_weight
    noise = rng.normal(0, 3, size=len(df))
    return (score + noise).clip(0, 100)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class InnovationModel:
    """Sklearn pipeline predicting innovation capability scores."""

    FEATURES = [
        "rd_spending",
        "product_launch_frequency",
        "industry_enc",
        "competition_enc",
    ]

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

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["competition_enc"] = (
            out["competition_level"].map(COMPETITION_ENCODING).fillna(1)
        )
        out["industry_enc"] = pd.Categorical(out["industry"], categories=INDUSTRIES).codes
        return out

    def fit(self, df: pd.DataFrame) -> "InnovationModel":
        encoded = self._encode_features(df)
        rng = np.random.default_rng(self._seed)
        X = encoded[self.FEATURES]
        y = _synthesise_innovation_score(encoded, rng)
        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return innovation scores in [0, 100]."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        encoded = self._encode_features(df)
        X = encoded[self.FEATURES]
        return self._pipeline.predict(X).clip(0, 100)
