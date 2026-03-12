"""
models/risk_model.py
---------------------
Trains and applies the Market Risk Level model.

Features : competition_level (encoded), profit_margin, market_growth,
           funding_stage (encoded)
Target   : risk_score  (0–100, higher = riskier, synthesised)
Algorithm: GradientBoostingRegressor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from config import COMPETITION_ENCODING, FUNDING_STAGE_ENCODING, RANDOM_SEED


# ---------------------------------------------------------------------------
# Target synthesis helper
# ---------------------------------------------------------------------------

def _synthesise_risk_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    competition_norm = df["competition_enc"] / (len(COMPETITION_ENCODING) - 1)
    # Negative profit margin raises risk, positive lowers it
    profit_risk = (50 - df["profit_margin"].clip(-15, 45)) / 60
    # Low market growth = higher risk
    growth_risk = 1 - (df["market_growth"].clip(-5, 35) + 5) / 40
    # Early-stage funding = higher risk (inverse of stage index)
    stage_risk = 1 - df["funding_stage_enc"] / (len(FUNDING_STAGE_ENCODING) - 1)

    score = (
        0.30 * competition_norm * 100
        + 0.30 * profit_risk * 100
        + 0.25 * growth_risk * 100
        + 0.15 * stage_risk * 100
    )
    noise = rng.normal(0, 3, size=len(df))
    return (score + noise).clip(0, 100)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RiskModel:
    """Sklearn pipeline predicting market risk scores."""

    FEATURES = [
        "competition_enc",
        "profit_margin",
        "market_growth",
        "funding_stage_enc",
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
        out["funding_stage_enc"] = (
            out["funding_stage"].map(FUNDING_STAGE_ENCODING).fillna(0)
        )
        return out

    def fit(self, df: pd.DataFrame) -> "RiskModel":
        encoded = self._encode_features(df)
        rng = np.random.default_rng(self._seed)
        X = encoded[self.FEATURES]
        y = _synthesise_risk_score(encoded, rng)
        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return risk scores in [0, 100]."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        encoded = self._encode_features(df)
        X = encoded[self.FEATURES]
        return self._pipeline.predict(X).clip(0, 100)
