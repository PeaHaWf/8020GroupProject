from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PRICE_COLUMNS, VOLUME_COLUMN


FEATURE_COLUMNS = [
    "open_return",
    "high_return",
    "low_return",
    "close_return",
    "log_return",
    "rolling_volatility",
    "volume_z",
    "intrabar_range",
]


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["hi1_close"].astype(float)

    for column in PRICE_COLUMNS:
        out[f"{column.split('_')[-1]}_return"] = out[column].astype(float).pct_change()

    out["log_return"] = np.log(close / close.shift(1))
    out["rolling_volatility"] = out["log_return"].rolling(30, min_periods=5).std()
    volume = out[VOLUME_COLUMN].astype(float)
    out["volume_z"] = (volume - volume.rolling(60, min_periods=10).mean()) / (
        volume.rolling(60, min_periods=10).std() + 1e-8
    )
    out["intrabar_range"] = (out["hi1_high"] - out["hi1_low"]) / (close + 1e-8)

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    featured = add_market_features(df)
    features = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if mean is None:
        mean = features.mean(axis=0, keepdims=True)
    if std is None:
        std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std
    close = featured["hi1_close"].to_numpy(dtype=np.float32)
    return features, close, mean.astype(np.float32), std.astype(np.float32)
