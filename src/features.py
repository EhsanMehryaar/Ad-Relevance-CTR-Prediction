from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, INTEGER_FEATURES


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create user/ad/context-style features from anonymized Criteo fields."""
    featured = df.copy()

    integer_frame = featured[INTEGER_FEATURES].apply(pd.to_numeric, errors="coerce")
    featured[INTEGER_FEATURES] = integer_frame

    featured["user_activity_score"] = integer_frame[["I1", "I2", "I3", "I4"]].fillna(0).sum(axis=1)
    featured["context_intensity_score"] = integer_frame[["I5", "I6", "I7", "I8"]].fillna(0).sum(axis=1)
    featured["ad_signal_score"] = integer_frame[["I9", "I10", "I11", "I12", "I13"]].fillna(0).sum(axis=1)
    featured["missing_numeric_count"] = integer_frame.isna().sum(axis=1)
    featured["known_category_count"] = featured[CATEGORICAL_FEATURES].notna().sum(axis=1)

    for column in CATEGORICAL_FEATURES:
        featured[column] = featured[column].fillna("__missing__").astype(str)

    return featured


def _log1p_abs(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        *INTEGER_FEATURES,
        "user_activity_score",
        "context_intensity_score",
        "ad_signal_score",
        "missing_numeric_count",
        "known_category_count",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log_scale", FunctionTransformer(_log1p_abs, feature_names_out="one-to-one")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
