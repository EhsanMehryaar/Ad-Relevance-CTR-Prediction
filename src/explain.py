from __future__ import annotations

import numpy as np
import pandas as pd


def get_feature_names(pipeline) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def extract_feature_importance(pipeline, top_n: int = 50) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
        importance_type = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
                "importance_type": importance_type,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance
