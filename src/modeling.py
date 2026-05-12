from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object


def available_model_specs(random_state: int = 42) -> dict[str, ModelSpec]:
    specs: dict[str, ModelSpec] = {
        "logistic_regression": ModelSpec(
            "Logistic Regression",
            LogisticRegression(
                max_iter=3000,
                solver="saga",
                tol=1e-3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "random_forest": ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        "neural_network": ModelSpec(
            "Neural Network",
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                batch_size="auto",
                early_stopping=True,
                random_state=random_state,
                max_iter=40,
            ),
        ),
    }

    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = ModelSpec(
            "XGBoost",
            XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        specs["lightgbm"] = ModelSpec(
            "LightGBM",
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                verbose=-1,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    except ImportError:
        pass

    return specs


def build_model_pipeline(model_key: str, random_state: int = 42) -> Pipeline:
    specs = available_model_specs(random_state=random_state)
    if model_key not in specs:
        valid = ", ".join(sorted(specs))
        raise ValueError(f"Unknown model '{model_key}'. Valid options: {valid}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", specs[model_key].estimator),
        ]
    )
