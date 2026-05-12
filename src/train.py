from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_RANDOM_STATE, LABEL_COLUMN, MODELS_DIR, RAW_DATA_PATH
from src.data import clean_criteo_data, load_criteo_data, sample_data
from src.explain import extract_feature_importance
from src.features import add_engineered_features
from src.metrics import calibration_table, compute_classification_metrics, lift_chart_table
from src.modeling import build_model_pipeline


def train(
    data_path: str | Path = RAW_DATA_PATH,
    model_key: str = "logistic_regression",
    sample_size: int | None = 200_000,
    test_size: float = 0.2,
    output_dir: str | Path = MODELS_DIR,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, float]:
    if sample_size is not None and sample_size <= 0:
        sample_size = None

    df = load_criteo_data(data_path)
    df = clean_criteo_data(df)
    df = sample_data(df, sample_size=sample_size, random_state=random_state)
    df = add_engineered_features(df)

    y = df[LABEL_COLUMN]
    x = df.drop(columns=[LABEL_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(model_key=model_key, random_state=random_state)
    pipeline.fit(x_train, y_train)

    y_score = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_score)
    metrics["model_key"] = model_key
    metrics["train_rows"] = int(len(x_train))
    metrics["test_rows"] = int(len(x_test))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "model_key": model_key,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    calibration_table(y_test, y_score).to_csv(output_dir / "calibration_curve.csv", index=False)
    lift_chart_table(y_test, y_score).to_csv(output_dir / "lift_chart.csv", index=False)
    extract_feature_importance(pipeline).to_csv(output_dir / "feature_importance.csv", index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ad relevance and CTR prediction model.")
    parser.add_argument("--data-path", default=str(RAW_DATA_PATH))
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm", "neural_network"],
    )
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train(
        data_path=args.data_path,
        model_key=args.model,
        sample_size=args.sample_size,
        test_size=args.test_size,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
