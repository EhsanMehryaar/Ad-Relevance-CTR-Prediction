from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import LABEL_COLUMN, MODELS_DIR, RAW_DATA_PATH
from src.data import clean_criteo_data, load_criteo_data, sample_data
from src.features import add_engineered_features
from src.metrics import calibration_table, compute_classification_metrics, lift_chart_table


def evaluate(
    data_path: str | Path = RAW_DATA_PATH,
    model_path: str | Path = MODELS_DIR / "model.joblib",
    sample_size: int | None = 100_000,
    output_dir: str | Path = MODELS_DIR,
) -> dict[str, float]:
    if sample_size is not None and sample_size <= 0:
        sample_size = None

    pipeline = joblib.load(model_path)
    df = load_criteo_data(data_path)
    df = clean_criteo_data(df)
    df = sample_data(df, sample_size=sample_size)
    df = add_engineered_features(df)

    y = df[LABEL_COLUMN]
    x = df.drop(columns=[LABEL_COLUMN])
    y_score = pipeline.predict_proba(x)[:, 1]

    metrics = compute_classification_metrics(y, y_score)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    calibration_table(y, y_score).to_csv(output_dir / "evaluation_calibration_curve.csv", index=False)
    lift_chart_table(y, y_score).to_csv(output_dir / "evaluation_lift_chart.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CTR prediction model.")
    parser.add_argument("--data-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--model-path", default=str(MODELS_DIR / "model.joblib"))
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate(
        data_path=args.data_path,
        model_path=args.model_path,
        sample_size=args.sample_size,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
