from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CRITEO_COLUMNS, LABEL_COLUMN


def load_criteo_data(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw Criteo CTR dataset with canonical column names."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Criteo train file there, "
            "or pass --data-path to a local sample."
        )

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=CRITEO_COLUMNS,
        nrows=nrows,
        low_memory=False,
    )


def clean_criteo_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight schema validation and label cleanup."""
    missing = set(CRITEO_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=[LABEL_COLUMN])
    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype(int)
    cleaned = cleaned[cleaned[LABEL_COLUMN].isin([0, 1])]
    return cleaned


def sample_data(
    df: pd.DataFrame,
    sample_size: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a reproducible sample when requested."""
    if sample_size is None or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=random_state)
