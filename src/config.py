from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "criteo_train.txt"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "criteo_sample.txt"
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_COLUMN = "label"
INTEGER_FEATURES = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_FEATURES = [f"C{i}" for i in range(1, 27)]
CRITEO_COLUMNS = [LABEL_COLUMN, *INTEGER_FEATURES, *CATEGORICAL_FEATURES]

DEFAULT_RANDOM_STATE = 42
