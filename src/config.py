"""Paths and colours shared by the dashboard, the tests and the notebooks."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "docs" / "figures"

CLEAN_DATA_PATH = DATA_DIR / "clean_sessions.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

FEATURE_SETTINGS_PATH = MODEL_DIR / "feature_columns.json"
REGRESSOR_PATH = MODEL_DIR / "best_regressor.pkl"
CLASSIFIER_PATH = MODEL_DIR / "best_classifier.pkl"

ACCENT = "#00895C"
SECONDARY = "#2F6F8F"
WARNING = "#E4572E"
