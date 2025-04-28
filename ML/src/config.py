# -*- coding: utf-8 -*-
from pathlib import Path

# --- Project Structure ---
# Assumes this file is in src/
# Resolve the project root directory (two levels up from src/config.py)
PROJECT_DIR = Path(__file__).resolve().parents[1]

# --- Data Paths ---
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "heloc_dataset_v1.csv"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed" # Optional: for saving intermediate data
MODELS_DIR = PROJECT_DIR / "models"

# --- Feature Engineering & Preprocessing ---
TARGET_VARIABLE = 'RiskPerformance'
SPECIAL_VALUES = [-9, -8, -7]
# Columns where special values might be particularly informative for indicators
INDICATOR_COLS = ['ExternalRiskEstimate', 'NumSatisfactoryTrades', 'NumTradesOpeninLast12M']

# --- Model Training & Tuning ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
POSITIVE_LABEL = 1 # Default: Assumes 'Bad' mapped to 1. Updated in preprocess_data if needed.
CV_FOLDS = 3 # Number of cross-validation folds for GridSearchCV
TUNING_SCORING = 'roc_auc' # Metric to optimize during tuning

# --- Logging ---
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE_PATH = PROJECT_DIR / "logs" / "model_training.log" # Define log file path

