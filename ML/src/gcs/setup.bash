#!/bin/bash

set -e

# --- Configuration ---
PYTHON_VERSION="3.10.0"
PYTHON_COMMAND="python3.10"
VENV_NAME="my_ml_venv_tuned_$(date +%Y%m%d)"
REPO_PATH="ML_classifications/ML"
REQUIREMENTS_FILE="$REPO_PATH/requirements.txt" # Ensure optuna is in this file
GCS_BUCKET="licencjat_ml_classification"
METADATA_URI="gs://$GCS_BUCKET/NeurIPS/metadata/preprocessing_metadata.json"
RESULTS_BASE_PREFIX="NeurIPS/results_tuned" # New prefix for tuned results

# --- Python Installation (if needed) ---
echo "Checking if $PYTHON_COMMAND is installed..."
if ! command -v $PYTHON_COMMAND &> /dev/null; then
    echo "$PYTHON_COMMAND not found. Proceeding with installation..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz -O Python-$PYTHON_VERSION.tgz
    tar -xzf Python-$PYTHON_VERSION.tgz
    cd Python-$PYTHON_VERSION
    ./configure --enable-optimizations --with-ensurepip=install
    make -j $(nproc)
    sudo make altinstall
    cd ..
    rm Python-$PYTHON_VERSION.tgz
    rm -rf Python-$PYTHON_VERSION
    if ! command -v $PYTHON_COMMAND &> /dev/null; then echo "ERROR: Python installation failed."; exit 1; fi
    echo "Python $PYTHON_VERSION installed successfully."
else
    echo "$PYTHON_COMMAND found. Skipping installation."
    sudo apt update
    sudo apt install -y python3-venv python3-pip # Ensure these are present
fi
$PYTHON_COMMAND --version

# --- Setup Virtual Environment and Install Requirements ---
echo "Creating Python virtual environment ($VENV_NAME) using $PYTHON_COMMAND..."
if [ -d "$VENV_NAME" ]; then rm -rf "$VENV_NAME"; fi
$PYTHON_COMMAND -m venv $VENV_NAME
source $VENV_NAME/bin/activate
pip install --upgrade pip
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE" # Make sure optuna, scikit-learn, xgboost are listed
else
    echo "ERROR: Requirements file not found at $REQUIREMENTS_FILE"; deactivate; exit 1;
fi
echo "Python environment setup complete!"

# --- Run Training Scripts with Tuning ---
SVM_SCRIPT_PATH="$REPO_PATH/src/gcs/svm_pipeline.py"
SVM_OUTPUT_PREFIX="$RESULTS_BASE_PREFIX/svm_run_$(date +%Y%m%d_%H%M%S)"

echo "Running SVM pipeline with tuning..."
if [ -f "$SVM_SCRIPT_PATH" ]; then
    python "$SVM_SCRIPT_PATH" \
        --gcs-bucket "$GCS_BUCKET" \
        --metadata-uri "$METADATA_URI" \
        --gcs-output-prefix "$SVM_OUTPUT_PREFIX" \
        --n-trials 10 \
        --svm-max-iter 1000 # Max iterations for LinearSVC used in trials & final model
else
    echo "ERROR: SVM pipeline script not found at $SVM_SCRIPT_PATH"; deactivate; exit 1;
fi
echo "Finished SVM training."
echo "------------------------------------"


XGB_SCRIPT_PATH="$REPO_PATH/src/gcs/xgboost_pipeline.py"
XGB_OUTPUT_PREFIX="$RESULTS_BASE_PREFIX/xgb_run_$(date +%Y%m%d_%H%M%S)"

echo "Running XGBoost pipeline with tuning..."
if [ -f "$XGB_SCRIPT_PATH" ]; then
    python "$XGB_SCRIPT_PATH" \
        --gcs-bucket "$GCS_BUCKET" \
        --metadata-uri "$METADATA_URI" \
        --gcs-output-prefix "$XGB_OUTPUT_PREFIX" \
        --n-trials 20 \
        --optimization-metric "aucpr" \
        --xgb-early-stopping-rounds 10
else
     echo "ERROR: XGBoost pipeline script not found at $XGB_SCRIPT_PATH"; deactivate; exit 1;
fi
echo "Finished XGBoost Training."
echo "------------------------------------"

echo "Script finished successfully!"