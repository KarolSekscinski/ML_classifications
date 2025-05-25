#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_VERSION="3.10.0"
PYTHON_COMMAND="python3.10"
VENV_NAME="my_ml_venv_$(date +%Y%m%d)"
REPO_PATH="ML_classifications/ML"
REQUIREMENTS_FILE="$REPO_PATH/requirements.txt" # Ensure optuna is in this file
GCS_BUCKET="licencjat_ml_classification"
METADATA_URI="gs://$GCS_BUCKET/FinBench/metadata/preprocessing_ohe_metadata.json"
RESULTS_BASE_PREFIX="FinBench/results"

# --- Check for Python 3.10 ---
echo "Checking if $PYTHON_COMMAND is installed..."

if ! command -v $PYTHON_COMMAND &> /dev/null
then
    echo "$PYTHON_COMMAND not found. Proceeding with installation..."
    echo "Updating and upgrading the system..."
    sudo apt update && sudo apt upgrade -y
    echo "Installing required build dependencies for Python..."
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl
    echo "Downloading Python $PYTHON_VERSION source code..."
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz -O Python-$PYTHON_VERSION.tgz
    echo "Extracting Python $PYTHON_VERSION source code..."
    tar -xzf Python-$PYTHON_VERSION.tgz
    echo "Building and installing Python $PYTHON_VERSION..."
    cd Python-$PYTHON_VERSION
    ./configure --enable-optimizations --with-ensurepip=install
    make -j $(nproc)
    sudo make altinstall
    cd ..
    echo "Cleaning up Python source files..."
    rm Python-$PYTHON_VERSION.tgz
    rm -rf Python-$PYTHON_VERSION
    echo "Verifying Python installation..."
    if ! command -v $PYTHON_COMMAND &> /dev/null
    then
        echo "ERROR: $PYTHON_COMMAND installation appears to have failed or is not in PATH."
        exit 1
    fi
    $PYTHON_COMMAND --version
    echo "Python $PYTHON_VERSION installed successfully."
else
    echo "$PYTHON_COMMAND found. Skipping installation."
    $PYTHON_COMMAND --version
    echo "Ensuring python3-venv and python3-pip packages are installed..."
    sudo apt update
    sudo apt install -y python3-venv python3-pip
fi

# --- Setup Virtual Environment and Install Requirements (Common Path) ---
echo "Creating Python virtual environment ($VENV_NAME) using $PYTHON_COMMAND..."
if [ -d "$VENV_NAME" ]; then
    echo "Removing existing virtual environment: $VENV_NAME"
    rm -rf "$VENV_NAME"
fi
$PYTHON_COMMAND -m venv $VENV_NAME

echo "Activating the Python virtual environment..."
source $VENV_NAME/bin/activate

echo "Upgrading pip within virtual environment..."
pip install --upgrade pip

echo "Installing Python packages from $REQUIREMENTS_FILE..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE" # Make sure optuna is listed here
else
    echo "ERROR: Requirements file not found at $REQUIREMENTS_FILE"
    deactivate
    exit 1
fi
echo "Python environment setup complete!"

# --- Run Training Scripts ---

#echo "Running svm pipeline (with Hyperparameter Tuning)..."
#SVM_SCRIPT_PATH="$REPO_PATH/src/gcs/finbench_svm.py"
#SVM_OUTPUT_PREFIX="$RESULTS_BASE_PREFIX/svm_tuned_run_$(date +%Y%m%d_%H%M%S)"
#
#if [ -f "$SVM_SCRIPT_PATH" ]; then
#    python "$SVM_SCRIPT_PATH" \
#        --gcs-bucket "$GCS_BUCKET" \
#        --metadata-uri "$METADATA_URI" \
#        --gcs-output-prefix "$SVM_OUTPUT_PREFIX"
#        # Removed --run-shap.
#        # --svm-c 10 # This might be used as a default or ignored if C is tuned.
#else
#    echo "ERROR: SVM pipeline script not found at $SVM_SCRIPT_PATH"
#    deactivate
#    exit 1
#fi
#echo "Finished SVM training."
#echo "------------------------------------"


echo "Running xgboost pipeline (with Hyperparameter Tuning)..."
XGB_SCRIPT_PATH="$REPO_PATH/src/gcs/finbench_xgboost.py"
XGB_OUTPUT_PREFIX="$RESULTS_BASE_PREFIX/xgb_tuned_run_$(date +%Y%m%d_%H%M%S)"

if [ -f "$XGB_SCRIPT_PATH" ]; then
    python "$XGB_SCRIPT_PATH" \
        --gcs-bucket "$GCS_BUCKET" \
        --metadata-uri "$METADATA_URI" \
        --gcs-output-prefix "$XGB_OUTPUT_PREFIX"
        # Removed --run-shap.
        # --xgb-n-estimators 1000 # This might be used as a default or max for trials.
else
     echo "ERROR: XGBoost pipeline script not found at $XGB_SCRIPT_PATH"
     deactivate
     exit 1
fi
echo "Finished XGBoost Training."
echo "------------------------------------"

echo "Script finished successfully!"