#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_VERSION="3.10.0"
PYTHON_COMMAND="python3.10" # The command we expect after installation/if already present
VENV_NAME="my_ml_venv_$(date +%Y%m%d)" # Virtual environment name (includes date)
REPO_PATH="ML_classifications/ML" # Adjust path to your repository if needed
REQUIREMENTS_FILE="$REPO_PATH/requirements.txt" # Adjust path to requirements if needed
GCS_BUCKET="licencjat_ml_classification" # Your GCS bucket
METADATA_URI="gs://$GCS_BUCKET/FinBench/metadata/preprocessing_ohe_metadata.json" # Metadata path
RESULTS_BASE_PREFIX="FinBench/results" # Base path for results

# --- Check for Python 3.10 ---
echo "Checking if $PYTHON_COMMAND is installed..."

if ! command -v $PYTHON_COMMAND &> /dev/null
then
    # --- Install Python 3.10 if not found ---
    echo "$PYTHON_COMMAND not found. Proceeding with installation..."

    echo "Updating and upgrading the system..."
    sudo apt update && sudo apt upgrade -y

    echo "Installing required build dependencies for Python..."
    # curl is added as it's generally useful, wget is used below
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
        libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl

    echo "Downloading Python $PYTHON_VERSION source code..."
    # Use -O to ensure the output filename is predictable
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz -O Python-$PYTHON_VERSION.tgz

    echo "Extracting Python $PYTHON_VERSION source code..."
    # Add 'z' flag for gzipped tar files (.tgz)
    tar -xzf Python-$PYTHON_VERSION.tgz

    echo "Building and installing Python $PYTHON_VERSION..."
    cd Python-$PYTHON_VERSION
    # Add --with-ensurepip=install to make sure pip is available for the new installation
    ./configure --enable-optimizations --with-ensurepip=install
    # Use make -j $(nproc) to speed up compilation using all available cores
    make -j $(nproc)
    # Use altinstall to avoid overwriting the default 'python3'
    sudo make altinstall
    cd .. # Go back to the original directory before removing files

    echo "Cleaning up Python source files..."
    rm Python-$PYTHON_VERSION.tgz
    rm -rf Python-$PYTHON_VERSION # Remove the extracted directory

    echo "Verifying Python installation..."
    # Re-check if the command is now available
    if ! command -v $PYTHON_COMMAND &> /dev/null
    then
        echo "ERROR: $PYTHON_COMMAND installation appears to have failed or is not in PATH."
        exit 1
    fi
    $PYTHON_COMMAND --version
    echo "Python $PYTHON_VERSION installed successfully."

else
    # --- Python 3.10 Found ---
    echo "$PYTHON_COMMAND found. Skipping installation."
    $PYTHON_COMMAND --version
    # Even if Python exists, make sure tools for venv/pip are installed system-wide if needed
    echo "Ensuring python3-venv and python3-pip packages are installed..."
    sudo apt update # Update list just in case
    sudo apt install -y python3-venv python3-pip
fi

# --- Setup Virtual Environment and Install Requirements (Common Path) ---

echo "Creating Python virtual environment ($VENV_NAME) using $PYTHON_COMMAND..."
# Remove existing venv with the same name if it exists, to ensure a clean environment
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
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "ERROR: Requirements file not found at $REQUIREMENTS_FILE"
    # Deactivate before exiting if requirements are missing
    deactivate
    exit 1
fi

echo "Python environment setup complete!"

# --- Run Training Scripts ---

echo "Running svm pipeline..."
# Use variables for paths and arguments for clarity
SVM_SCRIPT_PATH="$REPO_PATH/src/gcs/finbench_svm.py"
SVM_OUTPUT_PREFIX="$RESULTS_BASE_PREFIX/svm_run_$(date +%Y%m%d_%H%M%S)"

if [ -f "$SVM_SCRIPT_PATH" ]; then
    python "$SVM_SCRIPT_PATH" \
        --gcs-bucket "$GCS_BUCKET" \
        --metadata-uri "$METADATA_URI" \
        --gcs-output-prefix "$SVM_OUTPUT_PREFIX" \
        --run-shap # Add this flag to compute SHAP values (will be slow)
        # --svm-c 10 # Example: Override default hyperparameter
else
    echo "ERROR: SVM pipeline script not found at $SVM_SCRIPT_PATH"
    deactivate
    exit 1
fi
echo "Finished SVM training."
echo "------------------------------------"


echo "Running xgboost pipeline..."
XGB_SCRIPT_PATH="$REPO_PATH/src/gcs/finbench_xgboost.py"
XGB_OUTPUT_PREFIX="$RESULTS_BASE_PREFIX/xgb_run_$(date +%Y%m%d_%H%M%S)"

if [ -f "$XGB_SCRIPT_PATH" ]; then
    python "$XGB_SCRIPT_PATH" \
        --gcs-bucket "$GCS_BUCKET" \
        --metadata-uri "$METADATA_URI" \
        --gcs-output-prefix "$XGB_OUTPUT_PREFIX" \
        --run-shap # Add this flag to compute SHAP values
        # --xgb-n-estimators 1000 # Example: Override default hyperparameter
else
     echo "ERROR: XGBoost pipeline script not found at $XGB_SCRIPT_PATH"
     deactivate
     exit 1
fi
echo "Finished XGBoost Training."
echo "------------------------------------"

# Optional: Deactivate environment at the end of the script
# echo "Deactivating virtual environment..."
# deactivate

echo "Script finished successfully!"