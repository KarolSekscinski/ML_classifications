#!/bin/bash

# Ensure the script exits if any command fails
set -e

# --- Configuration ---

# Define the GCS bucket (replace with your actual bucket name)
GCS_BUCKET="licencjat_ml_classification"

# Define the base path for your code repository within the VM
REPO_PATH="ML_classifications/ML" # Adjust if your repo is cloned elsewhere

# Define the path to your requirements file (containing torch+cuda, rtdl, optuna etc.)
REQUIREMENTS_FILE="$REPO_PATH/requirements.txt" # Adjust path as needed
REQUIREMENTS_FILE_GPU="$REPO_PATH/gpu.txt" # Ensure optuna is in one of these

# Define GCS URIs for the metadata files
METADATA_URI_OHE="gs://$GCS_BUCKET/FinBench/metadata/preprocessing_ohe_metadata.json"
METADATA_URI_FT="gs://$GCS_BUCKET/FinBench/metadata/preprocessing_ft_metadata.json"

# Define base GCS prefixes for results
RESULTS_BASE_PREFIX="FinBench/results" # Store GPU results separately

# Define paths to the python pipeline scripts
MLP_SCRIPT_PATH="$REPO_PATH/src/gcs/finbench_mlp.py"
FT_SCRIPT_PATH="$REPO_PATH/src/gcs/finbench_ft_transformer.py"

# --- Environment Setup ---

echo "Updating package list..."
sudo apt update

echo "Ensuring necessary system packages are installed (python3-venv, pip)..."
sudo apt install -y python3-venv python3-pip

VENV_NAME="gpu_venv_$(date +%Y%m%d)"
echo "Creating Python virtual environment: $VENV_NAME ..."
python3 -m venv $VENV_NAME

echo "Activating the Python virtual environment..."
source $VENV_NAME/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python packages from $REQUIREMENTS_FILE and $REQUIREMENTS_FILE_GPU..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "ERROR: Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi
if [ -f "$REQUIREMENTS_FILE_GPU" ]; then
    pip install -r "$REQUIREMENTS_FILE_GPU" # Make sure optuna is listed here or in the main one
else
    echo "ERROR: GPU Requirements file not found at $REQUIREMENTS_FILE_GPU"
    exit 1
fi


# Optional: Verify PyTorch and CUDA setup
echo "Verifying PyTorch CUDA setup..."
python -c "import torch; print(f'--- PyTorch Info ---'); print(f'Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]; print(f'--------------------')"
if ! python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    echo "WARNING: PyTorch cannot detect CUDA. GPU acceleration will not be available."
fi

# --- Run GPU Pipelines ---

echo "Starting MLP pipeline execution (with Hyperparameter Tuning)..."
python "$MLP_SCRIPT_PATH" \
    --gcs-bucket "$GCS_BUCKET" \
    --metadata-uri "$METADATA_URI_OHE" \
    --gcs-output-prefix "$RESULTS_BASE_PREFIX/mlp_tuned_run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 50 \
    --batch-size 512 \
    --learning-rate 1e-4 \
    --mlp-hidden-dims "256,128,64"
    # Removed --run-shap. Other args might be used as defaults or for trials by the script.

echo "Finished MLP pipeline execution."
echo "------------------------------------"


echo "Starting FT-Transformer pipeline execution (with Hyperparameter Tuning)..."
python "$FT_SCRIPT_PATH" \
    --gcs-bucket "$GCS_BUCKET" \
    --metadata-uri "$METADATA_URI_FT" \
    --gcs-output-prefix "$RESULTS_BASE_PREFIX/ft_transformer_tuned_run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 30 \
    --batch-size 256 \
    --learning-rate 1e-4 \
    --ft-n-blocks 3 \
    --ft-d-token 192
    # Removed --run-shap. Other args might be used as defaults or for trials by the script.

echo "Finished FT-Transformer pipeline execution."
echo "------------------------------------"

echo "GPU pipeline execution script finished!"