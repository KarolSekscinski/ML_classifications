#!/bin/bash

# Ensure the script exits if any command fails
set -e

# --- Configuration ---

# Define the GCS bucket (replace with your actual bucket name)
GCS_BUCKET="licencjat_ml_classification"

# Define the base path for your code repository within the VM
REPO_PATH="ML_classifications/ML" # Adjust if your repo is cloned elsewhere

# Define the path to your requirements file (containing torch+cuda, rtdl, etc.)
REQUIREMENTS_FILE="$REPO_PATH/requirements.txt" # Adjust path as needed
REQUIREMENTS_FILE_GPU="$REPO_PATH/src/gcs/gpu.txt"

# Define GCS URIs for the metadata files
# Assumes the original preprocessing output (for MLP) is under 'NeurIPS/metadata'
METADATA_URI_OHE="gs://$GCS_BUCKET/NeurIPS/metadata/preprocessing_metadata.json"
# Assumes the FT-Transformer specific preprocessing output is under 'NeurIPS/metadata'
METADATA_URI_FT="gs://$GCS_BUCKET/NeurIPS/metadata/preprocessing_ft_metadata.json" # Adjust path if different

# Define base GCS prefixes for results
RESULTS_BASE_PREFIX="NeurIPS/results" # Store GPU results separately

# Define paths to the python pipeline scripts
MLP_SCRIPT_PATH="$REPO_PATH/src/gcs/mlp_pipeline.py" # Adjust path as needed
FT_SCRIPT_PATH="$REPO_PATH/src/gcs/ft_transformer_pipeline.py" # Adjust path as needed

# --- Environment Setup ---

echo "Updating package list..."
sudo apt update

echo "Ensuring necessary system packages are installed (python3-venv, pip)..."
# python3-pip might already be there, but ensure python3-venv is present
sudo apt install -y python3-venv python3-pip

# Create a Python virtual environment (using python3 assumes it's the desired version, e.g., 3.10 on the DL VM)
VENV_NAME="gpu_venv_$(date +%Y%m%d)"
echo "Creating Python virtual environment: $VENV_NAME ..."
python3 -m venv $VENV_NAME

# Activate the virtual environment
echo "Activating the Python virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip within the venv
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing Python packages from $REQUIREMENTS_FILE ..."
# Ensure requirements.txt specifies torch with CUDA support compatible with the VM's CUDA version (e.g., cu121)
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
    pip install -r "$REQUIREMENTS_FILE_GPU"
else
    echo "ERROR: Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi

# Optional: Verify PyTorch and CUDA setup
echo "Verifying PyTorch CUDA setup..."
python -c "import torch; print(f'--- PyTorch Info ---'); print(f'Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]; print(f'--------------------')"
if ! python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    echo "WARNING: PyTorch cannot detect CUDA. GPU acceleration will not be available."
    # Decide if you want to exit or continue on CPU
    # exit 1
fi

# --- Run GPU Pipelines ---

echo "Starting MLP pipeline execution..."
python "$MLP_SCRIPT_PATH" \
    --gcs-bucket "$GCS_BUCKET" \
    --metadata-uri "$METADATA_URI_OHE" \
    --gcs-output-prefix "$RESULTS_BASE_PREFIX/mlp_run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 50 \
    --batch-size 1024 \
    --learning-rate 1e-4 \
    --mlp-hidden-dims "256,128,64" \
    --run-shap # Add this flag to compute SHAP values

echo "Finished MLP pipeline execution."
echo "------------------------------------"


echo "Starting FT-Transformer pipeline execution..."
# Ensure the FT metadata URI points to the output of preprocess_ft.py
python "$FT_SCRIPT_PATH" \
    --gcs-bucket "$GCS_BUCKET" \
    --metadata-uri "$METADATA_URI_FT" \
    --gcs-output-prefix "$RESULTS_BASE_PREFIX/ft_transformer_run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 50 \
    --batch-size 512 \
    --learning-rate 1e-4 \
    --ft-n-blocks 3 \
    --ft-d-token 192 \
    --run-shap # Optional and experimental

echo "Finished FT-Transformer pipeline execution."
echo "------------------------------------"


# --- Cleanup ---

# Deactivate the virtual environment (optional, script will exit anyway)
# echo "Deactivating virtual environment..."
# deactivate

echo "GPU pipeline execution script finished!"