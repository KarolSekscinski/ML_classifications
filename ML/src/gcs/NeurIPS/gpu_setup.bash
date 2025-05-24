# gcs/gpu_setup.bash
#!/bin/bash

# Ensure the script exits if any command fails
set -e

# --- Configuration ---
GCS_BUCKET="licencjat_ml_classification"
REPO_PATH="ML_classifications/ML"
REQUIREMENTS_FILE="$REPO_PATH/requirements.txt"
REQUIREMENTS_FILE_GPU="$REPO_PATH/gpu.txt" # Assuming optuna will be in requirements.txt or gpu.txt

METADATA_URI_OHE="gs://$GCS_BUCKET/NeurIPS/metadata/preprocessing_metadata.json"
METADATA_URI_FT="gs://$GCS_BUCKET/NeurIPS/metadata/preprocessing_ft_metadata.json"

RESULTS_BASE_PREFIX="NeurIPS/results_tuned" # New prefix for tuned results

MLP_SCRIPT_PATH="$REPO_PATH/src/gcs/mlp_pipeline.py"
FT_SCRIPT_PATH="$REPO_PATH/src/gcs/ft_transformer_pipeline.py"

# --- Environment Setup ---
echo "Updating package list..."
sudo apt update
echo "Ensuring necessary system packages are installed (python3-venv, pip)..."
sudo apt install -y python3-venv python3-pip

VENV_NAME="gpu_venv_tuned_$(date +%Y%m%d)"
echo "Creating Python virtual environment: $VENV_NAME ..."
python3 -m venv $VENV_NAME
echo "Activating the Python virtual environment..."
source $VENV_NAME/bin/activate
echo "Upgrading pip..."
pip install --upgrade pip
echo "Installing Python packages..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
    # Ensure optuna is listed in requirements.txt or gpu.txt
    # Example: pip install optuna torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Make sure rtdl, torchmetrics, etc. are also in requirements
    if [ -f "$REQUIREMENTS_FILE_GPU" ]; then
        pip install -r "$REQUIREMENTS_FILE_GPU"
    fi
else
    echo "ERROR: Requirements file not found at $REQUIREMENTS_FILE"
    exit 1
fi

echo "Verifying PyTorch CUDA setup..."
python -c "import torch; print(f'--- PyTorch Info ---'); print(f'Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]; print(f'--------------------')"
if ! python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    echo "WARNING: PyTorch cannot detect CUDA. GPU acceleration will not be available."
fi

# --- Run GPU Pipelines with Tuning ---

echo "Starting MLP pipeline execution with tuning..."
python "$MLP_SCRIPT_PATH" \
    --gcs-bucket "$GCS_BUCKET" \
    --metadata-uri "$METADATA_URI_OHE" \
    --gcs-output-prefix "$RESULTS_BASE_PREFIX/mlp_run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 15 \ # Epochs per trial and for final model
    --n-trials 20 \ # Number of Optuna trials
    --optimization-metric "pr_auc" \
    --batch-size-default 512 \
    --num-workers 2
echo "Finished MLP pipeline execution."
echo "------------------------------------"


echo "Starting FT-Transformer pipeline execution with tuning..."
python "$FT_SCRIPT_PATH" \
    --gcs-bucket "$GCS_BUCKET" \
    --metadata-uri "$METADATA_URI_FT" \
    --gcs-output-prefix "$RESULTS_BASE_PREFIX/ft_transformer_run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 10 \ # Epochs per trial and for final model
    --n-trials 15 \ # Number of Optuna trials
    --optimization-metric "pr_auc" \
    --batch-size-default 256 \
    --num-workers 2
echo "Finished FT-Transformer pipeline execution."
echo "------------------------------------"

echo "GPU pipeline execution script finished!"