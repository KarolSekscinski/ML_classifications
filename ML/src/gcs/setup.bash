#!/bin/bash

# Script to run the fraud detection pipeline, managing a virtual environment,
# and then stop the VM.

# --- Configuration ---
PYTHON_SCRIPT_NAME="main.py"
VM_NAME="your-vm-name"                 # <--- REPLACE with your VM's name
VM_ZONE="your-vm-zone"                 # <--- REPLACE with your VM's zone (e.g., us-central1-a)
LOG_FILE="/app/pipeline_execution.log" # Path to a log file for the python script's output
VENV_DIR="/app/venv"                   # Directory for the virtual environment

# --- Ensure script is executable ---
# chmod +x /path/to/this_script.sh

echo "----------------------------------------------------"
echo "Starting Fraud Detection Pipeline Execution"
echo "Timestamp: $(date)"
echo "Python script: ${PYTHON_SCRIPT_NAME}"
echo "Virtual Environment: ${VENV_DIR}"
echo "----------------------------------------------------"

# --- Setup Virtual Environment ---
# Check if python3-venv is installed (apt-based systems)
# In a custom Docker image, this should ideally be part of the Dockerfile.
# If not running in Docker and it might be missing:
# if ! dpkg -s python3-venv > /dev/null 2>&1; then
#   echo "python3-venv not found. Attempting to install..."
#   apt-get update && apt-get install -y python3-venv
#   if [ $? -ne 0 ]; then
#     echo "ERROR: Failed to install python3-venv. Please install it manually and re-run."
#     exit 1
#   fi
# fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating Python virtual environment at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment."
    exit 1
  fi
else
  echo "Virtual environment already exists at ${VENV_DIR}."
fi

echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate virtual environment."
  exit 1
fi

echo "Installing dependencies from requirements.txt into virtual environment..."
pip install --no-cache-dir -r requirements.txt
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to install dependencies from requirements.txt."
  # Consider deactivating venv before exiting if appropriate, though script will end.
  exit 1
fi
echo "Dependencies installed."
echo "----------------------------------------------------"


# Run the Python script using the venv's python
echo "Executing Python script: ${PYTHON_SCRIPT_NAME} using python from venv..."
# The 'python3' command should now point to the venv's python due to activation
python3 "${PYTHON_SCRIPT_NAME}" > >(tee -a "${LOG_FILE}") 2> >(tee -a "${LOG_FILE}" >&2)

# Capture the exit code of the Python script
PYTHON_EXIT_CODE=$?

# Deactivate virtual environment (good practice, though script might end soon)
echo "Deactivating virtual environment..."
deactivate

if [ ${PYTHON_EXIT_CODE} -eq 0 ]; then
  echo "----------------------------------------------------"
  echo "Python script '${PYTHON_SCRIPT_NAME}' completed successfully."
  echo "Timestamp: $(date)"
  echo "----------------------------------------------------"
else
  echo "----------------------------------------------------"
  echo "ERROR: Python script '${PYTHON_SCRIPT_NAME}' failed with exit code ${PYTHON_EXIT_CODE}."
  echo "Timestamp: $(date)"
  echo "Check logs at ${LOG_FILE} for details."
  echo "----------------------------------------------------"
fi

# Regardless of Python script success or failure, attempt to stop the VM.
echo "----------------------------------------------------"
echo "Attempting to stop the VM: ${VM_NAME} in zone ${VM_ZONE}"
echo "Timestamp: $(date)"
echo "----------------------------------------------------"

# Command to stop the VM
# Ensure gcloud is configured and has permissions.
gcloud compute instances stop "${VM_NAME}" --zone="${VM_ZONE}" --quiet

# Check the exit code of the gcloud command (optional)
GCLOUD_EXIT_CODE=$?
if [ ${GCLOUD_EXIT_CODE} -eq 0 ]; then
  echo "VM stop command issued successfully for ${VM_NAME}."
  # Note: The script might terminate here as the VM shuts down.
else
  echo "ERROR: Failed to issue VM stop command for ${VM_NAME}. Exit code: ${GCLOUD_EXIT_CODE}."
  echo "The VM might still be running. Check Google Cloud Console."
fi

# The script might not reach here if the VM stops quickly.
echo "Script finished."
