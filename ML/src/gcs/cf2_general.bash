#!/bin/bash

# Get the directory where this script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT1="$DIR/gpu_setup.sh"
SCRIPT2="$DIR/cf2_gpu_setup.sh"

# Check if scripts are executable
if [[ ! -x "$SCRIPT1" ]]; then
  echo "Error: $SCRIPT1 is not executable or not found."
  exit 1
fi

if [[ ! -x "$SCRIPT2" ]]; then
  echo "Error: $SCRIPT2 is not executable or not found."
  exit 1
fi

# Run the scripts
echo "Running $SCRIPT1..."
"$SCRIPT1"

echo "Running $SCRIPT2..."
"$SCRIPT2"

echo "Both scripts have been executed."
