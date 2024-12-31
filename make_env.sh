#!/bin/bash

# This script assumes that the Conda environment is already activated

cuda_version_arg=$1

echo "Selected CUDA version: $cuda_version_arg"

# Check the passed CUDA version and install the appropriate PyTorch version
if [[ "$cuda_version_arg" == "11" ]]; then
    echo "Installing PyTorch 2.4.0 for CUDA 11.8..."
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
    pip uninstall torchvision
    pip install torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
elif [[ "$cuda_version_arg" == "12" ]]; then
    echo "Installing PyTorch 2.4.0 for CUDA 12.1..."
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "Invalid CUDA version argument. Please pass either '11' or '12'."
    exit 1
fi

echo "PyTorch installation complete."

# Upgrade pip
pip install --upgrade pip

# Install additional packages
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com


# Run the isaaclab installation script
./isaaclab.sh --install