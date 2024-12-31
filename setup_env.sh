#!/bin/bash


# Check if an argument is passed
if [ -z "$1" ]; then
    echo "No CUDA version argument provided. Please pass either '11' or '12' as the CUDA version."
else
    # Set the argument as the CUDA version
    cuda_version_arg=$1

    # Source the Conda profile script
    source ~/miniconda3/etc/profile.d/conda.sh
    # Create the Conda environment if it doesn't exist
    if ! conda info --envs | grep -q "isaaclab"; then
        echo "Creating conda environment 'isaaclab'..."
        conda create -n isaaclab python=3.10 -y
        conda install -c conda-forge libstdcxx-ng
    fi

    # Activate the Conda environment
    conda activate isaaclab

    echo "Conda environment 'isaaclab' is now activated."

    # After activation, source the second script for package installation
    source ./make_env.sh $cuda_version_arg
fi
