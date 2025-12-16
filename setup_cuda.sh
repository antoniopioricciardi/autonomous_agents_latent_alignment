#!/bin/bash
# Setup script for Linux with CUDA support

set -e

echo "Setting up environment with CUDA support..."

# Sync all dependencies except PyTorch
uv sync

# Activate environment
source .venv/bin/activate

# Uninstall CPU PyTorch
pip uninstall -y torch torchvision

# Install CUDA PyTorch (choose your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "✓ Environment ready with CUDA support!"
