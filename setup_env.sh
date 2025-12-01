#!/bin/bash
# Setup script for NeMo environment using uv
# Usage: bash setup_env.sh

set -e  # Exit on error

# Suppress PyTorch pynvml deprecation warning
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda"

echo "=============================================="
echo "Setting up NeMo environment with uv"
echo "=============================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Using uv: $(which uv)"
echo "uv version: $(uv --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "Creating virtual environment..."
    uv venv env --python 3.11
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies in correct order
echo ""
echo "=============================================="
echo "Installing dependencies..."
echo "=============================================="

# Step 1: Install PyTorch with CUDA support FIRST
# This MUST be done before other packages to ensure correct CUDA version
echo ""
echo "Step 1/3: Installing PyTorch with CUDA 12.1 (cu121)..."
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch
# uv pip install transformer-engine[pytorch] \
#     --extra-index-url https://pypi.nvidia.com \
#     --find-links https://github.com/NVIDIA/TransformerEngine/releases

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Step 2: Create constraints file to lock torch version
echo ""
echo "Step 2/3: Creating constraints file to prevent torch from being overwritten..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VERSION" > /tmp/torch_constraints.txt
echo "Torch version locked: $TORCH_VERSION"

# Step 3: Install all other dependencies from requirements.txt
# Now that PyTorch is installed with correct CUDA support, install everything else
# Use --constraint to prevent torch from being reinstalled
echo ""
echo "Step 3/3: Installing all other dependencies from requirements.txt..."
echo "Note: Using constraint file to preserve PyTorch CUDA 12.1 version"
uv pip install --constraint /tmp/torch_constraints.txt -r requirements.txt

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

# Verify key imports
python -c "
import torch
import nemo
import modelopt
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ NeMo version:', nemo.__version__)
print('✓ nvidia-modelopt installed')
print()
print('Testing NeMo imports...')
from nemo.collections.llm.recipes import gemma3_1b
print('✓ NeMo LLM recipes imported successfully')
print()
print('All checks passed!')
"

echo ""
echo "=============================================="
echo "Setup completed successfully!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source env/bin/activate"
echo ""
echo "To run training:"
echo "  python scripts/run_cpt_gemma3_1b.py --help"
echo ""
