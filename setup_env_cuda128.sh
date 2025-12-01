#!/bin/bash
# Setup script for Megatron-Bridge environment using uv
# Usage: bash setup_env_cuda128.sh

set -e  # Exit on error

# Suppress PyTorch pynvml deprecation warning
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda"

echo "=============================================="
echo "Setting up Megatron-Bridge environment with uv"
echo "=============================================="

ensure_uv_installed() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "Error: uv is not installed"
        echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    echo "Using uv: $(which uv)"
    echo "uv version: $(uv --version)"
}

create_virtualenv() {
    if [ ! -d "env" ]; then
        echo "Creating virtual environment..."
        uv venv env --python 3.11
    else
        echo "Virtual environment already exists"
    fi
}

activate_virtualenv() {
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source env/bin/activate
    echo "Python: $(which python)"
    echo "Python version: $(python --version)"
}

install_pytorch() {
    echo ""
    echo "Step 1/4: Installing PyTorch with CUDA 12.8 (cu128)..."
    uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
}

install_transformer_engine() {
    echo ""
    echo "Step 2/4: Installing transformer-engine[pytorch] for CUDA 12.8..."
    uv pip install --no-build-isolation transformer_engine[pytorch]
}

verify_pytorch_install() {
    echo ""
    echo "Verifying PyTorch installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
}

lock_torch_version() {
    echo ""
    echo "Step 3/4: Creating constraints file to prevent torch from being overwritten..."
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    TORCH_CONSTRAINTS=/tmp/torch_constraints.txt
    echo "torch==$TORCH_VERSION" > "$TORCH_CONSTRAINTS"
    echo "Torch version locked: $TORCH_VERSION"
}

install_remaining_requirements() {
    echo ""
    echo "Step 4/4: Installing all other dependencies from requirements_cuda128.txt..."
    echo "Note: Using constraint file to preserve PyTorch CUDA 12.8 version"
    uv pip install --constraint "$TORCH_CONSTRAINTS" -r requirements_cuda128.txt
}

verify_full_stack() {
    echo ""
    echo "=============================================="
    echo "Verifying installation..."
    echo "=============================================="

    python -c "
import torch
import transformer_engine
import megatron_bridge
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ CUDA reported version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')
print('✓ transformer-engine version:', getattr(transformer_engine, '__version__', 'unknown'))
print('✓ Megatron-Bridge version:', getattr(megatron_bridge, '__version__', 'unknown'))
print()
print('Testing Megatron-Bridge components...')
from megatron_bridge.config import ModelConfig
print('✓ Megatron-Bridge config import successful')
print()
print('All checks passed!')
"
}

print_completion_message() {
    echo ""
    echo "=============================================="
    echo "Setup completed successfully!"
    echo "=============================================="
    echo ""
    echo "To activate the environment:"
    echo "  source env/bin/activate"
    echo ""
    echo "Megatron-Bridge ready. Integrate with your training scripts via megatron_bridge APIs."
}

ensure_uv_installed
create_virtualenv
activate_virtualenv

echo ""
echo "=============================================="
echo "Installing dependencies..."
echo "=============================================="

install_pytorch
install_transformer_engine
verify_pytorch_install
lock_torch_version
install_remaining_requirements
verify_full_stack
print_completion_message
