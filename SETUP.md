# Environment Setup

## Prerequisites

- Python 3.11+
- CUDA 12.1 (available on Hopper cluster)
- `uv` package manager

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
bash setup_env.sh
```

This will:
1. Create a virtual environment (`env/`)
2. Install PyTorch with CUDA 12.1
3. Create a constraint file to lock the PyTorch version
4. Install NeMo toolkit and all other dependencies
5. Verify the installation

### Option 2: Manual Setup with uv

```bash
# Create virtual environment
uv venv env --python 3.11
source env/bin/activate

# Install PyTorch with CUDA 12.1 support
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch

# Create constraint file to prevent torch from being overwritten
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VERSION" > /tmp/torch_constraints.txt

# Install dependencies with constraint to preserve PyTorch version
uv pip install --constraint /tmp/torch_constraints.txt -r requirements.txt
```

### Option 3: Using pip (if uv is not available)

```bash
python -m venv env
source env/bin/activate

# Install PyTorch with CUDA support
pip install --index-url https://download.pytorch.org/whl/cu121 torch

# Create constraint file to lock torch version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VERSION" > /tmp/torch_constraints.txt

# Install other dependencies
pip install --constraint /tmp/torch_constraints.txt -r requirements.txt
```

## Verification

After installation, verify everything is working:

```bash
source env/bin/activate

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check NeMo
python -c "import nemo; print(f'NeMo: {nemo.__version__}')"

# Check NeMo recipes
python -c "from nemo.collections.llm.recipes import gemma3_1b; print('✓ NeMo recipes working')"
```

## Common Issues

### Issue 1: PyTorch gets overwritten during installation

**Cause**: Dependencies trying to reinstall PyTorch from PyPI

**Solution**: The setup script now uses a constraint file to prevent this. If installing manually:
```bash
# Install PyTorch first
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch

# Lock the version
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VERSION" > /tmp/torch_constraints.txt

# Install everything else with constraint
uv pip install --constraint /tmp/torch_constraints.txt -r requirements.txt
```

### Issue 2: CUDA version mismatch

**Cause**: PyTorch CUDA version doesn't match your GPU drivers

**Check your CUDA version**:
```bash
nvidia-smi  # Look for "CUDA Version: X.Y"
```

**Install matching PyTorch**:
- For CUDA 12.1 (Hopper): `--index-url https://download.pytorch.org/whl/cu121` ← **Use this**
- For CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`
- For CUDA 12.6: `--index-url https://download.pytorch.org/whl/cu126`
- For CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`

### Issue 3: NeMo 2.5.x not found

**Cause**: Old pip version can't see newer packages on PyPI

**Solution**: Use `uv` which has modern package resolution, or upgrade pip:
```bash
python -m pip install --upgrade pip
```

### Issue 4: Numpy version conflicts

**Error**: `numpy>=2.0 is not supported by nemo-toolkit`

**Solution**: Already fixed in requirements.txt with `numpy>=1.24.0,<2.1.0`

## Next Steps

Once setup is complete:

1. **Prepare your data**:
   ```bash
   python src/data_preprocessing/prepare_seapile.py
   ```

2. **Set WandB API key**:
   ```bash
   export WANDB_API_KEY="your-key-here"
   ```

3. **Run training**:
   ```bash
   # Interactive
   python scripts/run_cpt_gemma3_1b.py --devices 8
   
   # Or submit to cluster
   qsub jobs/submit_cpt_gemma3_1b.sh
   ```

## Environment Details

- **Python**: 3.11+
- **PyTorch**: Latest version with CUDA 12.1 support
- **NeMo**: 2.2.0+
- **GPU**: NVIDIA H100 (Hopper architecture)
- **CUDA**: 12.1 (toolkit version on cluster)
