# Environment Setup with uv

This project uses `uv` for fast, reproducible dependency management.

## Prerequisites

Install `uv` if you haven't already:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

## Reproducing the Environment

### On Your Linux Machine

1. **Clone the repository** (or copy the project files)

2. **Sync dependencies** using the lock file:
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment (`.venv`)
   - Install all dependencies with exact versions from `uv.lock`
   - Ensure identical environment to macOS development machine

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; import gymnasium; print('✓ Environment ready')"
   ```

## Quick Start

### Run PPO Training with Standard CNN
```bash
source .venv/bin/activate
python ppo_atari_lstm.py --env-id PongNoFrameskip-v4 --encoder-type cnn
```

### Run PPO Training with EfficientNet
```bash
source .venv/bin/activate
python ppo_atari_lstm.py \
    --env-id PongNoFrameskip-v4 \
    --encoder-type efficientnet \
    --efficientnet-variant b1 \
    --efficientnet-pretrained
```

## Adding New Dependencies

If you need to add packages:

```bash
# Add a dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update lock file
uv lock
```

## Platform-Specific Notes

### CUDA/GPU Support (Linux)

⚠️ **Important**: The lock file includes CPU-only PyTorch (generated on macOS).

For GPU/CUDA support on Linux, use the provided setup script:

```bash
# Option 1: Automated setup (recommended)
./setup_cuda.sh

# This will:
# 1. Install all dependencies via uv sync
# 2. Replace CPU PyTorch with CUDA PyTorch
# 3. Verify CUDA is working
```

**Manual setup** (if you prefer):
```bash
# 1. Sync dependencies
uv sync

# 2. Activate environment
source .venv/bin/activate

# 3. Replace PyTorch with CUDA version
pip uninstall -y torch torchvision

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Verify
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Check your CUDA version**:
```bash
nvidia-smi  # Check driver version
nvcc --version  # Check CUDA toolkit version
```

Choose PyTorch CUDA version based on your system:
- CUDA 11.8: `cu118` (most compatible)
- CUDA 12.1: `cu121` (newer, faster)
- CUDA 12.4: `cu124` (latest)

### Atari ROMs

If you encounter missing ROM errors:
```bash
source .venv/bin/activate
pip install "gymnasium[atari,accept-rom-license]"
```

## Files for Reproducibility

- **`pyproject.toml`**: Project metadata and dependency specifications
- **`uv.lock`**: Exact pinned versions for reproducibility
- **`.venv/`**: Virtual environment (created by `uv sync`, not committed to git)

## Troubleshooting

### "uv: command not found"
```bash
# Add uv to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Dependency conflicts
```bash
# Clear lock and regenerate
rm uv.lock
uv lock
uv sync
```

### Different Python version on Linux
Edit `pyproject.toml` to adjust `requires-python`:
```toml
requires-python = ">=3.10,<3.14"
```

Then regenerate lock:
```bash
uv lock
uv sync
```

## Alternative: Traditional pip

If `uv` is not available, export requirements:

```bash
uv pip compile pyproject.toml -o requirements.txt
pip install -r requirements.txt
```

## Benefits of uv

- **Fast**: 10-100× faster than pip
- **Reproducible**: Lock file ensures exact versions across machines
- **Simple**: One command (`uv sync`) sets up everything
- **Cross-platform**: Works identically on macOS, Linux, Windows

---

**Note**: The lock file was generated on macOS (Darwin). On Linux, `uv sync` will automatically resolve platform-specific wheels while maintaining version constraints.
