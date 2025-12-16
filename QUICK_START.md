# Quick Start Guide

## Setup on Linux (with GPU)

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Run automated setup with CUDA + Atari
./setup_cuda.sh

# That's it! Environment is ready with GPU support and Atari games.
```

**Note**: The setup script installs Atari support by default. For other environments:
```bash
# Manual setup with different extras
uv sync --extra procgen  # Procgen environments
uv sync --all-extras     # All environments
```

## Setup on macOS (CPU only)

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies with Atari support
uv sync --extra atari

# 3. Activate
source .venv/bin/activate
```

## Verify Installation

```bash
source .venv/bin/activate

# Check PyTorch and CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Check other dependencies
python -c "import gymnasium; import numpy; print('✓ All imports successful')"
```

## Run Training

### Quick Test (CPU, 10K steps)
```bash
source .venv/bin/activate
python ppo_atari_lstm.py \
    --env-id PongNoFrameskip-v4 \
    --encoder-type cnn \
    --total-timesteps 10000
```

### Full Training (GPU, 10M steps, with tracking)
```bash
source .venv/bin/activate
python ppo_atari_lstm.py \
    --env-id BreakoutNoFrameskip-v4 \
    --encoder-type efficientnet \
    --efficientnet-variant b1 \
    --efficientnet-pretrained \
    --total-timesteps 10000000 \
    --track \
    --wandb-project-name "r3l-atari"
```

### Available Encoder Options

**CNN (Fast - 5000+ SPS)**:
```bash
--encoder-type cnn
```

**EfficientNet-B0 (Balanced - 500-1000 SPS)**:
```bash
--encoder-type efficientnet --efficientnet-variant b0 --efficientnet-pretrained
```

**EfficientNet-B1 (Best Quality - 200-500 SPS, SCIL-validated)**:
```bash
--encoder-type efficientnet --efficientnet-variant b1 --efficientnet-pretrained
```

## Common Issues

### "CUDA out of memory"
Reduce batch size or number of environments:
```bash
--num-envs 4 --num-steps 64
```

### "No module named 'wandb'"
```bash
source .venv/bin/activate
pip install wandb
wandb login
```

### Atari ROM errors
```bash
source .venv/bin/activate
pip install "gymnasium[atari,accept-rom-license]"
```

### ImportError after updating dependencies
```bash
# Clean reinstall
rm -rf .venv
uv sync
./setup_cuda.sh  # On Linux with GPU
```

## Files Overview

- **`ppo_atari_lstm.py`**: Main PPO training script
- **`model/agent.py`**: Agent architecture (CNN/EfficientNet + LSTM)
- **`pyproject.toml`**: Dependency specifications
- **`uv.lock`**: Exact pinned versions
- **`setup_cuda.sh`**: Automated CUDA setup for Linux
- **`SETUP.md`**: Detailed setup documentation
- **`ENCODER_USAGE.md`**: Encoder architecture guide

## Next Steps

See detailed documentation:
- **Setup**: `SETUP.md`
- **Encoder options**: `ENCODER_USAGE.md`
- **SCIL/SAPS integration**: `SCIL_SAPS_report.md`
