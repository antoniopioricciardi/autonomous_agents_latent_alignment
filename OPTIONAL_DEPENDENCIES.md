# Optional Dependencies Guide

The project uses `uv`'s optional dependency groups to support multiple RL environments without bloating the base installation.

## Available Extras

### 🎮 Atari (`atari`)

**Environments**: Atari 2600 games (Pong, Breakout, Space Invaders, etc.)

**Install**:
```bash
uv sync --extra atari
```

**Includes**:
- `ale-py==0.8.1` - Arcade Learning Environment
- `AutoROM[accept-rom-license]~=0.4.2` - Atari ROMs (auto-licensed)
- `opencv-python>=4.6.0.66,<5` - Image processing
- `shimmy>=1.1.0` - Gym compatibility layer

**Use with**:
```python
import gymnasium as gym
env = gym.make("BreakoutNoFrameskip-v4")
```

---

### 🌳 Procgen (`procgen`)

**Environments**: 16 procedurally-generated games (CoinRun, StarPilot, etc.)

**Install**:
```bash
uv sync --extra procgen
```

**Includes**:
- `procgen>=0.10.7,<0.11` - Procgen environments

**Use with**:
```python
import gymnasium as gym
env = gym.make("procgen:procgen-coinrun-v0")
```

---

### 🤖 MuJoCo (`mujoco`)

**Environments**: MuJoCo physics simulations (Humanoid, Ant, HalfCheetah, etc.)

**Install**:
```bash
uv sync --extra mujoco
```

**Includes**:
- `mujoco<=2.3.3` - MuJoCo physics engine
- `imageio>=2.14.1,<3` - Video rendering

**Use with**:
```python
import gymnasium as gym
env = gym.make("Humanoid-v4")
```

---

### ⚡ EnvPool (`envpool`)

**Environments**: High-performance vectorized environments (10-100× faster)

**Install**:
```bash
uv sync --extra envpool
```

**Includes**:
- `envpool>=0.6.4,<0.7` - Fast environment pool
- `opencv-python>=4.6.0.66,<5` - Image processing

**Use with**:
```python
import envpool
envs = envpool.make("Pong-v5", env_type="gymnasium", num_envs=8)
```

---

## Install Multiple Extras

```bash
# Atari + Procgen
uv sync --extra atari --extra procgen

# Atari + MuJoCo (for diverse experiments)
uv sync --extra atari --extra mujoco

# Everything
uv sync --all-extras
```

## Alternative: pip-style Installation

If you prefer traditional pip syntax:

```bash
source .venv/bin/activate

# Install specific extras
uv pip install -e ".[atari]"
uv pip install -e ".[procgen]"
uv pip install -e ".[atari,mujoco]"

# Install all extras
uv pip install -e ".[atari,procgen,mujoco,envpool]"
```

## Recommendations by Use Case

### 🎯 This Project (Atari LSTM + SCIL)
```bash
uv sync --extra atari
```
**Why**: The project focuses on Atari environments with LSTM agents.

### 🔬 General RL Research
```bash
uv sync --extra atari --extra mujoco
```
**Why**: Covers both discrete (Atari) and continuous (MuJoCo) control.

### 🏃 Fast Prototyping
```bash
uv sync --extra atari --extra envpool
```
**Why**: EnvPool provides 10-100× speedup for Atari environments.

### 🌍 Maximum Compatibility
```bash
uv sync --all-extras
```
**Why**: Ensures you can run any experiment without reinstalling.

## Disk Space

| Extra | Additional Disk Space | Total Packages |
|-------|----------------------|----------------|
| Base (no extras) | ~2 GB | 94 |
| `+ atari` | ~200 MB | +4 packages |
| `+ procgen` | ~100 MB | +1 package |
| `+ mujoco` | ~500 MB | +2 packages |
| `+ envpool` | ~300 MB | +2 packages |
| **All extras** | ~3.1 GB | 124 packages |

## Lock File Behavior

The `uv.lock` file includes **all** optional dependencies (124 packages total), but `uv sync` only installs what you specify:

```bash
# Only installs base 94 packages
uv sync

# Installs base + atari (98 packages)
uv sync --extra atari

# Installs everything (124 packages)
uv sync --all-extras
```

This ensures:
- ✅ Reproducible environments (lock file pins all versions)
- ✅ Minimal disk usage (install only what you need)
- ✅ Fast switching (no re-downloading when adding extras)

## Updating Lock File

After modifying `pyproject.toml`:

```bash
# Regenerate lock file
uv lock

# Then install with desired extras
uv sync --extra atari
```

## Troubleshooting

### "Package not found" errors
Make sure you're using `--extra` (not `-e` or `--with`):
```bash
# ✓ Correct
uv sync --extra atari

# ✗ Wrong
uv sync -e atari
```

### ROMs not found (Atari)
The `AutoROM[accept-rom-license]` should auto-install ROMs. If not:
```bash
source .venv/bin/activate
python -m atari_py.import_roms
```

### MuJoCo license errors
MuJoCo 2.0+ is free and doesn't require a license. If you see license errors, ensure you have `mujoco<=2.3.3`.

### EnvPool compilation errors
EnvPool requires C++ compiler. On Linux:
```bash
sudo apt-get install build-essential cmake
```

On macOS:
```bash
xcode-select --install
```
