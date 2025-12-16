# Files to Commit for Linux Reproducibility

## ✅ Essential Files (Must Commit)

### Dependency Management
- **`pyproject.toml`** - Project metadata and dependency specifications
- **`uv.lock`** - Exact pinned versions for reproducible environments
- **`.gitignore`** - Excludes `.venv/`, `__pycache__/`, etc.

### Package Structure
- **`model/__init__.py`** - Makes model/ a Python package
- **`utils/__init__.py`** - Makes utils/ a Python package
- **`scil/__init__.py`** - Makes scil/ a Python package

### Setup Scripts
- **`setup_cuda.sh`** - Automated CUDA setup for Linux (executable)

### Documentation
- **`SETUP.md`** - Detailed setup instructions
- **`QUICK_START.md`** - Fast-track guide
- **`ENCODER_USAGE.md`** - Encoder architecture documentation
- **`GIT_COMMIT_CHECKLIST.md`** - This file

## 📝 Git Commands

```bash
# Add all essential files
git add pyproject.toml uv.lock .gitignore
git add model/__init__.py utils/__init__.py scil/__init__.py
git add setup_cuda.sh
git add SETUP.md QUICK_START.md ENCODER_USAGE.md GIT_COMMIT_CHECKLIST.md
git add ppo_atari_lstm.py  # Updated to remove cleanrl dependency

# Verify what will be committed
git status

# Commit
git commit -m "Add uv dependency management and Linux setup

- Add pyproject.toml with all dependencies
- Add uv.lock for reproducible environments
- Add __init__.py files for proper package structure
- Add setup_cuda.sh for automated CUDA setup on Linux
- Add comprehensive setup documentation
- Update .gitignore (excludes cleanrl, .venv, etc.)
- Remove unused cleanrl imports from ppo_atari_lstm.py"

# Push
git push
```

## 🐧 On Your Linux Machine

After pulling these changes:

```bash
# Quick setup (with GPU)
./setup_cuda.sh

# Or manual
uv sync
source .venv/bin/activate
```

## 🔍 Verify Before Committing

Run this checklist:

```bash
# 1. Check all __init__.py files exist
ls model/__init__.py utils/__init__.py scil/__init__.py

# 2. Verify cleanrl is in .gitignore
grep "^cleanrl/$" .gitignore

# 3. Verify pyproject.toml is valid
cat pyproject.toml | grep -E "name|version|dependencies"

# 4. Verify uv.lock exists and is recent
ls -lh uv.lock

# 5. Test that uv can build the project
uv sync  # Should succeed without errors

# 6. Verify setup_cuda.sh is executable
ls -la setup_cuda.sh | grep "x"

# 7. Check that cleanrl imports are removed
grep -q "cleanrl_utils" ppo_atari_lstm.py && echo "⚠️ cleanrl imports still present!" || echo "✓ cleanrl imports removed"
```

## ⚠️ What NOT to Commit

- **`cleanrl/`** - External CleanRL project (no longer needed)
- **`.venv/`** - Virtual environment (generated, not tracked)
- **`__pycache__/`** - Python bytecode (generated)
- **`*.pyc`, `*.pyo`** - Compiled Python files
- **`runs/`** - TensorBoard logs (experiment-specific)
- **`wandb/`** - Wandb logs (experiment-specific)
- **`checkpoints/`** - Model checkpoints (large, experiment-specific)
- **`videos/`** - Recorded gameplay (large)

All these are already in `.gitignore`.

## 🎯 Expected Result on Linux

After cloning and running `uv sync`:

```
Resolved 94 packages in 7ms
Installed 94 packages in 2s
  + torch==2.9.1 (CPU version)
  + gymnasium==0.29.1
  + numpy==<2.0.0
  + ... (91 more packages)
✓ Environment ready!
```

Then `./setup_cuda.sh` replaces CPU PyTorch with CUDA version.

## 📊 File Sizes

- `pyproject.toml`: ~1 KB
- `uv.lock`: ~368 KB
- `__init__.py` files: 0 bytes each
- `setup_cuda.sh`: ~500 bytes
- Documentation: ~20 KB total

**Total**: <400 KB of configuration files for full reproducibility!
