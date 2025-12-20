# Windows GPU Setup for RTX 5060

Your system: **NVIDIA GeForce RTX 5060** (CUDA capability sm_120 - Blackwell architecture)

## Current Issues

1. ❌ **PyTorch**: CUDA 11.8 doesn't support RTX 5060 (sm_120)
2. ❌ **XGBoost**: pip/uv install has NO GPU support on Windows

---

## Fix 1: PyTorch with CUDA 12.x (For RTX 5060)

Your RTX 5060 requires **CUDA 12.x**. PyTorch CUDA 11.8 is too old.

```bash
# Uninstall old PyTorch
uv pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 12.4 (supports RTX 5060)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify (should show no warnings):**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 5060
```

**No warning = working!**

---

## Fix 2: XGBoost GPU Support on Windows

### Option A: Use Conda (RECOMMENDED) ⭐

Windows pip/uv XGBoost does NOT include GPU support. Use conda:

**Install Miniconda:**
- Download: https://docs.conda.io/en/latest/miniconda.html
- Install to default location

**Create new environment with GPU XGBoost:**
```bash
# Create conda environment (optional - can use existing)
conda create -n price-stradamus python=3.13

# Activate
conda activate price-stradamus

# Install GPU-enabled XGBoost
conda install -c conda-forge py-xgboost-gpu

# Install other dependencies
uv pip install -e ".[dev]"
```

**Or add to existing environment:**
```bash
# Activate your environment
conda activate price-stradamus

# Install GPU XGBoost
conda install -c conda-forge py-xgboost-gpu
```

**Verify:**
```bash
python -c "from price_stradamus.utils.gpu_utils import check_xgboost_gpu; print(f'XGBoost GPU: {check_xgboost_gpu()}')"
```

Should print: `XGBoost GPU: True`

### Option B: Use CPU XGBoost (Temporary)

XGBoost CPU is still very fast! Use CPU for XGBoost, GPU for neural networks:

**Current setup:**
- ✅ XGBoost: CPU (still fast!)
- ✅ Neural Networks (N-BEATS, LSTM, etc.): GPU

**No changes needed** - code will automatically use:
- `tree_method='hist'` (CPU) for XGBoost
- CUDA for neural networks

**Performance:**
- XGBoost CPU: Still 2-10x faster than many other models
- Neural networks on GPU: 5-10x speedup vs CPU

---

## Verify Your Setup

```bash
# Run comprehensive GPU report
python -m price_stradamus.utils.gpu_utils
```

**Expected output (after fixes):**

```
============================================================
GPU SUPPORT REPORT
============================================================

[PyTorch]
  [OK] CUDA Available: Yes
  - CUDA Version: 12.4
  - PyTorch Version: 2.x.x+cu124
  - GPU Count: 1
  - GPU Name: NVIDIA GeForce RTX 5060
  - GPU Memory: 8.55 GB

[XGBoost]
  [OK] GPU Support: Yes  <- Only if using conda
  - Can use tree_method='gpu_hist'

  OR

  [FAIL] GPU Support: No  <- If using pip/uv (CPU mode)
  - Using CPU mode (still fast!)

[System]
  - Python: 3.13.9
  - NVIDIA Driver: 591.59
  - GPU (nvidia-smi): NVIDIA GeForce RTX 5060

============================================================
```

---

## Current Behavior (After Fix)

Your backtesting notebook will now:

1. **Neural Network Models** (N-BEATS, LSTM, TCN, TFT):
   - ✅ Use GPU automatically (via PyTorch CUDA 12.4)
   - Logs: `Training on device: cuda`

2. **XGBoost**:
   - With conda: ✅ `Using tree_method: gpu_hist (GPU)`
   - With pip/uv: ℹ️ `Using tree_method: hist (CPU)` (still fast!)

---

## Summary: What to Do

### Minimum (Neural networks on GPU, XGBoost on CPU):

```bash
# Fix PyTorch for RTX 5060
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Test
python -m price_stradamus.utils.gpu_utils
```

### Recommended (Everything on GPU):

```bash
# 1. Fix PyTorch
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install Miniconda (if not installed)
# Download from: https://docs.conda.io/en/latest/miniconda.html

# 3. Install GPU XGBoost via conda
conda install -c conda-forge py-xgboost-gpu

# 4. Test
python -m price_stradamus.utils.gpu_utils
```

---

## Performance Comparison

**With CPU XGBoost + GPU Neural Networks:**
- XGBoost training: ~5-10 seconds per fold (CPU is still fast!)
- Neural network training: 5-10x faster than CPU
- **Overall: Good performance, easy setup**

**With GPU XGBoost + GPU Neural Networks:**
- XGBoost training: ~2-5 seconds per fold (2-5x speedup)
- Neural network training: 5-10x faster than CPU
- **Overall: Best performance, requires conda**

---

## Troubleshooting

### Issue: PyTorch still shows warning about sm_120

**Cause:** Still using CUDA 11.8

**Fix:**
```bash
python -c "import torch; print(torch.__version__)"

# Should show: 2.x.x+cu124
# If it shows: 2.x.x+cu118, reinstall with cu124
```

### Issue: XGBoost GPU still fails with conda

**Check XGBoost version:**
```bash
python -c "import xgboost; print(xgboost.__version__)"
```

Should be 2.x.x or 3.x.x

**Reinstall:**
```bash
conda uninstall py-xgboost-gpu
conda install -c conda-forge py-xgboost-gpu
```

### Issue: Mixing conda and pip/uv

**Best practice:**
- Use conda ONLY for packages that need it (like py-xgboost-gpu)
- Use uv/pip for everything else

**Safe to mix:**
```bash
# Install GPU XGBoost with conda
conda install -c conda-forge py-xgboost-gpu

# Install other packages with uv
uv pip install pandas numpy loguru
```

---

## Quick Reference

```bash
# Check GPU status
python -m price_stradamus.utils.gpu_utils

# Check PyTorch CUDA version
python -c "import torch; print(torch.__version__)"

# Check if PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check XGBoost GPU
python -c "from price_stradamus.utils.gpu_utils import check_xgboost_gpu; print(check_xgboost_gpu())"

# Monitor GPU during training
nvidia-smi -l 1
```

---

## Next Steps

1. **Fix PyTorch** (required for RTX 5060):
   ```bash
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Choose XGBoost option**:
   - Easy: Use CPU (current setup works!)
   - Best: Install conda + py-xgboost-gpu

3. **Run backtest notebook** - should work now!

4. **Monitor GPU usage**:
   ```bash
   nvidia-smi -l 1
   ```

Your RTX 5060 is a very new GPU, so some compatibility issues are expected. The fixes above will get everything working! 🚀
