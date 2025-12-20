# Quick GPU Setup Guide

## 🚀 TL;DR - Enable GPU in 3 Steps

```bash
# 1. Check current GPU status
python -m price_stradamus.utils.gpu_utils

# 2. Install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Reinstall XGBoost (will auto-detect CUDA)
uv pip install --force-reinstall xgboost

# 4. Verify GPU is working
python -m price_stradamus.utils.gpu_utils
```

---

## ✅ What Changed in the Code

### Fixed Issues:
1. **XGBoost GPU detection now works correctly**
   - Old: Used `torch.cuda.is_available()` (wrong!)
   - New: Uses `check_xgboost_gpu()` which actually tests XGBoost GPU support

2. **Updated for XGBoost 3.1+ API**
   - Old: `gpu_id` parameter (deprecated)
   - New: `device="cuda:0"` parameter

3. **Created GPU utilities module**
   - New file: `src/price_stradamus/utils/gpu_utils.py`
   - Functions: `check_xgboost_gpu()`, `print_gpu_report()`, etc.

---

## 📊 Check Your GPU Status

Run this command to see a detailed report:

```bash
python -m price_stradamus.utils.gpu_utils
```

**Example output (GPU available):**
```
============================================================
GPU SUPPORT REPORT
============================================================

[PyTorch]
  [OK] CUDA Available: Yes
  - CUDA Version: 11.8
  - PyTorch Version: 2.1.0+cu118
  - GPU Count: 1
  - GPU Name: NVIDIA GeForce RTX 3080
  - GPU Memory: 10.0 GB

[XGBoost]
  [OK] GPU Support: Yes
  - Can use tree_method='gpu_hist'

[System]
  - Python: 3.13.3
  - NVIDIA Driver: 535.104.05
  - GPU (nvidia-smi): NVIDIA GeForce RTX 3080

============================================================
```

**Example output (No GPU):**
```
============================================================
GPU SUPPORT REPORT
============================================================

[PyTorch]
  [FAIL] CUDA Available: No
  - Install PyTorch with CUDA: https://pytorch.org/get-started/locally/

[XGBoost]
  [FAIL] GPU Support: No
  - Install with: pip install xgboost (will auto-detect if you have CUDA)

[System]
  - Python: 3.13.3
  - nvidia-smi: not available

============================================================
```

---

## 🔧 Installation Steps for GPU Support

### Step 1: Verify NVIDIA Driver

```bash
nvidia-smi
```

**If this fails:**
- Windows: Download drivers from [NVIDIA](https://www.nvidia.com/Download/index.aspx)
- Linux: `sudo apt install nvidia-driver-535`

### Step 2: Install PyTorch with CUDA

**Check which CUDA version your driver supports:**
```bash
nvidia-smi
# Look for "CUDA Version: XX.X" in the top right
```

**Install PyTorch:**

For CUDA 11.8:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify:**
```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print your GPU name
```

### Step 3: Install/Reinstall XGBoost

XGBoost will auto-detect CUDA and build with GPU support if available:

```bash
uv pip install --force-reinstall xgboost
```

**Verify:**
```python
from price_stradamus.utils.gpu_utils import check_xgboost_gpu
print(check_xgboost_gpu())  # Should print True
```

### Step 4: Test GPU in Notebook

Add this cell to your notebook to verify GPU usage:

```python
from price_stradamus.utils.gpu_utils import print_gpu_report, verify_gpu_usage_xgboost

# Print detailed GPU report
print_gpu_report()

# Run benchmark (GPU vs CPU comparison)
verify_gpu_usage_xgboost()
```

---

## 📝 Model-Specific GPU Usage

### XGBoost (Automatic)

XGBoost now **automatically detects and uses GPU** if available:

```python
from price_stradamus.models.ml.xgboost import XGBoostModel

model = XGBoostModel(
    input_chunk_length=120,
    output_chunk_length=10,
    n_estimators=100,
)

# GPU detection is automatic!
model.fit(train_data, past_covariates=train_cov)

# Check what device was used
print(model.get_model_summary())
# Will show: "Tree method: gpu_hist (GPU)" or "Tree method: hist (CPU)"
```

**In your logs, you should now see:**
```
INFO | Using tree_method: gpu_hist (GPU)
```

Instead of the old:
```
INFO | Using tree_method: auto
```

### Neural Network Models (N-BEATS, LSTM, TCN, TFT)

Neural models automatically use GPU if PyTorch CUDA is available:

```python
from price_stradamus.models.neural.nbeats import NBEATSModel
import torch

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

model = NBEATSModel(
    input_chunk_length=120,
    output_chunk_length=10,
    n_epochs=100,
)

# Will use GPU automatically if available
model.fit(train_data)
```

---

## 🔍 Monitor GPU Usage During Training

### Real-time GPU monitoring:

```bash
# Terminal 1: Monitor GPU
nvidia-smi -l 1

# Terminal 2: Run training
python -m price_stradamus.cli.commands train --model xgboost
```

You should see:
- **GPU Memory** increase (e.g., 500MB → 2GB)
- **GPU Utilization** spike (e.g., 0% → 80%)
- **Power Draw** increase (e.g., 50W → 150W)

### In Jupyter Notebook:

```python
# Cell 1: Start monitoring
!nvidia-smi

# Cell 2: Run training (in another cell)
model.fit(train_data)

# Cell 3: Check GPU was used
!nvidia-smi
```

---

## ⚠️ Troubleshooting

### Issue: "Using tree_method: hist (CPU)" even though I have GPU

**Cause:** XGBoost is not built with GPU support

**Fix:**
```bash
# Reinstall XGBoost
uv pip uninstall xgboost
uv pip install xgboost

# Verify
python -c "from price_stradamus.utils.gpu_utils import check_xgboost_gpu; print(check_xgboost_gpu())"
```

### Issue: PyTorch shows CUDA False

**Cause:** PyTorch is CPU-only version

**Fix:**
```bash
# Check current version
python -c "import torch; print(torch.__version__)"

# If it doesn't have "+cu118" or "+cu121", reinstall
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: CUDA out of memory

**Fix 1 - Reduce batch size:**
```python
model = NBEATSModel(
    batch_size=32,  # Reduce from default
)
```

**Fix 2 - Reduce input length:**
```python
model = XGBoostModel(
    input_chunk_length=60,  # Reduce from 120
)
```

### Issue: GPU not faster than CPU

**Check GPU utilization:**
```bash
nvidia-smi dmon -s u
```

If GPU utilization is low (<30%), the bottleneck is elsewhere (data loading, preprocessing).

---

## 📈 Expected Performance

With GPU enabled, you should see:

| Model | Dataset | GPU Speedup |
|-------|---------|-------------|
| XGBoost | 100k samples | **2-5x faster** |
| N-BEATS | 100k samples | **5-10x faster** |
| LSTM | 100k samples | **3-7x faster** |
| TCN | 100k samples | **4-8x faster** |

**Note:** Small datasets (<10k samples) might not show speedup due to GPU overhead.

---

## 📚 Full Documentation

For comprehensive GPU setup, see: [docs/GPU_SETUP.md](docs/GPU_SETUP.md)

---

## ✅ Summary Checklist

- [ ] `nvidia-smi` works and shows your GPU
- [ ] `python -m price_stradamus.utils.gpu_utils` shows `[OK]` for both PyTorch and XGBoost
- [ ] Training logs show `Using tree_method: gpu_hist (GPU)`
- [ ] `nvidia-smi -l 1` shows GPU usage during training
- [ ] GPU training is faster than CPU (run benchmark)

If all checked, you're good to go! 🚀
