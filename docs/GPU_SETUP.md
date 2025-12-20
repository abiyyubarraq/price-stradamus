# GPU Setup Guide for Price Stradamus

This guide explains how to enable and verify GPU acceleration for machine learning models in Price Stradamus.

---

## Quick GPU Check

Run this command to check your GPU setup:

```bash
python -m price_stradamus.utils.gpu_utils
```

This will print a comprehensive report showing:
- ✓ PyTorch CUDA availability
- ✓ XGBoost GPU support
- ✓ GPU hardware details
- ✓ Driver versions

---

## GPU Support by Model

| Model | GPU Support | Method |
|-------|-------------|--------|
| **XGBoost** | ✅ Yes | `tree_method='gpu_hist'` |
| **N-BEATS** | ✅ Yes | PyTorch CUDA |
| **LSTM** | ✅ Yes | PyTorch CUDA |
| **TCN** | ✅ Yes | PyTorch CUDA |
| **TFT** | ✅ Yes | PyTorch CUDA |
| **Prophet** | ❌ No | CPU only |
| **ARIMA** | ❌ No | CPU only |

---

## Installation

### 1. Install NVIDIA Drivers

**Check if you have NVIDIA GPU:**
```bash
nvidia-smi
```

If this fails, install NVIDIA drivers:
- **Windows**: Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- **Linux**: `sudo apt install nvidia-driver-535` (or latest version)

### 2. Install CUDA Toolkit (Optional)

PyTorch and XGBoost include CUDA libraries, so you usually **don't need** to install CUDA separately. But if you want system-wide CUDA:

- Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Recommended: CUDA 11.8 or 12.1

### 3. Install PyTorch with CUDA

**Uninstall CPU-only PyTorch first:**
```bash
uv pip uninstall torch torchvision torchaudio
```

**Install PyTorch with CUDA 11.8:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Or with CUDA 12.1:**
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch CUDA:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4. Install XGBoost with GPU Support

XGBoost usually auto-detects CUDA and builds with GPU support if available.

**Option 1: Standard installation (recommended)**
```bash
uv pip install xgboost
```

**Option 2: Explicit GPU build**
```bash
uv pip install xgboost[gpu]
```

**Verify XGBoost GPU:**
```python
from price_stradamus.utils.gpu_utils import check_xgboost_gpu
print(f"XGBoost GPU: {check_xgboost_gpu()}")
```

---

## Verify GPU Usage

### Run Comprehensive GPU Report

```bash
cd price-stradamus
python -m price_stradamus.utils.gpu_utils
```

This will:
- ✅ Check PyTorch CUDA availability
- ✅ Check XGBoost GPU support
- ✅ Show GPU hardware details
- ✅ Run performance benchmarks (GPU vs CPU)

### Check GPU Usage During Training

**Monitor GPU in real-time:**
```bash
# In one terminal, monitor GPU usage
nvidia-smi -l 1  # Update every 1 second

# In another terminal, run training
python -m price_stradamus.cli.commands train --model xgboost --epochs 100
```

You should see:
- GPU memory usage increase (e.g., 500MB → 2GB)
- GPU utilization spike (e.g., 0% → 80%)
- Power draw increase (e.g., 50W → 150W)

### Programmatic GPU Check

**In your notebook:**
```python
from price_stradamus.utils.gpu_utils import print_gpu_report, verify_gpu_usage_xgboost

# Print detailed report
print_gpu_report()

# Verify XGBoost is actually using GPU (with benchmark)
verify_gpu_usage_xgboost()
```

---

## Model-Specific GPU Configuration

### XGBoost

XGBoost **automatically uses GPU** if available. The model checks GPU support and sets `tree_method='gpu_hist'` automatically.

**Manual override (not needed):**
```python
from price_stradamus.models.ml.xgboost import XGBoostModel

model = XGBoostModel(
    input_chunk_length=120,
    output_chunk_length=10,
    n_estimators=100,
)

# GPU detection is automatic!
# No need to pass tree_method - it's handled internally
```

**Check what tree_method is being used:**
```python
model.fit(train_data)
print(model.get_model_summary())
# Will show: "Tree method: gpu_hist (GPU)" or "Tree method: hist (CPU)"
```

### Neural Network Models (N-BEATS, LSTM, TCN, TFT)

Neural models **automatically use GPU** if PyTorch CUDA is available.

**Verify device:**
```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# Models will automatically use this device
from price_stradamus.models.neural.nbeats import NBEATSModel

model = NBEATSModel(
    input_chunk_length=120,
    output_chunk_length=10,
    n_epochs=100,
)

model.fit(train_data)  # Uses GPU automatically if available
```

**Monitor GPU during neural network training:**
```bash
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Issue 1: `torch.cuda.is_available()` returns `False`

**Cause:** PyTorch is CPU-only version

**Fix:**
```bash
# Check current PyTorch
python -c "import torch; print(torch.__version__)"

# If it says "cpu" or no "+cu118", reinstall with CUDA
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: XGBoost says "Using tree_method: hist (CPU)"

**Cause 1:** XGBoost not built with GPU support

**Fix:**
```bash
uv pip uninstall xgboost
uv pip install xgboost  # Will auto-detect CUDA if available
```

**Cause 2:** No NVIDIA GPU detected

**Fix:** Check `nvidia-smi` works. If not, install NVIDIA drivers.

**Cause 3:** XGBoost GPU check fails

**Debug:**
```python
from price_stradamus.utils.gpu_utils import check_xgboost_gpu
import traceback

try:
    result = check_xgboost_gpu()
    print(f"GPU available: {result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
```

### Issue 3: `CUDA out of memory` error

**Cause:** Model or batch size too large for GPU memory

**Fix 1 - Reduce batch size (for neural models):**
```python
model = NBEATSModel(
    batch_size=32,  # Reduce from default 64
)
```

**Fix 2 - Reduce input chunk length:**
```python
model = NBEATSModel(
    input_chunk_length=60,  # Reduce from 120
)
```

**Fix 3 - Use gradient checkpointing (advanced):**
Enable in model configuration if available.

### Issue 4: GPU is detected but training is slow

**Possible causes:**
1. **Data transfer overhead** - Small datasets don't benefit from GPU
2. **Model too small** - GPU overhead > speedup for tiny models
3. **CPU bottleneck** - Data preprocessing on CPU is the bottleneck

**Check actual GPU usage:**
```bash
nvidia-smi dmon -s u  # Show GPU utilization
```

If GPU utilization is low (<30%), the bottleneck is elsewhere (CPU data loading, preprocessing, etc.).

### Issue 5: CUDA version mismatch

**Error:** `RuntimeError: CUDA error: no kernel image is available`

**Cause:** PyTorch CUDA version doesn't match your GPU driver

**Fix:**
1. Check your CUDA capability: `nvidia-smi` (look for "CUDA Version")
2. Install matching PyTorch:
   - CUDA 11.x: `--index-url https://download.pytorch.org/whl/cu118`
   - CUDA 12.x: `--index-url https://download.pytorch.org/whl/cu121`

---

## Performance Benchmarks

Expected speedups with GPU vs CPU:

| Model | Dataset Size | GPU Speedup |
|-------|--------------|-------------|
| XGBoost | 100k samples | 2-5x |
| N-BEATS | 100k samples | 5-10x |
| LSTM | 100k samples | 3-7x |
| TCN | 100k samples | 4-8x |
| TFT | 100k samples | 5-12x |

**Note:** Small datasets (<10k samples) might be slower on GPU due to overhead.

---

## Best Practices

### 1. Use GPU for Large Datasets

- ✅ GPU beneficial: >50k samples, >50 features
- ❌ GPU may be slower: <10k samples, simple models

### 2. Monitor GPU Memory

```bash
# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 3. Batch Processing for XGBoost

XGBoost loads entire dataset into GPU memory. For very large datasets:
- Use `max_bin` parameter to reduce memory
- Consider sampling for initial experiments

### 4. Mixed Precision Training (Advanced)

For neural models, use mixed precision (FP16) to save memory and speed up:
```python
# This is handled automatically by Darts/PyTorch Lightning
# No manual configuration needed
```

### 5. Multi-GPU Training (Future)

Currently single-GPU only. Multi-GPU support planned for Phase 5.

---

## Quick Reference Commands

```bash
# Check GPU hardware
nvidia-smi

# Monitor GPU usage (live)
nvidia-smi -l 1

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check XGBoost GPU
python -c "from price_stradamus.utils.gpu_utils import check_xgboost_gpu; print(f'XGBoost GPU: {check_xgboost_gpu()}')"

# Full GPU report
python -m price_stradamus.utils.gpu_utils

# Verify GPU usage with benchmark
python -c "from price_stradamus.utils.gpu_utils import verify_gpu_usage_xgboost; verify_gpu_usage_xgboost()"
```

---

## Additional Resources

- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)
- [XGBoost GPU Documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

---

## Questions?

If GPU is still not working after following this guide:

1. Run `python -m price_stradamus.utils.gpu_utils` and share the output
2. Share your `nvidia-smi` output
3. Share your PyTorch version: `python -c "import torch; print(torch.__version__)"`
4. Open an issue with these details
