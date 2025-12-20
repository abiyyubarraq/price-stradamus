"""GPU detection and verification utilities.

This module provides utilities to detect and verify GPU availability
for different frameworks (XGBoost, PyTorch, etc.).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Literal

import torch
from loguru import logger


def check_xgboost_gpu() -> bool:
    """Check if XGBoost is built with GPU support.

    XGBoost has its own GPU implementation independent of PyTorch.
    This function checks if xgboost can actually use GPU.

    Returns:
        True if XGBoost has GPU support, False otherwise
    """
    try:
        import xgboost as xgb

        # Try to create a DMatrix and train a small model with gpu_hist
        # This is the most reliable way to check GPU support
        import numpy as np

        # Create tiny dummy dataset
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        dtrain = xgb.DMatrix(X, label=y)

        # Try to use gpu_hist - will fail if GPU not available
        # Note: XGBoost 3.1+ uses 'device' instead of 'gpu_id'
        params = {
            "tree_method": "gpu_hist",
            "device": "cuda:0",  # Updated for XGBoost 3.1+
        }

        # This will raise an exception if GPU not available
        xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)

        logger.info("XGBoost GPU support verified")
        return True

    except Exception as e:
        logger.debug(f"XGBoost GPU check failed: {e}")
        return False


def check_pytorch_gpu() -> bool:
    """Check if PyTorch can use GPU (CUDA).

    Returns:
        True if PyTorch has CUDA available, False otherwise
    """
    available = torch.cuda.is_available()

    if available:
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        logger.info(f"✓ PyTorch GPU support: {device_count} device(s) - {device_name}")
    else:
        logger.info("✗ PyTorch GPU support: not available")

    return available


def get_gpu_info() -> dict[str, str | bool | int]:
    """Get detailed GPU information.

    Returns:
        Dictionary with GPU information
    """
    info: dict[str, str | bool | int] = {
        "pytorch_cuda_available": torch.cuda.is_available(),
        "xgboost_gpu_available": check_xgboost_gpu(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "cuda_version": torch.version.cuda or "unknown",
                "pytorch_version": torch.__version__,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                ),
            }
        )

    return info


def print_gpu_report() -> None:
    """Print a comprehensive GPU support report."""
    print("\n" + "=" * 60)
    print("GPU SUPPORT REPORT")
    print("=" * 60)

    info = get_gpu_info()

    # PyTorch section
    print("\n[PyTorch]")
    if info["pytorch_cuda_available"]:
        print("  [OK] CUDA Available: Yes")
        print(f"  - CUDA Version: {info.get('cuda_version', 'unknown')}")
        print(f"  - PyTorch Version: {info.get('pytorch_version', 'unknown')}")
        print(f"  - GPU Count: {info.get('gpu_count', 0)}")
        print(f"  - GPU Name: {info.get('gpu_name', 'unknown')}")
        print(f"  - GPU Memory: {info.get('gpu_memory_gb', 0)} GB")
    else:
        print("  [FAIL] CUDA Available: No")
        print("  - Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")

    # XGBoost section
    print("\n[XGBoost]")
    if info["xgboost_gpu_available"]:
        print("  [OK] GPU Support: Yes")
        print("  - Can use tree_method='gpu_hist'")
    else:
        print("  [FAIL] GPU Support: No")
        print("  - Install with: pip install xgboost[gpu] (CUDA 11.x)")
        print("  - Or: pip install xgboost (will auto-detect if you have CUDA)")

    # System info
    print("\n[System]")
    print(f"  - Python: {sys.version.split()[0]}")

    # Try to get nvidia-smi info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(",")
            print(f"  - NVIDIA Driver: {gpu_info[1].strip()}")
            print(f"  - GPU (nvidia-smi): {gpu_info[0].strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  - nvidia-smi: not available")

    print("\n" + "=" * 60 + "\n")


def verify_gpu_usage_xgboost() -> None:
    """Verify XGBoost is actually using GPU during training.

    This creates a small training task and monitors GPU usage.
    """
    import time

    import numpy as np

    try:
        import xgboost as xgb

        print("\n[Verifying XGBoost GPU usage]")
        print("   Training a small model with gpu_hist...")

        # Create larger dataset to make GPU usage visible
        X = np.random.rand(10000, 100)
        y = np.random.rand(10000)
        dtrain = xgb.DMatrix(X, label=y)

        # XGBoost 3.1+ uses 'device' instead of 'gpu_id'
        params = {
            "tree_method": "gpu_hist",
            "device": "cuda:0",
            "max_depth": 8,
        }

        print("   Monitor GPU usage with: nvidia-smi -l 1")
        print("   You should see GPU memory usage increase during training.")
        time.sleep(2)

        # Train with GPU
        start = time.time()
        xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        gpu_time = time.time() - start

        print(f"   [OK] GPU training completed in {gpu_time:.2f}s")

        # Compare with CPU
        params_cpu = {"tree_method": "hist", "max_depth": 8, "device": "cpu"}

        start = time.time()
        xgb.train(params_cpu, dtrain, num_boost_round=100, verbose_eval=False)
        cpu_time = time.time() - start

        print(f"   [OK] CPU training completed in {cpu_time:.2f}s")
        speedup = cpu_time / gpu_time
        print(f"\n   Speedup: {speedup:.2f}x (GPU vs CPU)")

        if speedup > 1.5:
            print("   [OK] GPU is significantly faster - GPU is working!")
        else:
            print(
                "   [WARN] GPU not much faster - might be overhead on small dataset or GPU not used"
            )

    except Exception as e:
        print(f"   [FAIL] Verification failed: {e}")


def get_recommended_device() -> Literal["cuda", "cpu"]:
    """Get recommended device for PyTorch models.

    Returns:
        "cuda" if GPU available, "cpu" otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_recommended_xgboost_tree_method() -> Literal["gpu_hist", "hist"]:
    """Get recommended tree method for XGBoost.

    Returns:
        "gpu_hist" if GPU available, "hist" (CPU) otherwise
    """
    if check_xgboost_gpu():
        return "gpu_hist"
    return "hist"


if __name__ == "__main__":
    # Run GPU report when executed directly
    print_gpu_report()

    # Optionally verify XGBoost GPU usage
    if check_xgboost_gpu():
        verify_gpu_usage_xgboost()
