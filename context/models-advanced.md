# Models - Advanced Topics (Phase 2-4)

---

## Overview

Advanced model optimization techniques for production scale and performance.

**For basic model usage**, see [models.md](models.md).

### Contents

1. [Model Explainability Methods](#model-explainability-methods)
2. [Transfer Learning](#transfer-learning)
3. [Model Compression](#model-compression)
4. [Model Registry Integration](#model-registry-integration)

These techniques are for **Phase 2-4** when you need to:
- Understand model predictions (explainability)
- Leverage pre-trained models (transfer learning)
- Reduce model size/latency (compression)
- Manage model versions in production (registry)

---

## Model Explainability Methods

### SHAP (SHapley Additive exPlanations)

```python
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap


class ModelExplainer:
    """Explain model predictions using SHAP and other methods."""

    def __init__(self, model: Any, background_data: np.ndarray) -> None:
        """Initialize explainer.

        Args:
            model: Trained model with predict method
            background_data: Background data for SHAP (100-1000 samples)
        """
        self.model = model
        self.background_data = background_data
        self._explainer = None

    def create_explainer(self, explainer_type: str = "kernel") -> None:
        """Create SHAP explainer."""
        def predict_fn(x: np.ndarray) -> np.ndarray:
            # Wrap model prediction
            return self.model.predict(x)

        if explainer_type == "kernel":
            self._explainer = shap.KernelExplainer(
                predict_fn,
                self.background_data,
            )
        elif explainer_type == "deep":
            # For neural networks
            self._explainer = shap.DeepExplainer(
                self.model.model.model,  # PyTorch model
                torch.tensor(self.background_data),
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")

    def explain_prediction(
        self,
        input_data: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Explain a single prediction."""
        if self._explainer is None:
            self.create_explainer()

        shap_values = self._explainer.shap_values(input_data)

        # Get feature importance
        if isinstance(shap_values, list):
            # Multi-output: average across outputs
            importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)

        # Rank features
        if feature_names is None:
            feature_names = [f"lag_{i}" for i in range(len(importance))]

        ranked_features = sorted(
            zip(feature_names, importance),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "shap_values": shap_values,
            "feature_importance": dict(ranked_features),
            "base_value": self._explainer.expected_value,
            "prediction": self.model.predict(input_data),
        }

    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        feature_names: list[str],
        max_features: int = 20,
    ) -> None:
        """Plot SHAP feature importance."""
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_features,
            plot_type="bar",
        )

    def plot_force(
        self,
        shap_values: np.ndarray,
        input_data: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Plot SHAP force plot for a single prediction."""
        shap.force_plot(
            self._explainer.expected_value,
            shap_values[0],
            input_data[0],
            feature_names=feature_names,
            matplotlib=True,
        )


# LIME (Local Interpretable Model-agnostic Explanations)
import lime.lime_tabular


class LIMEExplainer:
    """Explain predictions using LIME."""

    def __init__(
        self,
        model: Any,
        training_data: np.ndarray,
        feature_names: list[str],
    ) -> None:
        self.model = model
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=True,
        )

    def explain(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ) -> dict[str, Any]:
        """Explain a single prediction."""
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict,
            num_features=num_features,
        )

        return {
            "feature_weights": dict(explanation.as_list()),
            "prediction": self.model.predict(instance.reshape(1, -1))[0],
            "intercept": explanation.intercept[0],
            "local_prediction": explanation.local_pred[0],
            "r2_score": explanation.score,
        }
```

### Attention Visualization (for TFT)

```python
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


class AttentionVisualizer:
    """Visualize attention weights from TFT model."""

    def __init__(self, tft_model: Any) -> None:
        """Initialize with trained TFT model."""
        self.model = tft_model

    def extract_attention_weights(
        self,
        input_series: Any,
    ) -> dict[str, np.ndarray]:
        """Extract attention weights from prediction."""
        import torch

        # Get model internals during forward pass
        self.model.model.model.eval()

        with torch.no_grad():
            # This is model-specific - adjust based on Darts TFT implementation
            attention_weights = {
                "encoder_self_attention": None,
                "decoder_self_attention": None,
                "encoder_decoder_attention": None,
                "variable_selection": None,
            }

            # Hook to capture attention
            def hook_fn(module: Any, input: Any, output: Any) -> None:
                attention_weights["encoder_self_attention"] = output.detach().numpy()

            # Register hook (adjust layer name based on model)
            # handle = self.model.model.model.attention_layer.register_forward_hook(hook_fn)

            # Forward pass
            _ = self.model.predict(n=5, series=input_series)

            # handle.remove()

        return attention_weights

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        title: str = "Attention Weights",
    ) -> None:
        """Plot attention weights as heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
        ax.set_xlabel("Key Position (Input Timesteps)")
        ax.set_ylabel("Query Position (Output Timesteps)")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()
        return fig

    def plot_variable_importance(
        self,
        variable_selection_weights: np.ndarray,
        variable_names: list[str],
    ) -> None:
        """Plot variable selection weights from TFT."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sort by importance
        sorted_idx = np.argsort(variable_selection_weights)[::-1]
        sorted_weights = variable_selection_weights[sorted_idx]
        sorted_names = [variable_names[i] for i in sorted_idx]

        ax.barh(range(len(sorted_names)), sorted_weights)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Variable Selection Weight")
        ax.set_title("TFT Variable Importance")

        plt.tight_layout()
        return fig


class InterpretableForecast:
    """Generate interpretable forecasts with explanations."""

    def __init__(
        self,
        model: Any,
        explainer: ModelExplainer | None = None,
    ) -> None:
        self.model = model
        self.explainer = explainer

    def predict_with_explanation(
        self,
        input_data: Any,
        n: int = 5,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Make prediction with full explanation."""
        # Get prediction
        prediction = self.model.predict(n=n, series=input_data)

        result = {
            "predictions": prediction.values().tolist(),
            "timestamps": prediction.time_index.tolist(),
        }

        # Add SHAP explanation if available
        if self.explainer is not None:
            input_array = input_data.values()[-60:].reshape(1, -1)
            explanation = self.explainer.explain_prediction(
                input_array,
                feature_names,
            )
            result["explanation"] = explanation

        # Add prediction confidence (if model supports)
        if hasattr(self.model, "predict_quantiles"):
            quantiles = self.model.predict_quantiles(
                n=n,
                series=input_data,
                quantiles=[0.1, 0.5, 0.9],
            )
            result["confidence_interval"] = {
                "lower": quantiles[0.1].values().tolist(),
                "median": quantiles[0.5].values().tolist(),
                "upper": quantiles[0.9].values().tolist(),
            }

        return result
```

---

## Transfer Learning

### Pre-trained Model Fine-tuning

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from darts import TimeSeries


class TransferLearningManager:
    """Manage transfer learning for time series models."""

    def __init__(self, base_model_path: Path) -> None:
        """Initialize with pre-trained model path."""
        self.base_model_path = base_model_path
        self.base_model = None

    def load_pretrained(self) -> Any:
        """Load pre-trained model."""
        from darts.models import NBEATSModel

        self.base_model = NBEATSModel.load(str(self.base_model_path))
        return self.base_model

    def freeze_layers(
        self,
        freeze_pattern: str = "all_but_last",
    ) -> None:
        """Freeze model layers for transfer learning."""
        if self.base_model is None:
            raise ValueError("Load pretrained model first")

        pytorch_model = self.base_model.model.model

        if freeze_pattern == "all_but_last":
            # Freeze all layers except the last stack
            for name, param in pytorch_model.named_parameters():
                if "stack_29" not in name:  # Last stack
                    param.requires_grad = False

        elif freeze_pattern == "first_half":
            # Freeze first half of stacks
            for name, param in pytorch_model.named_parameters():
                stack_num = self._extract_stack_num(name)
                if stack_num is not None and stack_num < 15:
                    param.requires_grad = False

        elif freeze_pattern == "none":
            # Fine-tune all layers
            for param in pytorch_model.parameters():
                param.requires_grad = True

        # Log frozen/unfrozen parameters
        frozen = sum(
            p.numel() for p in pytorch_model.parameters() if not p.requires_grad
        )
        trainable = sum(
            p.numel() for p in pytorch_model.parameters() if p.requires_grad
        )
        print(f"Frozen parameters: {frozen:,}")
        print(f"Trainable parameters: {trainable:,}")

    def fine_tune(
        self,
        target_data: TimeSeries,
        epochs: int = 10,
        learning_rate: float = 1e-5,  # Lower LR for fine-tuning
    ) -> Any:
        """Fine-tune model on target domain data."""
        if self.base_model is None:
            raise ValueError("Load pretrained model first")

        # Update optimizer with lower learning rate
        self.base_model.model.optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                self.base_model.model.model.parameters(),
            ),
            lr=learning_rate,
        )

        # Fine-tune
        self.base_model.fit(
            series=target_data,
            epochs=epochs,
            verbose=True,
        )

        return self.base_model

    def _extract_stack_num(self, param_name: str) -> int | None:
        """Extract stack number from parameter name."""
        import re
        match = re.search(r"stack_(\d+)", param_name)
        if match:
            return int(match.group(1))
        return None


class DomainAdaptation:
    """Domain adaptation for different crypto assets."""

    def __init__(
        self,
        source_model: Any,
        source_symbol: str = "BTCUSDT",
    ) -> None:
        self.source_model = source_model
        self.source_symbol = source_symbol

    def adapt_to_target(
        self,
        target_data: TimeSeries,
        target_symbol: str,
        adaptation_strategy: str = "fine_tune",
    ) -> Any:
        """Adapt model to target domain (different crypto asset)."""
        import copy

        # Clone model
        adapted_model = copy.deepcopy(self.source_model)

        if adaptation_strategy == "fine_tune":
            # Simple fine-tuning
            adapted_model.fit(
                series=target_data,
                epochs=5,
            )

        elif adaptation_strategy == "domain_adversarial":
            # Domain adversarial neural network approach
            # (More complex - would require custom training loop)
            raise NotImplementedError("DANN not implemented yet")

        elif adaptation_strategy == "feature_alignment":
            # Align feature distributions
            # (Would require access to intermediate representations)
            raise NotImplementedError("Feature alignment not implemented yet")

        return adapted_model

    def evaluate_transfer(
        self,
        model: Any,
        test_data: TimeSeries,
    ) -> dict[str, float]:
        """Evaluate transfer learning performance."""
        from darts.metrics import mae, mape, rmse

        predictions = model.predict(n=len(test_data))

        return {
            "mae": mae(test_data, predictions),
            "rmse": rmse(test_data, predictions),
            "mape": mape(test_data, predictions),
        }
```

---

## Model Compression

### Quantization

```python
from __future__ import annotations

from typing import Any

import torch
import torch.quantization as quant


class ModelQuantizer:
    """Quantize models for faster inference and smaller size."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.quantized_model = None

    def dynamic_quantization(self) -> Any:
        """Apply dynamic quantization (weights only)."""
        pytorch_model = self.model.model.model

        self.quantized_model = quant.quantize_dynamic(
            pytorch_model,
            {torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8,
        )

        return self.quantized_model

    def static_quantization(
        self,
        calibration_data: torch.Tensor,
    ) -> Any:
        """Apply static quantization with calibration."""
        pytorch_model = self.model.model.model

        # Prepare for quantization
        pytorch_model.eval()
        pytorch_model.qconfig = quant.get_default_qconfig("fbgemm")
        quant.prepare(pytorch_model, inplace=True)

        # Calibrate with sample data
        with torch.no_grad():
            for batch in calibration_data:
                pytorch_model(batch)

        # Convert to quantized
        self.quantized_model = quant.convert(pytorch_model, inplace=False)

        return self.quantized_model

    def measure_compression(self) -> dict[str, Any]:
        """Measure compression ratio and speedup."""
        import tempfile
        from pathlib import Path

        # Save original model
        original_path = Path(tempfile.mktemp(suffix=".pt"))
        torch.save(self.model.model.model.state_dict(), original_path)
        original_size = original_path.stat().st_size

        # Save quantized model
        quantized_path = Path(tempfile.mktemp(suffix=".pt"))
        torch.save(self.quantized_model.state_dict(), quantized_path)
        quantized_size = quantized_path.stat().st_size

        # Cleanup
        original_path.unlink()
        quantized_path.unlink()

        return {
            "original_size_mb": original_size / (1024 * 1024),
            "quantized_size_mb": quantized_size / (1024 * 1024),
            "compression_ratio": original_size / quantized_size,
        }


class ModelPruner:
    """Prune model weights for smaller size."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def magnitude_pruning(
        self,
        sparsity: float = 0.5,
    ) -> Any:
        """Prune weights by magnitude."""
        import torch.nn.utils.prune as prune

        pytorch_model = self.model.model.model

        for name, module in pytorch_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")  # Make pruning permanent

        return pytorch_model

    def structured_pruning(
        self,
        prune_ratio: float = 0.3,
    ) -> Any:
        """Structured channel pruning."""
        import torch.nn.utils.prune as prune

        pytorch_model = self.model.model.model

        for name, module in pytorch_model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=prune_ratio,
                    n=2,
                    dim=0,
                )

        return pytorch_model

    def measure_sparsity(self) -> dict[str, float]:
        """Measure actual sparsity after pruning."""
        pytorch_model = self.model.model.model

        total_params = 0
        zero_params = 0

        for param in pytorch_model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()

        return {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "sparsity": zero_params / total_params,
        }
```

---

## Model Registry Integration

### Model Versioning and Tracking

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelVersion:
    """Version information for a model."""

    model_name: str
    version: str
    created_at: datetime
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    training_data_hash: str
    model_hash: str
    tags: list[str] = field(default_factory=list)
    description: str = ""
    stage: str = "development"  # development, staging, production, archived


class LocalModelRegistry:
    """Simple local model registry for tracking versions."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.index_path = registry_path / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load registry index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                self.index = json.load(f)
        else:
            self.index = {"models": {}}

    def _save_index(self) -> None:
        """Save registry index."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2, default=str)

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        training_data: Any,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Register a new model version."""
        # Generate version number
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = []
        version = f"v{len(self.index['models'][model_name]) + 1}"

        # Calculate hashes
        model_hash = self._hash_model(model)
        data_hash = self._hash_data(training_data)

        # Create version record
        version_record = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "training_data_hash": data_hash,
            "model_hash": model_hash,
            "tags": tags or [],
            "description": description,
            "stage": "development",
            "artifact_path": f"{model_name}/{version}/model.pkl",
        }

        # Save model artifact
        artifact_dir = self.registry_path / model_name / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(artifact_dir / "model.pkl"))

        # Update index
        self.index["models"][model_name].append(version_record)
        self._save_index()

        return version

    def get_model(
        self,
        model_name: str,
        version: str | None = None,
        stage: str | None = None,
    ) -> tuple[Any, dict]:
        """Get model by name and version or stage."""
        from darts.models import NBEATSModel

        if model_name not in self.index["models"]:
            raise ValueError(f"Model not found: {model_name}")

        versions = self.index["models"][model_name]

        if stage:
            # Get latest model in stage
            matching = [v for v in versions if v["stage"] == stage]
            if not matching:
                raise ValueError(f"No model in stage: {stage}")
            version_record = matching[-1]
        elif version:
            # Get specific version
            matching = [v for v in versions if v["version"] == version]
            if not matching:
                raise ValueError(f"Version not found: {version}")
            version_record = matching[0]
        else:
            # Get latest version
            version_record = versions[-1]

        # Load model
        artifact_path = self.registry_path / version_record["artifact_path"]
        model = NBEATSModel.load(str(artifact_path))

        return model, version_record

    def promote_model(
        self,
        model_name: str,
        version: str,
        to_stage: str,
    ) -> None:
        """Promote model to a new stage."""
        if model_name not in self.index["models"]:
            raise ValueError(f"Model not found: {model_name}")

        for v in self.index["models"][model_name]:
            if v["version"] == version:
                v["stage"] = to_stage
                break

        self._save_index()

    def compare_versions(
        self,
        model_name: str,
        version_a: str,
        version_b: str,
    ) -> dict[str, Any]:
        """Compare two model versions."""
        _, record_a = self.get_model(model_name, version_a)
        _, record_b = self.get_model(model_name, version_b)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metric_diff": {},
            "hyperparam_diff": {},
        }

        # Compare metrics
        for metric in record_a["metrics"]:
            if metric in record_b["metrics"]:
                diff = record_b["metrics"][metric] - record_a["metrics"][metric]
                comparison["metric_diff"][metric] = {
                    "a": record_a["metrics"][metric],
                    "b": record_b["metrics"][metric],
                    "diff": diff,
                    "improved": diff > 0,
                }

        return comparison

    def _hash_model(self, model: Any) -> str:
        """Generate hash for model weights."""
        import io
        import pickle

        buffer = io.BytesIO()
        pickle.dump(model.model.model.state_dict(), buffer)
        return hashlib.md5(buffer.getvalue()).hexdigest()

    def _hash_data(self, data: Any) -> str:
        """Generate hash for training data."""
        return hashlib.md5(str(data.values()).encode()).hexdigest()
```

---

*Last Updated: 2025-12-03*
