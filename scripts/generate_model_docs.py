"""Generate model documentation from registry."""

from __future__ import annotations

from price_stradamus.models.classical.arima import ARIMAModel  # noqa: F401
from price_stradamus.models.classical.prophet import ProphetModel  # noqa: F401
from price_stradamus.models.ml.random_forest import RandomForestModel  # noqa: F401
from price_stradamus.models.ml.xgboost import XGBoostModel  # noqa: F401
from price_stradamus.models.neural.lstm import LSTMModel  # noqa: F401

# Import all models to register them
from price_stradamus.models.neural.nbeats import NBEATSModel  # noqa: F401
from price_stradamus.models.neural.tcn import TCNModel  # noqa: F401
from price_stradamus.models.neural.tft import TFTModel  # noqa: F401
from price_stradamus.models.registry import ModelRegistry


def generate_model_docs() -> str:
    """Generate markdown documentation for all models.

    Returns:
        Markdown string with model documentation

    Example:
        >>> docs = generate_model_docs()
        >>> print(docs)
        # Available Models

        ## arima

        **Class**: `ARIMAModel`
        **Module**: `price_stradamus.models.classical.arima`

        ARIMA model for time series forecasting...

        ---
    """
    models = ModelRegistry.list_models()

    docs = ["# Available Models\n\n"]
    docs.append(f"**Total Models**: {len(models)}\n\n")
    docs.append("---\n\n")

    # Group models by category
    neural_models = []
    classical_models = []
    ml_models = []

    for model_name in sorted(models):
        info = ModelRegistry.get_model_info(model_name)
        module = info["module"]

        if "neural" in module:
            neural_models.append((model_name, info))
        elif "classical" in module:
            classical_models.append((model_name, info))
        elif "ml" in module:
            ml_models.append((model_name, info))

    # Neural Models Section
    if neural_models:
        docs.append("## Neural Models\n\n")
        docs.append("Deep learning models using PyTorch and neural architectures.\n\n")

        for model_name, info in neural_models:
            docs.append(f"### {model_name}\n\n")
            docs.append(f"**Class**: `{info['class']}`\n\n")
            docs.append(f"**Module**: `{info['module']}`\n\n")
            docs.append(f"{info['description']}\n\n")
            docs.append("---\n\n")

    # Classical Models Section
    if classical_models:
        docs.append("## Classical Models\n\n")
        docs.append("Statistical time series models.\n\n")

        for model_name, info in classical_models:
            docs.append(f"### {model_name}\n\n")
            docs.append(f"**Class**: `{info['class']}`\n\n")
            docs.append(f"**Module**: `{info['module']}`\n\n")
            docs.append(f"{info['description']}\n\n")
            docs.append("---\n\n")

    # ML Models Section
    if ml_models:
        docs.append("## Machine Learning Models\n\n")
        docs.append("Tree-based and ensemble models.\n\n")

        for model_name, info in ml_models:
            docs.append(f"### {model_name}\n\n")
            docs.append(f"**Class**: `{info['class']}`\n\n")
            docs.append(f"**Module**: `{info['module']}`\n\n")
            docs.append(f"{info['description']}\n\n")
            docs.append("---\n\n")

    # Usage Section
    docs.append("## Usage\n\n")
    docs.append("### Via CLI\n\n")
    docs.append("```bash\n")
    docs.append("# List all available models\n")
    docs.append("price-stradamus list-models\n\n")
    docs.append("# Train a specific model\n")
    docs.append("price-stradamus train --model MODEL_NAME --epochs 100\n\n")
    docs.append("# Evaluate a model\n")
    docs.append("price-stradamus evaluate --model MODEL_NAME\n\n")
    docs.append("# Compare models\n")
    docs.append("price-stradamus compare --models nbeats lstm xgboost\n")
    docs.append("```\n\n")

    docs.append("### Via Python API\n\n")
    docs.append("```python\n")
    docs.append("from price_stradamus.models.registry import ModelRegistry\n")
    docs.append("from darts import TimeSeries\n\n")
    docs.append("# Create model\n")
    docs.append('model = ModelRegistry.create_model("nbeats", n_epochs=100)\n\n')
    docs.append("# Train\n")
    docs.append("model.fit(train_data, val_data)\n\n")
    docs.append("# Predict\n")
    docs.append("predictions = model.predict(n=5)\n\n")
    docs.append("# Save\n")
    docs.append('model.save(Path("models/nbeats.pth"))\n')
    docs.append("```\n\n")

    return "".join(docs)


def generate_model_comparison_table() -> str:
    """Generate comparison table for all models.

    Returns:
        Markdown table comparing model characteristics

    Example:
        >>> table = generate_model_comparison_table()
        >>> print(table)
        | Model | Type | GPU | Epochs | Speed | Best For |
        |-------|------|-----|--------|-------|----------|
        ...
    """
    models = ModelRegistry.list_models()

    table = ["# Model Comparison\n\n"]
    table.append("| Model | Type | GPU | Training | Speed | Complexity |\n")
    table.append("|-------|------|-----|----------|-------|------------|\n")

    # Model characteristics (could be extended with actual measurements)
    characteristics = {
        "nbeats": {
            "type": "Neural",
            "gpu": "✓",
            "training": "Epochs",
            "speed": "Medium",
            "complexity": "High",
        },
        "lstm": {
            "type": "Neural",
            "gpu": "✓",
            "training": "Epochs",
            "speed": "Fast",
            "complexity": "Medium",
        },
        "tcn": {
            "type": "Neural",
            "gpu": "✓",
            "training": "Epochs",
            "speed": "Fast",
            "complexity": "Medium",
        },
        "tft": {
            "type": "Neural",
            "gpu": "✓",
            "training": "Epochs",
            "speed": "Slow",
            "complexity": "Very High",
        },
        "arima": {
            "type": "Classical",
            "gpu": "✗",
            "training": "Fit",
            "speed": "Medium",
            "complexity": "Low",
        },
        "prophet": {
            "type": "Classical",
            "gpu": "✗",
            "training": "Fit",
            "speed": "Slow",
            "complexity": "Low",
        },
        "xgboost": {
            "type": "ML",
            "gpu": "✓*",
            "training": "Boosting",
            "speed": "Very Fast",
            "complexity": "Medium",
        },
        "random_forest": {
            "type": "ML",
            "gpu": "✗",
            "training": "Bagging",
            "speed": "Very Fast",
            "complexity": "Low",
        },
    }

    for model_name in sorted(models):
        if model_name in characteristics:
            char = characteristics[model_name]
            table.append(
                f"| {model_name} | {char['type']} | {char['gpu']} | "
                f"{char['training']} | {char['speed']} | {char['complexity']} |\n"
            )

    table.append("\n*XGBoost GPU support depends on installation\n\n")

    return "".join(table)


if __name__ == "__main__":
    print("Generating model documentation...\n")

    # Generate full documentation
    docs = generate_model_docs()
    print(docs)

    # Generate comparison table
    print("\n" + "=" * 80 + "\n")
    table = generate_model_comparison_table()
    print(table)

    # Save to file
    output_file = "MODEL_DOCUMENTATION.md"
    with open(output_file, "w") as f:
        f.write(docs)
        f.write("\n" + "=" * 80 + "\n\n")
        f.write(table)

    print(f"\nDocumentation saved to {output_file}")
