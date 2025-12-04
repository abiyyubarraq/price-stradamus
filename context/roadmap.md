# Price Stradamus - Project Roadmap

## Vision

Build a production-ready, scalable Bitcoin price prediction system that achieves consistent directional accuracy >55% and enables profitable trading strategies.

---

## Roadmap Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase Dependencies                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1 ─────┬─────► Phase 2 ─────► Phase 3                               │
│  (Core)       │       (AutoML)       (Tournament)                          │
│               │                                                             │
│               ├─────► Phase 4 ─────► Phase 5 ─────► Phase 6                │
│               │       (API)          (Streaming)    (Cloud)                │
│               │                                                             │
│               └─────► Phase 7 ─────► Phase 8                               │
│                       (Advanced)     (Multi-Asset)                         │
│                                                                             │
│                              Phase 9 (Trading) - Optional, requires 4+5+6  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Maturity Levels

| Phase | Status | Maturity | Risk Level |
|-------|--------|----------|------------|
| Phase 1: Core System | Complete | Production | Low |
| Phase 2: AutoML | Not Started | Development | Medium |
| Phase 3: Tournament | Not Started | Research | High |
| Phase 4: Web API | Not Started | Development | Low |
| Phase 5: Streaming | Not Started | Development | Medium |
| Phase 6: Cloud | Not Started | Development | Medium |
| Phase 7: Advanced Models | Not Started | Research | High |
| Phase 8: Multi-Asset | Not Started | Research | High |
| Phase 9: Trading | Not Started | Development | Critical |

---

## Phase 1: Core System ✅ (Complete)

**Goal**: Build functional local prediction system

### Deliverables

- [x] Project setup and architecture
- [x] Binance data fetching pipeline
- [x] PostgreSQL storage with proper schema
- [x] Feature engineering with pandas-ta (50+ indicators)
- [x] N-BEATS model implementation
- [x] LSTM model implementation
- [x] TCN model implementation
- [x] TFT neural model implementation
- [x] Classical models (ARIMA, Prophet) implementation
- [x] ML models (XGBoost, RandomForest) implementation
- [x] Walk-forward backtesting
- [x] Comprehensive evaluation metrics (MAE, RMSE, MAPE, directional accuracy)
- [x] CLI interface (7 commands)
- [x] Docker setup (PostgreSQL + pgAdmin)
- [x] Testing infrastructure (pytest with fixtures)
- [x] Custom exception hierarchy
- [x] Model comparison utility

### Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Data fetch (30 days 1m candles) | Success | ✅ |
| N-BEATS training time | <30 min (GPU) | ✅ |
| Directional accuracy | >50% | ✅ |
| Test coverage | >80% | ✅ |
| Ruff/Pyright checks | Pass | ✅ |

### Technical Debt

- [ ] Database query optimization for large datasets
- [ ] Model checkpoint compression
- [ ] Feature caching layer

---

## Phase 2: AutoML Integration

**Goal**: Automate model selection and hyperparameter tuning

### Dependencies

```python
# Required from Phase 1
- ✅ BaseModel interface
- ✅ Evaluation metrics (MAE, RMSE, directional accuracy)
- ✅ Walk-forward validation framework
- ✅ Feature engineering pipeline
```

### Features

- [ ] Integrate auto-sklearn for ML models
- [ ] Automated feature selection
- [ ] Hyperparameter optimization with Optuna
- [ ] Ensemble model creation
- [ ] Model registry and versioning
- [ ] Experiment tracking (MLflow or Weights & Biases)
- [ ] Automated retraining pipeline

### Technical Implementation

#### Optuna Integration

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
from darts import TimeSeries

from price_stradamus.models.base import BaseModel
from price_stradamus.models.registry import model_registry


@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""

    model_type: str
    params: dict[str, tuple[str, Any, Any]]  # (type, min, max)


class OptunaOptimizer:
    """Hyperparameter optimization with Optuna."""

    def __init__(
        self,
        search_space: HyperparameterSpace,
        metric: str = "mae",
        direction: str = "minimize",
    ) -> None:
        self.search_space = search_space
        self.metric = metric
        self.direction = direction
        self.study: optuna.Study | None = None

    def create_objective(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries,
    ) -> callable:
        """Create Optuna objective function."""

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for name, (param_type, low, high) in self.search_space.params.items():
                match param_type:
                    case "int":
                        params[name] = trial.suggest_int(name, low, high)
                    case "float":
                        params[name] = trial.suggest_float(name, low, high)
                    case "loguniform":
                        params[name] = trial.suggest_float(name, low, high, log=True)
                    case "categorical":
                        params[name] = trial.suggest_categorical(name, low)

            # Create and train model
            model_class = model_registry.get(self.search_space.model_type)
            model = model_class(**params)
            model.fit(train_data)

            # Evaluate
            predictions = model.predict(len(val_data))
            metrics = model.evaluate(val_data, predictions)

            return metrics[self.metric]

        return objective

    def optimize(
        self,
        train_data: TimeSeries,
        val_data: TimeSeries,
        n_trials: int = 100,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run optimization.

        Args:
            train_data: Training data
            val_data: Validation data
            n_trials: Number of trials
            timeout: Optional timeout in seconds

        Returns:
            Best hyperparameters
        """
        self.study = optuna.create_study(direction=self.direction)
        objective = self.create_objective(train_data, val_data)

        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        return self.study.best_params


# Usage
search_space = HyperparameterSpace(
    model_type="nbeats",
    params={
        "input_chunk_length": ("int", 30, 120),
        "output_chunk_length": ("int", 1, 10),
        "num_stacks": ("int", 10, 50),
        "num_blocks": ("int", 1, 5),
        "learning_rate": ("loguniform", 1e-5, 1e-2),
    },
)

optimizer = OptunaOptimizer(search_space)
best_params = optimizer.optimize(train, val, n_trials=100)
```

#### Model Registry

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib


class ModelStatus(str, Enum):
    """Model lifecycle status."""

    TRAINING = "training"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model version with metadata."""

    model_id: str
    version: str
    model_type: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    training_data_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    status: ModelStatus = ModelStatus.TRAINING
    artifact_path: Path | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "training_data_hash": self.training_data_hash,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
        }


class ModelRegistry:
    """Version-controlled model registry."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path = registry_path / "models"
        self.models_path.mkdir(exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            self._registry = json.loads(registry_file.read_text())
        else:
            self._registry = {"models": {}, "production": {}}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"
        registry_file.write_text(json.dumps(self._registry, indent=2))

    def register_model(
        self,
        model: Any,
        model_type: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, float],
        training_data: Any,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model: Trained model object
            model_type: Type of model (e.g., "nbeats")
            hyperparameters: Model hyperparameters
            metrics: Evaluation metrics
            training_data: Training data (for hash)

        Returns:
            Model version info
        """
        # Generate model ID and version
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d')}"
        existing_versions = self._registry["models"].get(model_id, [])
        version = f"v{len(existing_versions) + 1}"

        # Hash training data for reproducibility
        data_hash = hashlib.sha256(
            str(training_data.values()).encode()
        ).hexdigest()[:12]

        # Save model artifact
        artifact_path = self.models_path / f"{model_id}_{version}.joblib"
        joblib.dump(model, artifact_path)

        # Create version entry
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            model_type=model_type,
            hyperparameters=hyperparameters,
            metrics=metrics,
            training_data_hash=data_hash,
            status=ModelStatus.VALIDATION,
            artifact_path=artifact_path,
        )

        # Update registry
        if model_id not in self._registry["models"]:
            self._registry["models"][model_id] = []
        self._registry["models"][model_id].append(model_version.to_dict())
        self._save_registry()

        return model_version

    def promote_to_production(
        self,
        model_id: str,
        version: str,
        symbol: str,
    ) -> bool:
        """Promote model version to production.

        Args:
            model_id: Model identifier
            version: Version string
            symbol: Trading symbol this model serves

        Returns:
            True if successful
        """
        # Find the version
        versions = self._registry["models"].get(model_id, [])
        for v in versions:
            if v["version"] == version:
                v["status"] = ModelStatus.PRODUCTION.value
                self._registry["production"][symbol] = {
                    "model_id": model_id,
                    "version": version,
                }
                self._save_registry()
                return True
        return False

    def get_production_model(self, symbol: str) -> Any | None:
        """Load production model for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Loaded model or None
        """
        prod_info = self._registry["production"].get(symbol)
        if not prod_info:
            return None

        model_id = prod_info["model_id"]
        version = prod_info["version"]
        artifact_path = self.models_path / f"{model_id}_{version}.joblib"

        if artifact_path.exists():
            return joblib.load(artifact_path)
        return None

    def rollback(self, symbol: str) -> bool:
        """Rollback to previous production model.

        Args:
            symbol: Trading symbol

        Returns:
            True if successful
        """
        prod_info = self._registry["production"].get(symbol)
        if not prod_info:
            return False

        model_id = prod_info["model_id"]
        versions = self._registry["models"].get(model_id, [])

        # Find previous production version
        current_version = prod_info["version"]
        previous_version = None

        for v in reversed(versions):
            if v["version"] != current_version and v["status"] == ModelStatus.ARCHIVED.value:
                previous_version = v["version"]
                break

        if previous_version:
            return self.promote_to_production(model_id, previous_version, symbol)

        return False
```

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| AutoML improvement over baseline | >10% MAE reduction | Compare best AutoML model vs Phase 1 best |
| Hyperparameter trials | 100+ per model type | Optuna trial count |
| Model registry | Functional | Version tracking, rollback working |
| Experiment tracking | Operational | MLflow/W&B dashboard accessible |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AutoML finds no improvement | Medium | Low | Keep Phase 1 models as fallback |
| Optuna search time too long | Medium | Medium | Implement early stopping, pruning |
| Memory issues with large search | Low | High | Use distributed optimization |
| Overfitting from exhaustive search | Medium | High | Validate on holdout set |

### Rollback Strategy

```python
class Phase2Rollback:
    """Rollback procedures for Phase 2."""

    @staticmethod
    def rollback_to_phase1() -> None:
        """Revert to Phase 1 configuration."""
        # 1. Disable AutoML CLI commands
        # 2. Use static hyperparameters from Phase 1
        # 3. Archive AutoML models
        # 4. Log rollback reason

    @staticmethod
    def partial_rollback() -> None:
        """Keep experiment tracking, disable AutoML."""
        # 1. Disable auto-sklearn integration
        # 2. Keep MLflow for manual experiments
        # 3. Use manual hyperparameter selection
```

### Technical Debt Budget

- **Acceptable**: <10% additional code complexity
- **New dependencies**: <5 new packages
- **Test coverage**: Maintain >80%
- **Performance**: <20% training time increase

---

## Phase 3: Bayesian Tournament System

**Goal**: Custom AutoML with evolutionary model selection

### Dependencies

```python
# Required from Phase 2
- ✅ Phase 1: Core System
- ✅ Phase 2: Optuna integration
- ✅ Phase 2: Model registry
- ✅ Phase 2: Experiment tracking
```

### Concept

Instead of grid/random search, use Bayesian optimization to run a "tournament" where models compete and evolve.

### Architecture

```
┌─────────────────────────────────────┐
│      Tournament Controller          │
│  (Bayesian Optimization Engine)     │
└────────────┬────────────────────────┘
             │
     ┌───────┴───────┐
     │               │
┌────▼────┐    ┌────▼────┐
│ Model A │    │ Model B │
│ (Parent)│    │ (Parent)│
└────┬────┘    └────┬────┘
     │               │
     └───────┬───────┘
             │
        ┌────▼────┐
        │ Model C │
        │ (Child) │
        │ Mutated │
        └─────────┘
```

### Features

- [ ] Bayesian hyperparameter optimization
- [ ] Model architecture search (NAS)
- [ ] Feature selection via genetic algorithm
- [ ] Multi-objective optimization (accuracy + speed + interpretability)
- [ ] Ensemble generation
- [ ] Performance-based model pruning

### Implementation

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


@dataclass
class Candidate:
    """Tournament candidate (model configuration)."""

    config: dict[str, Any]
    fitness: float | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Unique identifier based on config hash."""
        import hashlib
        config_str = str(sorted(self.config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


class AcquisitionFunction(ABC):
    """Base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        best_y: float,
    ) -> np.ndarray:
        """Compute acquisition values."""
        ...


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function."""

    def __init__(self, xi: float = 0.01) -> None:
        self.xi = xi

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        best_y: float,
    ) -> np.ndarray:
        """Compute Expected Improvement.

        Args:
            X: Points to evaluate
            gp: Fitted Gaussian Process
            best_y: Best observed value (to minimize)

        Returns:
            EI values for each point
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)

        # For minimization
        improvement = best_y - mu - self.xi
        Z = improvement / sigma

        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0

        return ei


class BayesianTournament:
    """Bayesian optimization tournament for model selection."""

    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.2,
    ) -> None:
        self.search_space = search_space
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
        )
        self.acquisition = ExpectedImprovement()

        self.population: list[Candidate] = []
        self.hall_of_fame: list[Candidate] = []
        self.history: list[Candidate] = []

    def _sample_config(self) -> dict[str, Any]:
        """Sample random configuration from search space."""
        config = {}
        for param, (low, high) in self.search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                config[param] = np.random.randint(low, high + 1)
            else:
                config[param] = np.random.uniform(low, high)
        return config

    def _config_to_array(self, config: dict[str, Any]) -> np.ndarray:
        """Convert config dict to numpy array."""
        return np.array([config[k] for k in sorted(self.search_space.keys())])

    def _array_to_config(self, arr: np.ndarray) -> dict[str, Any]:
        """Convert numpy array to config dict."""
        keys = sorted(self.search_space.keys())
        config = {}
        for i, key in enumerate(keys):
            low, high = self.search_space[key]
            if isinstance(low, int) and isinstance(high, int):
                config[key] = int(np.clip(arr[i], low, high))
            else:
                config[key] = float(np.clip(arr[i], low, high))
        return config

    def _mutate(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply mutation to configuration."""
        new_config = config.copy()
        for param, (low, high) in self.search_space.items():
            if np.random.random() < self.mutation_rate:
                if isinstance(low, int) and isinstance(high, int):
                    new_config[param] = np.random.randint(low, high + 1)
                else:
                    # Gaussian perturbation
                    std = (high - low) * 0.1
                    new_value = config[param] + np.random.normal(0, std)
                    new_config[param] = float(np.clip(new_value, low, high))
        return new_config

    def _crossover(
        self,
        parent1: Candidate,
        parent2: Candidate,
    ) -> dict[str, Any]:
        """Uniform crossover of two parent configurations."""
        child_config = {}
        for param in self.search_space:
            if np.random.random() < 0.5:
                child_config[param] = parent1.config[param]
            else:
                child_config[param] = parent2.config[param]
        return child_config

    def initialize_population(self) -> list[Candidate]:
        """Initialize random population."""
        return [
            Candidate(config=self._sample_config(), generation=0)
            for _ in range(self.population_size)
        ]

    def select_elite(self) -> list[Candidate]:
        """Select top performers."""
        evaluated = [c for c in self.population if c.fitness is not None]
        sorted_candidates = sorted(evaluated, key=lambda c: c.fitness)
        return sorted_candidates[:self.elite_size]

    def suggest_candidates(self, n: int) -> list[Candidate]:
        """Suggest new candidates using Bayesian optimization."""
        if len(self.history) < 5:
            # Not enough data, return random samples
            return [
                Candidate(config=self._sample_config(), generation=0)
                for _ in range(n)
            ]

        # Fit GP on history
        X = np.array([self._config_to_array(c.config) for c in self.history])
        y = np.array([c.fitness for c in self.history])
        self.gp.fit(X, y)

        # Generate candidate points
        best_y = min(y)
        candidates = []

        for _ in range(n):
            # Sample many random points
            random_configs = [self._sample_config() for _ in range(1000)]
            X_random = np.array([
                self._config_to_array(c) for c in random_configs
            ])

            # Select by acquisition function
            ei_values = self.acquisition(X_random, self.gp, best_y)
            best_idx = np.argmax(ei_values)

            candidates.append(Candidate(
                config=random_configs[best_idx],
                generation=max(c.generation for c in self.population) + 1,
            ))

        return candidates

    def evolve(
        self,
        elite: list[Candidate],
        suggested: list[Candidate],
    ) -> list[Candidate]:
        """Create next generation through evolution."""
        next_gen = list(elite)  # Keep elite

        # Add Bayesian suggestions
        next_gen.extend(suggested[:len(suggested)//2])

        # Fill rest with crossover + mutation
        while len(next_gen) < self.population_size:
            parent1, parent2 = np.random.choice(elite, size=2, replace=False)
            child_config = self._crossover(parent1, parent2)
            child_config = self._mutate(child_config)
            child = Candidate(
                config=child_config,
                generation=max(c.generation for c in elite) + 1,
                parent_ids=[parent1.id, parent2.id],
            )
            next_gen.append(child)

        return next_gen

    def run(
        self,
        evaluate_fn: callable,
        n_generations: int = 50,
        early_stopping_rounds: int = 10,
    ) -> Candidate:
        """Run tournament.

        Args:
            evaluate_fn: Function that takes config and returns fitness score
            n_generations: Maximum generations
            early_stopping_rounds: Stop if no improvement for this many rounds

        Returns:
            Best candidate found
        """
        self.population = self.initialize_population()
        best_fitness = float('inf')
        rounds_without_improvement = 0

        for generation in range(n_generations):
            # Evaluate population
            for candidate in self.population:
                if candidate.fitness is None:
                    candidate.fitness = evaluate_fn(candidate.config)
                    self.history.append(candidate)

            # Select elite
            elite = self.select_elite()

            # Update hall of fame
            if elite[0].fitness < best_fitness:
                best_fitness = elite[0].fitness
                self.hall_of_fame.append(elite[0])
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            # Early stopping
            if rounds_without_improvement >= early_stopping_rounds:
                break

            # Suggest new candidates
            suggested = self.suggest_candidates(self.population_size // 2)

            # Create next generation
            self.population = self.evolve(elite, suggested)

        return min(self.hall_of_fame, key=lambda c: c.fitness)
```

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Improvement over Phase 2 | >15% MAE reduction | Best tournament model vs Phase 2 best |
| Architecture search | Functional | Can discover novel architectures |
| Multi-objective | Pareto frontier | Trade-off curves generated |
| Tournament time | <24 hours | End-to-end runtime |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No improvement over Phase 2 | High | Medium | Phase 2 models remain valid |
| Compute cost too high | High | High | Use progressive search, cloud bursting |
| Overfitting to validation | Medium | High | Multiple holdout sets |
| Bayesian model collapse | Low | Medium | Ensemble of acquisition functions |

### Rollback Strategy

```python
class Phase3Rollback:
    """Rollback procedures for Phase 3."""

    @staticmethod
    def rollback_to_phase2() -> None:
        """Revert to Phase 2 AutoML."""
        # 1. Disable tournament system
        # 2. Use Optuna for hyperparameter search
        # 3. Archive tournament experiments
        # 4. Restore Phase 2 best models

    @staticmethod
    def partial_rollback() -> None:
        """Keep Bayesian optimization, disable evolution."""
        # 1. Use pure BO without crossover/mutation
        # 2. Simpler search strategy
```

### Technical Debt Budget

- **Acceptable**: <15% code complexity increase
- **New dependencies**: <3 packages (GPy, DEAP allowed)
- **Test coverage**: Maintain >75%
- **Compute budget**: <100 GPU hours per experiment

---

## Phase 4: Web API

**Goal**: Expose predictions via REST API

### Dependencies

```python
# Required from Phase 1
- ✅ Phase 1: Core System
- ✅ Phase 1: Prediction models
- ✅ Phase 1: Evaluation metrics
```

### Features

- [ ] FastAPI REST API
- [ ] Authentication and API keys
- [ ] Rate limiting
- [ ] Real-time predictions endpoint
- [ ] Historical predictions query
- [ ] Model metadata endpoint
- [ ] WebSocket for streaming predictions
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Monitoring dashboard

### API Design

```python
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

app = FastAPI(
    title="Price Stradamus API",
    version="1.0.0",
    description="Bitcoin price prediction API",
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


class Timeframe(str, Enum):
    """Supported timeframes."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"


class PredictionRequest(BaseModel):
    """Prediction request schema."""

    symbol: str = Field("BTCUSDT", description="Trading pair")
    timeframe: Timeframe = Field(Timeframe.ONE_MINUTE, description="Candle timeframe")
    steps: int = Field(5, ge=1, le=100, description="Prediction horizon")
    model: str = Field("nbeats", description="Model to use")


class PredictionResponse(BaseModel):
    """Prediction response schema."""

    symbol: str
    timeframe: str
    model: str
    predictions: list[float]
    confidence_intervals: list[tuple[float, float]] | None = None
    timestamp: datetime
    latency_ms: float


class ModelInfo(BaseModel):
    """Model information schema."""

    name: str
    version: str
    type: str
    metrics: dict[str, float]
    last_trained: datetime


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    api_key: Annotated[str, Depends(api_key_header)],
) -> PredictionResponse:
    """Generate price predictions.

    Args:
        request: Prediction parameters
        api_key: API key for authentication

    Returns:
        Prediction response with forecasts
    """
    # Implementation here
    ...


@app.get("/api/v1/models", response_model=list[ModelInfo])
async def list_models(
    api_key: Annotated[str, Depends(api_key_header)],
) -> list[ModelInfo]:
    """List available prediction models.

    Args:
        api_key: API key for authentication

    Returns:
        List of available models
    """
    ...


@app.get("/api/v1/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
```

### Infrastructure

- Docker Compose with API + DB + Redis (caching)
- Nginx reverse proxy
- SSL/TLS certificates
- Load balancing (multiple API instances)

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cached response latency | <100ms | p95 latency |
| New prediction latency | <2s | p95 latency |
| Uptime | 99.9% | Monthly average |
| Request capacity | 1000 req/min | Load test |
| Documentation | Complete | OpenAPI spec coverage |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Security vulnerabilities | Medium | Critical | Security audit, penetration testing |
| Performance degradation | Medium | High | Caching, load balancing |
| API abuse | High | Medium | Rate limiting, authentication |
| Documentation drift | Medium | Low | Auto-generated from code |

### Rollback Strategy

```python
class Phase4Rollback:
    """Rollback procedures for Phase 4."""

    @staticmethod
    def emergency_shutdown() -> None:
        """Shutdown API completely."""
        # 1. Return 503 for all endpoints
        # 2. Disable external access
        # 3. Alert on-call team

    @staticmethod
    def read_only_mode() -> None:
        """Disable prediction endpoints, keep health/info."""
        # 1. Return 503 for /predict
        # 2. Keep /health and /models active
        # 3. Serve cached predictions only
```

---

## Phase 5: Real-Time Streaming

**Goal**: Live predictions as new candles arrive

### Dependencies

```python
# Required
- ✅ Phase 1: Core System
- ✅ Phase 4: Web API
- ✅ Phase 4: WebSocket infrastructure
```

### Features

- [ ] Binance WebSocket integration
- [ ] Real-time feature computation
- [ ] Streaming prediction pipeline
- [ ] Redis for caching recent data
- [ ] Kafka for event streaming (optional)
- [ ] Live dashboard (Grafana or custom)

### Architecture

```
Binance WebSocket → Event Queue → Feature Engine → Model → Predictions → WebSocket Clients
                         ↓
                    PostgreSQL
```

### Implementation

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from binance import AsyncClient, BinanceSocketManager


@dataclass
class Candle:
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str


class StreamingPipeline:
    """Real-time prediction pipeline."""

    def __init__(
        self,
        symbols: list[str],
        model_name: str = "nbeats",
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        self.symbols = symbols
        self.model_name = model_name
        self.redis_url = redis_url
        self._client: AsyncClient | None = None
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    async def start(self) -> None:
        """Start streaming pipeline."""
        self._client = await AsyncClient.create()
        bsm = BinanceSocketManager(self._client)

        # Create tasks for each symbol
        tasks = [
            asyncio.create_task(self._stream_symbol(bsm, symbol))
            for symbol in self.symbols
        ]

        await asyncio.gather(*tasks)

    async def _stream_symbol(
        self,
        bsm: BinanceSocketManager,
        symbol: str,
    ) -> None:
        """Stream candles for a symbol."""
        async with bsm.kline_socket(symbol=symbol, interval="1m") as stream:
            while True:
                msg = await stream.recv()
                if msg["e"] == "kline" and msg["k"]["x"]:  # Closed candle
                    candle = self._parse_candle(msg)
                    await self._process_candle(candle)

    def _parse_candle(self, msg: dict[str, Any]) -> Candle:
        """Parse Binance WebSocket message to Candle."""
        k = msg["k"]
        return Candle(
            timestamp=datetime.fromtimestamp(k["t"] / 1000),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            symbol=k["s"],
        )

    async def _process_candle(self, candle: Candle) -> None:
        """Process new candle and generate prediction."""
        # 1. Update feature cache
        features = await self._update_features(candle)

        # 2. Generate prediction
        prediction = await self._predict(features)

        # 3. Store prediction
        await self._store_prediction(candle, prediction)

        # 4. Broadcast to subscribers
        await self._broadcast(candle.symbol, prediction)

    async def _update_features(self, candle: Candle) -> dict[str, float]:
        """Update feature cache with new candle."""
        # Implementation
        ...

    async def _predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Generate prediction from features."""
        # Implementation
        ...

    async def _store_prediction(
        self,
        candle: Candle,
        prediction: dict[str, Any],
    ) -> None:
        """Store prediction in database."""
        # Implementation
        ...

    async def _broadcast(
        self,
        symbol: str,
        prediction: dict[str, Any],
    ) -> None:
        """Broadcast prediction to subscribers."""
        queues = self._subscribers.get(symbol, [])
        for queue in queues:
            await queue.put(prediction)

    def subscribe(self, symbol: str) -> asyncio.Queue:
        """Subscribe to predictions for a symbol."""
        queue: asyncio.Queue = asyncio.Queue()
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []
        self._subscribers[symbol].append(queue)
        return queue
```

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Prediction latency | <1s after candle close | End-to-end timing |
| Missed candles | 0 | Monitoring alert |
| Concurrent symbols | 10+ | Load test |
| Dashboard | Operational | Visual confirmation |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| WebSocket disconnection | High | Medium | Auto-reconnect with backoff |
| Feature computation delay | Medium | High | Pre-compute, cache |
| Binance rate limits | Medium | High | Connection pooling |
| Memory leaks | Low | High | Profiling, bounded queues |

---

## Phase 6: Cloud Deployment (GCP)

**Goal**: Deploy to Google Cloud Platform for scalability

### Dependencies

```python
# Required
- ✅ Phase 1: Core System
- ✅ Phase 4: Web API
- ✅ Phase 5: Real-Time Streaming
```

### Architecture

```
┌─────────────────────────────────────────────┐
│        Google Cloud Platform                 │
│                                              │
│  ┌──────────────┐      ┌──────────────┐    │
│  │  Cloud Run   │◀────▶│  Cloud SQL   │    │
│  │  (API)       │      │  (Postgres)  │    │
│  └──────┬───────┘      └──────────────┘    │
│         │                                    │
│         ▼                                    │
│  ┌──────────────┐      ┌──────────────┐    │
│  │  Cloud       │      │  Cloud       │    │
│  │  Storage     │      │  Scheduler   │    │
│  │  (Models)    │      │  (Training)  │    │
│  └──────────────┘      └──────────────┘    │
│                                              │
│  ┌──────────────────────────────────┐       │
│  │  Vertex AI                       │       │
│  │  (Model Training & Serving)      │       │
│  └──────────────────────────────────┘       │
│                                              │
│  ┌──────────────┐      ┌──────────────┐    │
│  │  Cloud       │      │  Cloud       │    │
│  │  Monitoring  │      │  Logging     │    │
│  └──────────────┘      └──────────────┘    │
└─────────────────────────────────────────────┘
```

### Services

- **Cloud Run**: Containerized API (auto-scaling)
- **Cloud SQL**: Managed PostgreSQL
- **Cloud Storage**: Model artifacts, logs
- **Vertex AI**: Model training and serving
- **Cloud Scheduler**: Cron jobs for retraining
- **Cloud Monitoring**: Metrics and alerting
- **Cloud Logging**: Centralized logs

### Features

- [ ] Infrastructure as Code (Terraform)
- [ ] CI/CD pipeline (Cloud Build)
- [ ] Auto-scaling based on load
- [ ] Multi-region deployment
- [ ] Automated backups
- [ ] Disaster recovery plan
- [ ] Cost optimization

### Terraform Configuration

```hcl
# main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Cloud SQL
resource "google_sql_database_instance" "main" {
  name             = "price-stradamus-db"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier              = "db-custom-2-4096"
    availability_type = "REGIONAL"

    backup_configuration {
      enabled            = true
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
    }
  }
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "api" {
  name     = "price-stradamus-api"
  location = var.region

  template {
    containers {
      image = "gcr.io/${var.project_id}/price-stradamus-api:latest"

      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
      }

      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_url.id
            version = "latest"
          }
        }
      }
    }

    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }
  }
}
```

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| API latency (p95) | <1s | Cloud Monitoring |
| Monthly cost | <$500 | Billing reports |
| Uptime SLA | 99.99% | Status page |
| Auto-scaling | Functional | Load test verification |
| Disaster recovery | <4h RTO | DR drill |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cost overrun | High | High | Budget alerts, quotas |
| Region outage | Low | Critical | Multi-region deployment |
| Security breach | Low | Critical | IAM, VPC, security scanning |
| Vendor lock-in | Medium | Medium | Terraform abstraction |

---

## Phase 7: Advanced Models

**Goal**: Explore cutting-edge architectures

### Dependencies

```python
# Required
- ✅ Phase 1: Core System
- ✅ Phase 2: Experiment tracking
```

### Models to Explore

#### 1. Transformer-Based Models
- **Informer**: Efficient transformer for long sequences
- **Autoformer**: Auto-correlation mechanism
- **Fedformer**: Frequency-enhanced transformer
- **PatchTST**: Patching for transformers

#### 2. Diffusion Models
- **TimeGrad**: Autoregressive diffusion for time series
- **CSDI**: Conditional score-based diffusion
- **SSSD**: Structured state-space diffusion

#### 3. Graph Neural Networks
- **Temporal Graph Networks**: Model relationships between assets
- **Spatial-Temporal GNN**: Multi-asset forecasting
- **Correlation modeling**: Learn asset co-movements

#### 4. Neural ODE
- **Continuous-time models**: Neural Ordinary Differential Equations
- **Latent ODE**: Learn latent dynamics
- **Irregular time series**: Handle missing timestamps

#### 5. Foundation Models
- **TimesFM**: Google's time series foundation model
- **Chronos**: Amazon's time series model
- **Lag-Llama**: LLM-based forecasting

### Research Questions

- Can transformers beat N-BEATS on crypto data?
- Do diffusion models handle volatility better?
- Can GNNs leverage multi-asset correlations?
- Are foundation models good zero-shot predictors?

### Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Model improvement | >5% over baseline | MAE reduction |
| Novel architecture | 1+ working | Experiment results |
| Research contribution | Publishable | Internal review |

---

## Phase 8: Multi-Asset & Multi-Timeframe

**Goal**: Expand beyond Bitcoin 1-minute candles

### Dependencies

```python
# Required
- ✅ Phase 1: Core System
- ✅ Phase 4: Web API
- ✅ Phase 7: Advanced Models (for GNN)
```

### Features

- [ ] Support all major cryptocurrencies
- [ ] Multi-timeframe predictions (1m, 5m, 1h, 1d)
- [ ] Cross-asset correlation modeling
- [ ] Portfolio optimization
- [ ] Risk management system

### Architecture

```
┌─────────────────────────────────────┐
│   Multi-Asset Prediction System     │
│                                     │
│  BTC ─┐                             │
│  ETH ─┼─→ Correlation Model ──→ GNN │
│  SOL ─┤                         │   │
│  ... ─┘                         ↓   │
│                           Predictions│
│  1m ─┐                          │   │
│  5m ─┼─→ Multi-Scale Model ─────┘   │
│  1h ─┤                               │
│  1d ─┘                               │
└─────────────────────────────────────┘
```

---

## Phase 9: Trading Integration (Optional)

**Goal**: Execute trades based on predictions

### Dependencies

```python
# Required
- ✅ Phase 1: Core System
- ✅ Phase 4: Web API
- ✅ Phase 5: Real-Time Streaming
- ✅ Phase 6: Cloud Deployment
```

### Features

- [ ] Binance trading integration
- [ ] Order execution engine
- [ ] Position management
- [ ] Risk management (stop-loss, take-profit)
- [ ] Performance tracking
- [ ] Paper trading mode
- [ ] Backtesting with transaction costs
- [ ] Portfolio allocation

### Risk Warning

**Automated trading is extremely risky.** Implement extensive safeguards:

1. **Start with paper trading** - Test thoroughly before real money
2. **Small position sizes** - Never risk more than 1-2% per trade
3. **Stop-loss mandatory** - Always have exit strategy
4. **Max drawdown limits** - Halt trading if losses exceed threshold
5. **Manual approval** - Require human approval for large trades
6. **Circuit breakers** - Automatic shutdown on anomalies

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Total loss of capital | Medium | Critical | Position limits, paper trading first |
| API key compromise | Low | Critical | Key rotation, IP whitelist |
| Model failure | Medium | High | Fallback strategies, monitoring |
| Flash crash | Low | Critical | Circuit breakers, max position |
| Regulatory issues | Medium | Critical | Legal review, compliance |

---

## Success Metrics Summary

### Technical Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Directional Accuracy | >55% | 1+ |
| MAE | <0.5% of price | 1+ |
| Inference latency | <100ms | 4+ |
| API uptime | 99.9% | 4+ |
| Training time | <1 hour | 1+ |

### Infrastructure Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Monthly cost | <$1000 | 6 |
| Auto-scaling | Functional | 6 |
| Multi-region | 2+ regions | 6 |
| Disaster recovery | <4h RTO | 6 |

---

## Risk Mitigation Summary

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Model overfitting | Rigorous walk-forward validation |
| Data quality issues | Comprehensive validation pipeline |
| API rate limits | Caching, WebSocket streaming |
| Scalability | Cloud-native architecture |

### Market Risks

| Risk | Mitigation |
|------|------------|
| Crypto volatility | Risk management features |
| Regulatory changes | Adaptable architecture |
| Black swan events | Outlier detection, circuit breakers |

### Operational Risks

| Risk | Mitigation |
|------|------------|
| Team changes | Documentation, knowledge transfer |
| Dependency issues | Version pinning, security audits |
| Infrastructure failure | Multi-region, disaster recovery |

---

*Last Updated: 2025-12-03*
