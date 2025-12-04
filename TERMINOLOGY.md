# Price Stradamus - Terminology Guide

**Purpose**: Clarify the two different concepts of "phases" and "scale"

---

## Two Separate Concepts

### 1. Product Phases (WHAT features to build)

These are **feature milestones** from your product roadmap:

```
Phase 1: Core System ✅
├─ 8 forecasting models (N-BEATS, LSTM, TCN, TFT, ARIMA, Prophet, XGBoost, RF)
├─ PostgreSQL data storage
├─ Binance API integration
├─ Feature engineering (50+ indicators)
├─ Backtesting framework
└─ CLI interface

Phase 2: AutoML Integration ⏳
├─ auto-sklearn integration
├─ Automated hyperparameter tuning (Optuna)
├─ Ensemble methods
└─ Experiment tracking (MLflow)

Phase 3: Bayesian Tournament System ⏳
├─ Custom tournament selection
├─ Bayesian optimization
├─ Multi-model ensembles
└─ Adaptive weighting

Phase 4: FastAPI REST API ⏳
├─ REST endpoints
├─ Authentication (JWT)
├─ Rate limiting
└─ API documentation

Phase 5: Real-Time Streaming ⏳
├─ WebSocket connections
├─ Live price updates
├─ Streaming predictions
└─ Event-driven architecture

Phase 6: Google Cloud Deployment ⏳
├─ GKE (Kubernetes)
├─ Cloud SQL
├─ Cloud Storage
└─ Auto-scaling

Phase 7: Advanced Models ⏳
├─ Transformer models
├─ Diffusion models
├─ Multimodal inputs
└─ Alternative data integration
```

**Source**: [context/roadmap.md](context/roadmap.md)

---

### 2. Scale Levels (HOW MANY users)

These are **infrastructure tiers** based on user count:

```
Personal Use
├─ Scale: Just you
├─ Infrastructure: Local laptop + Docker
├─ Cost: $0/month
└─ Complexity: Simple

Team Scale
├─ Scale: <1k users
├─ Infrastructure: Single VPS
├─ Cost: ~$50/month
└─ Complexity: Still simple

Production Scale
├─ Scale: <10k users
├─ Infrastructure: Small cluster (2-3 servers)
├─ Cost: ~$200/month
└─ Complexity: Monitoring, CI/CD

Public API Scale
├─ Scale: <100k users
├─ Infrastructure: Load-balanced cluster
├─ Cost: ~$500/month
└─ Complexity: Multi-tenant, rate limiting

High Traffic Scale
├─ Scale: <500k users
├─ Infrastructure: Distributed system
├─ Cost: ~$1000/month
└─ Complexity: Caching, CDN, read replicas

Enterprise/Cloud Scale
├─ Scale: 1M+ users
├─ Infrastructure: GKE, multi-region
├─ Cost: $2000+/month
└─ Complexity: Service mesh, auto-scaling, chaos engineering
```

---

## How They Relate

**Product Phases** and **Scale Levels** are INDEPENDENT:

| Scenario | Product Phase | Scale Level |
|----------|---------------|-------------|
| **You right now** | Phase 1 ✅ | Personal Use |
| **You + automated tuning** | Phase 2 | Personal Use |
| **You sharing with friends** | Phase 1 ✅ | Team Scale |
| **Public API, many users** | Phase 4 | Public API Scale |
| **Cloud deployment** | Phase 6 | Enterprise Scale |

**Examples**:
- You can be at **Phase 3 (Bayesian)** but still **Personal Scale** (just you testing)
- You can be at **Phase 1 (Core)** but need **Team Scale** (50 users want access)
- Most likely path: **Phase 1 → Phase 2 → Phase 3** at **Personal Scale**, then scale up infrastructure later

---

## Decision Matrix

### Should I add a new feature? (Product Phases)

```
Ask: "Does this improve predictions or enable new capabilities?"

YES → Add the feature (move to next Product Phase)
NO  → Don't add it yet
```

### Should I scale infrastructure? (Scale Levels)

```
Ask: "Am I constrained by current infrastructure?"

User count exceeded? → Scale up
Performance issues? → Scale up
Still works fine? → Don't scale yet
```

---

## Documentation Markers

Throughout the docs, you'll see:

### Product Phase Markers

- `[PHASE 1]` = Core System features
- `[PHASE 2]` = AutoML features
- `[PHASE 3]` = Bayesian Tournament features
- `[PHASE 4-7]` = Future features

### Scale Level Markers

- `[PERSONAL]` = Local laptop patterns
- `[TEAM]` = Small VPS patterns
- `[PRODUCTION]` = Multi-server patterns
- `[CLOUD]` = GKE/enterprise patterns

### Combined Examples

```markdown
## AutoML Integration [PHASE 2] [PERSONAL]
You can use AutoML on your laptop

## Load Balancing [PHASE 4] [PUBLIC API]
Need this when serving many users

## Chaos Engineering [PHASE 6] [CLOUD]
Enterprise-scale reliability testing
```

---

## Current Status

**Product Phase**: Phase 1 - Core System ✅
- 8 models implemented
- Backtesting working
- CLI functional

**Scale Level**: Personal Use
- Just you
- Local laptop
- Docker PostgreSQL

---

## Next Steps

### Product Development Path
```
Phase 1 ✅ → Phase 2 ⏳ → Phase 3 ⏳ → Phase 4 ⏳ → ...
```

### Infrastructure Scaling Path
```
Personal ✅ → Team (if users) → Production (if popular) → Cloud (if huge)
```

**Key Insight**: You advance Product Phases based on features needed, not user count. You scale infrastructure based on user count, not features.

---

## Common Confusion (Fixed)

❌ **Before** (Confusing):
- "Phase 1" could mean Core System OR Personal Scale
- Wasn't clear if phases were features or infrastructure
- Mixing feature development with scaling concerns

✅ **After** (Clear):
- **Product Phase 1** = Core System features
- **Personal Scale** = Infrastructure for just you
- Separate concerns: features vs infrastructure

---

## Quick Reference

| Term | Meaning | Drives Decision |
|------|---------|-----------------|
| **Product Phase 1-7** | Feature milestones | "What should I build?" |
| **Personal Scale** | Just you, laptop | "How many users?" |
| **Team Scale** | <1k users, VPS | "How many users?" |
| **Production Scale** | <10k users, cluster | "How many users?" |
| **Cloud Scale** | 100k+ users, GKE | "How many users?" |

---

## Examples in Practice

### Scenario 1: Solo Developer
- Product Phase: 1 → 2 → 3 (adding features)
- Scale: Personal → Personal → Personal (still just you)
- **Action**: Add features, keep infrastructure simple

### Scenario 2: Got Users
- Product Phase: 1 (just core system)
- Scale: Personal → Team (100 users signed up!)
- **Action**: Add authentication, deploy to VPS, but don't add Phase 2 features yet

### Scenario 3: Popular Service
- Product Phase: 4 (REST API done)
- Scale: Team → Production (5000 users!)
- **Action**: Add monitoring, load balancing, but Phase 5 features can wait

---

## Key Takeaway

```
Product Phases = WHAT to build (features)
Scale Levels = HOW MUCH to build for (users/load)

They're independent dimensions!
```

**Your Current State**:
- ✅ Product Phase 1: Core System
- ✅ Personal Scale: Just you

**Focus**: Get >52% directional accuracy (Product Phase 1 goal) before anything else!

---

*Last Updated: 2025-12-03*
*Purpose: Clarify product phases vs scale levels*
