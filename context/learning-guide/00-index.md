# Price Stradamus - Complete Learning Guide

**For TypeScript/React/Node.js Developers Learning Python & Machine Learning**

Welcome! This comprehensive guide will take you from TypeScript developer to Python ML engineer through the lens of the Price Stradamus project. By the end, you'll understand every component and be able to rebuild this Bitcoin prediction system from scratch.

## 🎯 Learning Objectives

By completing this guide, you will:
- Master Python fundamentals from a TypeScript perspective
- Understand machine learning and time series forecasting concepts
- Work with ML libraries: PyTorch, Darts, pandas, numpy, scikit-learn
- Build production-ready ML systems with PostgreSQL
- Implement 8 different prediction models (neural networks, classical, ML)
- Evaluate models properly with walk-forward validation
- Follow ML engineering best practices

## 📚 Learning Path

This guide is structured as a progressive learning journey. Follow the modules in order:

### Phase 1: Python & Data Foundations (Week 1)
**Goal: Get comfortable with Python and data manipulation**

1. **[Python for TypeScript Developers](01-python-for-ts-devs.md)** ⭐ START HERE
   - Syntax comparison (TypeScript vs Python)
   - Type hints and Pydantic
   - Classes, decorators, comprehensions
   - **Time: 2-3 hours**

2. **[Async & Data Structures](02-async-and-data-structures.md)**
   - Async/await patterns (similar to Node.js)
   - Pandas DataFrames (like arrays on steroids)
   - Numpy arrays (mathematical operations)
   - Working with time series data
   - **Time: 3-4 hours**

3. **[ML Fundamentals](03-ml-fundamentals.md)**
   - What is machine learning? (concepts, not math)
   - Time series forecasting explained
   - Training, validation, testing
   - Overfitting and why it matters
   - **Time: 2-3 hours**

### Phase 2: Architecture & Data Pipeline (Week 2)
**Goal: Understand the project structure and data flow**

4. **[Project Architecture](04-project-architecture.md)**
   - Code organization (like a well-structured Node.js app)
   - Layer separation (CLI → Services → Models → Data)
   - How everything connects
   - File-by-file walkthrough
   - **Time: 2-3 hours**

5. **[Database & Data Pipeline](05-database-and-data-pipeline.md)**
   - PostgreSQL setup (like MongoDB but relational)
   - Async database operations
   - Fetching from Binance API
   - Data validation and cleaning
   - **Time: 3-4 hours**

6. **[Feature Engineering](06-feature-engineering.md)**
   - What are technical indicators?
   - RSI, MACD, Bollinger Bands explained
   - Using pandas-ta library
   - Creating lag features
   - **Time: 3-4 hours**

### Phase 3: Machine Learning Models (Week 3)
**Goal: Understand different model types and when to use them**

7. **[Model Fundamentals](07-model-fundamentals.md)**
   - Neural networks vs classical models vs ML models
   - When to use which model
   - Training loop explained
   - Hyperparameters demystified
   - **Time: 2-3 hours**

8. **[Neural Networks Deep Dive](08-neural-networks-deep-dive.md)**
   - N-BEATS (trend + seasonality)
   - LSTM (memory networks)
   - TCN (convolutional networks)
   - TFT (transformers)
   - PyTorch basics
   - **Time: 4-5 hours**

9. **[Classical & ML Models](09-classical-and-ml-models.md)**
   - ARIMA (statistical forecasting)
   - Prophet (Facebook's forecaster)
   - XGBoost (gradient boosting)
   - Random Forest (decision trees)
   - **Time: 3-4 hours**

### Phase 4: Evaluation & Integration (Week 4)
**Goal: Put it all together and evaluate properly**

10. **[Evaluation & Backtesting](10-evaluation-and-backtesting.md)**
    - Metrics: MAE, RMSE, MAPE, directional accuracy
    - Walk-forward validation (avoiding lookahead bias)
    - Interpreting results
    - Statistical significance
    - **Time: 3-4 hours**

11. **[CLI & Integration](11-cli-and-integration.md)**
    - Typer CLI framework
    - Commands: fetch, train, predict, evaluate
    - End-to-end workflows
    - Logging and debugging
    - **Time: 2-3 hours**

12. **[Best Practices & Exercises](12-best-practices-and-exercises.md)**
    - Code quality standards
    - Testing ML systems
    - Hands-on exercises
    - Mini-projects
    - Final challenge: Rebuild from scratch
    - **Time: Variable (practice as needed)**

## 🗺️ Learning Roadmap

```
┌──────────────────────────────────────────────────────────────────────┐
│                         LEARNING PATH                                │
│                                                                      │
│                   ┌─────────────────────────┐                       │
│                   │  Start: TS/React Dev    │                       │
│                   └───────────┬─────────────┘                       │
│                               │                                      │
│                               ▼                                      │
│        ╔═══════════════════════════════════════════╗                │
│        ║      PHASE 1: Python & Foundations        ║                │
│        ╚═══════════════════════════════════════════╝                │
│                   ┌───────────────────┐                             │
│                   │ 01: Python Basics │                             │
│                   └─────────┬─────────┘                             │
│                             │                                        │
│                   ┌─────────▼─────────┐                             │
│                   │ 02: Async & Data  │                             │
│                   └─────────┬─────────┘                             │
│                             │                                        │
│                   ┌─────────▼──────────┐                            │
│                   │ 03: ML Fundamentals│                            │
│                   └─────────┬──────────┘                            │
│                             │                                        │
│        ╔═══════════════════════════════════════════╗                │
│        ║   PHASE 2: Architecture & Data Pipeline   ║                │
│        ╚═══════════════════════════════════════════╝                │
│                   ┌─────────▼──────────┐                            │
│                   │ 04: Architecture   │                            │
│                   └─────────┬──────────┘                            │
│                             │                                        │
│                   ┌─────────▼─────────────┐                         │
│                   │ 05: Database Pipeline │                         │
│                   └─────────┬─────────────┘                         │
│                             │                                        │
│                   ┌─────────▼──────────────────┐                    │
│                   │ 06: Feature Engineering    │                    │
│                   └─────────┬──────────────────┘                    │
│                             │                                        │
│        ╔═══════════════════════════════════════════╗                │
│        ║      PHASE 3: Machine Learning Models     ║                │
│        ╚═══════════════════════════════════════════╝                │
│                   ┌─────────▼────────────┐                          │
│                   │ 07: Model Basics     │                          │
│                   └─────────┬────────────┘                          │
│                             │                                        │
│                   ┌─────────▼────────────────┐                      │
│                   │ 08: Neural Networks      │                      │
│                   └─────────┬────────────────┘                      │
│                             │                                        │
│                   ┌─────────▼──────────────────┐                    │
│                   │ 09: Classical/ML Models    │                    │
│                   └─────────┬──────────────────┘                    │
│                             │                                        │
│        ╔═══════════════════════════════════════════╗                │
│        ║    PHASE 4: Evaluation & Integration      ║                │
│        ╚═══════════════════════════════════════════╝                │
│                   ┌─────────▼────────────┐                          │
│                   │ 10: Evaluation       │                          │
│                   └─────────┬────────────┘                          │
│                             │                                        │
│                   ┌─────────▼─────────────────┐                     │
│                   │ 11: CLI Integration       │                     │
│                   └─────────┬─────────────────┘                     │
│                             │                                        │
│                   ┌─────────▼────────────────┐                      │
│                   │ 12: Best Practices       │                      │
│                   └─────────┬────────────────┘                      │
│                             │                                        │
│                             ▼                                        │
│                   ┌─────────────────────────┐                       │
│                   │ 🎯 Goal: Rebuild        │                       │
│                   │    Independently        │                       │
│                   └─────────────────────────┘                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## 📖 How to Use This Guide

### For Complete Python/ML Beginners
1. **Read sequentially** - Don't skip modules, each builds on the previous
2. **Code along** - Type out examples in a Python REPL or Jupyter notebook
3. **Run the actual code** - Install the project and run commands as you learn
4. **Do exercises** - Practice is crucial for retention
5. **Build challenges** - Apply knowledge to mini-projects

### For Quick Reference
- Use search (Ctrl+F) to find specific topics
- Each module has a "Quick Reference" section at the end
- Code examples are labeled with file paths for easy lookup
- Check the glossary in context/glossary.md

### Learning Style Tips

**🎓 Theory Learner?**
- Read the "Concepts" sections thoroughly
- Study the diagrams (we have lots!)
- Understand the "why" before the "how"

**⚡ Practical Learner?**
- Jump to "Code Walkthrough" sections
- Type out examples immediately
- Run commands and see results
- Experiment with modifications

**🔄 Comparative Learner?**
- Focus on "TypeScript vs Python" sections
- Use your TS/Node.js knowledge as anchor points
- Note similarities and differences

## 🛠️ Prerequisites

Before starting, ensure you have:

### Installed
- ✅ Python 3.13.9 (exact version required)
- ✅ uv package manager (`pip install uv`)
- ✅ VS Code with Python extension
- ✅ Docker Desktop (for PostgreSQL)
- ✅ Git

### Knowledge (You Already Have!)
- ✅ TypeScript/JavaScript
- ✅ React component patterns
- ✅ Async/await and Promises
- ✅ Node.js/Express basics
- ✅ REST APIs
- ✅ Basic Git

### Verify Installation
```bash
python --version  # Should show 3.13.9
uv --version
docker --version
git --version
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/price-stradamus.git
cd price-stradamus
```

## 📊 Progress Tracking

Track your progress through each module:

- [ ] Module 01: Python for TypeScript Developers
- [ ] Module 02: Async & Data Structures
- [ ] Module 03: ML Fundamentals
- [ ] Module 04: Project Architecture
- [ ] Module 05: Database & Data Pipeline
- [ ] Module 06: Feature Engineering
- [ ] Module 07: Model Fundamentals
- [ ] Module 08: Neural Networks Deep Dive
- [ ] Module 09: Classical & ML Models
- [ ] Module 10: Evaluation & Backtesting
- [ ] Module 11: CLI & Integration
- [ ] Module 12: Best Practices & Exercises

**Completion Goal:** ✅ Rebuild core features without AI assistance

## 🎯 Milestones

### Milestone 1: Python Comfort (After Module 3)
**Can you:**
- Write Python code without constantly looking up syntax?
- Use type hints and understand async/await?
- Read pandas DataFrames and numpy arrays?
- Explain ML concepts to someone else?

**Test:** Write a simple script that fetches data and calculates moving averages.

### Milestone 2: Architecture Understanding (After Module 6)
**Can you:**
- Explain the project structure?
- Set up the database and fetch data?
- Generate technical indicators?
- Trace data flow through the system?

**Test:** Add a new technical indicator to the feature pipeline.

### Milestone 3: Model Understanding (After Module 9)
**Can you:**
- Explain different model types?
- Train a model from scratch?
- Understand hyperparameters?
- Load and use a trained model?

**Test:** Train an XGBoost model and make predictions.

### Milestone 4: System Mastery (After Module 11)
**Can you:**
- Run end-to-end workflows?
- Evaluate models properly?
- Use all CLI commands?
- Debug issues independently?

**Test:** Fetch data, train 3 models, compare results.

### Final Milestone: Independence (After Module 12)
**Can you:**
- Rebuild Price Stradamus from scratch?
- Add a new model without guidance?
- Explain the system to another developer?
- Make architectural decisions confidently?

**Test:** Build a new feature (e.g., multi-asset support) from scratch.

## 💡 Study Tips

### Active Learning
1. **Type, don't copy-paste** - Muscle memory matters
2. **Run code frequently** - See results immediately
3. **Break things intentionally** - Learn from errors (Python errors are helpful!)
4. **Explain to yourself** - Teach concepts out loud
5. **Draw diagrams** - Visualize data flows and model architectures

### When You Get Stuck
1. Read error messages carefully (Python tracebacks are detailed!)
2. Check the actual project code for working examples
3. Review the relevant module section
4. Use print() statements for debugging (or Python debugger)
5. Consult official documentation (Python docs are excellent)

### Time Management
- **Focused learning:** 1-2 hour blocks with breaks
- **Daily practice:** Consistency beats intensity
- **Weekend deep dives:** Tackle complex topics when you have time
- **Review regularly:** Revisit earlier modules to reinforce

### Recommended Schedule (3-4 weeks)

**Week 1: Python & Foundations**
- Day 1-2: Module 01 (Python basics)
- Day 3-4: Module 02 (Async & data structures)
- Day 5-7: Module 03 (ML fundamentals)

**Week 2: Architecture & Pipeline**
- Day 1-2: Module 04 (Architecture)
- Day 3-4: Module 05 (Database & pipeline)
- Day 5-7: Module 06 (Feature engineering)

**Week 3: Models**
- Day 1-2: Module 07 (Model fundamentals)
- Day 3-4: Module 08 (Neural networks)
- Day 5-7: Module 09 (Classical & ML models)

**Week 4: Integration & Practice**
- Day 1-2: Module 10 (Evaluation)
- Day 3-4: Module 11 (CLI integration)
- Day 5-7: Module 12 (Best practices & exercises)

## 🔗 External Resources

### Python Learning
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/) - Excellent tutorials
- [Python Type Hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

### Data Science
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

### Machine Learning
- [Darts Documentation](https://unit8co.github.io/darts/) (Time series library)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Time Series Forecasting
- [Introduction to Time Series (free book)](https://otexts.com/fpp3/)
- [Time Series Analysis with Python](https://www.machinelearningplus.com/time-series/)

### Useful Tools
- [Python REPL](https://repl.it/languages/python3) - Online Python playground
- [Jupyter Lab](https://jupyter.org/) - Interactive notebooks (great for learning)
- [Python Tutor](http://pythontutor.com/) - Visualize code execution

## 📝 Project Context Files

Additional documentation in the project:
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines (after Module 4)
- [context/architecture.md](../architecture.md) - System architecture details
- [context/models.md](../models.md) - Model documentation
- [context/data-pipeline.md](../data-pipeline.md) - Data flow details
- [context/evaluation.md](../evaluation.md) - Evaluation methodology
- [context/glossary.md](../glossary.md) - ML and project terminology

## 🚀 Ready to Start?

**Begin your journey here:** [01: Python for TypeScript Developers →](01-python-for-ts-devs.md)

---

## 📝 Personal Notes Section

Use this space to track your learning journey:

**Key Takeaways:**
-

**Difficult Concepts to Review:**
-

**Questions for Deeper Exploration:**
-

**Mini-Projects Completed:**
-

**Code Snippets I Want to Remember:**
```python
# Add your favorite snippets here
```

---

## 🎓 Learning Philosophy

> **"The best way to learn ML is to build something real."**

This guide doesn't just teach you Python or ML in isolation - it teaches you through building a real production system. Every concept is immediately applied to Price Stradamus.

### Why This Approach Works

1. **Context matters** - You'll see WHY each concept is needed
2. **Immediate application** - Theory is immediately practiced
3. **Real code** - Not toy examples, but production-quality code
4. **Progressive complexity** - Start simple, build up naturally
5. **Goal-oriented** - Clear end goal: rebuild the system independently

### Remember

- Every expert was once a beginner
- Confusion is part of learning (embrace it!)
- Python is actually quite readable (you'll be surprised)
- ML concepts are simpler than they seem (mostly just math)
- This project is well-structured (easier to learn from good code)

**You've got this! Let's build some ML systems! 🚀📊**

---

*Last Updated: 2025-12-07*
*Estimated Total Time: 30-35 hours over 3-4 weeks*
*Difficulty: Beginner to Intermediate*
