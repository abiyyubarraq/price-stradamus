# Price Stradamus - Glossary

## Trading & Finance Terms

**OHLCV**
Open, High, Low, Close, Volume - the five data points that define a candlestick.

**Candlestick**
A chart representation showing open, high, low, and close prices for a specific time period.

**Timeframe**
The duration of each candlestick (e.g., 1m = 1 minute, 1h = 1 hour, 1d = 1 day).

**Symbol / Trading Pair**
The cryptocurrency pair being traded (e.g., BTCUSDT = Bitcoin vs US Dollar Tether).

**Bid / Ask**
Bid = highest price buyers are willing to pay. Ask = lowest price sellers are willing to accept.

**Spread**
Difference between bid and ask prices.

**Order Book**
List of all buy and sell orders at different price levels.

**Market Order**
Order executed immediately at current market price.

**Limit Order**
Order executed only at specified price or better.

**Long Position**
Buying an asset expecting price to increase.

**Short Position**
Selling an asset expecting price to decrease.

**Leverage**
Borrowing funds to increase position size (e.g., 10x leverage = $100 controls $1000).

**Liquidation**
Forced closure of leveraged position when losses exceed margin.

**Stop-Loss**
Automatic sell order to limit losses if price drops below threshold.

**Take-Profit**
Automatic sell order to lock in gains when price reaches target.

**Slippage**
Difference between expected and actual execution price.

**Volatility**
Measure of price fluctuation magnitude and frequency.

**Bull Market**
Market trending upward with optimistic sentiment.

**Bear Market**
Market trending downward with pessimistic sentiment.

**Support Level**
Price level where buying pressure prevents further decline.

**Resistance Level**
Price level where selling pressure prevents further increase.

**Breakout**
Price moving above resistance or below support level.

**Consolidation**
Period of sideways price movement with low volatility.

---

## Technical Indicators

**SMA (Simple Moving Average)**
Average price over a specified number of periods.

**EMA (Exponential Moving Average)**
Weighted moving average giving more weight to recent prices.

**RSI (Relative Strength Index)**
Momentum oscillator measuring speed and magnitude of price changes (0-100 scale).

**MACD (Moving Average Convergence Divergence)**
Trend-following momentum indicator showing relationship between two EMAs.

**Bollinger Bands**
Volatility bands plotted at standard deviations above/below moving average.

**ATR (Average True Range)**
Volatility indicator measuring average price range over specified period.

**Stochastic Oscillator**
Momentum indicator comparing closing price to price range over time.

**ROC (Rate of Change)**
Momentum indicator measuring percentage change between current and past prices.

**ADX (Average Directional Index)**
Trend strength indicator (0-100 scale).

**OBV (On-Balance Volume)**
Cumulative volume indicator showing buying/selling pressure.

**VWAP (Volume-Weighted Average Price)**
Average price weighted by volume.

**CCI (Commodity Channel Index)**
Oscillator measuring deviation from statistical mean.

**Aroon**
Trend indicator measuring time since highest/lowest price in period.

---

## Machine Learning Terms

**Model**
Mathematical representation that learns patterns from data to make predictions.

**Training**
Process of learning patterns from historical data.

**Inference / Prediction**
Using trained model to make predictions on new data.

**Epoch**
One complete pass through the entire training dataset.

**Batch**
Subset of training data processed together in one iteration.

**Learning Rate**
Step size for updating model parameters during training.

**Loss Function**
Metric measuring how wrong model's predictions are.

**Gradient Descent**
Optimization algorithm for minimizing loss function.

**Backpropagation**
Algorithm for calculating gradients in neural networks.

**Overfitting**
Model memorizes training data but fails to generalize to new data.

**Underfitting**
Model is too simple to capture patterns in data.

**Regularization**
Techniques to prevent overfitting (e.g., L1, L2, dropout).

**Dropout**
Randomly disabling neurons during training to prevent overfitting.

**Hyperparameters**
Configuration settings chosen before training (e.g., learning rate, hidden units).

**Validation Set**
Data used to tune hyperparameters and prevent overfitting.

**Test Set**
Data used for final evaluation (never seen during training).

**Cross-Validation**
Technique for assessing model performance using multiple train/test splits.

**Ensemble**
Combining multiple models to improve predictions.

**Feature Engineering**
Creating new input variables from raw data.

**Feature Selection**
Choosing most relevant features for modeling.

**Normalization**
Scaling features to similar ranges (e.g., 0-1 or mean 0, std 1).

**Baseline Model**
Simple model used as comparison benchmark.

---

## Time Series Terms

**Time Series**
Sequence of data points ordered by time.

**Univariate Time Series**
Single variable measured over time (e.g., Bitcoin price).

**Multivariate Time Series**
Multiple variables measured over time (e.g., price + volume + indicators).

**Stationarity**
Time series with constant mean, variance, and autocorrelation over time.

**Trend**
Long-term increase or decrease in time series.

**Seasonality**
Regular, predictable patterns that repeat over fixed periods.

**Autocorrelation**
Correlation of time series with lagged version of itself.

**Lag**
Time delay between observations (e.g., lag-1 = previous time step).

**Horizon / Forecast Horizon**
Number of time steps into the future to predict.

**Lookback Window**
Number of past time steps used as input for prediction.

**Walk-Forward Validation**
Evaluation method simulating real-world forecasting scenario.

**Lookahead Bias**
Using future information to make predictions (invalid!).

**Multi-Step Forecasting**
Predicting multiple future time steps (e.g., next 5 candles).

**One-Step-Ahead Forecasting**
Predicting only next time step.

**Recursive Forecasting**
Using previous predictions as inputs for future predictions.

**Direct Forecasting**
Training separate models for each forecast horizon.

**Differencing**
Subtracting previous value to make series stationary.

**Detrending**
Removing trend component from time series.

---

## Neural Network Architectures

**RNN (Recurrent Neural Network)**
Neural network with loops to process sequences.

**LSTM (Long Short-Term Memory)**
RNN variant with gates to handle long-term dependencies.

**GRU (Gated Recurrent Unit)**
Simplified version of LSTM.

**CNN (Convolutional Neural Network)**
Neural network using convolutions (typically for images, but also sequences).

**TCN (Temporal Convolutional Network)**
CNN architecture for sequence modeling using dilated causal convolutions.

**Transformer**
Architecture using self-attention mechanism (no recurrence).

**Attention Mechanism**
Allowing model to focus on relevant parts of input sequence.

**N-BEATS**
Neural Basis Expansion Analysis for Time Series - interpretable forecasting architecture.

**TFT (Temporal Fusion Transformer)**
Hybrid architecture combining LSTM and attention for multi-horizon forecasting.

**Encoder-Decoder**
Architecture with encoder compressing input and decoder generating output.

---

## Evaluation Metrics

**MAE (Mean Absolute Error)**
Average absolute difference between predicted and actual values.

**MSE (Mean Squared Error)**
Average squared difference (penalizes large errors more).

**RMSE (Root Mean Squared Error)**
Square root of MSE (same units as target variable).

**MAPE (Mean Absolute Percentage Error)**
Average percentage difference between predicted and actual values.

**sMAPE (Symmetric MAPE)**
Symmetric version of MAPE treating over/under-predictions equally.

**R² Score (Coefficient of Determination)**
Proportion of variance explained by model (0-1, where 1 = perfect).

**Directional Accuracy**
Percentage of correct up/down predictions.

**Sharpe Ratio**
Risk-adjusted return metric (higher = better).

**Max Drawdown**
Largest peak-to-trough decline in portfolio value.

**Sortino Ratio**
Similar to Sharpe but only considers downside volatility.

---

## AutoML Terms

**AutoML (Automated Machine Learning)**
Automating the process of model selection and hyperparameter tuning.

**Hyperparameter Optimization**
Automatically finding best hyperparameter values.

**Grid Search**
Exhaustively trying all combinations in hyperparameter grid.

**Random Search**
Randomly sampling hyperparameter combinations.

**Bayesian Optimization**
Using probabilistic model to intelligently select hyperparameters.

**Optuna**
Python library for hyperparameter optimization.

**auto-sklearn**
Automated machine learning library built on scikit-learn.

**Neural Architecture Search (NAS)**
Automatically designing neural network architectures.

**Model Selection**
Choosing best model type from candidates.

**Ensemble Selection**
Automatically combining multiple models.

---

## Project-Specific Terms

**Price Stradamus**
This project - Bitcoin price prediction system.

**Darts**
Time series forecasting library used in this project.

**Walk-Forward Backtester**
Our implementation of walk-forward validation.

**Base Model Interface**
Abstract class defining common interface for all models.

**Model Registry**
Central directory of available model implementations.

**Feature Engine**
Component generating technical indicators from raw OHLCV data.

**Tournament System**
Custom AutoML approach using Bayesian optimization (Phase 3).

---

## Database Terms

**PostgreSQL**
Open-source relational database used for storing data.

**Time Series Database**
Database optimized for time-ordered data.

**Partitioning**
Splitting large table into smaller pieces (e.g., by month).

**Indexing**
Creating data structures for fast lookups.

**Connection Pool**
Reusable database connections for better performance.

**ACID**
Atomicity, Consistency, Isolation, Durability - database transaction properties.

**Schema**
Structure and organization of database tables.

**Upsert**
Insert or update if already exists.

---

## DevOps Terms

**Docker**
Platform for containerizing applications.

**Container**
Lightweight, standalone executable package.

**Docker Compose**
Tool for defining multi-container applications.

**CI/CD (Continuous Integration/Continuous Deployment)**
Automated testing and deployment pipeline.

**Kubernetes**
Container orchestration platform (future use).

**Load Balancer**
Distributes traffic across multiple servers.

**API Gateway**
Entry point for API requests with routing, authentication, rate limiting.

**WebSocket**
Protocol for real-time bidirectional communication.

**Rate Limiting**
Restricting number of requests per time period.

---

## Statistics Terms

**Mean**
Average value.

**Median**
Middle value when sorted.

**Standard Deviation**
Measure of data spread around mean.

**Variance**
Square of standard deviation.

**Percentile**
Value below which a percentage of data falls (e.g., 95th percentile).

**IQR (Interquartile Range)**
Range between 25th and 75th percentiles.

**Z-Score**
Number of standard deviations from mean.

**P-Value**
Probability of observing results under null hypothesis.

**Null Hypothesis**
Assumption of no effect or no difference.

**Statistical Significance**
Result unlikely to occur by chance (typically p < 0.05).

**Confidence Interval**
Range likely to contain true parameter value.

---

## MLOps Terms

**MLOps (Machine Learning Operations)**
Practices combining ML development with DevOps for reliable model deployment and monitoring. *See also: DevOps Terms*

**Model Registry**
Centralized repository for tracking model versions, metadata, and artifacts. *See also: Feature Store*

**Feature Store**
Centralized platform for storing, serving, and managing ML features across training and inference. *See also: Feature Engineering*

**Model Serving**
Infrastructure for deploying models and handling prediction requests in production.

**Model Card**
Standardized documentation describing a model's intended use, limitations, and performance metrics (Google AI standard).

**Experiment Tracking**
Recording hyperparameters, metrics, and artifacts across ML experiments (e.g., MLflow, Weights & Biases).

**ML Pipeline**
Automated workflow orchestrating data processing, training, evaluation, and deployment.

**Model Lineage**
Tracing a model's origins including training data, features, and hyperparameters.

**A/B Testing**
Comparing two model versions by routing traffic between them and measuring performance. *See also: Canary Deployment*

**Shadow Mode / Shadow Deployment**
Running new model alongside production model without affecting users, for comparison.

**Model Drift**
Change in model performance over time due to data distribution shifts. *See also: Concept Drift, Data Drift*

**Concept Drift**
When the relationship between inputs and outputs changes over time.

**Data Drift**
When the statistical properties of input data change over time.

**Retraining Trigger**
Automated signal to retrain model based on performance degradation or time.

**Champion/Challenger**
Pattern where current production model (champion) is compared against new candidates (challengers).

**Feature Importance**
Ranking of features by their contribution to model predictions. *See also: SHAP, LIME*

**SHAP (SHapley Additive exPlanations)**
Method for explaining individual predictions by computing feature contributions.

**LIME (Local Interpretable Model-agnostic Explanations)**
Technique for explaining predictions by approximating model locally with interpretable model.

---

## Cryptocurrency Terms

**Blockchain**
Distributed ledger technology recording transactions in immutable blocks.

**DeFi (Decentralized Finance)**
Financial services built on blockchain without traditional intermediaries.

**DEX (Decentralized Exchange)**
Exchange where trading occurs directly between users via smart contracts. *Contrast: CEX*

**CEX (Centralized Exchange)**
Traditional exchange like Binance or Coinbase with centralized order matching.

**Smart Contract**
Self-executing code deployed on blockchain that automatically enforces agreements.

**Perpetual Swap / Perpetual Futures**
Derivative contract with no expiration date, tracking spot price through funding mechanism.

**Funding Rate**
Periodic payment between long and short position holders in perpetual contracts.

**Spot Market**
Market for immediate delivery of asset at current price. *Contrast: Futures Market*

**Futures Market**
Market for contracts to buy/sell asset at predetermined price on future date.

**Options**
Contracts giving right (not obligation) to buy (call) or sell (put) at specific price.

**Open Interest**
Total number of outstanding derivative contracts not yet settled.

**Liquidation Cascade**
Chain reaction of forced liquidations causing rapid price movement.

**Whale**
Large holder whose trading can significantly impact market price.

**HODL**
Long-term holding strategy (originated as misspelling of "hold").

**Pump and Dump**
Market manipulation involving artificial price inflation followed by selling.

**Rug Pull**
Scam where developers abandon project after raising funds.

**TVL (Total Value Locked)**
Total value of crypto assets deposited in DeFi protocol.

**APY (Annual Percentage Yield)**
Annualized return including compound interest.

**Gas Fee**
Transaction fee on blockchain network (e.g., Ethereum).

**Mempool**
Queue of unconfirmed transactions waiting to be added to blockchain.

**On-Chain Analysis**
Analyzing blockchain transaction data for trading signals.

**Order Flow**
Sequence and pattern of buy/sell orders entering market.

**Market Microstructure**
Study of how markets operate including order matching, price formation, and market makers.

**Maker / Taker**
Maker adds liquidity (limit orders), taker removes liquidity (market orders).

**Market Maker**
Entity providing liquidity by continuously quoting buy and sell prices.

**Flash Loan**
Uncollateralized loan that must be borrowed and repaid within single transaction.

**Arbitrage**
Profiting from price differences across markets or instruments.

---

## Risk Management Terms

**VaR (Value at Risk)**
Maximum expected loss at given confidence level over specific time period. *See also: CVaR*

**CVaR (Conditional Value at Risk) / Expected Shortfall**
Average loss in worst-case scenarios beyond VaR threshold. More conservative than VaR.

**Risk-Free Rate**
Return on investment with zero risk (typically government bonds).

**Beta**
Measure of asset's volatility relative to market.

**Alpha**
Excess return above benchmark adjusted for risk.

**Sharpe Ratio**
Risk-adjusted return: (Return - Risk-Free Rate) / Volatility. *See also: Sortino Ratio*

**Sortino Ratio**
Like Sharpe but uses downside deviation instead of total volatility.

**Calmar Ratio**
Return divided by maximum drawdown.

**Information Ratio**
Active return divided by tracking error versus benchmark.

**Maximum Drawdown**
Largest peak-to-trough decline in portfolio value.

**Drawdown Duration**
Time from peak to recovery of previous high.

**Position Sizing**
Determining how much capital to allocate to each trade. *See also: Kelly Criterion*

**Kelly Criterion**
Optimal bet sizing formula maximizing long-term growth rate.

**Fixed Fractional Position Sizing**
Risking fixed percentage of capital per trade.

**Volatility Targeting**
Adjusting position size to maintain constant portfolio volatility.

**Risk Parity**
Allocating based on risk contribution rather than capital.

**Regime Detection**
Identifying different market states (trending, mean-reverting, volatile).

**Tail Risk**
Risk of extreme events beyond normal distribution expectations.

**Black Swan**
Unpredictable, rare event with extreme consequences.

**Fat Tails / Leptokurtosis**
Distribution with more extreme outcomes than normal distribution.

**Greeks (Options)**
Sensitivity measures: Delta (price), Gamma (delta change), Theta (time), Vega (volatility).

**Correlation Risk**
Risk that correlations change unexpectedly (often increase during crises).

**Counterparty Risk**
Risk that other party in transaction fails to meet obligations.

**Liquidity Risk**
Risk of being unable to exit position without significant price impact.

**Execution Risk**
Risk of adverse price movement between decision and execution.

**Slippage Cost**
Difference between expected and actual execution price.

**Market Impact**
Price movement caused by own trading activity.

**Monte Carlo Simulation**
Generating many random scenarios to estimate distributions of outcomes.

**Stress Testing**
Evaluating performance under extreme but plausible scenarios.

**Scenario Analysis**
Examining impact of specific hypothetical events.

---

## Data Engineering Terms

**ETL (Extract, Transform, Load)**
Process of moving data from sources to destination with transformations. *See also: ELT*

**ELT (Extract, Load, Transform)**
Loading raw data first, then transforming in destination (common with data warehouses).

**Data Pipeline**
Automated workflow moving data between systems.

**Data Lineage**
Tracing data from origin through all transformations to final use.

**Data Catalog**
Searchable inventory of available datasets with metadata.

**Data Contract**
Agreement between data producer and consumer on schema, quality, and SLAs.

**Schema Registry**
Centralized repository for managing and versioning data schemas.

**Schema Evolution**
Managing changes to data structure while maintaining compatibility.

**Schema-on-Read vs Schema-on-Write**
Validation at read time (flexible) vs write time (strict).

**CDC (Change Data Capture)**
Tracking and capturing changes in source database for incremental loading.

**Idempotency**
Operation that produces same result regardless of how many times executed.

**Backpressure**
Mechanism for handling when data producer is faster than consumer.

**Dead Letter Queue (DLQ)**
Storage for messages that fail processing for later investigation.

**Data Quality**
Accuracy, completeness, consistency, and timeliness of data.

**Data Observability**
Monitoring data health including freshness, volume, and schema changes.

**Great Expectations**
Python library for data validation, documentation, and profiling.

**Data Versioning**
Tracking changes to datasets over time (e.g., DVC, LakeFS).

**DVC (Data Version Control)**
Git-like version control for ML data and experiments.

**Lakehouse**
Architecture combining data lake flexibility with data warehouse reliability.

**Medallion Architecture**
Layered data quality pattern: Bronze (raw) → Silver (cleansed) → Gold (business-ready).

**Data Mesh**
Decentralized data architecture with domain-oriented ownership.

**Partitioning**
Dividing data into segments for efficient querying (by time, hash, range).

**Compaction**
Consolidating small files into larger ones for query performance.

**Late-Arriving Data**
Data that arrives after the processing window has closed.

**Watermark**
Threshold tracking event time progress for handling late data.

**Event Time vs Processing Time**
When event occurred vs when it was processed.

**Exactly-Once Semantics**
Guarantee that each message is processed exactly one time.

**At-Least-Once vs At-Most-Once**
Delivery guarantees trading off duplicates vs losses.

**Stream Processing**
Real-time processing of data as it arrives. *Contrast: Batch Processing*

**Batch Processing**
Processing data in discrete time intervals.

**Lambda Architecture**
Combining batch and stream processing for completeness and low latency.

**Kappa Architecture**
Stream-only architecture treating everything as real-time.

---

## Infrastructure & Cloud Terms

**IaC (Infrastructure as Code)**
Managing infrastructure through code (e.g., Terraform, Pulumi).

**GitOps**
Using Git as single source of truth for declarative infrastructure and applications.

**Service Mesh**
Infrastructure layer handling service-to-service communication.

**Sidecar Pattern**
Deploying helper container alongside main application container.

**Blue-Green Deployment**
Switching traffic between two identical environments for zero-downtime updates.

**Canary Deployment**
Gradually rolling out changes to subset of users before full release. *See also: A/B Testing*

**Rolling Deployment**
Incrementally updating instances while keeping service available.

**Horizontal Scaling**
Adding more instances/machines. *Contrast: Vertical Scaling*

**Vertical Scaling**
Adding more resources to existing machine.

**Auto-Scaling**
Automatically adjusting resources based on demand.

**SLO (Service Level Objective)**
Target metric for service reliability (e.g., 99.9% uptime).

**SLA (Service Level Agreement)**
Contractual commitment with penalties for not meeting SLOs.

**SLI (Service Level Indicator)**
Metric measuring service performance against SLO.

**Error Budget**
Allowed downtime before violating SLO (100% - SLO).

**MTTR (Mean Time to Recovery)**
Average time to restore service after failure.

**MTTF (Mean Time to Failure)**
Average time between failures.

**RTO (Recovery Time Objective)**
Maximum acceptable downtime after disaster.

**RPO (Recovery Point Objective)**
Maximum acceptable data loss measured in time.

**Observability**
Ability to understand system state from external outputs (logs, metrics, traces).

**Three Pillars of Observability**
Logs (events), Metrics (measurements), Traces (request flows).

**Distributed Tracing**
Tracking requests across multiple services. *See also: OpenTelemetry*

**OpenTelemetry**
Standard for telemetry data (traces, metrics, logs) collection and export.

**Circuit Breaker**
Pattern preventing cascade failures by stopping calls to failing service.

**Bulkhead**
Isolating components so one failure doesn't affect others.

**Chaos Engineering**
Deliberately injecting failures to test system resilience.

---

## Acronyms

**API** - Application Programming Interface
**APY** - Annual Percentage Yield
**ATR** - Average True Range
**CDC** - Change Data Capture
**CEX** - Centralized Exchange
**CI/CD** - Continuous Integration/Continuous Deployment
**CLI** - Command Line Interface
**CSV** - Comma-Separated Values
**CVaR** - Conditional Value at Risk
**DB** - Database
**DEX** - Decentralized Exchange
**DeFi** - Decentralized Finance
**DLQ** - Dead Letter Queue
**DVC** - Data Version Control
**ELT** - Extract, Load, Transform
**ETL** - Extract, Transform, Load
**GPU** - Graphics Processing Unit
**HTTP** - HyperText Transfer Protocol
**IaC** - Infrastructure as Code
**JSON** - JavaScript Object Notation
**LIME** - Local Interpretable Model-agnostic Explanations
**ML** - Machine Learning
**MLOps** - Machine Learning Operations
**MLP** - Multi-Layer Perceptron
**MTTR** - Mean Time to Recovery
**NAS** - Neural Architecture Search
**OOP** - Object-Oriented Programming
**REST** - Representational State Transfer
**RPO** - Recovery Point Objective
**RTO** - Recovery Time Objective
**SDK** - Software Development Kit
**SHAP** - SHapley Additive exPlanations
**SLA** - Service Level Agreement
**SLI** - Service Level Indicator
**SLO** - Service Level Objective
**SQL** - Structured Query Language
**SSL** - Secure Sockets Layer
**TLS** - Transport Layer Security
**TVL** - Total Value Locked
**URL** - Uniform Resource Locator
**UUID** - Universally Unique Identifier
**VaR** - Value at Risk
**VRAM** - Video Random Access Memory
**YAML** - YAML Ain't Markup Language

---

*Last Updated: 2025-12-03*
