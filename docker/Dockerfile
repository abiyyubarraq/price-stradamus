# Multi-stage build for Price Stradamus
# Stage 1: Builder
FROM python:3.13.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY README.md .

# Create virtual environment and install dependencies
RUN uv venv .venv
RUN uv pip install -e ".[dev]"

# Stage 2: Runtime
FROM python:3.13.9-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash stradamus

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY --chown=stradamus:stradamus src/ /app/src/
COPY --chown=stradamus:stradamus pyproject.toml /app/

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs && \
    chown -R stradamus:stradamus /app

# Switch to non-root user
USER stradamus

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "-m", "price_stradamus.cli.commands"]
