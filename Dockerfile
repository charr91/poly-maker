# Dockerfile for poly-maker Polymarket market making bot
# Multi-stage build for optimized image size

# Stage 1: Build stage with UV package manager
FROM python:3.9-slim AS builder

# Install UV package manager
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies only (skip building local package which needs README.md)
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime stage
FROM python:3.9-slim AS runtime

# Install curl for health checks and Node.js for poly_merger
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r polybot && useradd -r -g polybot polybot

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=polybot:polybot . .

# Install poly_merger Node.js dependencies
RUN cd poly_merger && npm ci --production && cd ..

# Create data directories
RUN mkdir -p /app/data /app/positions && \
    chown -R polybot:polybot /app/data /app/positions

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER polybot

# Health check - verifies the bot process is running
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD pgrep -f "python main.py" > /dev/null || exit 1

# Default command
CMD ["python", "main.py"]
