# Multi-stage Dockerfile for AI Trading Platform

# Stage 1: Builder - install all dependencies
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY setup.py requirements.txt README.md ./
COPY src ./src

RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY . .

RUN useradd -m -u 1000 trading && \
    chown -R trading:trading /app && \
    mkdir -p /app/logs && \
    chown -R trading:trading /app/logs

USER trading

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-c", "import sys; sys.path.insert(0, '/app'); from src.presentation.api.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"]
