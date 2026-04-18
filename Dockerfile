# Multi-stage Dockerfile for AI Trading Platform

# Stage 1: Builder - install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY requirements-ml.txt ./

# Install to a virtual env so we can cleanly copy it to runtime stage.
# setuptools + wheel are upgraded up front so requirements.txt's own
# setuptools constraint isn't contradicted by an older version being
# pinned here. Leaving the version floor unspecified — requirements.txt
# is the source of truth.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Heavy ML stack (tensorflow + torch + transformers) is opt-in because
# it adds ~4 GB to the build and a single layer ENOSPCs on small root
# volumes. Enable with `--build-arg INCLUDE_HEAVY_ML=true` once the host
# has >= 25 GB free on / (EBS resize on the current t4g.medium).
ARG INCLUDE_HEAVY_ML=false
RUN if [ "$INCLUDE_HEAVY_ML" = "true" ]; then \
      pip install --no-cache-dir -r requirements-ml.txt; \
    else \
      echo "Skipping heavy ML (INCLUDE_HEAVY_ML=false)"; \
    fi

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

COPY . .

RUN useradd -m -u 1000 trading && \
    chown -R trading:trading /app && \
    mkdir -p /app/logs /app/models && \
    chown -R trading:trading /app/logs /app/models

USER trading

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/ready || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
