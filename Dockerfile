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

# Install CPU-only PyTorch from the official CPU index, then the rest.
# Install to a virtual env so we can cleanly copy it to runtime stage.
# setuptools + wheel are upgraded up front so requirements.txt's own
# setuptools constraint (>=78.1.1,<81.0.0 for CVE PYSEC-2025-49) isn't
# contradicted by an older version being pinned here. Leaving the version
# floor unspecified: requirements.txt is the source of truth.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

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
