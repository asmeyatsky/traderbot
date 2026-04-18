"""
Observability — tracing, metrics, and correlation IDs.

Architectural Intent:
- Single entry point `setup_observability(app)` wires FastAPI into OTel + Prometheus.
- Correlation IDs flow end-to-end via `correlation_id_var`. Middleware sets it
  from the `X-Request-Id` header (or a fresh UUID), and `JSONFormatter` picks it
  up in every log line for the request's lifetime — including Claude API call
  telemetry — without having to thread it through call signatures.
- OTel OTLP export is only enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is set,
  so local dev stays silent while production ships traces to Grafana Cloud.
- Prometheus `/metrics` is always exposed (internal network only — surface via
  reverse proxy ACLs).

2026 rules mapped:
- §6: OpenTelemetry tracing through service calls.
- §6: RED metrics per endpoint.
- §6: Structured JSON logs with correlation IDs, zero PII.
"""
from __future__ import annotations

import logging
import os
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Contextvar is the async-safe way to carry per-request state across awaits.
# Default empty string (not None) so `extra={"correlation_id": get_correlation_id()}`
# never raises; JSONFormatter drops the field if empty.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

REQUEST_ID_HEADER = "X-Request-Id"


def get_correlation_id() -> str:
    """Return the current request's correlation ID, or empty string outside a request."""
    return correlation_id_var.get()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request.

    Preserves an incoming `X-Request-Id` (e.g. from Caddy) so traces stitch
    together across services; generates a short UUID fragment otherwise.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        cid = incoming if incoming else uuid.uuid4().hex[:16]
        token = correlation_id_var.set(cid)
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = cid
        return response


def _configure_tracing(app: FastAPI) -> None:
    """Wire OpenTelemetry tracing if an OTLP endpoint is configured.

    Silent no-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset — avoids spurious
    HTTP retries to localhost:4318 during local dev and CI.
    """
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.info("OTel tracing disabled — OTEL_EXPORTER_OTLP_ENDPOINT not set")
        return

    # Imports are local so the app still boots if the optional OTel packages
    # aren't installed in a lightweight environment.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.getenv("OTEL_SERVICE_NAME", "traderbot-api")
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.getenv("APP_VERSION", "unknown"),
            "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)

    # SQLAlchemy + httpx instrumentation are best-effort — skipped silently if
    # the packages aren't installed or the engine isn't bound yet.
    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

        SQLAlchemyInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001 — opt-in, keep app booting
        logger.warning("SQLAlchemy instrumentation skipped: %s", exc)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logger.warning("httpx instrumentation skipped: %s", exc)

    logger.info("OTel tracing enabled → %s (service=%s)", endpoint, service_name)


def _configure_metrics(app: FastAPI) -> None:
    """Expose Prometheus RED metrics at `/metrics`.

    Enabled by default — gate via reverse-proxy ACLs, not by this code, so we
    always have data available when an on-call needs it.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not installed — /metrics disabled")
        return

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,  # drop 404s to /random-path — reduces cardinality
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        inprogress_name="traderbot_requests_in_progress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        tags=["observability"],
    )
    logger.info("Prometheus /metrics endpoint exposed")


def setup_observability(app: FastAPI) -> None:
    """Entry point — call from main.py after FastAPI app is created.

    Order matters: correlation middleware MUST run outside tracing so spans
    carry the same ID that logs will emit.
    """
    app.add_middleware(CorrelationIdMiddleware)
    _configure_tracing(app)
    _configure_metrics(app)
