"""
Main Application Module for AI Trading Platform

This module initializes the FastAPI application with all middleware,
routers, and configurations following clean architecture principles.

Features:
- Security headers middleware (HSTS, CSP, X-Frame-Options)
- CORS middleware with configurable origins
- Audit logging middleware for compliance
- Rate limiting on API endpoints
- Comprehensive API documentation via OpenAPI
- Structured logging and error handling
- AWS Secrets Manager integration
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
import logging

# Load environment variables before importing other modules
load_dotenv()
load_dotenv(".env")  # explicitly load .env file in current directory

# Load secrets from AWS Secrets Manager if configured
from src.infrastructure.config.aws_secrets import load_secrets_to_environment

load_secrets_to_environment()

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from src.infrastructure.config.settings import settings
from src.infrastructure.logging import setup_logging
from src.infrastructure.middleware.audit_logging import AuditLoggingMiddleware
from src.presentation.api.routers import (
    orders,
    portfolio,
    users,
    risk,
    dashboard,
    market_data,
    performance,
    brokers,
    alternative_data,
    ml,
    rl,
    trading_activity,
)

# Setup logging
logger = setup_logging()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize the FastAPI application
app = FastAPI(
    title="AI Trading Platform API",
    description="Autonomous AI-powered trading platform with real-time market data aggregation, "
    "sentiment analysis, automated trading execution, and advanced risk management.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redocs",
    openapi_url="/api/openapi.json",
)

# Add rate limiter to app state
app.state.limiter = limiter

# Parse allowed origins from settings
allowed_origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]

is_production = settings.ENVIRONMENT == "production"

# Disable API docs in production (prevents leaking endpoint schemas publicly)
if is_production:
    app.docs_url = None
    app.redoc_url = None
    app.openapi_url = None

# ============================================================================
# Security Middleware Stack (order matters — outermost first)
# ============================================================================

# 1. HTTPS redirect in production (ALB terminates TLS, but enforce on app level)
if is_production:
    app.add_middleware(HTTPSRedirectMiddleware)

# 2. Trusted Host middleware in production
if is_production:
    allowed_hosts = [h for h in allowed_origins if "://" not in h]
    # Also allow ALB health checks (no Host header filtering for those)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # ALB handles host validation; tighten when domain is set
    )

# 3. CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,  # 10 minutes
)

# 4. Audit logging middleware
app.add_middleware(AuditLoggingMiddleware)


# 5. Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next) -> Response:
    """Add security headers to all responses per OWASP recommendations."""
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "0"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=()"
    )

    # HSTS only in production (when behind TLS-terminating ALB)
    if is_production:
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )

    # Remove server identification header
    if "server" in response.headers:
        del response.headers["server"]

    return response


# ============================================================================
# Health Check Endpoints (locked down — no internal details exposed)
# ============================================================================


@app.get(
    "/",
    tags=["status"],
    summary="Welcome message",
    responses={200: {"description": "Welcome message"}},
)
async def root():
    """Welcome endpoint providing basic API information."""
    return {
        "message": "Welcome to the AI Trading Platform API",
        "docs": "/api/docs",
        "status": "operational",
    }


@app.get(
    "/health",
    tags=["status"],
    summary="Health check",
    responses={200: {"description": "Service health status"}},
)
async def health_check():
    """
    Health check endpoint for monitoring and orchestration.
    Returns OK if the service is running (no dependency checks).
    No internal details exposed (environment, version, etc.).
    """
    return {"status": "healthy"}


@app.get(
    "/ready",
    tags=["status"],
    summary="Readiness check",
    responses={
        200: {"description": "Service ready"},
        503: {"description": "Service not ready"},
    },
)
async def readiness_check():
    """
    Readiness check endpoint for ALB/ECS.
    Verifies DB and Redis connectivity before receiving traffic.
    Returns only 200/503 — no dependency names or internal state exposed.
    """
    from src.infrastructure.database import get_database_manager
    from src.infrastructure.cache import get_cache_manager

    all_ready = True

    try:
        db_manager = get_database_manager()
        if not db_manager.health_check():
            all_ready = False
    except Exception:
        all_ready = False

    try:
        cache_mgr = get_cache_manager()
        if cache_mgr.client:
            cache_mgr.client.ping()
        else:
            all_ready = False
    except Exception:
        all_ready = False

    if all_ready:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )


# ============================================================================
# API Routers Registration
# ============================================================================

# Include all API routers
app.include_router(orders.router)
app.include_router(portfolio.router)
app.include_router(users.router)
app.include_router(risk.router)
app.include_router(dashboard.router)
app.include_router(market_data.router)
app.include_router(performance.router)
app.include_router(brokers.router)
app.include_router(alternative_data.router)
app.include_router(ml.router)
app.include_router(rl.router)
app.include_router(trading_activity.router)


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too Many Requests",
            "detail": "Rate limit exceeded. Please try again later.",
            "retry_after": 60,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle uncaught exceptions — no internal details leaked."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred.",
        },
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("=" * 80)
    logger.info("AI Trading Platform API Starting Up")
    logger.info("=" * 80)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")

    # Initialize database connection
    from src.infrastructure.database import initialize_database
    try:
        initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Start risk monitoring services
    from src.infrastructure.di_container import container
    try:
        risk_manager = container.adapters.risk_manager()
        risk_manager.start_monitoring()
        circuit_breaker = container.adapters.circuit_breaker_service()
        circuit_breaker.start_monitoring()
        logger.info("Risk monitoring services started")
    except Exception as e:
        logger.error(f"Failed to start risk monitoring: {e}")

    # Start autonomous trading scheduler if enabled
    if settings.AUTO_TRADING_ENABLED:
        try:
            from src.infrastructure.scheduler import start_scheduler
            from src.application.services.autonomous_trading_service import AutonomousTradingService
            from src.infrastructure.api_clients.market_data import MarketDataService

            trading_service = AutonomousTradingService(
                user_repository=container.repositories.user_repository(),
                portfolio_repository=container.repositories.portfolio_repository(),
                position_repository=container.repositories.position_repository(),
                order_repository=container.repositories.order_repository(),
                activity_log_repository=container.repositories.activity_log_repository(),
                ml_model_service=container.services.ml_model_service(),
                broker_service=container.adapters.alpaca_broker_service(),
                risk_manager=container.adapters.risk_manager(),
                circuit_breaker=container.adapters.circuit_breaker_service(),
                market_data_service=MarketDataService(),
                confidence_threshold=settings.AUTO_TRADING_CONFIDENCE_THRESHOLD,
            )
            start_scheduler(trading_service, cycle_minutes=settings.AUTO_TRADING_CYCLE_MINUTES)
            logger.info("Autonomous trading scheduler started")
        except Exception as e:
            logger.error(f"Failed to start autonomous trading: {e}")
    else:
        logger.info("Autonomous trading is disabled (AUTO_TRADING_ENABLED=false)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("=" * 80)
    logger.info("AI Trading Platform API Shutting Down")
    logger.info("=" * 80)

    # Stop scheduler
    try:
        from src.infrastructure.scheduler import stop_scheduler
        stop_scheduler()
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")

    # Stop risk monitoring
    from src.infrastructure.di_container import container
    try:
        container.adapters.risk_manager().stop_monitoring()
        container.adapters.circuit_breaker_service().stop_monitoring()
    except Exception as e:
        logger.error(f"Error stopping risk monitoring: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "src.presentation.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=(settings.ENVIRONMENT == "development"),
        log_level=settings.LOG_LEVEL.lower(),
    )
