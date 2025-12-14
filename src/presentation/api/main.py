"""
Main Application Module for AI Trading Platform

This module initializes the FastAPI application with all middleware,
routers, and configurations following clean architecture principles.

Features:
- CORS middleware with configurable origins
- Rate limiting on API endpoints
- Comprehensive API documentation via OpenAPI
- Structured logging and error handling
- Security and authentication enforcement
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
import logging

# Load environment variables before importing other modules
load_dotenv()
load_dotenv('.env')  # explicitly load .env file in current directory

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.infrastructure.config.settings import settings
from src.infrastructure.logging import setup_logging
from src.presentation.api.routers import orders, portfolio, users, risk, dashboard, market_data, performance, brokers, alternative_data, ml, rl

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

# Add CORS middleware with proper security configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,  # 10 minutes
)


# ============================================================================
# Health Check Endpoints
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
        "version": "1.0.0",
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

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "AI Trading Platform API",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
    }


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


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    """Handle rate limit exceeded errors."""
    return {
        "error": "Too Many Requests",
        "detail": "Rate limit exceeded. Please try again later.",
        "retry_after": 60,
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "error": "Internal Server Error",
        "detail": "An unexpected error occurred. Please contact support.",
        "request_id": getattr(request, "id", "unknown"),
    }


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
    logger.info(f"Allowed Origins: {allowed_origins}")

    # Initialize infrastructure components here if needed
    # - Database connections
    # - Cache connections
    # - External service connections


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("=" * 80)
    logger.info("AI Trading Platform API Shutting Down")
    logger.info("=" * 80)

    # Cleanup infrastructure here if needed
    # - Close database connections
    # - Cleanup cache connections
    # - Close external service connections


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import os

    # Use PORT environment variable from Cloud Run, default to 8000 for local dev
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "src.presentation.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=(settings.ENVIRONMENT == "development"),
        log_level=settings.LOG_LEVEL.lower(),
    )