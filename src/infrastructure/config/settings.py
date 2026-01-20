"""
Application Settings Configuration

This module loads and manages application settings from environment variables.
Following Pydantic v2 patterns with proper validation.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field, field_validator
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation."""

    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # API Keys - Required
    POLYGON_API_KEY: str = Field(..., min_length=1)
    ALPHA_VANTAGE_API_KEY: str = Field(..., min_length=1)
    MARKETAUX_API_KEY: str = Field(..., min_length=1)
    FINNHUB_API_KEY: str = Field(..., min_length=1)

    # Broker API Keys - Required
    ALPACA_API_KEY: str = Field(..., min_length=1)
    ALPACA_SECRET_KEY: str = Field(..., min_length=1)
    INTERACTIVE_BROKERS_API_KEY: Optional[str] = Field(default=None)

    # Database Configuration - Required
    DATABASE_URL: str = Field(..., min_length=1)
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # ML Model Configuration
    TF_SERVING_HOST: str = Field(default="tf-serving")
    TF_SERVING_PORT: int = Field(default=8501, ge=1, le=65535)

    # Security
    JWT_SECRET_KEY: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=1)

    # CORS Configuration
    ALLOWED_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8000")

    # Application Settings
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # Risk Management Configuration - Conservative
    RISK_CONSERVATIVE_MAX_DRAWDOWN: float = Field(default=10.0, ge=0, le=100)
    RISK_CONSERVATIVE_POSITION_LIMIT_PCT: float = Field(default=5.0, ge=0, le=100)
    RISK_CONSERVATIVE_VOLATILITY_THRESHOLD: float = Field(default=2.0, ge=0)

    # Risk Management Configuration - Moderate
    RISK_MODERATE_MAX_DRAWDOWN: float = Field(default=15.0, ge=0, le=100)
    RISK_MODERATE_POSITION_LIMIT_PCT: float = Field(default=10.0, ge=0, le=100)
    RISK_MODERATE_VOLATILITY_THRESHOLD: float = Field(default=2.5, ge=0)

    # Risk Management Configuration - Aggressive
    RISK_AGGRESSIVE_MAX_DRAWDOWN: float = Field(default=25.0, ge=0, le=100)
    RISK_AGGRESSIVE_POSITION_LIMIT_PCT: float = Field(default=20.0, ge=0, le=100)
    RISK_AGGRESSIVE_VOLATILITY_THRESHOLD: float = Field(default=3.0, ge=0)

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_VOLATILITY_THRESHOLD: float = Field(default=5.0, ge=0)
    CIRCUIT_BREAKER_RESET_MINUTES: int = Field(default=30, ge=1)

    @field_validator('DATABASE_URL')
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        """Validate database URL format"""
        if not v.startswith(('postgresql://', 'sqlite://', 'mysql://')):
            raise ValueError('Invalid database URL format. Must start with postgresql://, sqlite://, or mysql://')
        return v

    @field_validator('REDIS_URL')
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format"""
        if not v.startswith('redis://'):
            raise ValueError('Invalid Redis URL. Must start with redis://')
        return v


# Import at the top of the file to make sure they're loaded
from functools import lru_cache

@lru_cache(maxsize=1)
def get_settings():
    """
    Get settings instance with caching to avoid repeated loading
    """
    try:
        return Settings()
    except Exception as e:
        raise RuntimeError(f"Failed to load application settings: {e}")

# Defer initialization until first access
def __getattr__(name):
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")