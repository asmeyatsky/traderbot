"""
Application Settings Configuration

This module loads and manages application settings from environment variables.
Following Pydantic v2 patterns with proper validation.
"""
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional


class Settings(BaseModel):
    """Application settings loaded from environment variables with validation."""

    model_config = ConfigDict(env_file=".env", case_sensitive=True)

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


# Create settings instance - this will fail fast if required env vars are missing
try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(f"Failed to load application settings: {e}")