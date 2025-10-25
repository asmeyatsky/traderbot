"""
Application Settings Configuration

This module loads and manages application settings from environment variables.
"""
from pydantic import BaseSettings


class Settings(BaseSettings):
    # API Keys
    POLYGON_API_KEY: str
    ALPHA_VANTAGE_API_KEY: str
    MARKETAUX_API_KEY: str
    FINNHUB_API_KEY: str

    # Broker API Keys
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    INTERACTIVE_BROKERS_API_KEY: str

    # Database Configuration
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"

    # ML Model Configuration
    TF_SERVING_HOST: str = "tf-serving"
    TF_SERVING_PORT: int = 8501

    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Application Settings
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()