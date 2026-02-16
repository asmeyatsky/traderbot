"""
AWS Secrets Manager Integration

This module handles fetching configuration from AWS Secrets Manager
for production deployments. Falls back to environment variables in development.

Usage:
    - In AWS: Set AWS_SECRETS_NAME env var to your secret name
    - In development: Uses .env file (no AWS calls)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_secret(secret_name: str, region: str = "us-east-1") -> Optional[dict[str, Any]]:
    """
    Fetch secret from AWS Secrets Manager.

    Args:
        secret_name: Name of the secret
        region: AWS region

    Returns:
        Secret dict or None if not available
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, BotoCoreError

        client = boto3.client("secretsmanager", region_name=region)

        try:
            response = client.get_secret_value(SecretId=secret_name)
            secret = response["SecretString"]
            return json.loads(secret)
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ResourceNotFoundException":
                logger.warning(f"Secret {secret_name} not found in Secrets Manager")
            else:
                logger.error(f"AWS Secrets Manager error: {e}")
            return None
        except (BotoCoreError, json.JSONDecodeError) as e:
            logger.error(f"Failed to fetch secret: {e}")
            return None

    except ImportError:
        logger.warning("boto3 not installed, cannot fetch from Secrets Manager")
        return None


def load_secrets_to_environment(secret_name: Optional[str] = None) -> None:
    """
    Load secrets from AWS Secrets Manager and set as environment variables.
    Call this before loading application settings.

    Args:
        secret_name: Name of the secret. If None, reads from AWS_SECRETS_NAME env var.
    """
    secret_name = secret_name or os.environ.get("AWS_SECRETS_NAME")

    if not secret_name:
        logger.debug("No AWS secret name configured, using environment variables")
        return

    if os.environ.get("ENVIRONMENT") == "development":
        logger.debug("Running in development mode, skipping Secrets Manager")
        return

    region = os.environ.get("AWS_REGION", "eu-west-2")
    logger.info(f"Loading secrets from AWS Secrets Manager: {secret_name} (region: {region})")
    secret = get_secret(secret_name, region=region)

    if secret:
        for key, value in secret.items():
            os.environ[key] = str(value)
        logger.info(
            f"Successfully loaded {len(secret)} secrets from AWS Secrets Manager"
        )
    else:
        logger.warning(
            "Failed to load secrets from AWS Secrets Manager, using environment variables"
        )


def get_required_secrets() -> list[str]:
    """
    Return list of required secret keys that must be configured.
    """
    return [
        "POLYGON_API_KEY",
        "ALPHA_VANTAGE_API_KEY",
        "MARKETAUX_API_KEY",
        "FINNHUB_API_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "JWT_SECRET_KEY",
        "DATABASE_URL",
    ]


def validate_secrets() -> bool:
    """
    Validate that all required secrets are available.
    Returns True if all secrets are set.
    """
    missing = []
    for key in get_required_secrets():
        if not os.environ.get(key):
            missing.append(key)

    if missing:
        logger.warning(f"Missing required secrets: {missing}")
        return False

    return True
