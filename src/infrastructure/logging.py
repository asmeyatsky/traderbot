"""
Logging Configuration for the Trading Platform

Sets up structured logging with rotating file handlers, console output,
JSON format for CloudWatch, and sensitive data filtering for production use.

Architectural Intent:
- Structured JSON logging for CloudWatch integration and analysis
- Sensitive data masking for PCI DSS and GDPR compliance
- Extended PII pattern detection (email, SSN, card numbers, phone, address)
- Separate audit logger for security events
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Optional
from logging import Formatter


class JSONFormatter(Formatter):
    """JSON formatter for CloudWatch Logs integration."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        # Include audit-specific fields if present
        for field in (
            "client_ip", "method", "path", "status_code",
            "duration_ms", "security_event", "success", "user_agent",
        ):
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        return json.dumps(log_data)


class SensitiveDataFilter(logging.Filter):
    """
    Filter that masks sensitive information in log records.

    Covers PCI DSS (card data), GDPR (PII), and general secrets.
    """

    SENSITIVE_PATTERNS = [
        ("api_key", "API_KEY"),
        ("secret", "SECRET"),
        ("password", "PASSWORD"),
        ("token", "TOKEN"),
        ("bearer", "BEARER"),
        ("auth", "AUTH"),
        ("email", "EMAIL"),
        ("ssn", "SSN"),
        ("social_security", "SSN"),
        ("card_number", "CARD"),
        ("credit_card", "CARD"),
        ("account_number", "ACCOUNT"),
        ("phone", "PHONE"),
        ("address", "ADDRESS"),
        ("street", "ADDRESS"),
        ("zip_code", "ADDRESS"),
        ("postal_code", "ADDRESS"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record and mask sensitive data."""
        if isinstance(record.msg, str):
            record.msg = self._mask_sensitive_data(record.msg)
        if isinstance(record.args, dict):
            for key, value in record.args.items():
                if isinstance(value, str) and self._is_sensitive(key):
                    record.args[key] = "***MASKED***"
        return True

    @staticmethod
    def _is_sensitive(key: str) -> bool:
        """Check if a key represents sensitive data."""
        key_lower = key.lower()
        return any(
            pattern in key_lower
            for pattern, _ in SensitiveDataFilter.SENSITIVE_PATTERNS
        )

    @staticmethod
    def _mask_sensitive_data(text: str) -> str:
        """Mask sensitive data in text."""
        import re

        for pattern, replacement in SensitiveDataFilter.SENSITIVE_PATTERNS:
            # Match various formats: key=value, key: value, "key": "value"
            text = re.sub(
                rf'({pattern})["\']?\s*[:=]\s*["\']?[^"\'\s,}}\]]+',
                f"{replacement}=***MASKED***",
                text,
                flags=re.IGNORECASE,
            )
        return text


def setup_logging(
    log_level: Optional[str] = None,
    log_file: str = "logs/app.log",
    use_json: bool = None,
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        use_json: Use JSON format for CloudWatch (auto-detected from AWS_CW_LOG_GROUP)

    Returns:
        Configured root logger
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    if use_json is None:
        use_json = bool(os.getenv("AWS_CW_LOG_GROUP"))

    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Log format - use JSON for CloudWatch, otherwise human-readable
    if use_json:
        formatter = JSONFormatter()
    else:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
        formatter = logging.Formatter(log_format)

    # Create sensitive data filter
    sensitive_filter = SensitiveDataFilter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)
    root_logger.addHandler(console_handler)

    # File handler with rotation (only if not using JSON and log_file provided)
    if log_file and not use_json:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10 MB
                backupCount=5,
            )
            file_handler.setLevel(getattr(logging, log_level))
            file_handler.setFormatter(formatter)
            file_handler.addFilter(sensitive_filter)
            root_logger.addHandler(file_handler)
        except IOError as e:
            root_logger.warning(f"Could not configure file logging: {e}")

    # Suppress verbose logging from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    if use_json:
        root_logger.info("JSON logging enabled for CloudWatch")

    return root_logger


# Configure logging when module is imported
logger = setup_logging()
