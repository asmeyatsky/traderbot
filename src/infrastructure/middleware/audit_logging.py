"""
Audit Logging Middleware

Logs all HTTP requests with security-relevant context for compliance auditing.
Captures: timestamp, user_id, IP, method, path, status_code, response_time.

Architectural Intent:
- Provides non-repudiation audit trail for PCI DSS and ISO 27001
- Captures security events (login, password change) with extra context
- Integrates with structured JSON logging for CloudWatch analysis
- Middleware pattern keeps audit concerns separate from business logic
"""
from __future__ import annotations

import time
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("audit")

# Paths that generate security audit events
SECURITY_PATHS = {
    "/api/v1/users/login": "AUTH_LOGIN",
    "/api/v1/users/register": "AUTH_REGISTER",
    "/api/v1/users/logout": "AUTH_LOGOUT",
    "/api/v1/users/me/change-password": "PASSWORD_CHANGE",
    "/api/v1/users/me/data": "GDPR_DATA_DELETE",
    "/api/v1/users/me/export": "GDPR_DATA_EXPORT",
}


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs all requests for audit compliance.

    Captures request metadata, response status, and timing information.
    Security-sensitive endpoints receive additional audit event classification.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.monotonic()
        client_ip = self._get_client_ip(request)
        user_id = self._extract_user_id(request)

        response: Optional[Response] = None
        try:
            response = await call_next(request)
            return response
        finally:
            duration_ms = round((time.monotonic() - start_time) * 1000, 2)
            status_code = response.status_code if response else 500

            audit_entry = {
                "client_ip": client_ip,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_agent": request.headers.get("user-agent", ""),
            }

            if user_id:
                audit_entry["user_id"] = user_id

            # Classify security events
            security_event = SECURITY_PATHS.get(request.url.path)
            if security_event and request.method == "POST":
                audit_entry["security_event"] = security_event
                audit_entry["success"] = 200 <= status_code < 400

                if security_event == "AUTH_LOGIN" and status_code == 401:
                    logger.warning("Failed login attempt", extra=audit_entry)
                else:
                    logger.info(
                        f"Security event: {security_event}",
                        extra=audit_entry,
                    )
            elif security_event and request.method in ("GET", "DELETE"):
                audit_entry["security_event"] = security_event
                logger.info(
                    f"Security event: {security_event}",
                    extra=audit_entry,
                )
            else:
                logger.info("Request processed", extra=audit_entry)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For from ALB."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    @staticmethod
    def _extract_user_id(request: Request) -> Optional[str]:
        """
        Extract user_id from JWT in Authorization header without full validation.
        This is best-effort for logging only — actual auth is handled by dependencies.
        """
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return None

        try:
            import jwt as pyjwt
            token = auth_header[7:]
            # Decode without verification — just for logging context
            payload = pyjwt.decode(
                token, options={"verify_signature": False}
            )
            return payload.get("sub")
        except Exception:
            return None
