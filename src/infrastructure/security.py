"""
Security and Authentication Module

This module handles JWT token generation/validation, password hashing,
and authentication middleware for the application.

Following security best practices and FastAPI patterns.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from pydantic import BaseModel
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials

from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer security
security = HTTPBearer()


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # Subject (user_id)
    exp: datetime  # Expiration time
    iat: datetime  # Issued at
    type: str  # Token type (access, refresh)


class TokenResponse(BaseModel):
    """Token response DTO."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int  # Seconds until expiration


class SecurityManager:
    """Manages authentication, token, and password operations."""

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt.

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against its hash.

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> tuple[str, datetime]:
        """
        Create a JWT access token.

        Args:
            user_id: User ID to encode in token
            expires_delta: Token expiration delta (defaults to settings)

        Returns:
            Tuple of (token, expiration_datetime)
        """
        if expires_delta is None:
            expires_delta = timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        expire = datetime.utcnow() + expires_delta
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
        }

        try:
            encoded_jwt = jwt.encode(
                payload,
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )
            return encoded_jwt, expire
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise

    @staticmethod
    def create_refresh_token(user_id: str) -> tuple[str, datetime]:
        """
        Create a JWT refresh token.

        Args:
            user_id: User ID to encode in token

        Returns:
            Tuple of (token, expiration_datetime)
        """
        expires_delta = timedelta(days=7)  # Refresh tokens last 7 days
        expire = datetime.utcnow() + expires_delta
        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }

        try:
            encoded_jwt = jwt.encode(
                payload,
                settings.JWT_SECRET_KEY,
                algorithm=settings.JWT_ALGORITHM,
            )
            return encoded_jwt, expire
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise

    @staticmethod
    def verify_token(token: str) -> TokenPayload:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenPayload with decoded token data

        Raises:
            HTTPException if token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                )
            return TokenPayload(
                sub=user_id,
                exp=datetime.fromtimestamp(payload.get("exp")),
                iat=datetime.fromtimestamp(payload.get("iat")),
                type=payload.get("type", "access"),
            )
        except PyJWTError as e:
            logger.warning(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

    @staticmethod
    def is_token_expired(token_payload: TokenPayload) -> bool:
        """Check if a token is expired."""
        return datetime.utcnow() > token_payload.exp


# Dependency injection functions for FastAPI

async def get_current_user(
    credentials: HTTPAuthCredentials = Depends(security),
) -> str:
    """
    FastAPI dependency to get current authenticated user.

    Args:
        credentials: HTTP Bearer credentials from request

    Returns:
        User ID string

    Raises:
        HTTPException if authentication fails
    """
    token = credentials.credentials
    try:
        token_payload = SecurityManager.verify_token(token)

        if token_payload.type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        if SecurityManager.is_token_expired(token_payload):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
            )

        return token_payload.sub

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthCredentials] = Depends(security),
) -> Optional[str]:
    """
    FastAPI dependency to optionally get current user.
    Returns None if no credentials provided.

    Args:
        credentials: Optional HTTP Bearer credentials

    Returns:
        User ID string or None
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Security manager instance
security_manager = SecurityManager()
