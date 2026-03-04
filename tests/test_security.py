"""
Security Module Tests

Tests JWT token generation, validation, password hashing, and authentication
middleware to ensure the security layer works correctly.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pytest
import jwt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def security_manager():
    """Return a SecurityManager instance."""
    from src.infrastructure.security import SecurityManager
    return SecurityManager()


@pytest.fixture
def jwt_settings():
    """Return test-safe JWT settings."""
    from src.infrastructure.config.settings import settings
    return settings


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    """Verify bcrypt password hashing and verification."""

    def test_hash_password_returns_bcrypt_hash(self, security_manager):
        hashed = security_manager.hash_password("testpassword123")
        assert hashed.startswith("$2b$")
        assert len(hashed) == 60

    def test_verify_correct_password(self, security_manager):
        hashed = security_manager.hash_password("secureP@ss1")
        assert security_manager.verify_password("secureP@ss1", hashed) is True

    def test_verify_wrong_password(self, security_manager):
        hashed = security_manager.hash_password("secureP@ss1")
        assert security_manager.verify_password("wrongpassword", hashed) is False

    def test_different_hashes_for_same_password(self, security_manager):
        h1 = security_manager.hash_password("same")
        h2 = security_manager.hash_password("same")
        assert h1 != h2  # Different salts

    def test_empty_password_still_hashes(self, security_manager):
        hashed = security_manager.hash_password("")
        assert hashed.startswith("$2b$")


# ---------------------------------------------------------------------------
# JWT Token Creation
# ---------------------------------------------------------------------------

class TestTokenCreation:
    """Verify JWT access and refresh token generation."""

    def test_create_access_token_returns_string_and_expiry(self, security_manager):
        token, expire = security_manager.create_access_token("user-123")
        assert isinstance(token, str)
        assert isinstance(expire, datetime)
        assert expire > datetime.utcnow()

    def test_access_token_contains_correct_claims(self, security_manager, jwt_settings):
        token, _ = security_manager.create_access_token("user-456")
        payload = jwt.decode(
            token,
            jwt_settings.JWT_SECRET_KEY,
            algorithms=[jwt_settings.JWT_ALGORITHM],
        )
        assert payload["sub"] == "user-456"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_create_refresh_token_type(self, security_manager, jwt_settings):
        token, expire = security_manager.create_refresh_token("user-789")
        payload = jwt.decode(
            token,
            jwt_settings.JWT_SECRET_KEY,
            algorithms=[jwt_settings.JWT_ALGORITHM],
        )
        assert payload["type"] == "refresh"
        # Refresh tokens expire in 7 days
        assert expire > datetime.utcnow() + timedelta(days=6)

    def test_custom_expiration_delta(self, security_manager):
        delta = timedelta(minutes=5)
        token, expire = security_manager.create_access_token("u1", expires_delta=delta)
        assert expire < datetime.utcnow() + timedelta(minutes=6)


# ---------------------------------------------------------------------------
# JWT Token Verification
# ---------------------------------------------------------------------------

class TestTokenVerification:
    """Verify JWT token decoding and validation."""

    def test_verify_valid_token(self, security_manager):
        token, _ = security_manager.create_access_token("user-ok")
        payload = security_manager.verify_token(token)
        assert payload.sub == "user-ok"
        assert payload.type == "access"

    def test_verify_expired_token_raises(self, security_manager):
        from fastapi import HTTPException
        token, _ = security_manager.create_access_token(
            "user-exp", expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(HTTPException) as exc:
            security_manager.verify_token(token)
        assert exc.value.status_code == 401

    def test_verify_tampered_token_raises(self, security_manager):
        from fastapi import HTTPException
        token, _ = security_manager.create_access_token("user-tamper")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(HTTPException) as exc:
            security_manager.verify_token(tampered)
        assert exc.value.status_code == 401

    def test_verify_token_wrong_secret_raises(self, security_manager, jwt_settings):
        from fastapi import HTTPException
        payload = {
            "sub": "user-wrong",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "type": "access",
        }
        token = jwt.encode(payload, "wrong-secret-key", algorithm=jwt_settings.JWT_ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            security_manager.verify_token(token)
        assert exc.value.status_code == 401

    def test_verify_token_missing_sub_raises(self, security_manager, jwt_settings):
        from fastapi import HTTPException
        payload = {
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "type": "access",
        }
        token = jwt.encode(payload, jwt_settings.JWT_SECRET_KEY, algorithm=jwt_settings.JWT_ALGORITHM)
        with pytest.raises(HTTPException) as exc:
            security_manager.verify_token(token)
        assert exc.value.status_code == 401


# ---------------------------------------------------------------------------
# Token Blacklist (fail-closed)
# ---------------------------------------------------------------------------

class TestTokenBlacklist:
    """Verify that token blacklist check is fail-closed."""

    @pytest.mark.asyncio
    async def test_blacklisted_token_is_rejected(self):
        from src.infrastructure.security import SecurityManager, get_current_user
        from fastapi import HTTPException
        from fastapi.security.http import HTTPAuthorizationCredentials

        sm = SecurityManager()
        token, _ = sm.create_access_token("blacklisted-user")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        # Mock cache to return a blacklisted token
        with patch("src.infrastructure.cache.get_cache_manager") as mock_cache:
            cache_instance = MagicMock()
            cache_instance.get.return_value = "revoked"
            mock_cache.return_value = cache_instance

            with pytest.raises(HTTPException) as exc:
                await get_current_user(creds)
            assert exc.value.status_code == 401
            assert "revoked" in exc.value.detail.lower()

    @pytest.mark.asyncio
    async def test_cache_unavailable_denies_token(self):
        """When cache is down, tokens should be DENIED (fail-closed)."""
        from src.infrastructure.security import SecurityManager, get_current_user
        from fastapi import HTTPException
        from fastapi.security.http import HTTPAuthorizationCredentials

        sm = SecurityManager()
        token, _ = sm.create_access_token("cache-down-user")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

        with patch("src.infrastructure.cache.get_cache_manager") as mock_cache:
            mock_cache.side_effect = ConnectionError("Redis unavailable")

            with pytest.raises(HTTPException) as exc:
                await get_current_user(creds)
            assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Token Expiry Check
# ---------------------------------------------------------------------------

class TestTokenExpiry:
    """Test is_token_expired utility."""

    def test_not_expired(self, security_manager):
        from src.infrastructure.security import TokenPayload
        payload = TokenPayload(
            sub="u1",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            type="access",
        )
        assert security_manager.is_token_expired(payload) is False

    def test_expired(self, security_manager):
        from src.infrastructure.security import TokenPayload
        payload = TokenPayload(
            sub="u1",
            exp=datetime.utcnow() - timedelta(hours=1),
            iat=datetime.utcnow() - timedelta(hours=2),
            type="access",
        )
        assert security_manager.is_token_expired(payload) is True
