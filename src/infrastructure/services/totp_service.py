"""
TOTP service — generate and verify per-user TOTP secrets for live trading.

Architectural Intent (ADR-002):
- Live-mode enablement requires 2FA (TOTP). Every user who flips to live mode
  has a TOTP secret generated and stored encrypted.
- Every live-trading action (order placement) is protected by a TOTP challenge
  OR a fresh re-auth within the last 5 minutes — enforced at the router layer.
- Encryption uses Fernet with JWT_SECRET_KEY as the key material. It's good
  enough for launch (one operator, single EC2, secrets off disk per Phase 3).
  Phase 8 can swap to a dedicated KMS key if regulatory scrutiny warrants.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from typing import Tuple

import pyotp
from cryptography.fernet import Fernet, InvalidToken

from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

TOTP_ISSUER = "TraderBot"


def _fernet() -> Fernet:
    """Derive a 32-byte Fernet key from JWT_SECRET_KEY.

    We hash the secret key so its raw length doesn't matter — only that it's
    present. JWT_SECRET_KEY is ≥ 32 chars per settings validation, so key
    material entropy is sufficient.
    """
    key_material = settings.JWT_SECRET_KEY.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(key_material).digest())
    return Fernet(key)


def generate_totp_secret() -> Tuple[str, str]:
    """Return (plaintext_secret, encrypted_secret).

    The plaintext is shown to the user ONCE during enablement so they can add
    it to their authenticator app; only the encrypted form is persisted.
    """
    plaintext = pyotp.random_base32()
    encrypted = _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")
    return plaintext, encrypted


def decrypt_totp_secret(encrypted: str) -> str:
    """Decrypt a stored TOTP secret. Raises on tampering."""
    try:
        return _fernet().decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # Either the secret key rotated or the ciphertext was tampered with.
        # Either way we refuse — the user will need to re-enrol.
        logger.error("totp_decrypt_failed — secret unreadable with current key")
        raise


def provisioning_uri(plaintext_secret: str, user_email: str) -> str:
    """otpauth:// URI for display as a QR code in the enrolment UI."""
    return pyotp.TOTP(plaintext_secret).provisioning_uri(
        name=user_email, issuer_name=TOTP_ISSUER
    )


def verify_totp(encrypted_secret: str, code: str) -> bool:
    """Check a 6-digit TOTP code against the stored secret.

    pyotp's `valid_window=1` accepts the previous and next 30s windows — ~90s
    total drift tolerance — which absorbs clock skew without being so wide that
    a code stays valid after the user's tab is gone.
    """
    if not code or not code.isdigit() or len(code) != 6:
        return False
    try:
        plaintext = decrypt_totp_secret(encrypted_secret)
    except Exception:  # noqa: BLE001
        return False
    return pyotp.TOTP(plaintext).verify(code, valid_window=1)
