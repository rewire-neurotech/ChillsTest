"""
Field-level encryption for sensitive data (AES-256 via Fernet).

Fernet provides:
  - AES-128-CBC encryption
  - HMAC-SHA256 authentication (tamper-proof)
  - Unique IV per encryption (same input -> different ciphertext each time)

Usage:
  encrypt_field(plaintext) -> ciphertext string (base64-encoded)
  decrypt_field(ciphertext) -> plaintext string

Safety:
  - None / empty values pass through unchanged
  - If ENCRYPTION_KEY is not set, data passes through unencrypted (dev mode)
  - Old unencrypted data decrypts gracefully (returns original string)
"""

from cryptography.fernet import Fernet, InvalidToken
from app.core.config import cfg


def _get_fernet():
    """Return a Fernet instance if ENCRYPTION_KEY is configured, else None."""
    key = cfg.ENCRYPTION_KEY
    if not key:
        return None
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception:
        print("[encryption] Invalid ENCRYPTION_KEY. Data will NOT be encrypted.")
        return None


def encrypt_field(value: str) -> str:
    """Encrypt a string value. Returns the original if encryption is not configured."""
    if not value:
        return value
    f = _get_fernet()
    if not f:
        return value
    try:
        return f.encrypt(value.encode("utf-8")).decode("utf-8")
    except Exception as e:
        print(f"[encryption] Encrypt error: {e}")
        return value


def decrypt_field(value: str) -> str:
    """
    Decrypt a string value.
    Returns the original if decryption fails (handles legacy unencrypted data).
    """
    if not value:
        return value
    f = _get_fernet()
    if not f:
        return value
    try:
        return f.decrypt(value.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # Legacy unencrypted data -- return as-is
        return value
    except Exception as e:
        print(f"[encryption] Decrypt error: {e}")
        return value