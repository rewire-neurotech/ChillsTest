"""
Field-level encryption for sensitive data (AES-256 via Fernet).

Fernet provides:
  - AES-128-CBC encryption
  - HMAC-SHA256 authentication (tamper-proof)
  - Unique IV per encryption (same input -> different ciphertext each time)

Usage:
  encrypt_field(plaintext) -> ciphertext string (base64-encoded)
  decrypt_field(ciphertext) -> plaintext string
  encrypt_file(filepath) -> encrypts a file in place
  decrypt_file_to_bytes(filepath) -> returns decrypted file contents as bytes

Safety:
  - None / empty values pass through unchanged
  - If ENCRYPTION_KEY is not set, data passes through unencrypted (dev mode)
  - Old unencrypted data decrypts gracefully (returns original string/bytes)
"""

from pathlib import Path
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


def encrypt_file(filepath: str) -> bool:
    """
    Encrypt a file in place. Returns True if encrypted, False if skipped/failed.
    If ENCRYPTION_KEY is not set, the file is left unencrypted.
    """
    f = _get_fernet()
    if not f:
        return False
    try:
        p = Path(filepath)
        raw = p.read_bytes()
        encrypted = f.encrypt(raw)
        p.write_bytes(encrypted)
        print(f"[encryption] File encrypted: {p.name}")
        return True
    except Exception as e:
        print(f"[encryption] File encrypt error: {e}")
        return False


def decrypt_file_to_bytes(filepath: str) -> bytes:
    """
    Read a file and return decrypted bytes.
    If decryption fails (unencrypted legacy file), returns raw bytes as-is.
    """
    p = Path(filepath)
    raw = p.read_bytes()
    f = _get_fernet()
    if not f:
        return raw
    try:
        return f.decrypt(raw)
    except InvalidToken:
        # Legacy unencrypted file -- return as-is
        return raw
    except Exception as e:
        print(f"[encryption] File decrypt error: {e}")
        return raw