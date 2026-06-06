"""OTP issue + verify logic.

- Codes are stored hashed (sha256). Plaintext is returned to the caller only
  long enough to email it; it never touches the DB in clear form.
- One unconsumed OTP per (email, purpose) at a time: re-issuing supersedes the
  previous code by deleting all unconsumed rows for that pair.
- Rate-limited via OTP_RESEND_COOLDOWN_SECONDS — the route layer raises 429.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy.orm import Session

from app.config import (
    OTP_CODE_LENGTH,
    OTP_MAX_ATTEMPTS,
    OTP_RESEND_COOLDOWN_SECONDS,
    OTP_TTL_MINUTES,
    SECRET_KEY,
)
from app.database import OtpCode

OtpPurpose = Literal["signup", "reset"]

VALID_PURPOSES = {"signup", "reset"}


class OtpError(Exception):
    """Base class for OTP failures. `code` maps to an HTTP response detail."""

    def __init__(self, message: str, code: str = "otp_error"):
        super().__init__(message)
        self.code = code


class OtpCooldownError(OtpError):
    def __init__(self, retry_after_seconds: int):
        super().__init__(
            f"Please wait {retry_after_seconds}s before requesting a new code.",
            code="otp_cooldown",
        )
        self.retry_after_seconds = retry_after_seconds


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _hash_code(code: str) -> str:
    """Keyed sha256 — pepper with SECRET_KEY so leaking the DB alone isn't enough."""
    return hmac.new(
        SECRET_KEY.encode("utf-8"),
        code.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _generate_code() -> str:
    upper = 10 ** OTP_CODE_LENGTH
    return f"{secrets.randbelow(upper):0{OTP_CODE_LENGTH}d}"


def _cleanup_expired(db: Session) -> None:
    """Best-effort cleanup of stale rows. Cheap, runs on each issue."""
    cutoff = datetime.utcnow() - timedelta(days=1)
    db.query(OtpCode).filter(OtpCode.expires_at < cutoff).delete(
        synchronize_session=False
    )


def issue_otp(db: Session, email: str, purpose: OtpPurpose) -> str:
    """Generate a new OTP, store its hash, and return the plaintext code.

    Caller is responsible for emailing the returned code. Plaintext is never
    persisted.
    """
    if purpose not in VALID_PURPOSES:
        raise OtpError(f"Invalid OTP purpose: {purpose}", code="otp_invalid_purpose")

    email = _normalize_email(email)
    now = datetime.utcnow()

    latest = (
        db.query(OtpCode)
        .filter(
            OtpCode.email == email,
            OtpCode.purpose == purpose,
            OtpCode.consumed_at.is_(None),
        )
        .order_by(OtpCode.created_at.desc())
        .first()
    )
    if latest is not None:
        age = (now - latest.created_at).total_seconds()
        if age < OTP_RESEND_COOLDOWN_SECONDS:
            raise OtpCooldownError(int(OTP_RESEND_COOLDOWN_SECONDS - age))

    db.query(OtpCode).filter(
        OtpCode.email == email,
        OtpCode.purpose == purpose,
        OtpCode.consumed_at.is_(None),
    ).delete(synchronize_session=False)

    _cleanup_expired(db)

    code = _generate_code()
    record = OtpCode(
        email=email,
        code_hash=_hash_code(code),
        purpose=purpose,
        expires_at=now + timedelta(minutes=OTP_TTL_MINUTES),
        attempts=0,
        consumed_at=None,
        created_at=now,
    )
    db.add(record)
    db.commit()
    return code


def verify_otp(db: Session, email: str, code: str, purpose: OtpPurpose) -> bool:
    """Verify `code` for (email, purpose). Marks the row consumed on success.

    Returns True on match; raises `OtpError` with a descriptive code on failure.
    """
    if purpose not in VALID_PURPOSES:
        raise OtpError(f"Invalid OTP purpose: {purpose}", code="otp_invalid_purpose")

    email = _normalize_email(email)
    code = (code or "").strip()
    if not code:
        raise OtpError("Code is required.", code="otp_missing")

    record = (
        db.query(OtpCode)
        .filter(
            OtpCode.email == email,
            OtpCode.purpose == purpose,
            OtpCode.consumed_at.is_(None),
        )
        .order_by(OtpCode.created_at.desc())
        .first()
    )

    if record is None:
        raise OtpError(
            "No active code for this email. Request a new one.",
            code="otp_not_found",
        )

    if record.expires_at < datetime.utcnow():
        raise OtpError("Code expired. Request a new one.", code="otp_expired")

    if record.attempts >= OTP_MAX_ATTEMPTS:
        raise OtpError(
            "Too many incorrect attempts. Request a new code.",
            code="otp_too_many_attempts",
        )

    expected_hash = _hash_code(code)
    if not hmac.compare_digest(record.code_hash, expected_hash):
        record.attempts += 1
        db.commit()
        remaining = max(0, OTP_MAX_ATTEMPTS - record.attempts)
        raise OtpError(
            f"Incorrect code. {remaining} attempt(s) remaining.",
            code="otp_incorrect",
        )

    record.consumed_at = datetime.utcnow()
    db.commit()
    return True
