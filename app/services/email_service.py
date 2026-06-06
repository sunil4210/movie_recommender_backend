"""Email delivery for OTP codes.

Uses Python's stdlib `smtplib` so the backend has zero extra dependencies and
works out-of-the-box against any SMTP relay (Brevo, Mailtrap, Gmail, etc.).

Falls back to logging the OTP to stdout when `SMTP_HOST` is unset, so the
project still works in a clean local environment without any email creds.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from pathlib import Path

from app.config import (
    BASE_DIR,
    SMTP_FROM,
    SMTP_FROM_NAME,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USER,
    SMTP_USE_TLS,
    OTP_TTL_MINUTES,
)

logger = logging.getLogger(__name__)


_SUBJECTS = {
    "signup": "Verify your CineMatch email",
    "reset": "Your CineMatch password reset code",
}

_LOGO_PATH: Path = BASE_DIR / "assets" / "cinematch-logo.png"


def _load_logo_bytes() -> bytes | None:
    """Read the brand logo once. Returns None if the asset is missing so the
    email still goes out with a text header instead of failing the send."""
    try:
        return _LOGO_PATH.read_bytes()
    except FileNotFoundError:
        logger.warning("Logo asset missing at %s — email will skip the logo", _LOGO_PATH)
        return None


_LOGO_BYTES = _load_logo_bytes()


def _build_html(code: str, intro: str, action: str, logo_cid: str | None) -> str:
    """Build the light-mode HTML body. `logo_cid` is a bare message id (no <>)
    that the <img> references via `cid:` — leave None to skip the logo block."""
    if logo_cid:
        logo_block = (
            f'<img src="cid:{logo_cid}" alt="CineMatch" width="56" height="56" '
            f'style="display:block;width:56px;height:56px;max-width:56px;'
            f'border:0;outline:none;text-decoration:none;border-radius:12px;'
            f'margin:0 auto;">'
        )
    else:
        logo_block = ""

    return f"""\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CineMatch</title>
  </head>
  <body style="margin:0;padding:0;background:#f4f6fb;
               font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
               Helvetica,Arial,sans-serif;color:#1a1d29;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
           border="0" style="background:#f4f6fb;padding:40px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="600" cellpadding="0" cellspacing="0"
                 border="0" style="max-width:600px;width:100%;background:#ffffff;
                 border-radius:16px;box-shadow:0 4px 24px rgba(33,40,72,0.08);
                 overflow:hidden;">
            <!-- Header strip -->
            <tr>
              <td style="background:#ffffff;padding:36px 40px 8px 40px;
                         text-align:center;">
                {logo_block}
                <h1 style="margin:12px 0 0 0;color:#2196F3;font-size:22px;
                           font-weight:700;letter-spacing:0.2px;">
                  CineMatch
                </h1>
              </td>
            </tr>
            <!-- Body -->
            <tr>
              <td style="padding:40px 48px 16px 48px;">
                <h2 style="margin:0 0 16px 0;color:#1a1d29;font-size:22px;
                           font-weight:700;line-height:1.3;">
                  {intro}
                </h2>
                <p style="margin:0 0 8px 0;color:#4a5060;font-size:16px;
                          line-height:1.6;">
                  {action}
                </p>
              </td>
            </tr>
            <!-- OTP code box -->
            <tr>
              <td style="padding:24px 48px 8px 48px;">
                <div style="background:#f4f6fb;border:1px solid #e3e7f0;
                            border-radius:12px;padding:28px 16px;
                            text-align:center;">
                  <div style="font-size:13px;font-weight:600;color:#6b7280;
                              letter-spacing:1.5px;text-transform:uppercase;
                              margin-bottom:12px;">
                    Verification code
                  </div>
                  <div style="font-size:42px;font-weight:700;letter-spacing:14px;
                              color:#2196F3;font-family:'SF Mono',Menlo,Consolas,
                              monospace;line-height:1.2;">
                    {code}
                  </div>
                </div>
              </td>
            </tr>
            <!-- Expiry note -->
            <tr>
              <td style="padding:20px 48px 8px 48px;">
                <p style="margin:0;color:#6b7280;font-size:14px;line-height:1.6;
                          text-align:center;">
                  This code expires in
                  <strong style="color:#1a1d29;">{OTP_TTL_MINUTES} minutes</strong>.
                </p>
              </td>
            </tr>
            <!-- Disclaimer -->
            <tr>
              <td style="padding:24px 48px 40px 48px;">
                <div style="border-top:1px solid #eef0f5;padding-top:20px;">
                  <p style="margin:0;color:#9aa0aa;font-size:13px;line-height:1.6;">
                    Didn't request this email? You can safely ignore it — no
                    changes will be made to your account.
                  </p>
                </div>
              </td>
            </tr>
          </table>
          <!-- Footer -->
          <p style="margin:24px 0 0 0;color:#9aa0aa;font-size:12px;
                    text-align:center;">
            &copy; CineMatch &middot; Sent because of activity on your account
          </p>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


def _build_message(to: str, code: str, purpose: str) -> EmailMessage:
    subject = _SUBJECTS.get(purpose, "Your CineMatch verification code")

    if purpose == "reset":
        intro = "Reset your password"
        action = "Use the code below to set a new password for your CineMatch account."
    else:
        intro = "Verify your email"
        action = "Welcome to CineMatch! Enter the code below to finish setting up your account."

    text_body = (
        f"{intro}\n\n"
        f"{action}\n\n"
        f"Verification code: {code}\n\n"
        f"This code expires in {OTP_TTL_MINUTES} minutes. "
        "If you didn't request this email, you can safely ignore it."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((SMTP_FROM_NAME, SMTP_FROM))
    msg["To"] = to
    msg.set_content(text_body)

    logo_cid = None
    if _LOGO_BYTES is not None:
        # make_msgid wraps in <>; strip them for the cid: URL reference.
        logo_cid = make_msgid(domain="cinematch.local")[1:-1]

    msg.add_alternative(_build_html(code, intro, action, logo_cid), subtype="html")

    if _LOGO_BYTES is not None and logo_cid is not None:
        html_part = msg.get_payload()[1]
        html_part.add_related(
            _LOGO_BYTES,
            maintype="image",
            subtype="png",
            cid=f"<{logo_cid}>",
            filename="cinematch-logo.png",
        )

    return msg


def _send_sync(msg: EmailMessage) -> None:
    """Blocking SMTP send. Runs in a worker thread via `asyncio.to_thread`."""
    if SMTP_USE_TLS:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            if SMTP_USER:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            if SMTP_USER:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)


async def send_otp_email(to: str, code: str, purpose: str) -> None:
    """Deliver an OTP to `to`.

    When SMTP is not configured, the code is logged at WARNING level so a
    developer can copy it out of the server log during local testing.
    """
    if not SMTP_HOST:
        logger.warning(
            "[email_service] SMTP not configured — OTP for %s (%s): %s",
            to, purpose, code,
        )
        return

    msg = _build_message(to, code, purpose)
    try:
        await asyncio.to_thread(_send_sync, msg)
        logger.info("OTP email sent to %s (purpose=%s)", to, purpose)
    except Exception as exc:
        logger.error("Failed to send OTP email to %s: %s", to, exc)
        raise
