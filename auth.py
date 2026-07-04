# auth.py
"""
Lightweight WebSocket authentication for RealTimeSpeechTranslator.

Why authenticate the WebSocket endpoint?
─────────────────────────────────────────
Without auth, anyone who finds your server's IP/domain can connect to
/ws/translate and use your GPU for free — including running up VRAM,
triggering model inference, and reading translation output.

How it works
────────────
The frontend sends a signed token in the WebSocket URL:
  ws://your-server:8000/ws/translate?token=ABC123

The backend validates the token before accepting the connection.
Invalid or missing tokens get a 403 close and are never added to the
connection manager.

Two token modes (pick one based on your deployment)
─────────────────────────────────────────────────────

MODE A — STATIC SHARED SECRET (current implementation, simplest)
  One secret token is set via the SESSION_TOKEN env variable.
  All authorized users share the same token.
  Good for: internal demo, single-team deployment.
  Bad for:  multi-user / multi-department production (token leaks = full access).

  Setup:
    export SESSION_TOKEN=some-long-random-string-here
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"

MODE B — PER-USER JWT (commented out below, upgrade path)
  Each user gets a JWT signed with your server's secret.
  The JWT carries user_id, department_id, and expiry.
  Your auth server (or a simple /login endpoint) issues JWTs.
  The WebSocket handler decodes the JWT and loads department_id from it
  instead of from config.py.
  Good for: real multi-tenant deployment.

Frontend change needed (add token to URL)
─────────────────────────────────────────
  // In frontend/index.html, change:
  const WS_URL = `${protocol}//${host}/ws/translate`;

  // To:
  const TOKEN = "your-token-here";  // or read from sessionStorage after login
  const WS_URL = `${protocol}//${host}/ws/translate?token=${TOKEN}`;
"""

import hmac
import logging
import os
import secrets

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger("ispeak.auth")

# ── Load the shared secret ────────────────────────────────────────────────────
# In production: export SESSION_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
# In development: the fallback "dev-insecure-token" is used so the app still
# starts, but a WARNING is printed so you don't forget to set it in production.

_SESSION_TOKEN: str = os.environ.get("SESSION_TOKEN", "")

if not _SESSION_TOKEN:
    _SESSION_TOKEN = "dev-insecure-token"
    logger.warning(
        "[Auth] SESSION_TOKEN env variable is not set. "
        "Using insecure dev token. Set SESSION_TOKEN before deploying."
    )

# ── Runtime token (generated per server restart) ──────────────────────────────
# This token is injected into the frontend HTML at serve-time so it never
# appears in source code or git history. It rotates on every server restart.
_RUNTIME_TOKEN: str = ""


def set_runtime_token(token: str) -> None:
    """Called once at startup to register the auto-generated runtime token."""
    global _RUNTIME_TOKEN
    _RUNTIME_TOKEN = token


def validate_ws_token(token: str | None) -> bool:
    """
    Returns True if the token is valid, False otherwise.

    Accepts EITHER the static SESSION_TOKEN (from env) OR the runtime token
    (auto-generated per server restart and injected into the frontend HTML).

    Uses hmac.compare_digest() for constant-time comparison — this prevents
    timing attacks where an attacker could guess the token one character at
    a time by measuring response latency.
    """
    if not token:
        return False
    token_bytes = token.encode()
    return (
        hmac.compare_digest(token_bytes, _SESSION_TOKEN.encode())
        or (_RUNTIME_TOKEN and hmac.compare_digest(token_bytes, _RUNTIME_TOKEN.encode()))
    )


async def reject_websocket(websocket: WebSocket, reason: str) -> None:
    """
    Accepts the WebSocket handshake (required by the protocol) then
    immediately closes it with a 4003 application-level close code.

    We MUST accept first — the HTTP upgrade has already happened before
    FastAPI calls our handler, so we can't send a plain 403 HTTP response.
    Sending a 4003 close code is the WebSocket-idiomatic equivalent of 403.
    """
    await websocket.accept()
    logger.warning(f"[Auth] Rejected WebSocket connection: {reason}")
    await websocket.close(code=4003)


# ─────────────────────────────────────────────────────────────────────────────
# MODE B — JWT upgrade path (commented out)
# Install: pip install python-jose[cryptography]
# ─────────────────────────────────────────────────────────────────────────────
# from jose import JWTError, jwt
# from datetime import datetime
#
# JWT_SECRET = os.environ.get("JWT_SECRET", "")
# JWT_ALGORITHM = "HS256"
#
# def validate_jwt_token(token: str | None) -> dict | None:
#     """
#     Returns the decoded payload dict if valid, None if invalid/expired.
#     Payload typically contains: {"sub": user_id, "dept": department_id, "exp": timestamp}
#     """
#     if not token:
#         return None
#     try:
#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
#         if datetime.utcnow().timestamp() > payload.get("exp", 0):
#             return None
#         return payload
#     except JWTError:
#         return None
