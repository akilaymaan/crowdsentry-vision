"""API authentication and credential hygiene.

CrowdSentry has one credential: a shared API key from ``settings.api_key`` (the
``API_KEY`` environment variable). When it is empty -- the default -- every check
passes, which keeps local development friction-free. Set it for anything reachable
over a network.

HTTP clients authenticate with either ``Authorization: Bearer <key>`` or an
``X-API-Key`` header. WebSocket clients (``/ws/live``) may use the same headers --
non-browser clients -- or a ``?api_key=`` query parameter, since the browser
WebSocket API cannot set headers.

This is shared-secret auth, appropriate for an ops dashboard behind network
controls. It is not per-user identity: if you need roles or audit attribution, put
the API behind an authenticating reverse proxy (oauth2-proxy, an IdP-aware ingress)
and keep this as the second layer.
"""

from __future__ import annotations

import hmac
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, WebSocket
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("api.security")

# auto_error=False so the dependency can pick whichever credential the client sent
# rather than FastAPI raising on the first scheme that is absent.
_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

#: Returned by the checks below; kept as a constant so call sites stay readable.
AUTH_FAILED = "missing or invalid API key"


def _matches(candidate: str | None) -> bool:
    """Constant-time comparison; empty keys never match even an empty setting."""
    if not candidate:
        return False
    return hmac.compare_digest(candidate, settings.api_key)


def _credentials_ok(bearer, api_key) -> bool:
    if not settings.api_key:
        return True
    return _matches(api_key) or _matches(
        bearer.credentials if bearer is not None else None
    )


async def require_api_key(
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer),
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """FastAPI dependency; attach to routers. Raises 401 unless the key matches."""
    from fastapi import HTTPException, status

    if _credentials_ok(bearer, api_key):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=AUTH_FAILED,
        headers={"WWW-Authenticate": "Bearer"},
    )


def websocket_authorized(websocket: WebSocket) -> bool:
    """Whether an incoming /ws/live handshake carries the configured key.

    Accepts the ``api_key`` query parameter (browsers), the ``X-API-Key`` header,
    or an ``Authorization: Bearer`` header (both for non-browser clients).
    """
    if not settings.api_key:
        return True

    if _matches(websocket.query_params.get("api_key")):
        return True
    if _matches(websocket.headers.get("x-api-key")):
        return True

    authorization = websocket.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and _matches(token):
        return True
    return False


def redact_url(url: str | None) -> str | None:
    """Strip ``user:pass@`` credentials out of a URL for logs and status output.

    RTSP camera URLs routinely embed credentials (``rtsp://admin:pw@cam/stream``);
    those must never reach an API response or a log line. Non-URL specs -- file
    paths, device indices -- pass through unchanged.
    """
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.password is None and parts.username is None:
        return url
    host = parts.hostname or ""
    try:
        port = parts.port  # ValueError on a malformed port, e.g. rtsp://u:p@cam:x/y
    except ValueError:
        port = None
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


__all__ = ["AUTH_FAILED", "redact_url", "require_api_key", "websocket_authorized"]
