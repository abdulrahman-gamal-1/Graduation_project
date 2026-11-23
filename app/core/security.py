"""Service-to-service authentication.

The Node.js backend sends a shared secret in `X-Internal-API-Key`. This service must only be
reachable from the private network; the key is defence in depth, not the only control.
"""
import hmac

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

_api_key_header = APIKeyHeader(
    name="X-Internal-API-Key",
    auto_error=False,
    description="Shared secret between the GlucoCare Node.js backend and this service.",
)


async def require_internal_key(key: str | None = Security(_api_key_header)) -> None:
    expected = settings.internal_api_key
    if not expected:
        if settings.env == "dev":
            return
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Service authentication is not configured."
        )
    if not key or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing internal API key.")
