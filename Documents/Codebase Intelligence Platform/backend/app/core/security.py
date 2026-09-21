import logging
import secrets
from fastapi import Request, HTTPException, status

from app.core.config import settings

logger = logging.getLogger("security")


def _extract_key(request: Request) -> str:
    """Reads the API key from the configured header, falling back to a bearer token."""
    provided = request.headers.get(settings.API_KEY_HEADER, "").strip()
    if provided:
        return provided

    authorization = request.headers.get("Authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[len("bearer "):].strip()

    return ""


def is_authenticated(request: Request) -> bool:
    """
    Non-raising check, for endpoints that stay public but reveal more to an authenticated caller.
    Returns True when auth is disabled, since in that mode there is no caller to distinguish.
    """
    if not settings.API_KEY:
        return True
    provided = _extract_key(request)
    return bool(provided) and secrets.compare_digest(provided, settings.API_KEY)


async def require_api_key(request: Request) -> None:
    """
    FastAPI dependency guarding the API routes.

    Every endpoint was previously unauthenticated: anyone who could reach the port could upload
    archives, list and read every indexed repository, read or delete any conversation by id, and
    dump the whole dependency graph — while the service stores third-party source code and,
    as uploaded archives showed, their secrets.

    Auth is opt-in so local development is unaffected; setting API_KEY turns it on everywhere.
    """
    if not settings.API_KEY:
        return

    provided = _extract_key(request)

    # compare_digest keeps the comparison constant-time, so a wrong key cannot be discovered
    # byte-by-byte from response timing.
    if not provided or not secrets.compare_digest(provided, settings.API_KEY):
        logger.warning(
            "Rejected unauthenticated request to %s from %s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid API key. Provide it in the {settings.API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
