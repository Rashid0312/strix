"""Shared OAuth 2.0 + PKCE logic for multi-provider authentication.

Handles token exchange, refresh, storage, and expiry detection for any OIDC provider
(OpenAI, Anthropic, etc.). Provider-specific constants and client building live
in dedicated modules (codex.py, claude_oauth.py).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import secrets
import threading
import time
import urllib.parse
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from strix.utils.secret_files import write_secret_text


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


# Kept separate from cli-config.json so OAuth tokens never land in the env-var config.
AUTH_PATH = Path.home() / ".strix" / "subscription-auth.json"

_TOKEN_TIMEOUT = 30
_EXPIRY_SKEW_S = 300

_refresh_lock = threading.Lock()


class OAuthError(Exception):
    """Generic OAuth error with provider-agnostic code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def _read_store() -> dict[str, Any]:
    """Read entire auth store (all providers)."""
    try:
        data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_store(data: dict[str, Any]) -> None:
    """Write entire auth store with secure permissions."""
    write_secret_text(AUTH_PATH, json.dumps(data, indent=2))


def read_record(provider: str) -> dict[str, Any] | None:
    """Read OAuth record for a specific provider."""
    record = _read_store().get(provider)
    if not isinstance(record, dict) or record.get("type") != "oauth":
        return None
    if not (record.get("access") and record.get("refresh") and record.get("account_id")):
        return None
    return record


def is_authenticated(provider: str) -> bool:
    """Check if provider auth exists and is valid."""
    return read_record(provider) is not None


def save_record(provider: str, record: dict[str, Any]) -> None:
    """Save OAuth record for a provider."""
    data = _read_store()
    data[provider] = record
    _write_store(data)


def logout(provider: str) -> None:
    """Remove OAuth record for a provider."""
    data = _read_store()
    if provider not in data:
        return
    del data[provider]
    if data:
        _write_store(data)
        return
    with contextlib.suppress(OSError):
        AUTH_PATH.unlink()


@contextlib.contextmanager
def refresh_guard() -> Generator[None, None, None]:
    """Serialize token refresh within (lock) and across (flock) processes,
    so concurrent runs can't both spend the single-use refresh token."""
    with _refresh_lock:
        try:
            import fcntl

            lock_path = AUTH_PATH.with_suffix(".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("w")
        except (ImportError, OSError):
            yield
            return
        try:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _b64url(raw: bytes) -> str:
    """Base64-URL encode without padding."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate PKCE (code_verifier, code_challenge)."""
    verifier = _b64url(secrets.token_bytes(64))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def create_state() -> str:
    """Generate OAuth state for CSRF protection."""
    return secrets.token_hex(16)


def parse_redirect_input(value: str) -> tuple[str | None, str | None]:
    """Extract ``(code, state)`` from a pasted redirect URL, ``code#state``,
    query string, or bare code."""
    value = (value or "").strip()
    if not value:
        return None, None
    with contextlib.suppress(ValueError):
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme and parsed.query:
            query = urllib.parse.parse_qs(parsed.query)
            return _first(query, "code"), _first(query, "state")
    if "#" in value:
        code, _, state = value.partition("#")
        return code or None, state or None
    if "code=" in value:
        query = urllib.parse.parse_qs(value)
        return _first(query, "code"), _first(query, "state")
    return value, None


def _first(query: dict[str, list[str]], key: str) -> str | None:
    """Get first value from query dict."""
    values = query.get(key)
    return values[0] if values else None


def _post_form(url: str, payload: dict[str, str]) -> dict[str, Any]:
    """POST form data to token endpoint, return JSON response."""
    detail = ""
    try:
        with requests.post(
            url,
            data=payload,
            headers={"Accept": "application/json"},
            timeout=_TOKEN_TIMEOUT,
        ) as response:
            status_code = response.status_code
            body = response.content
            if status_code >= 400:
                detail = response.text[:300]
    except requests.RequestException as exc:
        raise OAuthError("unavailable", str(exc)) from exc
    if status_code >= 400:
        raise OAuthError("token_http_error", f"HTTP {status_code}: {detail}")
    data = json.loads(body or b"{}")
    if not isinstance(data, dict):
        raise OAuthError("bad_response", "token endpoint returned non-object")
    return data


def account_id_from_jwt(token: str | None, account_claim: str) -> str | None:
    """Extract account_id from JWT payload without verification.

    Args:
        token: JWT token (or None/empty)
        account_claim: Provider-specific claim key to read (e.g., "https://api.openai.com/auth")

    Returns:
        account_id string or None if not found
    """
    if not token or token.count(".") != 2:
        return None
    payload_b64 = token.split(".")[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    # Provider-specific claim (e.g., "https://api.openai.com/auth")
    auth = payload.get(account_claim)
    if isinstance(auth, dict):
        account_id = auth.get("chatgpt_account_id") or auth.get("user_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    # Fallback: organizations list
    organizations = payload.get("organizations")
    if isinstance(organizations, list) and organizations and isinstance(organizations[0], dict):
        org_id = organizations[0].get("id")
        if isinstance(org_id, str) and org_id:
            return org_id
    return None


def _near_expiry(record: dict[str, Any]) -> bool:
    """Check if token is near expiry (within skew window)."""
    expires_at = record.get("expires_at")
    if not isinstance(expires_at, int | float):
        return True
    return expires_at - _EXPIRY_SKEW_S <= time.time()


def record_from_token_response(
    data: dict[str, Any],
    provider: str,
    account_claim: str,
    refresh_fallback: str | None = None,
) -> dict[str, Any]:
    """Convert token response to record format.

    Args:
        data: JSON response from token endpoint
        provider: provider name (e.g., "codex", "claude")
        account_claim: provider-specific claim key for JWT parsing
        refresh_fallback: fallback refresh_token if response omits it (for refresh case)

    Returns:
        OAuth record dict with type, provider, access, refresh, account_id, expires_at
    """
    access = data.get("access_token")
    # A refresh response may omit refresh_token when it isn't rotated; keep the old one.
    refresh = data.get("refresh_token") or refresh_fallback
    expires_in = data.get("expires_in")
    if not isinstance(access, str) or not access:
        raise OAuthError("bad_response", "token response missing access_token")
    if not isinstance(refresh, str) or not refresh:
        raise OAuthError("bad_response", "token response missing refresh_token")
    account_id = account_id_from_jwt(access, account_claim) or account_id_from_jwt(
        data.get("id_token") if isinstance(data.get("id_token"), str) else "", account_claim
    )
    if not account_id:
        raise OAuthError("no_account_id", f"could not read account_id from {provider} token")
    ttl = expires_in if isinstance(expires_in, int | float) else 3600
    return {
        "type": "oauth",
        "provider": provider,
        "access": access,
        "refresh": refresh,
        "account_id": account_id,
        "expires_at": time.time() + ttl,
    }
