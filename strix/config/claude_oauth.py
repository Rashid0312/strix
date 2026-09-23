"""Claude (Anthropic) subscription auth: OAuth login, token refresh, and the Anthropic
client that routes inference through Claude's backend.

Mirrors OpenAI's Codex OAuth but routes to Anthropic's auth endpoints. Uses OAuth 2.0 + PKCE
against ``auth.anthropic.com``, with the access token sent as a ``Bearer`` token to
``api.anthropic.com``.

Note: Anthropic OAuth constants (CLIENT_ID, endpoints) must be obtained from Anthropic.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.parse
from typing import TYPE_CHECKING, Any

from strix.config import oauth_base
from strix.utils.secret_files import write_secret_text


if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


logger = logging.getLogger(__name__)


PROVIDER = "claude"

# TODO: Obtain from Anthropic (similar to OpenAI's Codex CLI client ID)
CLIENT_ID = "TODO_ANTHROPIC_OAUTH_CLIENT_ID"
AUTHORIZE_URL = "https://auth.anthropic.com/oauth/authorize"  # TODO: verify endpoint
TOKEN_URL = "https://auth.anthropic.com/oauth/token"  # noqa: S105  # nosec B105 - URL, not a secret
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 1456  # Different from OpenAI to avoid port conflict
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPE = "openid profile email offline_access"

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# TODO: verify Anthropic's account_id claim in JWT
_ACCOUNT_CLAIM = "https://api.anthropic.com/auth"


def read_record() -> dict[str, Any] | None:
    """Read Claude OAuth record."""
    return oauth_base.read_record(PROVIDER)


def is_authenticated() -> bool:
    """Check if Claude auth exists and is valid."""
    return oauth_base.is_authenticated(PROVIDER)


def save_record(record: dict[str, Any]) -> None:
    """Save Claude OAuth record."""
    oauth_base.save_record(PROVIDER, record)


def logout() -> None:
    """Remove Claude OAuth record."""
    oauth_base.logout(PROVIDER)


def build_authorize_url(challenge: str, state: str) -> str:
    """Build OAuth authorize URL with PKCE challenge."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str, verifier: str) -> dict[str, Any]:
    """Exchange authorization code for tokens."""
    data = oauth_base._post_form(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        },
    )
    return oauth_base.record_from_token_response(data, PROVIDER, _ACCOUNT_CLAIM)


def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    """Refresh access token using refresh token."""
    data = oauth_base._post_form(
        TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        },
    )
    return oauth_base.record_from_token_response(
        data, PROVIDER, _ACCOUNT_CLAIM, refresh_fallback=refresh_token
    )


def get_valid_token() -> tuple[str, str]:
    """Return ``(access_token, account_id)``, refreshing under cross-process guard
    if near expiry."""
    record = read_record()
    if record is None:
        raise oauth_base.OAuthError("not_authenticated", "not signed in; run: strix auth login claude")
    if not oauth_base._near_expiry(record):
        return record["access"], record["account_id"]
    with oauth_base.refresh_guard():
        record = read_record()
        if record is None:
            raise oauth_base.OAuthError("not_authenticated", "not signed in; run: strix auth login claude")
        if not oauth_base._near_expiry(record):
            return record["access"], record["account_id"]
        try:
            refreshed = refresh_tokens(record["refresh"])
        except oauth_base.OAuthError:
            # A peer process may have already spent this single-use refresh token.
            latest = read_record()
            if latest and latest["refresh"] != record["refresh"] and not oauth_base._near_expiry(latest):
                return latest["access"], latest["account_id"]
            raise
        save_record(refreshed)
        return refreshed["access"], refreshed["account_id"]


def build_anthropic_client() -> AsyncAnthropic:
    """An ``AsyncAnthropic`` for Claude models via OAuth. A per-request hook
    re-stamps a fresh bearer token so long scans survive token expiry."""
    import httpx
    from anthropic import AsyncAnthropic

    from strix.llm import request_log

    get_valid_token()  # fail fast at configure time if the sign-in is dead

    async def _auth_hook(request: httpx.Request) -> None:
        access, _ = await asyncio.to_thread(get_valid_token)
        request.headers["Authorization"] = f"Bearer {access}"

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        event_hooks={"request": [_auth_hook]},
    )
    request_log.observe_http_client(http_client)
    return AsyncAnthropic(
        api_key="strix-claude-oauth",  # placeholder; the hook overwrites Authorization
        base_url=ANTHROPIC_BASE_URL,
        http_client=http_client,
    )


_subscription_client: AsyncAnthropic | None = None


def get_subscription_client() -> AsyncAnthropic:
    global _subscription_client  # noqa: PLW0603
    if _subscription_client is None:
        _subscription_client = build_anthropic_client()
    return _subscription_client


SUBSCRIPTION_PREFIX = "claude/"


def subscription_model(model_name: str | None) -> str | None:
    """The model slug behind a ``claude/<model>`` STRIX_LLM, or None."""
    name = (model_name or "").strip()
    if not name.lower().startswith(SUBSCRIPTION_PREFIX):
        return None
    return name[len(SUBSCRIPTION_PREFIX) :] or None


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"
