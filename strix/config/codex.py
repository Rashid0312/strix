"""ChatGPT (Codex) subscription auth: OAuth login, token refresh, and the OpenAI
client that routes inference through the ChatGPT backend.

Mirrors OpenAI's Codex CLI: OAuth 2.0 + PKCE against ``auth.openai.com``, with the
access token sent as a ``Bearer`` token to ``chatgpt.com/backend-api/codex``. Using
a ChatGPT subscription outside OpenAI's own products is not officially supported by
OpenAI; the user chooses this path knowingly. The OAuth constants are OpenAI's own
Codex CLI values (the backend only accepts that client).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import urllib.parse

from strix.config import oauth_base


if TYPE_CHECKING:
    from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


PROVIDER = "codex"

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105  # nosec B105 - URL, not a secret
CALLBACK_HOST = "localhost"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPE = "openid profile email offline_access"

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
ORIGINATOR = "codex_cli_rs"
_ACCOUNT_CLAIM = "https://api.openai.com/auth"


class CodexAuthError(oauth_base.OAuthError):
    """OpenAI/Codex-specific auth error (wraps generic OAuthError)."""

    pass


class CodexContentGuardrailError(Exception):
    """The ChatGPT backend refused a request via its content guardrail.
    Terminal — retrying identical content never clears the block."""

    def __init__(self, model: str, original: BaseException | None = None) -> None:
        self.model = model
        self.original = original
        super().__init__(
            f"'{model}' was blocked by ChatGPT's content guardrails "
            f"(flagged as a possible cybersecurity risk). "
            f"Set STRIX_LLM to a model that isn't blocked and re-run."
        )


_GUARDRAIL_MARKERS = (
    "flagged for possible cybersecurity risk",
    "trusted access for cyber",
)


def is_content_guardrail_error(exc: BaseException) -> bool:
    if isinstance(exc, CodexContentGuardrailError):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _GUARDRAIL_MARKERS)


def read_record() -> dict[str, Any] | None:
    return oauth_base.read_record(PROVIDER)


def is_authenticated() -> bool:
    return oauth_base.is_authenticated(PROVIDER)


def save_record(record: dict[str, Any]) -> None:
    oauth_base.save_record(PROVIDER, record)


def logout() -> None:
    oauth_base.logout(PROVIDER)


def generate_pkce() -> tuple[str, str]:
    return oauth_base.generate_pkce()


def create_state() -> str:
    return oauth_base.create_state()


def build_authorize_url(challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",  # nosec B105 - boolean flag, not a secret
        "codex_cli_simplified_flow": "true",
        "originator": ORIGINATOR,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def parse_redirect_input(value: str) -> tuple[str | None, str | None]:
    """Extract ``(code, state)`` from a pasted redirect URL, ``code#state``,
    query string, or bare code."""
    return oauth_base.parse_redirect_input(value)


def exchange_code(code: str, verifier: str) -> dict[str, Any]:
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
    """Return ``(access_token, account_id)``, refreshing under the cross-process
    guard if near expiry."""
    record = read_record()
    if record is None:
        raise CodexAuthError("not_authenticated", "not signed in; run: strix auth login")
    if not oauth_base._near_expiry(record):
        return record["access"], record["account_id"]
    with oauth_base.refresh_guard():
        record = read_record()
        if record is None:
            raise CodexAuthError("not_authenticated", "not signed in; run: strix auth login")
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


def build_openai_client() -> AsyncOpenAI:
    """An ``AsyncOpenAI`` for the ChatGPT backend. A per-request hook re-stamps a
    fresh bearer token so long scans survive token expiry."""
    import asyncio

    import httpx
    from openai import AsyncOpenAI

    from strix.llm import request_log

    get_valid_token()  # fail fast at configure time if the sign-in is dead

    async def _auth_hook(request: httpx.Request) -> None:
        access, account_id = await asyncio.to_thread(get_valid_token)
        request.headers["Authorization"] = f"Bearer {access}"
        request.headers["chatgpt-account-id"] = account_id

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        event_hooks={"request": [_auth_hook]},
    )
    request_log.observe_http_client(http_client)
    return AsyncOpenAI(
        api_key="strix-codex-oauth",  # placeholder; the hook overwrites Authorization
        base_url=CODEX_BASE_URL,
        http_client=http_client,
        default_headers={
            "OpenAI-Beta": "responses=experimental",
            "originator": ORIGINATOR,
        },
    )


_subscription_client: AsyncOpenAI | None = None


def get_subscription_client() -> AsyncOpenAI:
    global _subscription_client  # noqa: PLW0603
    if _subscription_client is None:
        _subscription_client = build_openai_client()
    return _subscription_client


SUBSCRIPTION_PREFIX = "chatgpt/"


def subscription_model(model_name: str | None) -> str | None:
    """The model slug behind a ``chatgpt/<model>`` STRIX_LLM, or None."""
    name = (model_name or "").strip()
    if not name.lower().startswith(SUBSCRIPTION_PREFIX):
        return None
    return name[len(SUBSCRIPTION_PREFIX) :] or None


def auth_mode(model_name: str | None) -> str:
    return "subscription" if subscription_model(model_name) else "api_key"
