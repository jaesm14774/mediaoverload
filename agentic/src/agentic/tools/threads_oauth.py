from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, quote_plus, urlencode, urlsplit

import requests
from dotenv import dotenv_values, set_key


GRAPH_API_BASE = "https://graph.threads.net"
GRAPH_API_VERSION = "v1.0"
LONG_LIVED_TOKEN_SECONDS = 60 * 24 * 60 * 60
TOKEN_REFRESH_WINDOW_SECONDS = 7 * 24 * 60 * 60


class ThreadsOAuthError(RuntimeError):
    """Raised when Threads token exchange or refresh cannot be completed."""


def _meta_error_details(response: requests.Response, sensitive_values: list[str]) -> str:
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""

    fields = [
        ("type", error.get("type")),
        ("code", error.get("code")),
        ("subcode", error.get("error_subcode")),
        ("message", error.get("message")),
        ("fbtrace_id", error.get("fbtrace_id")),
    ]
    details: list[str] = []
    redactions = {
        variant
        for value in sensitive_values
        if value
        for variant in (value, quote(value, safe=""), quote_plus(value, safe=""))
    }
    for name, raw_value in fields:
        if raw_value is None or str(raw_value) == "":
            continue
        value = str(raw_value).replace("\r", " ").replace("\n", " ")
        if name == "message":
            for secret in sorted(redactions, key=len, reverse=True):
                value = value.replace(secret, "[redacted]")
            value = value[:500]
        details.append(f"{name}={value}")
    return f" Meta response: {'; '.join(details)}" if details else ""


def _request_json(
    *,
    url: str,
    params: dict[str, str],
    access_token: str,
    operation: str,
) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ThreadsOAuthError(
            f"Threads {operation} request failed ({type(exc).__name__})."
        ) from None

    if not 200 <= response.status_code < 300:
        # Never include the request URL or raw response body: exchange params
        # contain the app secret. Only selected fields are returned after redaction.
        sensitive_values = [access_token]
        sensitive_values.extend(
            value
            for key, value in params.items()
            if "secret" in key.lower() or "token" in key.lower()
        )
        details = _meta_error_details(response, sensitive_values)
        raise ThreadsOAuthError(
            f"Threads {operation} failed with HTTP {response.status_code}.{details}"
        )
    if not response.content:
        return {}
    try:
        body = response.json()
    except requests.JSONDecodeError:
        raise ThreadsOAuthError(
            f"Threads {operation} returned invalid JSON."
        ) from None
    if not isinstance(body, dict):
        raise ThreadsOAuthError(f"Threads {operation} returned an invalid response.")
    return body


def _post_json(
    *,
    url: str,
    params: dict[str, str],
    sensitive_values: list[str],
    operation: str,
) -> dict[str, Any]:
    try:
        response = requests.post(url, params=params, timeout=30)
    except requests.RequestException as exc:
        raise ThreadsOAuthError(
            f"Threads {operation} request failed ({type(exc).__name__})."
        ) from None
    if not 200 <= response.status_code < 300:
        details = _meta_error_details(response, sensitive_values)
        raise ThreadsOAuthError(
            f"Threads {operation} failed with HTTP {response.status_code}.{details}"
        )
    if not response.content:
        return {}
    try:
        body = response.json()
    except requests.JSONDecodeError:
        raise ThreadsOAuthError(f"Threads {operation} returned invalid JSON.") from None
    if not isinstance(body, dict):
        raise ThreadsOAuthError(f"Threads {operation} returned an invalid response.")
    return body


def _write_credentials(credentials_path: Path, values: dict[str, str]) -> None:
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    existing = credentials_path.read_text(encoding="utf-8") if credentials_path.exists() else ""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{credentials_path.name}.",
        suffix=".tmp",
        dir=credentials_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(existing, encoding="utf-8")
        for key, value in values.items():
            set_key(temporary_path, key, value, quote_mode="always", encoding="utf-8")
        os.replace(temporary_path, credentials_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fetch_profile(access_token: str, *, api_base: str) -> dict[str, Any]:
    return _request_json(
        url=f"{api_base.rstrip('/')}/{GRAPH_API_VERSION}/me",
        params={"fields": "id,username"},
        access_token=access_token,
        operation="profile lookup",
    )


def exchange_short_lived_token(
    short_lived_token: str,
    app_secret: str,
    credentials_path: str | Path,
    *,
    api_base: str = GRAPH_API_BASE,
    now: int | None = None,
) -> dict[str, str | int]:
    """Exchange an authorized short-lived token and atomically save its successor."""
    if not short_lived_token.strip() or not app_secret.strip():
        raise ThreadsOAuthError("A short-lived token and Threads App Secret are required.")

    body = _request_json(
        url=f"{api_base.rstrip('/')}/access_token",
        params={
            "grant_type": "th_exchange_token",
            "client_secret": app_secret,
        },
        access_token=short_lived_token,
        operation="long-lived token exchange",
    )
    access_token = str(body.get("access_token") or "").strip()
    try:
        expires_in = int(body.get("expires_in") or 0)
    except (TypeError, ValueError):
        expires_in = 0
    if not access_token or expires_in <= 0:
        raise ThreadsOAuthError(
            "Threads token exchange did not return an access token and expiry."
        )

    profile = _fetch_profile(access_token, api_base=api_base)
    user_id = str(profile.get("id") or "").strip()
    if not user_id:
        raise ThreadsOAuthError("Threads profile lookup did not return a user id.")

    expires_at = int(now if now is not None else time.time()) + expires_in
    _write_credentials(
        Path(credentials_path),
        {
            "THREADS_ACCESS_TOKEN": access_token,
            "THREADS_ACCESS_TOKEN_EXPIRES_AT": str(expires_at),
            "THREADS_USER_ID": user_id,
        },
    )
    return {
        "user_id": user_id,
        "username": str(profile.get("username") or ""),
        "expires_at": expires_at,
    }


def exchange_authorization_code(
    code: str,
    app_id: str,
    app_secret: str,
    redirect_uri: str,
    credentials_path: str | Path,
    *,
    api_base: str = GRAPH_API_BASE,
    now: int | None = None,
) -> dict[str, str | int]:
    if not all(value.strip() for value in (code, app_id, app_secret, redirect_uri)):
        raise ThreadsOAuthError(
            "An authorization code, Threads App ID, App Secret, and redirect URI are required."
        )
    body = _post_json(
        url=f"{api_base.rstrip('/')}/oauth/access_token",
        params={
            "client_id": app_id,
            "client_secret": app_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        sensitive_values=[code, app_secret],
        operation="authorization code exchange",
    )
    short_lived_token = str(body.get("access_token") or "").strip()
    if not short_lived_token:
        raise ThreadsOAuthError(
            "Threads authorization exchange did not return a user access token."
        )
    return exchange_short_lived_token(
        short_lived_token,
        app_secret,
        credentials_path,
        api_base=api_base,
        now=now,
    )


def refresh_token_if_due(
    access_token: str,
    expires_at: int,
    credentials_path: str | Path,
    *,
    api_base: str = GRAPH_API_BASE,
    now: int | None = None,
) -> tuple[str, int, bool]:
    """Refresh an unexpired token within seven days of expiry and persist it."""
    current_time = int(now if now is not None else time.time())
    remaining = int(expires_at) - current_time
    if remaining <= 0:
        raise ThreadsOAuthError(
            "Threads access token has expired; reauthorize the Threads account."
        )
    if remaining > TOKEN_REFRESH_WINDOW_SECONDS:
        return access_token, int(expires_at), False

    body = _request_json(
        url=f"{api_base.rstrip('/')}/refresh_access_token",
        params={"grant_type": "th_refresh_token"},
        access_token=access_token,
        operation="access token refresh",
    )
    refreshed_token = str(body.get("access_token") or access_token).strip()
    try:
        expires_in = int(body.get("expires_in") or LONG_LIVED_TOKEN_SECONDS)
    except (TypeError, ValueError):
        expires_in = LONG_LIVED_TOKEN_SECONDS
    if not refreshed_token or expires_in <= 0:
        raise ThreadsOAuthError("Threads token refresh returned invalid token metadata.")

    refreshed_expiry = current_time + expires_in
    _write_credentials(
        Path(credentials_path),
        {
            "THREADS_ACCESS_TOKEN": refreshed_token,
            "THREADS_ACCESS_TOKEN_EXPIRES_AT": str(refreshed_expiry),
        },
    )
    return refreshed_token, refreshed_expiry, True


def exchange_credentials_file(credentials_path: str | Path, app_secret: str) -> dict[str, str | int]:
    values = dotenv_values(credentials_path)
    short_lived_token = str(values.get("THREADS_ACCESS_TOKEN") or "").strip()
    if not short_lived_token or short_lived_token.startswith("your_threads_"):
        raise ThreadsOAuthError(
            "THREADS_ACCESS_TOKEN is missing from the credentials file."
        )
    return exchange_short_lived_token(
        short_lived_token,
        app_secret,
        credentials_path,
    )


def refresh_credentials_file(credentials_path: str | Path) -> tuple[int, bool]:
    values = dotenv_values(credentials_path)
    access_token = str(values.get("THREADS_ACCESS_TOKEN") or "").strip()
    raw_expiry = str(values.get("THREADS_ACCESS_TOKEN_EXPIRES_AT") or "").strip()
    if not access_token or not raw_expiry:
        raise ThreadsOAuthError(
            "THREADS_ACCESS_TOKEN and THREADS_ACCESS_TOKEN_EXPIRES_AT are required."
        )
    try:
        expires_at = int(raw_expiry)
    except ValueError:
        raise ThreadsOAuthError("THREADS_ACCESS_TOKEN_EXPIRES_AT must be a Unix timestamp.") from None
    _, refreshed_expiry, refreshed = refresh_token_if_due(
        access_token,
        expires_at,
        credentials_path,
    )
    return refreshed_expiry, refreshed


def _authorization_url(app_id: str, redirect_uri: str, state: str) -> str:
    query = urlencode(
        {
            "client_id": app_id,
            "redirect_uri": redirect_uri,
            "scope": "threads_basic,threads_content_publish",
            "response_type": "code",
            "state": state,
        }
    )
    return f"https://threads.net/oauth/authorize?{query}"


def _code_from_callback(callback_url: str, expected_state: str) -> str:
    query = parse_qs(urlsplit(callback_url.strip()).query)
    if query.get("error"):
        error = str(query["error"][0])
        raise ThreadsOAuthError(f"Threads authorization was denied ({error}).")
    code = str((query.get("code") or [""])[0]).strip()
    state = str((query.get("state") or [""])[0]).strip()
    if not code:
        raise ThreadsOAuthError("The callback URL did not contain an authorization code.")
    if not state or state != expected_state:
        raise ThreadsOAuthError("Threads OAuth state did not match; start authorization again.")
    return code


def _default_credentials_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "configs" / "social_media" / "credentials" / "kirby" / "threads.env"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exchange or refresh Threads OAuth tokens.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("exchange", "refresh", "authorize"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--credentials",
            type=Path,
            default=_default_credentials_path(),
            help="Threads env file (defaults to this repository's Kirby credentials).",
        )
        if command == "authorize":
            command_parser.add_argument("--app-id")
            command_parser.add_argument("--redirect-uri")
    args = parser.parse_args(argv)
    credentials_path = args.credentials.expanduser().resolve()

    try:
        if args.command == "authorize":
            app_id = (args.app_id or os.getenv("THREADS_APP_ID") or input("Threads App ID: ")).strip()
            redirect_uri = (
                args.redirect_uri
                or os.getenv("THREADS_REDIRECT_URI")
                or input("Registered Threads OAuth redirect URI: ")
            ).strip()
            app_secret = os.getenv("THREADS_APP_SECRET") or ""
            if not app_secret:
                if not sys.stdin.isatty():
                    raise ThreadsOAuthError(
                        "Run this command in a local terminal to enter App Secret securely."
                    )
                app_secret = getpass.getpass(
                    "Threads App Secret (input hidden; not saved): "
                ).strip()
            state = secrets.token_urlsafe(32)
            authorize_url = _authorization_url(app_id, redirect_uri, state)
            print("Opening Threads authorization. Approve with the invited Threads account.")
            print(authorize_url)
            webbrowser.open(authorize_url, new=2)
            callback_url = getpass.getpass(
                "After approval, paste the full redirected callback URL (input hidden): "
            ).strip()
            code = _code_from_callback(callback_url, state)
            result = exchange_authorization_code(
                code,
                app_id,
                app_secret,
                redirect_uri,
                credentials_path,
            )
            expiry = time.strftime(
                "%Y-%m-%d %H:%M:%S UTC",
                time.gmtime(int(result["expires_at"])),
            )
            print(
                "Threads long-lived token saved. "
                f"User @{result['username']} ({result['user_id']}), expires {expiry}."
            )
            return 0

        if args.command == "exchange":
            app_secret = os.getenv("THREADS_APP_SECRET") or ""
            if not app_secret:
                if not sys.stdin.isatty():
                    raise ThreadsOAuthError(
                        "Run this command in a local terminal to enter App Secret securely."
                    )
                app_secret = getpass.getpass(
                    "Threads App Secret (input hidden; not saved): "
                ).strip()
            result = exchange_credentials_file(credentials_path, app_secret)
            expiry = time.strftime(
                "%Y-%m-%d %H:%M:%S UTC",
                time.gmtime(int(result["expires_at"])),
            )
            print(
                "Threads long-lived token saved. "
                f"User @{result['username']} ({result['user_id']}), expires {expiry}."
            )
            return 0

        expires_at, refreshed = refresh_credentials_file(credentials_path)
        expiry = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(expires_at))
        print(
            f"Threads token {'refreshed' if refreshed else 'still valid'}; "
            f"expires {expiry}."
        )
        return 0
    except ThreadsOAuthError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
