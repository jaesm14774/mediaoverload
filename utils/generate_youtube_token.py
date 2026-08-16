from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
YOUTUBE_SCOPE = ["https://www.googleapis.com/auth/youtube.upload"]


def _read_client_config(client_secret_path: Path) -> tuple[str, str]:
    payload = json.loads(client_secret_path.read_text(encoding="utf-8"))
    installed = dict(payload.get("installed") or {})
    client_id = str(installed.get("client_id") or "").strip()
    client_secret = str(installed.get("client_secret") or "").strip()
    if not client_id or not client_secret:
        raise ValueError("client secret JSON is missing installed.client_id or installed.client_secret")
    return client_id, client_secret


def _resolve_character_dir(character: str) -> Path:
    directory = ROOT_DIR / "configs" / "social_media" / "credentials" / character
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def generate_token(character: str, client_secret_path: Path) -> Path:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise RuntimeError("Missing YouTube OAuth dependency; install google-auth-oauthlib") from exc

    client_id, client_secret = _read_client_config(client_secret_path)
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret_path), scopes=YOUTUBE_SCOPE)
    credentials = flow.run_local_server(
        host="localhost",
        port=0,
        authorization_prompt_message="Open the browser to authorize YouTube upload access.",
        success_message="YouTube authorization succeeded. You can close this tab.",
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError("OAuth flow did not return a refresh token; revoke access and try again with prompt=consent")

    target_path = _resolve_character_dir(character) / "youtube.env"
    lines = [
        f"YOUTUBE_CLIENT_ID={client_id}",
        f"YOUTUBE_CLIENT_SECRET={client_secret}",
        f"YOUTUBE_REFRESH_TOKEN={credentials.refresh_token}",
        "YOUTUBE_PRIVACY_STATUS=public",
        "YOUTUBE_CATEGORY_ID=22",
        "YOUTUBE_NOTIFY_SUBSCRIBERS=false",
        "YOUTUBE_MADE_FOR_KIDS=false",
        "YOUTUBE_CONTAINS_SYNTHETIC_MEDIA=true",
    ]
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and store a YouTube OAuth refresh token")
    parser.add_argument(
        "--character",
        default="kirby",
        help="Character credential folder name under configs/social_media/credentials",
    )
    parser.add_argument(
        "--client-secret-json",
        required=True,
        help="Path to the Google OAuth client_secret JSON downloaded from Google Cloud",
    )
    args = parser.parse_args()

    output_path = generate_token(args.character, Path(args.client_secret_json).resolve())
    print(f"Wrote YouTube OAuth credentials to {output_path}")


if __name__ == "__main__":
    main()
