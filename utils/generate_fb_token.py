"""Facebook 長期 Page Access Token 產生器

流程：短期 User Token → 長期 User Token (60天)
短期 Token 可從 Graph API Explorer 取得：https://developers.facebook.com/tools/explorer/
需具備 pages_manage_posts, pages_read_engagement 權限
"""
from pathlib import Path

import requests
from dotenv import dotenv_values

root_dir = Path(__file__).parent.parent

GRAPH_API_VERSION = "v24.0"
GRAPH_API_BASE = "https://graph.facebook.com"


def _exchange_long_lived_user_token(app_id: str, app_secret: str, short_token: str) -> str | None:
    """短期 User Token 交換為長期 User Token (效期約 60 天)"""
    url = f"{GRAPH_API_BASE}/{GRAPH_API_VERSION}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": short_token,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    if "access_token" not in data:
        print("錯誤：無法取得長期 User Token", data)
        return None
    return data["access_token"]


def _select_credentials_file(
    credentials_root: Path,
    *,
    choice: str | None = None,
) -> Path | None:
    characters = [directory.name for directory in credentials_root.iterdir() if directory.is_dir()]
    if not characters:
        print("錯誤：找不到任何角色目錄於 configs/social_media/credentials/")
        return None

    print("可用角色：")
    for index, character in enumerate(characters, 1):
        print(f"  {index}. {character}")
    selected_choice = (choice if choice is not None else input("\n請選擇角色編號 (預設 1): ")).strip() or "1"
    try:
        character_dir = credentials_root / characters[int(selected_choice) - 1]
    except (ValueError, IndexError):
        print("無效的選擇")
        return None

    env_file = character_dir / "facebook.env"
    if not env_file.exists():
        print(f"錯誤：找不到 {env_file}")
        return None
    return env_file


def _load_facebook_credentials(env_file: Path) -> dict[str, str]:
    values = dotenv_values(env_file)
    return {
        "app_id": str(values.get("APP_ID") or "").strip().strip("'\""),
        "app_secret": str(values.get("APP_SECRET") or "").strip().strip("'\""),
        "page_id": str(values.get("FB_PAGE_ID") or "").strip(),
        "short_token": str(values.get("FB_SHORT_LIVED_TOKEN") or "").strip().strip("'\""),
    }


def _save_refreshed_token(env_file: Path, long_user_token: str) -> None:
    lines = []
    with env_file.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip().startswith("FB_PAGE_ACCESS_TOKEN="):
                lines.append(f"FB_PAGE_ACCESS_TOKEN={long_user_token}\n")
            elif line.strip().startswith("FB_SHORT_LIVED_TOKEN="):
                lines.append(f"FB_SHORT_LIVED_TOKEN={long_user_token}\n")
            else:
                lines.append(line)
    with env_file.open("w", encoding="utf-8") as destination:
        destination.writelines(lines)


def generate_fb_token():
    """產生長期 Page Access Token 並可選擇寫入 facebook.env"""
    print("=== Facebook 長期 Token 產生器 ===\n")

    credentials_root = root_dir / "configs" / "social_media" / "credentials"
    env_file = _select_credentials_file(credentials_root)
    if env_file is None:
        return

    credentials = _load_facebook_credentials(env_file)
    app_id = credentials["app_id"]
    app_secret = credentials["app_secret"]
    page_id = credentials["page_id"]
    short_token = credentials["short_token"]

    if not app_id or not app_secret:
        print("錯誤：facebook.env 缺少 APP_ID 或 APP_SECRET")
        return
    if not page_id:
        print("錯誤：facebook.env 缺少 FB_PAGE_ID")
        return

    if not short_token:
        short_token = input("請貼上短期 User Token (從 Graph API Explorer 取得): ").strip()
    if not short_token:
        print("錯誤：需要短期 User Token 才能交換")
        return

    print("\n--- 交換長期 User Token ---")
    long_user_token = _exchange_long_lived_user_token(app_id, app_secret, short_token)
    if not long_user_token:
        return
    print("成功取得長期 User Token (效期約 60 天)")

    save = input("是否寫入 facebook.env？(y/N): ").strip().lower()
    if save == "y":
        _save_refreshed_token(env_file, long_user_token)
        print(f"已更新 {env_file}")
    else:
        print("請手動將上述 Token 複製到 facebook.env 的 FB_PAGE_ACCESS_TOKEN")


if __name__ == "__main__":
    generate_fb_token()
