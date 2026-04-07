from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def generate_session() -> None:
    print("=== Instagram Graph 設定工具 ===")
    print("舊 private session 流程已淘汰，請改用 instagram_graph.env。")

    credentials_root = ROOT_DIR / "configs" / "social_media" / "credentials"
    credentials_root.mkdir(parents=True, exist_ok=True)
    characters = [d.name for d in credentials_root.iterdir() if d.is_dir()]

    print("\n可用角色：")
    for index, char in enumerate(characters, 1):
        print(f"{index}. {char}")
    print(f"{len(characters) + 1}. 新增角色")

    choice = input("\n請選擇角色編號 (預設 1): ").strip() or "1"
    if choice == str(len(characters) + 1):
        char_name = input("輸入新角色名稱: ").strip()
        if not char_name:
            print("角色名稱不可為空")
            return
        char_dir = credentials_root / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            char_name = characters[int(choice) - 1]
            char_dir = credentials_root / char_name
        except (ValueError, IndexError):
            print("無效的選擇")
            return

    env_file = char_dir / "instagram_graph.env"
    existing = _read_env_file(env_file)
    access_token = _prompt("IG_GRAPH_ACCESS_TOKEN", existing.get("IG_GRAPH_ACCESS_TOKEN", ""))
    user_id = _prompt("IG_USER_ID", existing.get("IG_USER_ID", ""))
    media_base_url = _prompt("IG_GRAPH_MEDIA_BASE_URL", existing.get("IG_GRAPH_MEDIA_BASE_URL", ""))

    if not access_token or not user_id:
        print("IG_GRAPH_ACCESS_TOKEN 與 IG_USER_ID 為必填")
        return

    lines = [
        f"IG_GRAPH_ACCESS_TOKEN={access_token}",
        f"IG_USER_ID={user_id}",
    ]
    if media_base_url:
        lines.append(f"IG_GRAPH_MEDIA_BASE_URL={media_base_url}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n已寫入: {env_file}")
    print("後續可直接供 agentic-native Instagram Graph 發佈使用。")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _prompt(key: str, current: str) -> str:
    if current:
        value = input(f"{key} [{current}]: ").strip()
        return value or current
    return input(f"{key}: ").strip()

if __name__ == "__main__":
    generate_session()
