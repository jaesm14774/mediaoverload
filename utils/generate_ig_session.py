import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# 將專案根目錄加入路徑，以便導入 lib
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from lib.instagram import Client
from lib.instagram.exceptions import ChallengeRequired, FeedbackRequired, LoginRequired, TwoFactorRequired

def generate_session():
    print("=== Instagram Session 產生器 ===")
    
    # 1. 選擇角色
    credentials_root = root_dir / "configs" / "social_media" / "credentials"
    characters = [d.name for d in credentials_root.iterdir() if d.is_dir()]
    
    if not characters:
        print("錯誤：找不到任何角色目錄於 configs/social_media/credentials/")
        return

    print("\n可用角色：")
    for i, char in enumerate(characters, 1):
        print(f"{i}. {char}")
    print(f"{len(characters) + 1}. 新增角色")
    
    choice = input("\n請選擇角色編號 (預設 1): ").strip() or "1"
    
    if choice == str(len(characters) + 1):
        char_name = input("輸入新角色名稱: ").strip()
        char_dir = credentials_root / char_name
        char_dir.mkdir(parents=True, exist_ok=True)
    else:
        try:
            char_name = characters[int(choice) - 1]
            char_dir = credentials_root / char_name
        except (ValueError, IndexError):
            print("無效的選擇")
            return

    # 2. 載入或詢問憑證
    env_file = char_dir / "ig.env"
    username = ""
    password = ""
    proxy = ""

    if env_file.exists():
        # 清除環境變數中的快取，避免重複執行時抓到舊的
        if "IG_USERNAME" in os.environ: del os.environ["IG_USERNAME"]
        if "IG_PASSWORD" in os.environ: del os.environ["IG_PASSWORD"]
        if "IG_PROXY" in os.environ: del os.environ["IG_PROXY"]
        
        load_dotenv(env_file)
        username = os.getenv("IG_USERNAME", "")
        password = os.getenv("IG_PASSWORD", "")
        proxy = os.getenv("IG_PROXY", "")
        print(f"\n已從 {env_file} 載入設定")
    else:
        print(f"\n找不到 {env_file}，請手動輸入：")

    if not username:
        username = input("Instagram 使用者名稱: ").strip()
    else:
        u_input = input(f"Instagram 使用者名稱 [{username}]: ").strip()
        if u_input: username = u_input

    if not password:
        password = input("Instagram 密碼: ").strip()
    else:
        p_input = input(f"Instagram 密碼 [已存在]: ").strip()
        if p_input: password = p_input

    if not proxy:
        proxy = input("Proxy (選填, 例如 http://user:pass@host:port): ").strip()
    else:
        pr_input = input(f"Proxy [{proxy}]: ").strip()
        if pr_input: proxy = pr_input

    # 儲存到 .env (如果資訊有更新)
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(f"IG_USERNAME={username}\n")
        f.write(f"IG_PASSWORD={password}\n")
        if proxy:
            f.write(f"IG_PROXY={proxy}\n")
        f.write("IG_ACCOUNT_COOKIE_FILE_PATH=ig_account.json\n")

    # 3. 登入並產生 Session
    print(f"\n正在嘗試為 {username} 登入...")
    cl = Client()
    if proxy:
        cl.set_proxy(proxy)
    
    session_file = char_dir / "instagram_session.json"
    
    try:
        # 嘗試登入
        cl.login(username, password)
        print("登入成功！")
        
        # 儲存 Session
        cl.dump_settings(str(session_file))
        print(f"Session 已儲存至: {session_file}")
        
    except TwoFactorRequired as e:
        print(f"\n需要兩步驟驗證 (2FA): {e}")
        code = input("請輸入驗證碼: ").strip()
        if code:
            try:
                cl.login(username, password, verification_code=code)
                print("2FA 登入成功！")
                cl.dump_settings(str(session_file))
                print(f"Session 已儲存至: {session_file}")
            except Exception as e2:
                print(f"2FA 登入失敗: {e2}")
    except ChallengeRequired as e:
        print(f"\n需要驗證 (Challenge Required): {e}")
        print("請嘗試在手機 App 上登入並確認是您本人，或者檢查郵箱完成驗證。")
        print("instagrapi 可能需要更複雜的處理來解決 challenge，建議先在手機上排除。")
    except Exception as e:
        print(f"\n登入失敗: {e}")

if __name__ == "__main__":
    generate_session()
