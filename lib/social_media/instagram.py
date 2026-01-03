import os
import time
import tempfile
import random
from dotenv import load_dotenv

from .models import MediaPost
from .base import SocialMediaPlatform

class InstagramPlatform(SocialMediaPlatform):
    """Instagram-specific implementation"""
    
    def __init__(self, config_folder_path: str, prefix=''):
        from utils.logger import setup_logger
        from lib.services.implementations.ffmpeg_service import FFmpegService
        self.logger = setup_logger(__name__)
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.client = None
        self.ffmpeg_service = FFmpegService()
        self.temp_files = []
        self.load_config()
        self.authenticate()
    
    def _get_session_file_path(self) -> str:
        """取得 session 檔案路徑"""
        if self.prefix:
            return f"{self.config_folder_path}/{self.prefix}/instagram_session.json"
        return f"{self.config_folder_path}/instagram_session.json"
    
    def authenticate(self, force_renew: bool = False) -> None:
        """使用 instagrapi 進行 Instagram 認證
        
        Args:
            force_renew: 是否強制重新驗證（刪除舊 session）
        """
        try:
            from lib.instagram import Client
            from lib.instagram.exceptions import LoginRequired, ClientLoginRequired, BadCredentials, ProxyAddressIsBlocked
            
            username = os.getenv('IG_USERNAME')
            password = os.getenv('IG_PASSWORD')
            
            if not username or not password:
                missing = [k for k, v in {
                    'IG_USERNAME': username,
                    'IG_PASSWORD': password
                }.items() if not v]
                error_msg = f"缺少必要的 Instagram API 憑證: {', '.join(missing)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            session_file = self._get_session_file_path()
            proxy = self._get_proxy()
            
            self.client = Client(proxy=proxy)
            
            if os.path.exists(session_file) and not force_renew:
                try:
                    self.client.load_settings(session_file)
                    self.client.login(username, password)
                    self.logger.debug("Instagram 客戶端已從 session 文件載入")
                    return
                except (LoginRequired, ClientLoginRequired) as e:
                    error_msg = (
                        f"Session 已過期，需要重新登入。\n"
                        f"Session 檔案: {session_file}\n"
                        f"錯誤詳情: {str(e)}\n"
                        "請手動執行重新登入，或使用 generate_ig_session.py 重新產生 session。"
                    )
                    self.logger.error(error_msg)
                    raise LoginRequired(error_msg) from e
                except ProxyAddressIsBlocked as e:
                    error_msg = (
                        "IP 地址已被 Instagram 封鎖。解決方案：\n"
                        "1. 在 ig.env 中設定 IG_PROXY 使用代理伺服器\n"
                        "2. 更換網路環境（使用 VPN 或不同的網路）\n"
                        "3. 等待一段時間後再試（通常需要數小時到數天）\n"
                        f"錯誤詳情: {str(e)}"
                    )
                    self.logger.error(error_msg)
                    raise ProxyAddressIsBlocked(error_msg) from e
                except BadCredentials as e:
                    error_msg = (
                        "Instagram 憑證錯誤。可能原因：\n"
                        "1. 帳號被鎖定，需要通過電子郵件驗證\n"
                        "2. 密碼錯誤\n"
                        "3. 帳號需要兩步驟驗證\n"
                        f"Session 檔案: {session_file}\n"
                        f"錯誤詳情: {str(e)}"
                    )
                    self.logger.error(error_msg)
                    raise BadCredentials(error_msg) from e
                except Exception as e:
                    error_str = str(e).lower()
                    # 檢測 proxy 連接錯誤
                    if "socks" in error_str or "proxy" in error_str:
                        if "name or service not known" in error_str or "failed to establish" in error_str:
                            error_msg = (
                                "Proxy 連接失敗。可能原因：\n"
                                "1. Proxy 伺服器地址錯誤或無法訪問\n"
                                "2. Proxy 伺服器已關閉或無法連接\n"
                                "3. DNS 解析失敗（檢查 proxy 主機名稱是否正確）\n"
                                "4. 網路連線問題\n"
                                f"當前 Proxy 設定: {proxy or '未設定'}\n"
                                f"錯誤詳情: {str(e)}\n"
                                "解決方案：\n"
                                "- 檢查 ig.env 中的 IG_PROXY 設定是否為真實的 proxy 地址\n"
                                "- 確認 proxy 伺服器正在運行且可訪問\n"
                                "- 嘗試使用 HTTP proxy 而非 SOCKS5（如果可能）\n"
                                "- 暫時移除 IG_PROXY 設定以使用直連"
                            )
                            self.logger.error(error_msg)
                            raise ConnectionError(error_msg) from e
                        else:
                            error_msg = (
                                f"Proxy 連接錯誤: {str(e)}\n"
                                f"當前 Proxy 設定: {proxy or '未設定'}\n"
                                "請檢查 proxy 設定是否正確"
                            )
                            self.logger.error(error_msg)
                            raise ConnectionError(error_msg) from e
                    elif "email" in error_str or "blacklist" in error_str or "blocked" in error_str:
                        error_msg = (
                            "帳號或 IP 可能被封鎖。解決方案：\n"
                            "1. 檢查電子郵件並完成 Instagram 驗證\n"
                            "2. 在 ig.env 中設定 IG_PROXY 使用代理伺服器\n"
                            "3. 更換網路環境\n"
                            f"Session 檔案: {session_file}\n"
                            f"錯誤詳情: {str(e)}"
                        )
                        self.logger.error(error_msg)
                        raise ValueError(error_msg) from e
                    else:
                        error_msg = (
                            f"使用 session 登入失敗。\n"
                            f"Session 檔案: {session_file}\n"
                            f"錯誤詳情: {str(e)}\n"
                            "請檢查錯誤訊息並手動處理，或使用 generate_ig_session.py 重新產生 session。"
                        )
                        self.logger.error(error_msg, exc_info=True)
                        raise Exception(error_msg) from e
            else:
                # 首次登入或強制重新驗證
                self._renew_session(username, password, session_file, force_renew=force_renew)
                
        except ImportError:
            error_msg = "請確保 lib.instagram 模組存在"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            self.logger.error(f"Instagram 認證錯誤: {str(e)}", exc_info=True)
            raise
    
    def _renew_session(self, username: str, password: str, session_file: str, force_renew: bool = False) -> None:
        """重新建立 session（僅在首次登入或強制重新驗證時使用）
        
        Args:
            username: Instagram 使用者名稱
            password: Instagram 密碼
            session_file: Session 檔案路徑
            force_renew: 是否為強制重新驗證（登入成功後才刪除舊檔案）
        """
        import shutil
        backup_file = None
        
        try:
            if force_renew:
                self.logger.info(f"強制重新驗證：正在重新登入 Instagram，使用者名稱: {username}")
            else:
                self.logger.info(f"正在重新登入 Instagram，使用者名稱: {username}")
            # Instagram API 的 username 欄位可以接受使用者名稱、email 或電話號碼
            # 檢查是否為 email 格式
            is_email = '@' in username
            login_identifier = "email" if is_email else "使用者名稱"
            self.logger.debug(f"使用 {login_identifier} 登入: {username}")
            
            # 如果舊 session 檔案存在，先備份
            if os.path.exists(session_file):
                backup_file = f"{session_file}.backup"
                try:
                    shutil.copy2(session_file, backup_file)
                    self.logger.debug(f"已備份舊 session 檔案至: {backup_file}")
                except Exception as backup_error:
                    self.logger.warning(f"無法備份舊 session 檔案: {backup_error}")
            
            # 嘗試登入
            self.client.login(username, password, relogin=True)
            
            # 只有在登入成功後才保存新的 session
            os.makedirs(os.path.dirname(session_file), exist_ok=True)
            self.client.dump_settings(session_file)
            self.logger.info("Instagram 客戶端已重新登入並保存 session")
            
            # 登入成功後刪除備份檔案
            if backup_file and os.path.exists(backup_file):
                try:
                    os.remove(backup_file)
                    self.logger.debug(f"已刪除備份檔案: {backup_file}")
                except Exception as cleanup_error:
                    self.logger.warning(f"無法刪除備份檔案: {cleanup_error}")
                    
        except Exception as e:
            error_str = str(e).lower()
            
            # 如果登入失敗且有備份檔案，恢復舊的 session 檔案
            if backup_file and os.path.exists(backup_file):
                try:
                    if os.path.exists(session_file):
                        os.remove(session_file)
                    shutil.copy2(backup_file, session_file)
                    self.logger.info(f"登入失敗，已恢復舊 session 檔案: {session_file}")
                    os.remove(backup_file)
                except Exception as restore_error:
                    self.logger.warning(f"無法恢復舊 session 檔案: {restore_error}")
            
            # 檢測找不到帳號的錯誤
            if "can't find an account" in error_str or "找不到帳號" in error_str:
                error_msg = (
                    f"重新登入失敗：找不到帳號 '{username}'\n"
                    "可能原因：\n"
                    "1. IG_USERNAME 設定錯誤（請確認是否為正確的使用者名稱、email 或電話號碼）\n"
                    "2. 如果使用使用者名稱登入失敗，請嘗試使用 email 登入\n"
                    "3. 帳號可能需要驗證（檢查電子郵件或簡訊）\n"
                    "4. 帳號可能已被停用或刪除\n"
                    f"當前登入識別碼: {username}\n"
                    f"Session 檔案: {session_file}\n"
                    f"建議：如果這是使用者名稱，請在 ig.env 中改用 email 或電話號碼"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg) from e
            # 檢測 proxy 連接錯誤
            if "socks" in error_str or ("proxy" in error_str and "connection" in error_str):
                if "name or service not known" in error_str or "failed to establish" in error_str:
                    proxy = self._get_proxy()
                    error_msg = (
                        "重新登入失敗：Proxy 連接錯誤\n"
                        "可能原因：\n"
                        "1. Proxy 伺服器地址錯誤或無法訪問\n"
                        "2. Proxy 伺服器已關閉或無法連接\n"
                        "3. DNS 解析失敗（檢查 proxy 主機名稱是否正確）\n"
                        "4. 網路連線問題\n"
                        f"當前 Proxy 設定: {proxy or '未設定'}\n"
                        f"Session 檔案: {session_file}\n"
                        f"錯誤詳情: {str(e)}\n"
                        "解決方案：\n"
                        "- 檢查 ig.env 中的 IG_PROXY 設定是否為真實的 proxy 地址（不是範例值）\n"
                        "- 確認 proxy 伺服器正在運行且可訪問\n"
                        "- 嘗試使用 HTTP proxy 而非 SOCKS5（如果可能）\n"
                        "- 暫時移除或註解掉 IG_PROXY 設定以使用直連"
                    )
                    self.logger.error(error_msg)
                    raise ConnectionError(error_msg) from e
            error_msg = (
                f"重新登入失敗。\n"
                f"Session 檔案: {session_file}\n"
                f"錯誤詳情: {str(e)}\n"
                "請檢查錯誤訊息並手動處理，或使用 generate_ig_session.py 重新產生 session。"
            )
            self.logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e
    
    def reauthenticate(self) -> None:
        """強制重新驗證（刪除舊 session 並重新登入）"""
        self.logger.info("執行強制重新驗證...")
        self.authenticate(force_renew=True)
    
    def upload_post(self, post: MediaPost) -> bool:
        """上傳內容到 Instagram"""
        try:
            if not self.client:
                error_msg = "Instagram 客戶端未初始化，請先進行認證"
                self.logger.error(error_msg)
                return False
            
            return self._upload_post_with_retry(post)
        except Exception as e:
            error_msg = f"Instagram 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._cleanup_temp_files()
            return False
    
    def _upload_post_with_retry(self, post: MediaPost, retry_count: int = 1) -> bool:
        """上傳貼文（含 session 過期自動重試）"""
        from lib.instagram.exceptions import LoginRequired, ClientLoginRequired, ProxyAddressIsBlocked
        
        try:
            # 組合標題內容
            caption = post.caption or ''
            if post.hashtags:
                caption = f"{caption}\n{post.hashtags}" if caption else post.hashtags
            
            # Instagram 標題長度限制為 2200 字元
            if len(caption) > 2200:
                caption = caption[:2197] + "..."
                self.logger.warning(f"Instagram 標題過長，已截斷為 2200 字元")
            
            self.logger.info(f"準備發布 Instagram 貼文，內容長度: {len(caption)} 字元，媒體數量: {len(post.media_paths) if post.media_paths else 0}")
            
            # 處理媒體上傳
            if not post.media_paths:
                self.logger.warning("沒有媒體文件可供上傳")
                return False
            
            # 檢查媒體文件是否存在
            valid_media_paths = [path for path in post.media_paths if os.path.exists(path)]
            if not valid_media_paths:
                self.logger.error("所有媒體文件都不存在")
                return False
            
            # 判斷媒體類型並上傳
            if len(valid_media_paths) == 1:
                # 單一媒體
                media_path = valid_media_paths[0]
                if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
                    # 影片 - 如果是 GIF，先轉換為 MP4
                    upload_path = media_path
                    if media_path.lower().endswith('.gif'):
                        self.logger.info(f"檢測到 GIF 檔案，轉換為 MP4 以符合 Instagram 格式要求")
                        temp_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                        temp_mp4.close()
                        upload_path = self.ffmpeg_service.gif_to_mp4(
                            gif_path=media_path,
                            output_path=temp_mp4.name,
                            fps=None  # 自動從 GIF 讀取 fps
                        )
                        self.temp_files.append(upload_path)
                        self.logger.info(f"GIF 已轉換為 MP4: {upload_path}")
                    
                    self.logger.info(f"正在上傳影片: {upload_path}")
                    media = self.client.clip_upload(upload_path, caption)
                else:
                    # 圖片
                    self.logger.info(f"正在上傳圖片: {media_path}")
                    media = self.client.photo_upload(media_path, caption)
            else:
                # 多媒體（相簿）
                # Instagram 最多支援 10 個媒體
                media_paths = valid_media_paths[:10]
                if len(valid_media_paths) > 10:
                    self.logger.warning(f"媒體數量超過限制 (10)，將只使用前 10 個")
                
                # album_upload 支援的格式：圖片 (.jpg, .jpeg, .webp) 和影片 (.mp4)
                # 將 GIF 轉換為 MP4 後加入相簿
                converted_media = []
                for media_path in media_paths:
                    if media_path.lower().endswith('.gif'):
                        # GIF 轉換為 MP4
                        self.logger.info(f"檢測到 GIF 檔案，轉換為 MP4 以符合 Instagram 格式要求")
                        temp_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                        temp_mp4.close()
                        converted_path = self.ffmpeg_service.gif_to_mp4(
                            gif_path=media_path,
                            output_path=temp_mp4.name,
                            fps=None  # 自動從 GIF 讀取 fps
                        )
                        self.temp_files.append(converted_path)
                        converted_media.append(converted_path)
                        self.logger.info(f"GIF 已轉換為 MP4: {converted_path}")
                    elif media_path.lower().endswith(('.jpg', '.jpeg', '.webp', '.mp4')):
                        # 直接支援的格式
                        converted_media.append(media_path)
                
                if converted_media:
                    image_count = len([p for p in converted_media if p.lower().endswith(('.jpg', '.jpeg', '.webp'))])
                    video_count = len([p for p in converted_media if p.lower().endswith('.mp4')])
                    self.logger.info(f"正在上傳相簿，包含 {image_count} 張圖片和 {video_count} 個影片")
                    media = self.client.album_upload(converted_media, caption)
                else:
                    # 如果沒有支援的媒體，只上傳第一個媒體
                    self.logger.warning("沒有相簿支援的格式，將只上傳第一個媒體")
                    media_path = media_paths[0]
                    if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
                        # 如果是 GIF，先轉換為 MP4
                        upload_path = media_path
                        if media_path.lower().endswith('.gif'):
                            self.logger.info(f"檢測到 GIF 檔案，轉換為 MP4 以符合 Instagram 格式要求")
                            temp_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                            temp_mp4.close()
                            upload_path = self.ffmpeg_service.gif_to_mp4(
                                gif_path=media_path,
                                output_path=temp_mp4.name,
                                fps=None,  # 自動從 GIF 讀取 fps
                                loop=0
                            )
                            self.temp_files.append(upload_path)
                            self.logger.info(f"GIF 已轉換為 MP4: {upload_path}")
                        media = self.client.clip_upload(upload_path, caption)
                    else:
                        media = self.client.photo_upload(media_path, caption)
            
            if media:
                media_id = media.pk if hasattr(media, 'pk') else media.id if hasattr(media, 'id') else None
                self.logger.info(f"Instagram 貼文發布成功，Media ID: {media_id}")
                
                # 檢查是否需要分享到 Story
                if post.additional_params and post.additional_params.get('share_to_story', False):
                    self.logger.info("開始分享到 Story")
                    # 只選擇圖片（不選擇影片）進行隨機分享
                    image_paths = [path for path in valid_media_paths 
                                  if not path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm'))]
                    
                    if image_paths:
                        # 隨機選擇一張圖片
                        random_image = random.choice(image_paths)
                        self.logger.info(f"隨機選擇圖片分享到 Story: {random_image}")
                        
                        story_post = MediaPost(
                            media_paths=[random_image],
                            caption=post.caption or '',
                            hashtags=post.hashtags
                        )
                        story_result = self.share_story(story_post)
                        if story_result:
                            self.logger.info("Story 分享成功")
                        else:
                            self.logger.warning("Story 分享失敗")
                    else:
                        self.logger.warning("沒有可用的圖片分享到 Story")
                
                # 清理臨時檔案
                self._cleanup_temp_files()
                return True
            else:
                self.logger.error("Instagram 貼文發布失敗：未返回媒體數據")
                # 清理臨時檔案
                self._cleanup_temp_files()
                return False
                
        except (LoginRequired, ClientLoginRequired) as e:
            if retry_count > 0:
                self.logger.warning(f"Session 過期，嘗試重新驗證後重試 (剩餘重試次數: {retry_count})")
                try:
                    self.reauthenticate()
                    return self._upload_post_with_retry(post, retry_count - 1)
                except Exception as reauth_error:
                    self.logger.error(f"重新驗證失敗: {str(reauth_error)}")
                    self._cleanup_temp_files()
                    return False
            else:
                self.logger.error("Session 過期且重試次數已用完")
                self._cleanup_temp_files()
                return False
        except ProxyAddressIsBlocked as e:
            error_msg = (
                "上傳失敗：IP 地址已被 Instagram 封鎖。\n"
                "請在 ig.env 中設定 IG_PROXY 使用代理伺服器，或更換網路環境。"
            )
            self.logger.error(error_msg)
            self._cleanup_temp_files()
            return False
        except Exception as e:
            error_msg = f"Instagram 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # 清理臨時檔案
            self._cleanup_temp_files()
            return False

    def share_story(self, post: MediaPost) -> bool:
        """分享 Story 到 Instagram（只上傳第一張媒體）"""
        try:
            if not self.client:
                error_msg = "Instagram 客戶端未初始化，請先進行認證"
                self.logger.error(error_msg)
                return False

            if not post.media_paths:
                self.logger.warning("沒有媒體文件可供上傳 Story")
                return False

            valid_media_paths = [path for path in post.media_paths if os.path.exists(path)]
            if not valid_media_paths:
                self.logger.error("所有媒體文件都不存在")
                return False

            return self._share_story_with_retry(post, valid_media_paths[0], retry_count=1)

        except Exception as e:
            error_msg = f"Instagram Story 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._cleanup_temp_files()
            return False
    
    def _share_story_with_retry(self, post: MediaPost, media_path: str, retry_count: int = 1) -> bool:
        """分享 Story（含 session 過期自動重試）"""
        from lib.instagram.exceptions import LoginRequired, ClientLoginRequired, ProxyAddressIsBlocked
        
        try:
            if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
                upload_path = media_path
                if media_path.lower().endswith('.gif'):
                    self.logger.info(f"檢測到 GIF 檔案，轉換為 MP4 以符合 Instagram 格式要求")
                    temp_mp4 = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                    temp_mp4.close()
                    upload_path = self.ffmpeg_service.gif_to_mp4(
                        gif_path=media_path,
                        output_path=temp_mp4.name,
                        fps=None
                    )
                    self.temp_files.append(upload_path)
                    self.logger.info(f"GIF 已轉換為 MP4: {upload_path}")
                self.logger.info(f"正在上傳影片 Story: {upload_path}")
                self.client.video_upload_to_story(upload_path, caption=post.caption)
            else:
                self.logger.info(f"正在上傳圖片 Story: {media_path}")
                self.client.photo_upload_to_story(media_path, caption=post.caption)
            
            self._cleanup_temp_files()
            return True
        except (LoginRequired, ClientLoginRequired) as e:
            if retry_count > 0:
                self.logger.warning(f"Session 過期，嘗試重新驗證後重試 (剩餘重試次數: {retry_count})")
                try:
                    self.reauthenticate()
                    return self._share_story_with_retry(post, media_path, retry_count - 1)
                except Exception as reauth_error:
                    self.logger.error(f"重新驗證失敗: {str(reauth_error)}")
                    self._cleanup_temp_files()
                    return False
            else:
                self.logger.error("Session 過期且重試次數已用完")
                self._cleanup_temp_files()
                return False
        except ProxyAddressIsBlocked as e:
            error_msg = (
                "上傳 Story 失敗：IP 地址已被 Instagram 封鎖。\n"
                "請在 ig.env 中設定 IG_PROXY 使用代理伺服器，或更換網路環境。"
            )
            self.logger.error(error_msg)
            self._cleanup_temp_files()
            return False
        except Exception as e:
            self.logger.error(f"上傳 Story 失敗 ({media_path}): {str(e)}")
            self._cleanup_temp_files()
            return False

    def _cleanup_temp_files(self):
        """清理臨時轉換的 MP4 檔案"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.logger.debug(f"已刪除臨時檔案: {temp_file}")
            except Exception as e:
                self.logger.warning(f"無法刪除臨時檔案 {temp_file}: {e}")
        self.temp_files = []
    
    def load_config(self) -> None:
        """載入 Instagram 配置"""
        self.config_path = f"{self.config_folder_path}/{self.prefix}" if self.prefix else f"{self.config_folder_path}"
        # Instagram 配置文件是 ig.env（不是 instagram.env）
        env_file = f'{self.config_path}/ig.env'
        if os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            self.logger.warning(f"Instagram 配置文件不存在: {env_file}")
    
    def _get_proxy(self) -> str | None:
        """從環境變數取得 proxy 設定並驗證"""
        proxy = os.getenv('IG_PROXY')
        if proxy:
            proxy = proxy.strip()
            # 檢查是否為範例值
            if 'example.com' in proxy.lower() or proxy == '':
                self.logger.warning(
                    "檢測到範例 proxy 設定，已忽略。請在 ig.env 中設定真實的 proxy 地址。"
                )
                return None
            
            # 驗證 proxy URL 格式
            from urllib.parse import urlparse
            parsed = urlparse(proxy)
            
            if not parsed.scheme:
                self.logger.warning(f"Proxy URL 缺少協議（scheme），已忽略: {proxy}")
                return None
            
            if parsed.scheme not in ['http', 'https', 'socks5', 'socks5h']:
                self.logger.warning(
                    f"不支援的 proxy 協議: {parsed.scheme}。"
                    f"支援的協議: http, https, socks5, socks5h"
                )
                return None
            
            if not parsed.hostname:
                self.logger.warning(f"Proxy URL 缺少主機名稱，已忽略: {proxy}")
                return None
            
            self.logger.info(f"使用 Proxy: {parsed.scheme}://{parsed.hostname}:{parsed.port or 'default'}")
            return proxy
        
        return None
