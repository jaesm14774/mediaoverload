import os
import time
from dotenv import load_dotenv

from .models import MediaPost
from .base import SocialMediaPlatform

class InstagramPlatform(SocialMediaPlatform):
    """Instagram-specific implementation"""
    
    def __init__(self, config_folder_path: str, prefix=''):
        from utils.logger import setup_logger
        self.logger = setup_logger(__name__)
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.client = None
        self.load_config()
        self.authenticate()
    
    def authenticate(self) -> None:
        """使用 instagrapi 進行 Instagram 認證"""
        try:
            # 檢查必要的環境變數（使用 IG_ 前綴，與配置文件一致）
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
            
            # 嘗試導入 lib.instagram
            try:
                from lib.instagram import Client
            except ImportError:
                error_msg = "請確保 lib.instagram 模組存在"
                self.logger.error(error_msg)
                raise ImportError(error_msg)
            
            # 創建客戶端並登入
            self.client = Client()
            
            # 嘗試從 session 文件載入（如果存在）
            session_file = f"{self.config_folder_path}/{self.prefix}/instagram_session.json" if self.prefix else f"{self.config_folder_path}/instagram_session.json"
            
            try:
                if os.path.exists(session_file):
                    self.client.load_settings(session_file)
                    self.client.login(username, password)
                    self.logger.debug("Instagram 客戶端已從 session 文件載入")
                else:
                    # 首次登入
                    self.client.login(username, password)
                    # 保存 session
                    os.makedirs(os.path.dirname(session_file), exist_ok=True)
                    self.client.dump_settings(session_file)
                    self.logger.debug("Instagram 客戶端已登入並保存 session")
            except Exception as e:
                self.logger.warning(f"使用 session 登入失敗，嘗試重新登入: {str(e)}")
                # 如果 session 登入失敗，嘗試重新登入
                self.client.login(username, password)
                # 保存新的 session
                os.makedirs(os.path.dirname(session_file), exist_ok=True)
                self.client.dump_settings(session_file)
                self.logger.debug("Instagram 客戶端已重新登入並保存 session")
                
        except Exception as e:
            self.logger.error(f"Instagram 認證錯誤: {str(e)}", exc_info=True)
            raise
    
    def upload_post(self, post: MediaPost) -> bool:
        """上傳內容到 Instagram"""
        try:
            if not self.client:
                error_msg = "Instagram 客戶端未初始化，請先進行認證"
                self.logger.error(error_msg)
                return False
            
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
                    # 影片
                    self.logger.info(f"正在上傳影片: {media_path}")
                    media = self.client.clip_upload(media_path, caption)
                else:
                    # 圖片
                    self.logger.info(f"正在上傳圖片: {media_path}")
                    media = self.client.photo_upload(media_path, caption)
            else:
                # 多媒體（相簿）
                # Instagram 最多支援 10 張圖片
                media_paths = valid_media_paths[:10]
                if len(valid_media_paths) > 10:
                    self.logger.warning(f"媒體數量超過限制 (10)，將只使用前 10 個")
                
                # 檢查是否都是圖片（相簿只能包含圖片）
                image_paths = [p for p in media_paths if not p.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm'))]
                
                if len(image_paths) == len(media_paths):
                    # 所有都是圖片，可以創建相簿
                    self.logger.info(f"正在上傳相簿，包含 {len(image_paths)} 張圖片")
                    media = self.client.album_upload(image_paths, caption)
                else:
                    # 混合媒體，只上傳第一張
                    self.logger.warning("相簿只能包含圖片，將只上傳第一張媒體")
                    media_path = media_paths[0]
                    if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
                        media = self.client.clip_upload(media_path, caption)
                    else:
                        media = self.client.photo_upload(media_path, caption)
            
            if media:
                media_id = media.pk if hasattr(media, 'pk') else media.id if hasattr(media, 'id') else None
                self.logger.info(f"Instagram 貼文發布成功，Media ID: {media_id}")
                return True
            else:
                self.logger.error("Instagram 貼文發布失敗：未返回媒體數據")
                return False
                
        except Exception as e:
            error_msg = f"Instagram 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False

    def share_story(self, post: MediaPost) -> bool:
        """分享 Story 到 Instagram"""
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

            success_count = 0
            for media_path in valid_media_paths:
                try:
                    if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
                        self.logger.info(f"正在上傳影片 Story: {media_path}")
                        self.client.video_upload_to_story(media_path, caption=post.caption)
                    else:
                        self.logger.info(f"正在上傳圖片 Story: {media_path}")
                        self.client.photo_upload_to_story(media_path, caption=post.caption)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"上傳 Story 失敗 ({media_path}): {str(e)}")
            
            return success_count > 0

        except Exception as e:
            error_msg = f"Instagram Story 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False
    
    def load_config(self) -> None:
        """載入 Instagram 配置"""
        self.config_path = f"{self.config_folder_path}/{self.prefix}" if self.prefix else f"{self.config_folder_path}"
        # Instagram 配置文件是 ig.env（不是 instagram.env）
        env_file = f'{self.config_path}/ig.env'
        if os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            self.logger.warning(f"Instagram 配置文件不存在: {env_file}")
