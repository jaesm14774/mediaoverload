import os
import time
import tempfile
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
                    # 影片 - 如果是 GIF，先轉換為 MP4
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
                            fps=None,  # 自動從 GIF 讀取 fps
                            loop=0
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
                # 清理臨時檔案
                self._cleanup_temp_files()
                return True
            else:
                self.logger.error("Instagram 貼文發布失敗：未返回媒體數據")
                # 清理臨時檔案
                self._cleanup_temp_files()
                return False
                
        except Exception as e:
            error_msg = f"Instagram 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # 清理臨時檔案
            self._cleanup_temp_files()
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
                        self.logger.info(f"正在上傳影片 Story: {upload_path}")
                        self.client.video_upload_to_story(upload_path, caption=post.caption)
                    else:
                        self.logger.info(f"正在上傳圖片 Story: {media_path}")
                        self.client.photo_upload_to_story(media_path, caption=post.caption)
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"上傳 Story 失敗 ({media_path}): {str(e)}")
            
            # 清理臨時檔案
            self._cleanup_temp_files()
            return success_count > 0

        except Exception as e:
            error_msg = f"Instagram Story 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # 清理臨時檔案
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
