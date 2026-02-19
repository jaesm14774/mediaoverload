"""Instagram Graph API 發布實作（Instagram Login 專用）

參考: https://developers.facebook.com/docs/instagram-platform/content-publishing

- Host: graph.instagram.com
- Token: Instagram User access token
- 權限: instagram_business_basic, instagram_business_content_publish
- 上傳方式: image_url / video_url（API 會 cURL 媒體，需可公開存取）
"""
import os
import time
import tempfile
from dotenv import load_dotenv
import requests

from .models import MediaPost
from .base import SocialMediaPlatform


class InstagramGraphPlatform(SocialMediaPlatform):
    """Instagram Graph API 發布實作，使用官方 Meta Instagram API（非 Facebook）"""

    GRAPH_API_VERSION = "v25.0"
    GRAPH_API_BASE = "https://graph.instagram.com"
    IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
    VIDEO_EXTS = ("mp4", "avi", "mov", "webm")
    CAPTION_MAX = 2200

    def __init__(self, config_folder_path: str, prefix: str = ""):
        from utils.logger import setup_logger
        from lib.services.implementations.ffmpeg_service import FFmpegService
        from lib.services.implementations.cloudinary_upload_service import CloudinaryUploadService
        self.logger = setup_logger(__name__)
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.ig_user_id = None
        self.access_token = None
        self.media_base_url = None
        self.cloudinary = CloudinaryUploadService()
        self.ffmpeg_service = FFmpegService()
        self.temp_files = []
        self.load_config()
        self.authenticate()

    def _get_config_path(self) -> str:
        if self.prefix:
            return f"{self.config_folder_path}/{self.prefix}"
        return self.config_folder_path

    def load_config(self) -> None:
        """載入 Instagram Graph 配置"""
        config_path = self._get_config_path()
        env_file = f"{config_path}/instagram_graph.env"
        if os.path.exists(env_file):
            load_dotenv(env_file)
        for env in ("media_overload.env", ".env"):
            if os.path.exists(env):
                load_dotenv(env)
                break
        self.media_base_url = (os.getenv("IG_GRAPH_MEDIA_BASE_URL") or "").rstrip("/")

    def authenticate(self) -> None:
        """驗證並取得 IG User ID（透過 /me 端點）"""
        self.access_token = os.getenv("IG_GRAPH_ACCESS_TOKEN")
        self.ig_user_id = os.getenv("IG_USER_ID") or os.getenv("IG_GRAPH_USER_ID")

        if not self.access_token:
            raise ValueError("缺少 IG_GRAPH_ACCESS_TOKEN，請使用 Instagram User access token")

        if not self.ig_user_id:
            self.ig_user_id = self._fetch_ig_user_id_from_me()

        if not self.ig_user_id:
            raise ValueError("無法取得 IG User ID，請設定 IG_USER_ID 或 IG_GRAPH_USER_ID")

        try:
            url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}"
            resp = requests.get(
                url,
                params={"fields": "username", "access_token": self.access_token},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            self.logger.debug(f"Instagram Graph 驗證成功，帳號: {data.get('username', self.ig_user_id)}")
        except requests.RequestException as e:
            self.logger.error(f"Instagram Graph 驗證失敗: {e}")
            raise ValueError(f"Instagram Graph 驗證失敗: {e}") from e

    def _fetch_ig_user_id_from_me(self) -> str | None:
        """從 /me 端點取得 IG 商業帳號 ID"""
        try:
            url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me"
            resp = requests.get(
                url,
                params={"fields": "user_id,username,id", "access_token": self.access_token},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            uid = data.get("user_id") or data.get("id")
            if uid:
                return str(uid)
            if isinstance(data.get("data"), list) and data["data"]:
                item = data["data"][0]
                return str(item.get("user_id") or item.get("id") or "")
        except requests.RequestException as e:
            self.logger.warning(f"無法從 /me 取得 IG User ID: {e}")
        return None

    def _get_ext(self, path: str) -> str:
        return path.lower().rsplit(".", 1)[-1] if "." in path else ""

    def _get_media_url(self, local_path: str) -> str | None:
        """取得媒體的公開 URL（Cloudinary 上傳或 IG_GRAPH_MEDIA_BASE_URL）"""
        if not hasattr(self, "_url_cache"):
            self._url_cache = {}
        if local_path in self._url_cache:
            return self._url_cache[local_path]
        url = None
        if self.cloudinary.is_configured():
            url = self.cloudinary.upload(local_path)
        if not url and self.media_base_url:
            url = f"{self.media_base_url}/{os.path.basename(local_path)}"
        if url:
            self._url_cache[local_path] = url
        return url

    def upload_post(self, post: MediaPost) -> bool:
        """上傳內容到 Instagram（Graph API）"""
        self._url_cache = {}
        try:
            caption = post.caption or ""
            if post.hashtags:
                caption = f"{caption}\n{post.hashtags}" if caption else post.hashtags
            if len(caption) > self.CAPTION_MAX:
                caption = caption[: self.CAPTION_MAX - 3] + "..."
                self.logger.warning("Instagram 標題過長，已截斷")

            valid_paths = [p for p in (post.media_paths or []) if os.path.exists(p)]
            if not valid_paths:
                self.logger.warning("沒有可用的媒體檔案")
                return False

            if len(valid_paths) == 1:
                return self._upload_single(valid_paths[0], caption)
            return self._upload_carousel(valid_paths[:10], caption)
        except Exception as e:
            self.logger.error(f"Instagram Graph 上傳失敗: {e}", exc_info=True)
            self._cleanup_temp()
            return False

    def _upload_single(self, media_path: str, caption: str) -> bool:
        """上傳單一媒體（圖片或影片）"""
        ext = self._get_ext(media_path)
        is_video = ext in self.VIDEO_EXTS or ext == "gif"

        if ext == "gif":
            media_path = self._gif_to_mp4(media_path)

        if is_video:
            if self._get_media_url(media_path):
                return self._publish_video_url(media_path, caption)
            self.logger.error("影片發布需設定 Cloudinary 或 IG_GRAPH_MEDIA_BASE_URL")
            return False
        return self._publish_image_url(media_path, caption)

    def _publish_image_url(self, image_path: str, caption: str) -> bool:
        """透過 image_url 發布圖片（需公開 URL，官方僅支援 JPEG）"""
        media_url = self._get_media_url(image_path)
        if not media_url:
            self.logger.error("圖片發布需設定 Cloudinary 或 IG_GRAPH_MEDIA_BASE_URL")
            return False

        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        data = {"image_url": media_url, "caption": caption, "access_token": self.access_token}
        resp = requests.post(url, data=data, timeout=60)
        resp.raise_for_status()
        container_id = resp.json().get("id")
        if not container_id:
            self.logger.error("建立圖片容器失敗")
            return False
        return self._publish_container(container_id)

    def _publish_video_url(self, video_path: str, caption: str) -> bool:
        """透過 video_url 發布影片（需公開 URL）"""
        media_url = self._get_media_url(video_path)
        if not media_url:
            self.logger.error("影片需設定 IG_GRAPH_MEDIA_BASE_URL")
            return False
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        data = {
            "media_type": "REELS",
            "video_url": media_url,
            "caption": caption,
            "access_token": self.access_token,
        }
        resp = requests.post(url, data=data, timeout=60)
        resp.raise_for_status()
        container_id = resp.json().get("id")
        if not container_id:
            return False
        if not self._wait_container_ready(container_id):
            return False
        return self._publish_container(container_id)

    def _create_carousel_item(self, media_path: str) -> str | None:
        """建立輪播項目容器，回傳 container id"""
        ext = self._get_ext(media_path)
        is_video = ext in self.VIDEO_EXTS or ext == "gif"

        if ext == "gif":
            media_path = self._gif_to_mp4(media_path)

        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        if is_video:
            media_url = self._get_media_url(media_path)
            if media_url:
                data = {
                    "media_type": "VIDEO",
                    "is_carousel_item": "true",
                    "video_url": media_url,
                    "access_token": self.access_token,
                }
                resp = requests.post(url, data=data, timeout=60)
                resp.raise_for_status()
                container_id = resp.json().get("id")
            else:
                self.logger.error("輪播影片需設定 Cloudinary 或 IG_GRAPH_MEDIA_BASE_URL")
                return None
            if not self._wait_container_ready(container_id):
                return None
            return container_id
        else:
            media_url = self._get_media_url(media_path)
            if not media_url:
                self.logger.error("輪播圖片需設定 Cloudinary 或 IG_GRAPH_MEDIA_BASE_URL")
                return None
            data = {
                "image_url": media_url,
                "is_carousel_item": "true",
                "access_token": self.access_token,
            }
            resp = requests.post(url, data=data, timeout=60)
            resp.raise_for_status()
            return resp.json().get("id")

    def _upload_carousel(self, media_paths: list, caption: str) -> bool:
        """上傳輪播貼文"""
        children = []
        for path in media_paths:
            ext = self._get_ext(path)
            if ext == "gif":
                path = self._gif_to_mp4(path)
            cid = self._create_carousel_item(path)
            if cid:
                children.append(cid)
            time.sleep(1)
        if not children:
            self.logger.error("輪播項目建立失敗")
            self._cleanup_temp()
            return False

        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        data = {
            "media_type": "CAROUSEL",
            "caption": caption,
            "children": ",".join(children),
            "access_token": self.access_token,
        }
        resp = requests.post(url, data=data, timeout=60)
        resp.raise_for_status()
        container_id = resp.json().get("id")
        if not container_id:
            self._cleanup_temp()
            return False
        if not self._wait_container_ready(container_id):
            self._cleanup_temp()
            return False
        result = self._publish_container(container_id)
        self._cleanup_temp()
        return result

    def _wait_container_ready(self, container_id: str, max_wait: int = 120) -> bool:
        """等待容器處理完成"""
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{container_id}"
        for _ in range(max_wait):
            resp = requests.get(
                url,
                params={"fields": "status_code", "access_token": self.access_token},
                timeout=30
            )
            resp.raise_for_status()
            status = resp.json().get("status_code", "")
            if status == "FINISHED":
                return True
            if status == "ERROR":
                self.logger.error(f"容器處理失敗: {container_id}")
                return False
            time.sleep(2)
        self.logger.error("等待容器處理逾時")
        return False

    def _publish_container(self, container_id: str) -> bool:
        """發布已建立的容器"""
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media_publish"
        data = {"creation_id": container_id, "access_token": self.access_token}
        resp = requests.post(url, data=data, timeout=60)
        resp.raise_for_status()
        self.logger.info(f"Instagram Graph 貼文發布成功，ID: {resp.json().get('id')}")
        return True

    def _gif_to_mp4(self, gif_path: str) -> str:
        """GIF 轉 MP4"""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        out = self.ffmpeg_service.gif_to_mp4(gif_path, tmp.name, fps=None)
        self.temp_files.append(out)
        return out

    def _cleanup_temp(self) -> None:
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except Exception as e:
                self.logger.warning(f"無法刪除臨時檔案 {f}: {e}")
        self.temp_files = []
