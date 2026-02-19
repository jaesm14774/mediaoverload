"""Facebook 粉絲專頁發布實作"""
import json
import os
import time
import tempfile
from dotenv import load_dotenv
import requests

from .models import MediaPost
from .base import SocialMediaPlatform


class FacebookPlatform(SocialMediaPlatform):
    """Facebook 粉絲專頁發布實作，使用 Graph API"""

    GRAPH_API_VERSION = "v24.0"
    GRAPH_API_BASE = "https://graph.facebook.com"
    IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
    VIDEO_EXTS = ("mp4", "avi", "mov", "webm")

    def __init__(self, config_folder_path: str, prefix: str = ""):
        from utils.logger import setup_logger
        from lib.services.implementations.ffmpeg_service import FFmpegService
        self.logger = setup_logger(__name__)
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.page_id = None
        self.page_access_token = None
        self.ffmpeg_service = FFmpegService()
        self.temp_files = []
        self.load_config()
        self.authenticate()

    def _get_config_path(self) -> str:
        if self.prefix:
            return f"{self.config_folder_path}/{self.prefix}"
        return self.config_folder_path

    def load_config(self) -> None:
        """載入 Facebook 配置"""
        config_path = self._get_config_path()
        env_file = f"{config_path}/facebook.env"
        if os.path.exists(env_file):
            load_dotenv(env_file)
        else:
            self.logger.warning(f"Facebook 配置文件不存在: {env_file}")

    def authenticate(self) -> None:
        """驗證 Page Access Token"""
        self.page_id = os.getenv("FB_PAGE_ID")
        self.page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")

        if not self.page_id or not self.page_access_token:
            missing = [k for k, v in {
                "FB_PAGE_ID": self.page_id,
                "FB_PAGE_ACCESS_TOKEN": self.page_access_token,
            }.items() if not v]
            raise ValueError(f"缺少 Facebook 憑證: {', '.join(missing)}")

        try:
            url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me"
            resp = requests.get(url, params={"access_token": self.page_access_token}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            me_id = data.get("id")
            if me_id:
                self.page_id = str(me_id)
            self.logger.debug(f"Facebook 驗證成功，Page: {data.get('name', self.page_id)}")
        except requests.RequestException as e:
            self.logger.error(f"Facebook 驗證失敗: {e}")
            raise ValueError(f"Facebook 驗證失敗: {e}") from e

    def _get_ext(self, path: str) -> str:
        return path.lower().rsplit(".", 1)[-1] if "." in path else ""

    def upload_post(self, post: MediaPost) -> bool:
        """上傳內容到 Facebook 粉絲專頁"""
        try:
            caption = post.caption or ""
            if post.hashtags:
                caption = f"{caption}\n{post.hashtags}" if caption else post.hashtags

            if len(caption) > 63206:
                caption = caption[:63203] + "..."
                self.logger.warning("Facebook 貼文過長，已截斷")

            valid_paths = [p for p in (post.media_paths or []) if os.path.exists(p)]
            if not valid_paths:
                self.logger.warning("沒有可用的媒體檔案")
                return False

            if len(valid_paths) == 1:
                return self._upload_single(valid_paths[0], caption)
            return self._upload_multiple(valid_paths, caption)
        except Exception as e:
            self.logger.error(f"Facebook 上傳失敗: {e}", exc_info=True)
            self._cleanup_temp()
            return False

    def _upload_single(self, media_path: str, caption: str) -> bool:
        """上傳單一媒體（圖片或影片）"""
        ext = self._get_ext(media_path)
        is_video = ext in self.VIDEO_EXTS or ext == "gif"

        if ext == "gif":
            media_path = self._gif_to_mp4(media_path)

        node = "me"
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{node}"
        endpoint = "videos" if is_video else "photos"
        text_key = "description" if is_video else "message"

        with open(media_path, "rb") as f:
            files = {"source": (os.path.basename(media_path), f, self._get_content_type(media_path))}
            data = {text_key: caption, "access_token": self.page_access_token}

            resp = requests.post(f"{url}/{endpoint}", files=files, data=data, timeout=300)
            resp.raise_for_status()
            result = resp.json()

        post_id = result.get("id") or result.get("post_id")
        self.logger.info(f"Facebook 貼文發布成功，ID: {post_id}")
        self._cleanup_temp()
        return True

    def _upload_multiple(self, media_paths: list, caption: str) -> bool:
        """上傳多個媒體：圖片/影片皆走 unpublished → attached_media 一次發布"""
        image_paths = []
        video_paths = []

        for p in media_paths[:10]:
            ext = self._get_ext(p)
            if ext in self.IMAGE_EXTS:
                image_paths.append(p)
            elif ext == "gif":
                video_paths.append(self._gif_to_mp4(p))
            elif ext in self.VIDEO_EXTS:
                video_paths.append(p)
            else:
                self.logger.warning(f"不支援的檔案格式，跳過: {p}")

        success = False
        caption_used = False

        if len(image_paths) >= 2:
            if self._post_images_album(image_paths, caption):
                success = True
            caption_used = True
        elif len(image_paths) == 1 and not video_paths:
            return self._upload_single(image_paths[0], caption)

        if len(video_paths) >= 2:
            if self._post_videos_album(video_paths, caption):
                success = True
            caption_used = True
        elif len(video_paths) == 1:
            vc = caption if not caption_used else ""
            if self._upload_single(video_paths[0], vc):
                success = True
            caption_used = True

        if len(image_paths) == 1 and video_paths:
            ic = caption if not caption_used else ""
            if self._upload_single(image_paths[0], ic):
                success = True

        self._cleanup_temp()
        return success

    def _post_videos_album(self, video_paths: list, caption: str) -> bool:
        """透過 unpublished videos + feed attached_media 發布多影片貼文"""
        media_fbids = []
        upload_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/videos"

        for path in video_paths:
            try:
                with open(path, "rb") as f:
                    files = {"source": (os.path.basename(path), f, self._get_content_type(path))}
                    data = {"published": "false", "access_token": self.page_access_token}
                    resp = requests.post(upload_url, files=files, data=data, timeout=300)
                    resp.raise_for_status()
                    vid = resp.json().get("id") or resp.json().get("video_id")
                    if vid:
                        media_fbids.append(vid)
                        self.logger.debug(f"上傳未發布影片成功，ID: {vid}")
                    else:
                        self.logger.warning(f"上傳影片未返回 ID: {path}")
            except requests.RequestException as e:
                self.logger.error(f"上傳影片失敗: {path}, {e}")
            time.sleep(2)

        if not media_fbids:
            self.logger.error("所有影片上傳失敗")
            return False

        feed_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/feed"
        data = {"message": caption, "access_token": self.page_access_token}
        for i, fid in enumerate(media_fbids):
            data[f"attached_media[{i}]"] = json.dumps({"media_fbid": fid})

        resp = requests.post(feed_url, data=data, timeout=60)
        resp.raise_for_status()
        self.logger.info(f"Facebook 多影片貼文發布成功，ID: {resp.json().get('id')}")
        return True

    def _post_images_album(self, image_paths: list, caption: str) -> bool:
        """透過 unpublished photos + feed attached_media 發布多圖貼文"""
        media_fbids = []
        upload_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/photos"

        for path in image_paths:
            try:
                with open(path, "rb") as f:
                    files = {"source": (os.path.basename(path), f, self._get_content_type(path))}
                    data = {"published": "false", "access_token": self.page_access_token}
                    resp = requests.post(upload_url, files=files, data=data, timeout=120)
                    resp.raise_for_status()
                    photo_id = resp.json().get("id")
                    if photo_id:
                        media_fbids.append(photo_id)
                        self.logger.debug(f"上傳未發布圖片成功，ID: {photo_id}")
                    else:
                        self.logger.warning(f"上傳圖片未返回 ID: {path}")
            except requests.RequestException as e:
                self.logger.error(f"上傳圖片失敗: {path}, {e}")
            time.sleep(1)

        if not media_fbids:
            self.logger.error("所有圖片上傳失敗")
            return False

        feed_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/feed"
        data = {"message": caption, "access_token": self.page_access_token}
        for i, fid in enumerate(media_fbids):
            data[f"attached_media[{i}]"] = json.dumps({"media_fbid": fid})

        resp = requests.post(feed_url, data=data, timeout=60)
        resp.raise_for_status()
        self.logger.info(f"Facebook 多圖貼文發布成功，ID: {resp.json().get('id')}")
        return True

    def _gif_to_mp4(self, gif_path: str) -> str:
        """GIF 轉 MP4"""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        out = self.ffmpeg_service.gif_to_mp4(gif_path, tmp.name, fps=None)
        self.temp_files.append(out)
        return out

    def _get_content_type(self, path: str) -> str:
        ext = self._get_ext(path)
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif", "mp4": "video/mp4",
        }
        return mime.get(ext, "application/octet-stream")

    def _cleanup_temp(self) -> None:
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except Exception as e:
                self.logger.warning(f"無法刪除臨時檔案 {f}: {e}")
        self.temp_files = []
