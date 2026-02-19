"""Cloudinary 媒體上傳服務

上傳圖片/影片到 Cloudinary，取得公開 URL 供 IG Graph API 使用。
"""
import os
from typing import Optional

from dotenv import load_dotenv


def _get_logger():
    try:
        from utils.logger import setup_logger
        return setup_logger(__name__)
    except Exception:
        return None


class CloudinaryUploadService:
    IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")
    VIDEO_EXTS = ("mp4", "avi", "mov", "webm")
    FOLDER = "mediaoverload"

    def __init__(self):
        for env in ("media_overload.env", ".env"):
            if os.path.exists(env):
                load_dotenv(env)
                break
        load_dotenv()
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        self.api_key = os.getenv("CLOUDINARY_API_KEY")
        self.api_secret = os.getenv("CLOUDINARY_API_SECRET") or os.getenv("cloudinary_token")

    def is_configured(self) -> bool:
        return bool(self.cloud_name and self.api_key and self.api_secret)

    def _get_resource_type(self, path: str) -> str:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return "video" if ext in self.VIDEO_EXTS else "image"

    def upload(self, local_path: str) -> Optional[str]:
        """上傳媒體到 Cloudinary，回傳公開 URL"""
        if not self.is_configured():
            return None
        if not os.path.exists(local_path):
            return None

        try:
            import cloudinary
            import cloudinary.uploader

            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )

            resource_type = self._get_resource_type(local_path)
            result = cloudinary.uploader.upload(
                local_path,
                resource_type=resource_type,
                folder=self.FOLDER,
            )
            return result.get("secure_url")
        except ImportError:
            log = _get_logger()
            if log:
                log.warning("cloudinary 未安裝，執行 pip install cloudinary")
            return None
        except Exception as e:
            log = _get_logger()
            if log:
                log.warning(f"Cloudinary 上傳失敗: {e}")
            return None
