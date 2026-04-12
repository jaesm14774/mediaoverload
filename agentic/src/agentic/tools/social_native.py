from __future__ import annotations

import os
import tempfile
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
from dotenv import load_dotenv

from agentic.tools.ffmpeg_adapter import FFmpegAdapter


@dataclass(slots=True)
class MediaPost:
    media_paths: list[str]
    caption: str
    hashtags: str | None = None
    additional_params: dict[str, Any] | None = None


class SocialPlatform(Protocol):
    def authenticate(self) -> None: ...
    def upload_post(self, post: MediaPost) -> bool: ...


class SocialMediaManager:
    def __init__(self) -> None:
        self.platforms: dict[str, SocialPlatform] = {}

    def register_platform(self, name: str, platform: SocialPlatform) -> None:
        self.platforms[name] = platform
        platform.authenticate()

    def upload_to_platform(self, platform_name: str, post: MediaPost) -> bool:
        if platform_name not in self.platforms:
            raise ValueError(f"Platform {platform_name} not registered")
        try:
            return self.platforms[platform_name].upload_post(post)
        except Exception as exc:
            raise RuntimeError(f"Publishing failed for {platform_name}: {_describe_publish_exception(exc)}") from exc

    def upload_to_all(self, post: MediaPost) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for platform_name in self.platforms:
            results[platform_name] = self.upload_to_platform(platform_name, post)
        return results


def _describe_publish_exception(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        response = exc.response
        body = (response.text or "").strip().replace("\n", " ")
        if len(body) > 300:
            body = body[:297] + "..."
        request_url = response.request.url if response.request is not None else response.url
        return f"HTTP {response.status_code} {response.reason} | url={request_url} | body={body}"
    return f"{type(exc).__name__}: {exc}"


class CloudinaryUploadService:
    FOLDER = "mediaoverload"

    def __init__(self) -> None:
        for env_path in ("media_overload.env", ".env"):
            if Path(env_path).exists():
                load_dotenv(env_path)
                break
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
        self.api_key = os.getenv("CLOUDINARY_API_KEY")
        self.api_secret = os.getenv("CLOUDINARY_API_SECRET") or os.getenv("cloudinary_token")

    def is_configured(self) -> bool:
        return bool(self.cloud_name and self.api_key and self.api_secret)

    def upload(self, local_path: str) -> str | None:
        if not self.is_configured() or not Path(local_path).exists():
            return None
        try:
            import cloudinary
            import cloudinary.uploader

            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
            resource_type = "video" if Path(local_path).suffix.lower() in {".mp4", ".avi", ".mov", ".webm"} else "image"
            result = cloudinary.uploader.upload(local_path, resource_type=resource_type, folder=self.FOLDER)
            return str(result.get("secure_url"))
        except Exception:
            return None


class BaseConfigPlatform:
    def __init__(self, config_folder_path: str, prefix: str = "") -> None:
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.ffmpeg = FFmpegAdapter()

    def _config_path(self) -> Path:
        return Path(self.config_folder_path) / self.prefix if self.prefix else Path(self.config_folder_path)


class TwitterPlatform(BaseConfigPlatform):
    def __init__(self, config_folder_path: str, prefix: str = "") -> None:
        super().__init__(config_folder_path, prefix)
        self.client_v1 = None
        self.client_v2 = None
        self.load_config()
        self.authenticate()

    def load_config(self) -> None:
        load_dotenv(self._config_path() / "twitter.env")

    def authenticate(self) -> None:
        import tweepy

        api_key = os.getenv("TWITTER_API_KEY")
        api_secret = os.getenv("TWITTER_API_SECRET")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if not all([api_key, api_secret, access_token, access_token_secret]):
            missing = [key for key, value in {
                "TWITTER_API_KEY": api_key,
                "TWITTER_API_SECRET": api_secret,
                "TWITTER_ACCESS_TOKEN": access_token,
                "TWITTER_ACCESS_TOKEN_SECRET": access_token_secret,
            }.items() if not value]
            raise ValueError(f"Missing Twitter credentials: {', '.join(missing)}")
        auth = tweepy.OAuth1UserHandler(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
        )
        self.client_v1 = tweepy.API(auth, wait_on_rate_limit=False)
        self.client_v2 = tweepy.Client(
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            bearer_token=bearer_token if bearer_token else None,
            wait_on_rate_limit=False,
        )

    def upload_post(self, post: MediaPost) -> bool:
        text = post.caption or ""
        if post.hashtags:
            text = f"{text}\n{post.hashtags}" if text else str(post.hashtags)
        if len(text) > 280:
            text = text[:277] + "..."

        media_ids: list[int] = []
        if post.media_paths:
            for media_path in post.media_paths[:4]:
                if not Path(media_path).exists():
                    continue
                if str(media_path).lower().endswith(".mp4"):
                    media = self.client_v1.media_upload(media_path, media_category="tweet_video")
                    media_ids.append(int(media.media_id))
                    self._wait_for_video_processing(int(media.media_id))
                else:
                    media = self.client_v1.media_upload(media_path)
                    media_ids.append(int(media.media_id))

        try:
            if media_ids:
                tweet = self.client_v2.create_tweet(text=text, media_ids=media_ids)
            else:
                tweet = self.client_v2.create_tweet(text=text)
            return bool(tweet.data)
        except Exception:
            if media_ids:
                tweet = self.client_v1.update_status(status=text, media_ids=media_ids)
            else:
                tweet = self.client_v1.update_status(status=text)
            return bool(tweet)

    def _wait_for_video_processing(self, media_id: int, max_wait_time: int = 300) -> None:
        import tweepy

        started = time.time()
        while time.time() - started < max_wait_time:
            status = self.client_v1.get_media_upload_status(media_id)
            processing = getattr(status, "processing_info", None)
            if not processing:
                return
            state = processing.get("state")
            if state == "succeeded":
                return
            if state == "failed":
                raise RuntimeError(f"Twitter video processing failed for media_id={media_id}")
            time.sleep(int(processing.get("check_after_secs", 5)))
        raise tweepy.TweepyException(f"Twitter video processing timed out for media_id={media_id}")


class FacebookPlatform(BaseConfigPlatform):
    GRAPH_API_VERSION = "v24.0"
    GRAPH_API_BASE = "https://graph.facebook.com"

    def __init__(self, config_folder_path: str, prefix: str = "") -> None:
        super().__init__(config_folder_path, prefix)
        self.page_id: str | None = None
        self.page_access_token: str | None = None
        self.temp_files: list[str] = []
        self.load_config()
        self.authenticate()

    def load_config(self) -> None:
        load_dotenv(self._config_path() / "facebook.env")

    def authenticate(self) -> None:
        self.page_id = os.getenv("FB_PAGE_ID")
        self.page_access_token = os.getenv("FB_PAGE_ACCESS_TOKEN")
        if not self.page_id or not self.page_access_token:
            missing = [key for key, value in {
                "FB_PAGE_ID": self.page_id,
                "FB_PAGE_ACCESS_TOKEN": self.page_access_token,
            }.items() if not value]
            raise ValueError(f"Missing Facebook credentials: {', '.join(missing)}")

    def upload_post(self, post: MediaPost) -> bool:
        caption = post.caption or ""
        if post.hashtags:
            caption = f"{caption}\n{post.hashtags}" if caption else str(post.hashtags)
        valid_paths = [path for path in post.media_paths if Path(path).exists()]
        if not valid_paths:
            return False
        if len(valid_paths) == 1:
            return self._upload_single(valid_paths[0], caption)
        return self._upload_multiple(valid_paths[:10], caption)

    def _upload_single(self, media_path: str, caption: str) -> bool:
        ext = Path(media_path).suffix.lower()
        is_video = ext in {".mp4", ".avi", ".mov", ".webm", ".gif"}
        upload_path = media_path
        if ext == ".gif":
            upload_path = self._gif_to_mp4(media_path)
            is_video = True
        endpoint = "videos" if is_video else "photos"
        text_key = "description" if is_video else "message"
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/{endpoint}"
        with open(upload_path, "rb") as handle:
            files = {"source": (Path(upload_path).name, handle)}
            data = {text_key: caption, "access_token": self.page_access_token}
            response = requests.post(url, files=files, data=data, timeout=300)
            response.raise_for_status()
        return True

    def _upload_multiple(self, media_paths: list[str], caption: str) -> bool:
        image_paths: list[str] = []
        video_paths: list[str] = []
        for media_path in media_paths:
            ext = Path(media_path).suffix.lower()
            if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                image_paths.append(media_path)
            elif ext == ".gif":
                video_paths.append(self._gif_to_mp4(media_path))
            elif ext in {".mp4", ".avi", ".mov", ".webm"}:
                video_paths.append(media_path)

        success = False
        caption_used = False
        if len(image_paths) >= 2:
            success = self._post_images_album(image_paths, caption) or success
            caption_used = True
        elif len(image_paths) == 1 and not video_paths:
            return self._upload_single(image_paths[0], caption)

        if len(video_paths) >= 2:
            success = self._post_videos_album(video_paths, caption if not caption_used else "") or success
            caption_used = True
        elif len(video_paths) == 1:
            success = self._upload_single(video_paths[0], caption if not caption_used else "") or success
            caption_used = True

        if len(image_paths) == 1 and video_paths:
            success = self._upload_single(image_paths[0], caption if not caption_used else "") or success

        self._cleanup_temp()
        return success

    def _post_images_album(self, image_paths: list[str], caption: str) -> bool:
        media_fbids: list[str] = []
        upload_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/photos"
        for path in image_paths:
            with open(path, "rb") as handle:
                files = {"source": (Path(path).name, handle)}
                data = {"published": "false", "access_token": self.page_access_token}
                response = requests.post(upload_url, files=files, data=data, timeout=120)
                response.raise_for_status()
                photo_id = response.json().get("id")
                if photo_id:
                    media_fbids.append(str(photo_id))
            time.sleep(1)
        if not media_fbids:
            return False
        feed_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/feed"
        data = {"message": caption, "access_token": self.page_access_token}
        for index, media_fbid in enumerate(media_fbids):
            data[f"attached_media[{index}]"] = json.dumps({"media_fbid": media_fbid})
        response = requests.post(feed_url, data=data, timeout=60)
        response.raise_for_status()
        return True

    def _post_videos_album(self, video_paths: list[str], caption: str) -> bool:
        media_fbids: list[str] = []
        upload_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/videos"
        for path in video_paths:
            with open(path, "rb") as handle:
                files = {"source": (Path(path).name, handle)}
                data = {"published": "false", "access_token": self.page_access_token}
                response = requests.post(upload_url, files=files, data=data, timeout=300)
                response.raise_for_status()
                video_id = response.json().get("id") or response.json().get("video_id")
                if video_id:
                    media_fbids.append(str(video_id))
            time.sleep(2)
        if not media_fbids:
            return False
        feed_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/feed"
        data = {"message": caption, "access_token": self.page_access_token}
        for index, media_fbid in enumerate(media_fbids):
            data[f"attached_media[{index}]"] = json.dumps({"media_fbid": media_fbid})
        response = requests.post(feed_url, data=data, timeout=60)
        response.raise_for_status()
        return True

    def _gif_to_mp4(self, gif_path: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        out = self.ffmpeg.gif_to_mp4(gif_path, tmp.name)
        self.temp_files.append(out)
        return out

    def _cleanup_temp(self) -> None:
        for path in self.temp_files:
            if Path(path).exists():
                Path(path).unlink()
        self.temp_files = []


class InstagramGraphPlatform(BaseConfigPlatform):
    GRAPH_API_VERSION = "v25.0"
    GRAPH_API_BASE = "https://graph.instagram.com"
    CAPTION_MAX = 2200

    def __init__(self, config_folder_path: str, prefix: str = "") -> None:
        super().__init__(config_folder_path, prefix)
        self.ig_user_id: str | None = None
        self.access_token: str | None = None
        self.media_base_url: str | None = None
        self.cloudinary = CloudinaryUploadService()
        self.temp_files: list[str] = []
        self._url_cache: dict[str, str] = {}
        self.load_config()
        self.authenticate()

    def load_config(self) -> None:
        load_dotenv(self._config_path() / "instagram_graph.env")
        for env_path in ("media_overload.env", ".env"):
            if Path(env_path).exists():
                load_dotenv(env_path)
                break
        self.media_base_url = (os.getenv("IG_GRAPH_MEDIA_BASE_URL") or "").rstrip("/")

    def authenticate(self) -> None:
        self.access_token = os.getenv("IG_GRAPH_ACCESS_TOKEN")
        self.ig_user_id = os.getenv("IG_USER_ID") or os.getenv("IG_GRAPH_USER_ID")
        if not self.access_token:
            raise ValueError("Missing IG_GRAPH_ACCESS_TOKEN")
        if not self.ig_user_id:
            self.ig_user_id = self._fetch_ig_user_id_from_me()
        if not self.ig_user_id:
            raise ValueError("Missing IG user id for Instagram Graph")

    def upload_post(self, post: MediaPost) -> bool:
        caption = post.caption or ""
        if post.hashtags:
            caption = f"{caption}\n{post.hashtags}" if caption else str(post.hashtags)
        if len(caption) > self.CAPTION_MAX:
            caption = caption[: self.CAPTION_MAX - 3] + "..."
        valid_paths = [path for path in post.media_paths if Path(path).exists()]
        if not valid_paths:
            return False
        self._url_cache = {}
        if len(valid_paths) == 1:
            try:
                return self._upload_single(valid_paths[0], caption)
            finally:
                self._cleanup_temp()
        try:
            return self._upload_carousel(valid_paths[:10], caption)
        finally:
            self._cleanup_temp()

    def _upload_single(self, media_path: str, caption: str) -> bool:
        ext = Path(media_path).suffix.lower()
        if ext == ".gif":
            media_path = self._gif_to_mp4(media_path)
            ext = ".mp4"
        if ext in {".mp4", ".avi", ".mov", ".webm"}:
            return self._publish_video_url(media_path, caption)
        return self._publish_image_url(media_path, caption)

    def _publish_image_url(self, image_path: str, caption: str) -> bool:
        media_url = self._require_media_url(image_path)
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        response = requests.post(
            url,
            data={"image_url": media_url, "caption": caption, "access_token": self.access_token},
            timeout=60,
        )
        response.raise_for_status()
        container_id = response.json().get("id")
        if not container_id:
            raise RuntimeError(f"Instagram Graph did not return an image container id for {Path(image_path).name}")
        return self._publish_container(str(container_id))

    def _publish_video_url(self, video_path: str, caption: str) -> bool:
        media_url = self._require_media_url(video_path)
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        response = requests.post(
            url,
            data={
                "media_type": "REELS",
                "video_url": media_url,
                "caption": caption,
                "access_token": self.access_token,
            },
            timeout=60,
        )
        response.raise_for_status()
        container_id = str(response.json().get("id", ""))
        if not container_id:
            raise RuntimeError(f"Instagram Graph did not return a video container id for {Path(video_path).name}")
        if not self._wait_container_ready(container_id):
            raise RuntimeError(f"Instagram Graph video container was not ready: {container_id}")
        return self._publish_container(container_id)

    def _upload_carousel(self, media_paths: list[str], caption: str) -> bool:
        children: list[str] = []
        for path in media_paths:
            child = self._create_carousel_item(path)
            if child:
                children.append(child)
            time.sleep(1)
        if len(children) < 2:
            raise RuntimeError("Instagram Graph carousel requires at least two ready child containers")
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        response = requests.post(
            url,
            data={
                "media_type": "CAROUSEL",
                "caption": caption,
                "children": ",".join(children),
                "access_token": self.access_token,
            },
            timeout=60,
        )
        response.raise_for_status()
        container_id = str(response.json().get("id", ""))
        if not container_id:
            raise RuntimeError("Instagram Graph did not return a carousel container id")
        if not self._wait_container_ready(container_id):
            raise RuntimeError(f"Instagram Graph carousel container was not ready: {container_id}")
        return self._publish_container(container_id)

    def _create_carousel_item(self, media_path: str) -> str | None:
        ext = Path(media_path).suffix.lower()
        is_video = ext in {".mp4", ".avi", ".mov", ".webm", ".gif"}
        if ext == ".gif":
            media_path = self._gif_to_mp4(media_path)
            is_video = True
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media"
        if is_video:
            media_url = self._require_media_url(media_path)
            response = requests.post(
                url,
                data={
                    "media_type": "VIDEO",
                    "is_carousel_item": "true",
                    "video_url": media_url,
                    "access_token": self.access_token,
                },
                timeout=60,
            )
            response.raise_for_status()
            container_id = str(response.json().get("id", ""))
            if not container_id:
                raise RuntimeError(f"Instagram Graph did not return a carousel video container id for {Path(media_path).name}")
            if not self._wait_container_ready(container_id):
                raise RuntimeError(f"Instagram Graph carousel video container was not ready: {container_id}")
            return container_id
        media_url = self._require_media_url(media_path)
        response = requests.post(
            url,
            data={"image_url": media_url, "is_carousel_item": "true", "access_token": self.access_token},
            timeout=60,
        )
        response.raise_for_status()
        container_id = str(response.json().get("id", ""))
        if not container_id:
            raise RuntimeError(f"Instagram Graph did not return a carousel image container id for {Path(media_path).name}")
        return container_id

    def _publish_container(self, container_id: str) -> bool:
        response = requests.post(
            f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media_publish",
            data={"creation_id": container_id, "access_token": self.access_token},
            timeout=60,
        )
        response.raise_for_status()
        return True

    def _wait_container_ready(self, container_id: str, max_wait: int = 120) -> bool:
        url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{container_id}"
        for _ in range(max_wait):
            response = requests.get(url, params={"fields": "status_code", "access_token": self.access_token}, timeout=30)
            response.raise_for_status()
            status = str(response.json().get("status_code", ""))
            if status == "FINISHED":
                return True
            if status == "ERROR":
                return False
            time.sleep(2)
        return False

    def _fetch_ig_user_id_from_me(self) -> str | None:
        response = requests.get(
            f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me",
            params={"fields": "user_id,username,id", "access_token": self.access_token},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        user_id = body.get("user_id") or body.get("id") or ""
        return str(user_id) if user_id else None

    def _get_media_url(self, local_path: str) -> str | None:
        cached = self._url_cache.get(local_path)
        if cached:
            return cached
        uploaded = self.cloudinary.upload(local_path)
        if uploaded:
            self._url_cache[local_path] = uploaded
            return uploaded
        if self.media_base_url:
            resolved = f"{self.media_base_url}/{Path(local_path).name}"
            self._url_cache[local_path] = resolved
            return resolved
        return None

    def _require_media_url(self, local_path: str) -> str:
        media_url = self._get_media_url(local_path)
        if media_url:
            return media_url
        raise RuntimeError(
            f"Instagram Graph requires a public media URL for {Path(local_path).name}; "
            "configure Cloudinary or IG_GRAPH_MEDIA_BASE_URL"
        )

    def _gif_to_mp4(self, gif_path: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        out = self.ffmpeg.gif_to_mp4(gif_path, tmp.name)
        self.temp_files.append(out)
        return out

    def _cleanup_temp(self) -> None:
        for path in self.temp_files:
            if Path(path).exists():
                Path(path).unlink()
        self.temp_files = []


class PublishingService:
    def __init__(self) -> None:
        self.social_media_manager = SocialMediaManager()

    def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
        from utils.image import ImageConverter

        converter = ImageConverter(quality=95)
        processed_paths: list[str] = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for media_path in media_paths:
            if not Path(media_path).exists():
                continue
            suffix = Path(media_path).suffix.lower()
            if suffix in {".mp4", ".avi", ".mov", ".gif", ".webm", ".jpg"}:
                processed_paths.append(media_path)
                continue
            converted = converter.convert_to_jpg(media_path, output_dir=None)
            if converted and Path(converted).exists():
                processed_paths.append(converted)
                try:
                    Path(media_path).unlink()
                except Exception:
                    pass
            else:
                processed_paths.append(media_path)
        return list(dict.fromkeys(processed_paths))

    def publish_to_social_media(self, post: MediaPost, platforms: list[str] | None = None) -> dict[str, bool]:
        if platforms:
            return {platform: self.social_media_manager.upload_to_platform(platform, post) for platform in platforms}
        return self.social_media_manager.upload_to_all(post)

    def register_platform(self, platform_name: str, platform_config: dict[str, Any]) -> None:
        normalized = platform_name.lower()
        if normalized == "twitter":
            platform = TwitterPlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        elif normalized == "facebook":
            platform = FacebookPlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        elif normalized == "instagram_graph":
            platform = InstagramGraphPlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        else:
            raise ValueError(f"Unsupported agentic-native publishing platform: {platform_name}")
        self.social_media_manager.register_platform(platform_name, platform)
