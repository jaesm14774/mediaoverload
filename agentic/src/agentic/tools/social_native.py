from __future__ import annotations

import os
import tempfile
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
        self.publish_receipts: dict[str, dict[str, Any]] = {}

    def register_platform(self, name: str, platform: SocialPlatform) -> None:
        self.platforms[name] = platform
        self.publish_receipts.pop(name, None)
        platform.authenticate()

    def upload_to_platform(self, platform_name: str, post: MediaPost) -> bool:
        if platform_name not in self.platforms:
            raise ValueError(f"Platform {platform_name} not registered")
        platform = self.platforms[platform_name]
        if hasattr(platform, "last_publish_receipt"):
            platform.last_publish_receipt = None  # type: ignore[attr-defined]
        try:
            result = bool(platform.upload_post(post))
            receipt = getattr(platform, "last_publish_receipt", None)
            if isinstance(receipt, dict):
                self.publish_receipts[platform_name] = dict(receipt)
            if platform_name.lower() in {"facebook", "youtube"}:
                if not result:
                    raise RuntimeError(f"{platform_name} publisher returned false")
                if not isinstance(receipt, dict) or not receipt.get("verified"):
                    raise RuntimeError(
                        f"{platform_name} upload returned without a verified external artifact"
                    )
            return result
        except Exception as exc:
            receipt = getattr(platform, "last_publish_receipt", None)
            if isinstance(receipt, dict):
                self.publish_receipts[platform_name] = dict(receipt)
            raise RuntimeError(f"Publishing failed for {platform_name}: {_describe_publish_exception(exc)}") from exc

    def get_publish_receipts(self) -> dict[str, dict[str, Any]]:
        return {name: dict(receipt) for name, receipt in self.publish_receipts.items()}

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
        request_url = _redact_url_credentials(request_url)
        return f"HTTP {response.status_code} {response.reason} | url={request_url} | body={body}"
    return f"{type(exc).__name__}: {exc}"


def _redact_url_credentials(url: str) -> str:
    parsed = urlsplit(str(url))
    sensitive_keys = {"access_token", "refresh_token", "client_secret", "api_key", "token"}
    query = [
        (key, "<redacted>" if key.lower() in sensitive_keys else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))


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
        self.last_publish_receipt: dict[str, Any] | None = None

    def _record_publish_receipt(
        self,
        *,
        platform: str,
        external_id: str,
        status: str,
        verified: bool,
        visibility: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt = {
            "platform": platform,
            "external_id": str(external_id),
            "status": status,
            "verified": bool(verified),
            "visibility": visibility,
            "details": details or {},
        }
        self.last_publish_receipt = receipt
        return receipt

    def _config_path(self) -> Path:
        return Path(self.config_folder_path) / self.prefix if self.prefix else Path(self.config_folder_path)

    def _prepare_reel_canvas(self, video_path: str) -> str:
        """Normalize non-Reels video into one API-safe 9:16 canvas.

        H3 stays at its low-VRAM native size during inference. The platform
        adapter owns delivery formatting so Instagram and Facebook do not
        grow separate, contradictory padding rules.
        """
        probe = self.ffmpeg.probe_media(video_path)
        width = int(probe.get("width") or 0)
        height = int(probe.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"Reels video has no readable dimensions: {Path(video_path).name}")
        aspect = width / height
        if (9 / 16) <= aspect <= 1.91:
            return video_path
        vertical_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        vertical_path = vertical_file.name
        vertical_file.close()
        self.ffmpeg.pad_video_to_aspect(
            video_path,
            vertical_path,
            target_width=720,
            target_height=1280,
        )
        temp_files = getattr(self, "temp_files", None)
        if isinstance(temp_files, list):
            temp_files.append(vertical_path)
        return vertical_path


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
    GRAPH_API_VERSION = "v25.0"
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
        additional = dict(post.additional_params or {})
        video_extensions = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v", ".gif"}
        has_video = any(Path(path).suffix.lower() in video_extensions for path in valid_paths)
        use_reels = _coerce_bool(
            additional.get("facebook_use_reels"),
            # A short generated video is a Reel by default. The previous
            # default routed it to the legacy Page video endpoint, which
            # returned True without a receipt and was therefore correctly
            # rejected by the verified-publication gate.
            default=_coerce_bool(os.getenv("FB_USE_REELS"), default=has_video),
        )
        if use_reels:
            video_path = next(
                (
                    path
                    for path in valid_paths
                    if Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v", ".gif"}
                ),
                "",
            )
            if not video_path:
                raise ValueError("Facebook Reels publishing requires a local video file")
            try:
                return self._upload_reel(video_path, caption, additional)
            finally:
                self._cleanup_temp()
        if len(valid_paths) == 1:
            return self._upload_single(valid_paths[0], caption)
        return self._upload_multiple(valid_paths[:10], caption)

    def _upload_reel(self, media_path: str, caption: str, additional: dict[str, Any]) -> bool:
        upload_path = media_path
        if Path(media_path).suffix.lower() == ".gif":
            upload_path = self._gif_to_mp4(media_path)
        probe = self.ffmpeg.probe_media(upload_path)
        duration = float(probe.get("duration") or 0.0)
        if duration < 4 or duration > 60:
            raise ValueError(f"Facebook Reels video duration must be 4-60 seconds, got {duration:.3f}")
        upload_path = self._prepare_reel_canvas(upload_path)

        video_state = str(additional.get("facebook_video_state") or "PUBLISHED").strip().upper()
        if video_state not in {"DRAFT", "SCHEDULED", "PUBLISHED"}:
            raise ValueError(f"Unsupported Facebook Reels video_state: {video_state}")

        start_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/me/video_reels"
        start_response = requests.post(
            start_url,
            params={"access_token": self.page_access_token, "upload_phase": "start"},
            timeout=60,
        )
        start_response.raise_for_status()
        start_body = start_response.json()
        video_id = str(start_body.get("video_id") or "")
        upload_url = str(start_body.get("upload_url") or "")
        if not video_id or not upload_url:
            raise RuntimeError("Facebook Reels start phase did not return video_id and upload_url")
        self._record_publish_receipt(
            platform="facebook",
            external_id=video_id,
            status="uploading",
            verified=False,
            visibility=video_state.lower(),
            details={"start_response": {"video_id": video_id, "upload_url": upload_url}},
        )

        file_size = Path(upload_path).stat().st_size
        with open(upload_path, "rb") as handle:
            upload_response = requests.post(
                upload_url,
                headers={
                    "Authorization": f"OAuth {self.page_access_token}",
                    "offset": "0",
                    "file_size": str(file_size),
                    "Content-Type": "application/octet-stream",
                },
                data=handle,
                timeout=600,
            )
        upload_response.raise_for_status()
        upload_body = upload_response.json()
        if upload_body.get("success") is not True:
            self._record_publish_receipt(
                platform="facebook",
                external_id=video_id,
                status="upload_failed",
                verified=False,
                details={"upload_response": upload_body},
            )
            raise RuntimeError(f"Facebook Reels upload did not confirm success for video_id={video_id}")

        # Facebook documents status polling as optional and requires the finish
        # phase to end the upload and start assembly/encoding. Waiting for READY
        # here deadlocks because processing_phase remains not_started until the
        # finish request is sent.
        finish_params = {
            "access_token": self.page_access_token,
            "video_id": video_id,
            "upload_phase": "finish",
            "video_state": video_state,
            "description": caption,
        }
        title = str(additional.get("facebook_title") or "").strip()
        if title:
            finish_params["title"] = title[:100]
        finish_response = requests.post(start_url, params=finish_params, timeout=60)
        finish_error: dict[str, Any] | None = None
        try:
            finish_response.raise_for_status()
            finish_body = finish_response.json()
        except requests.RequestException as exc:
            finish_body = {}
            try:
                response_body = finish_response.json()
            except ValueError:
                response_body = {}
            finish_error = {
                "error_type": type(exc).__name__,
                "status_code": finish_response.status_code,
                "body": response_body if isinstance(response_body, dict) else {},
            }
            verified, status_body = self._wait_for_reel_ready(video_id, video_state=video_state)
            if verified:
                receipt_status = "draft_ready" if video_state == "DRAFT" else "published_ready"
                self._record_publish_receipt(
                    platform="facebook",
                    external_id=video_id,
                    status=receipt_status,
                    verified=True,
                    visibility=video_state.lower(),
                    details={
                        "finish_error": finish_error,
                        "status_response": status_body,
                        "reconciled_after_finish_error": True,
                    },
                )
                return True
            self._record_publish_receipt(
                platform="facebook",
                external_id=video_id,
                status="finish_failed",
                verified=False,
                visibility=video_state.lower(),
                details={"finish_error": finish_error, "status_response": status_body},
            )
            raise RuntimeError(f"Facebook Reels finish failed for video_id={video_id}") from exc
        if finish_body.get("success") is not True:
            self._record_publish_receipt(
                platform="facebook",
                external_id=video_id,
                status="finish_failed",
                verified=False,
                visibility=video_state.lower(),
                details={"finish_response": finish_body},
            )
            raise RuntimeError(f"Facebook Reels finish did not confirm success for video_id={video_id}")

        verified, status_body = self._wait_for_reel_ready(video_id, video_state=video_state)
        receipt_status = "draft_ready" if video_state == "DRAFT" else "published_ready"
        self._record_publish_receipt(
            platform="facebook",
            external_id=video_id,
            status=receipt_status if verified else "processing_timeout",
            verified=verified,
            visibility=video_state.lower(),
            details={"finish_response": finish_body, "status_response": status_body},
        )
        if not verified:
            raise RuntimeError(
                f"Facebook Reels artifact was not verified after finish: video_id={video_id}"
            )
        return True

    def _wait_for_reel_ready(
        self,
        video_id: str,
        *,
        video_state: str,
    ) -> tuple[bool, dict[str, Any]]:
        timeout_seconds = max(1, int(os.getenv("FACEBOOK_VERIFY_TIMEOUT_SECONDS", "180")))
        interval_seconds = max(0.5, float(os.getenv("FACEBOOK_VERIFY_INTERVAL_SECONDS", "3")))
        started = time.monotonic()
        last_body: dict[str, Any] = {}
        status_url = f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{video_id}"
        while time.monotonic() - started < timeout_seconds:
            response = requests.get(
                status_url,
                params={"fields": "status", "access_token": self.page_access_token},
                timeout=30,
            )
            response.raise_for_status()
            last_body = response.json()
            status = last_body.get("status") or {}
            processing_phase = status.get("processing_phase") or {}
            publishing_phase = status.get("publishing_phase") or {}
            video_status = str(status.get("video_status") or "").lower()
            processing_status = str(processing_phase.get("status") or "").lower()
            publishing_status = str(publishing_phase.get("status") or "").lower()
            publish_status = str(publishing_phase.get("publish_status") or "").lower()

            if video_status in {"failed", "error", "rejected"}:
                return False, last_body
            if processing_status in {"failed", "error", "rejected"}:
                return False, last_body

            processing_ready = processing_status in {"complete", "completed"} or video_status in {"ready", "published"}
            if processing_ready:
                if video_state == "DRAFT":
                    return True, last_body
                if video_state == "PUBLISHED" and (publishing_status in {"complete", "completed"} or publish_status == "published"):
                    return True, last_body
                if video_state == "SCHEDULED" and (publishing_status in {"complete", "completed"} or publish_status == "scheduled"):
                    return True, last_body
            time.sleep(interval_seconds)
        return False, last_body

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
        body = response.json() if getattr(response, "content", b"") else {}
        external_id = str(body.get("id") or body.get("video_id") or "") if isinstance(body, dict) else ""
        self._record_publish_receipt(
            platform="facebook",
            external_id=external_id,
            status="published" if external_id else "uploaded_unverified",
            verified=bool(external_id),
            visibility="published" if external_id else "unknown",
            details={"endpoint": endpoint, "response": body},
        )
        return bool(external_id)

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
        body = response.json() if getattr(response, "content", b"") else {}
        external_id = str(body.get("id") or "") if isinstance(body, dict) else ""
        self._record_publish_receipt(
            platform="facebook",
            external_id=external_id,
            status="published" if external_id else "uploaded_unverified",
            verified=bool(external_id),
            visibility="published" if external_id else "unknown",
            details={"endpoint": "feed", "response": body, "media_count": len(media_fbids)},
        )
        return bool(external_id)

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
        body = response.json() if getattr(response, "content", b"") else {}
        external_id = str(body.get("id") or "") if isinstance(body, dict) else ""
        self._record_publish_receipt(
            platform="facebook",
            external_id=external_id,
            status="published" if external_id else "uploaded_unverified",
            verified=bool(external_id),
            visibility="published" if external_id else "unknown",
            details={"endpoint": "feed", "response": body, "media_count": len(media_fbids)},
        )
        return bool(external_id)

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
        configured_user_id = os.getenv("IG_USER_ID") or os.getenv("IG_GRAPH_USER_ID")
        if not self.access_token:
            raise ValueError("Missing IG_GRAPH_ACCESS_TOKEN")
        fetched_user_id = self._fetch_ig_user_id_from_me()
        if configured_user_id and fetched_user_id and str(configured_user_id) != fetched_user_id:
            raise RuntimeError(
                f"Instagram token belongs to unexpected user: expected={configured_user_id} actual={fetched_user_id}"
            )
        self.ig_user_id = fetched_user_id or configured_user_id
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
        additional = dict(post.additional_params or {})
        publish_mode = str(additional.get("instagram_publish_mode") or "").strip().lower()
        if publish_mode == "container_only":
            video_path = next(
                (
                    path
                    for path in valid_paths
                    if Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v", ".gif"}
                ),
                "",
            )
            if not video_path:
                raise ValueError("Instagram container-only POC requires a local video file")
            try:
                return self._upload_single(video_path, caption, publish=False)
            finally:
                self._cleanup_temp()
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

    def _upload_single(self, media_path: str, caption: str, *, publish: bool = True) -> bool:
        ext = Path(media_path).suffix.lower()
        if ext == ".gif":
            media_path = self._gif_to_mp4(media_path)
            ext = ".mp4"
        if ext in {".mp4", ".avi", ".mov", ".webm"}:
            return self._publish_video_url(media_path, caption, publish=publish)
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
        media_id = self._publish_container(str(container_id))
        self._record_publish_receipt(
            platform="instagram_graph",
            external_id=media_id,
            status="published",
            verified=True,
            visibility="published",
            details={"container_id": str(container_id)},
        )
        return True

    def _publish_video_url(self, video_path: str, caption: str, *, publish: bool = True) -> bool:
        video_path = self._prepare_reel_canvas(video_path)
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
        if not publish:
            self._record_publish_receipt(
                platform="instagram_graph",
                external_id=container_id,
                status="container_ready",
                verified=True,
                visibility="container_only",
                details={"container_id": container_id},
            )
            return True
        media_id = self._publish_container(container_id)
        self._record_publish_receipt(
            platform="instagram_graph",
            external_id=media_id,
            status="published",
            verified=True,
            visibility="published",
            details={"container_id": container_id},
        )
        return True

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
        media_id = self._publish_container(container_id)
        self._record_publish_receipt(
            platform="instagram_graph",
            external_id=media_id,
            status="published",
            verified=True,
            visibility="published",
            details={"container_id": container_id},
        )
        return True

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

    def _publish_container(self, container_id: str) -> str:
        response = requests.post(
            f"{self.GRAPH_API_BASE}/{self.GRAPH_API_VERSION}/{self.ig_user_id}/media_publish",
            data={"creation_id": container_id, "access_token": self.access_token},
            timeout=60,
        )
        response.raise_for_status()
        media_id = str(response.json().get("id") or "")
        if not media_id:
            raise RuntimeError(f"Instagram Graph publish did not return a media id for container={container_id}")
        return media_id

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


class YouTubePlatform(BaseConfigPlatform):
    TOKEN_URI = "https://oauth2.googleapis.com/token"
    UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
    READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"}
    TITLE_MAX = 100
    DESCRIPTION_MAX = 5000

    def __init__(self, config_folder_path: str, prefix: str = "") -> None:
        super().__init__(config_folder_path, prefix)
        self.client_id: str | None = None
        self.client_secret: str | None = None
        self.refresh_token: str | None = None
        self.channel_id: str | None = None
        self.default_privacy_status = "public"
        self.default_category_id = "22"
        self.default_notify_subscribers = False
        self.default_made_for_kids = False
        self.default_contains_synthetic_media = True
        self.service = None
        self.load_config()
        self.authenticate()

    def load_config(self) -> None:
        load_dotenv(self._config_path() / "youtube.env")

    def authenticate(self) -> None:
        self.client_id = os.getenv("YOUTUBE_CLIENT_ID")
        self.client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
        self.refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
        self.channel_id = os.getenv("YOUTUBE_CHANNEL_ID")
        self.default_privacy_status = str(os.getenv("YOUTUBE_PRIVACY_STATUS", "public")).strip().lower() or "public"
        self.default_category_id = str(os.getenv("YOUTUBE_CATEGORY_ID", "22")).strip() or "22"
        self.default_notify_subscribers = _coerce_bool(os.getenv("YOUTUBE_NOTIFY_SUBSCRIBERS"), default=False)
        self.default_made_for_kids = _coerce_bool(os.getenv("YOUTUBE_MADE_FOR_KIDS"), default=False)
        self.default_contains_synthetic_media = _coerce_bool(
            os.getenv("YOUTUBE_CONTAINS_SYNTHETIC_MEDIA"),
            default=True,
        )
        missing = [
            key
            for key, value in {
                "YOUTUBE_CLIENT_ID": self.client_id,
                "YOUTUBE_CLIENT_SECRET": self.client_secret,
                "YOUTUBE_REFRESH_TOKEN": self.refresh_token,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing YouTube credentials: {', '.join(missing)}")
        self.service = self._build_service()

    def upload_post(self, post: MediaPost) -> bool:
        video_path = next(
            (
                path
                for path in post.media_paths
                if Path(path).exists() and Path(path).suffix.lower() in self.VIDEO_EXTENSIONS
            ),
            "",
        )
        if not video_path:
            raise ValueError("YouTube publishing requires at least one local video file")

        body, notify_subscribers = self._build_video_request(post, video_path)
        media_body = self._build_media_upload(video_path)
        request = self.service.videos().insert(  # type: ignore[union-attr]
            part="snippet,status",
            body=body,
            notifySubscribers=notify_subscribers,
            media_body=media_body,
        )
        response = None
        while response is None:
            _, response = request.next_chunk()
        video_id = str(response.get("id") or "").strip()
        if not video_id:
            self._record_publish_receipt(
                platform="youtube",
                external_id="",
                status="upload_failed",
                verified=False,
                visibility=body.get("status", {}).get("privacyStatus"),
                details={"insert_response": response},
            )
            raise RuntimeError("YouTube videos.insert did not return a video id")
        privacy_status = str(body.get("status", {}).get("privacyStatus") or "private")
        self._record_publish_receipt(
            platform="youtube",
            external_id=video_id,
            status="uploaded_unverified",
            verified=False,
            visibility=privacy_status,
            details={"insert_response": response},
        )
        if self.channel_id and str(response.get("snippet", {}).get("channelId", "")).strip():
            uploaded_channel = str(response["snippet"]["channelId"]).strip()
            if uploaded_channel != self.channel_id:
                self._record_publish_receipt(
                    platform="youtube",
                    external_id=video_id,
                    status="unexpected_channel",
                    verified=False,
                    visibility=privacy_status,
                    details={
                        "expected_channel_id": self.channel_id,
                        "actual_channel_id": uploaded_channel,
                        "insert_response": response,
                    },
                )
                raise RuntimeError(
                    f"YouTube upload landed on unexpected channel: expected={self.channel_id} actual={uploaded_channel}"
                )
        verified, verification_body = self._wait_for_uploaded_video(video_id, privacy_status=privacy_status)
        self._record_publish_receipt(
            platform="youtube",
            external_id=video_id,
            status="processed" if verified else "processing_timeout",
            verified=verified,
            visibility=privacy_status,
            details={"insert_response": response, "verification_response": verification_body},
        )
        if not verified:
            raise RuntimeError(f"YouTube uploaded video was not verified: video_id={video_id}")
        return True

    def _wait_for_uploaded_video(
        self,
        video_id: str,
        *,
        privacy_status: str,
    ) -> tuple[bool, dict[str, Any]]:
        timeout_seconds = max(1, int(os.getenv("YOUTUBE_VERIFY_TIMEOUT_SECONDS", "180")))
        interval_seconds = max(0.5, float(os.getenv("YOUTUBE_VERIFY_INTERVAL_SECONDS", "5")))
        started = time.monotonic()
        last_body: dict[str, Any] = {}
        while time.monotonic() - started < timeout_seconds:
            response = self.service.videos().list(  # type: ignore[union-attr]
                part="id,snippet,status,processingDetails",
                id=video_id,
            ).execute()
            items = response.get("items") or []
            if not items:
                return False, response
            item = items[0]
            last_body = item
            status = item.get("status") or {}
            processing = item.get("processingDetails") or {}
            upload_status = str(status.get("uploadStatus") or "").lower()
            actual_privacy = str(status.get("privacyStatus") or "").lower()
            processing_status = str(processing.get("processingStatus") or "").lower()
            if upload_status in {"failed", "rejected", "deleted"} or processing_status in {"failed", "terminated"}:
                return False, last_body
            if actual_privacy != privacy_status.lower():
                return False, last_body
            if upload_status in {"uploaded", "processed"} and processing_status in {"", "succeeded"}:
                return True, last_body
            time.sleep(interval_seconds)
        return False, last_body

    def _build_service(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise RuntimeError(
                "Missing YouTube client dependencies; install google-api-python-client and google-auth-oauthlib"
            ) from exc

        credentials = Credentials(
            token=None,
            refresh_token=self.refresh_token,
            token_uri=self.TOKEN_URI,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=[self.UPLOAD_SCOPE, self.READ_SCOPE],
        )
        credentials.refresh(Request())
        return build("youtube", "v3", credentials=credentials, cache_discovery=False)

    @staticmethod
    def _build_media_upload(video_path: str):
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RuntimeError(
                "Missing YouTube client dependencies; install google-api-python-client and google-auth-oauthlib"
            ) from exc
        return MediaFileUpload(video_path, chunksize=-1, resumable=True)

    def _build_video_request(self, post: MediaPost, video_path: str) -> tuple[dict[str, Any], bool]:
        additional = dict(post.additional_params or {})
        title = str(additional.get("youtube_title") or self._derive_title(post.caption, video_path)).strip()
        description = str(
            additional.get("youtube_description")
            or self._derive_description(post.caption, post.hashtags)
        ).strip()
        tags = self._derive_tags(additional.get("youtube_tags"), post.hashtags)
        privacy_status = str(additional.get("youtube_privacy_status") or self.default_privacy_status).strip().lower()
        category_id = str(additional.get("youtube_category_id") or self.default_category_id).strip() or "22"
        notify_subscribers = _coerce_bool(
            additional.get("youtube_notify_subscribers"),
            default=self.default_notify_subscribers,
        )
        made_for_kids = _coerce_bool(
            additional.get("youtube_made_for_kids"),
            default=self.default_made_for_kids,
        )
        contains_synthetic_media = _coerce_bool(
            additional.get("youtube_contains_synthetic_media"),
            default=self.default_contains_synthetic_media,
        )
        publish_at = str(additional.get("youtube_publish_at") or "").strip()

        snippet: dict[str, Any] = {
            "title": title[: self.TITLE_MAX] or Path(video_path).stem[: self.TITLE_MAX],
            "description": description[: self.DESCRIPTION_MAX],
            "categoryId": category_id,
        }
        if tags:
            snippet["tags"] = tags
        status: dict[str, Any] = {
            "privacyStatus": privacy_status or "private",
            "selfDeclaredMadeForKids": made_for_kids,
            "containsSyntheticMedia": contains_synthetic_media,
        }
        if publish_at and status["privacyStatus"] == "private":
            status["publishAt"] = publish_at
        return {"snippet": snippet, "status": status}, notify_subscribers

    @staticmethod
    def _derive_title(caption: str, video_path: str) -> str:
        lines = [line.strip() for line in str(caption or "").splitlines() if line.strip()]
        if lines:
            return lines[0]
        return Path(video_path).stem.replace("_", " ").replace("-", " ")

    @staticmethod
    def _derive_description(caption: str, hashtags: str | None) -> str:
        parts = [str(caption or "").strip(), str(hashtags or "").strip()]
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _derive_tags(raw_tags: Any, hashtags: str | None) -> list[str]:
        tags: list[str] = []
        if isinstance(raw_tags, list):
            tags.extend(str(tag).strip().lstrip("#") for tag in raw_tags if str(tag).strip())
        elif isinstance(raw_tags, str):
            tags.extend(token.strip().lstrip("#") for token in raw_tags.replace(",", " ").split() if token.strip())
        if hashtags:
            tags.extend(token.strip().lstrip("#") for token in str(hashtags).split() if token.strip())
        deduped: list[str] = []
        for tag in tags:
            normalized = tag.strip()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped[:500]


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


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

    def get_publish_receipts(self) -> dict[str, dict[str, Any]]:
        return self.social_media_manager.get_publish_receipts()

    def register_platform(self, platform_name: str, platform_config: dict[str, Any]) -> None:
        normalized = platform_name.lower()
        if normalized == "twitter":
            platform = TwitterPlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        elif normalized == "facebook":
            platform = FacebookPlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        elif normalized == "instagram_graph":
            platform = InstagramGraphPlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        elif normalized == "youtube":
            platform = YouTubePlatform(platform_config["config_folder_path"], platform_config.get("prefix", ""))
        else:
            raise ValueError(f"Unsupported agentic-native publishing platform: {platform_name}")
        self.social_media_manager.register_platform(platform_name, platform)
