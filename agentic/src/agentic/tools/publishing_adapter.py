from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic.tools.social_native import MediaPost as NativeMediaPost
from agentic.tools.social_native import PublishingService


@dataclass(slots=True)
class MediaPost:
    media_paths: list[str]
    caption: str
    hashtags: str | None = None
    additional_params: dict[str, Any] | None = None


class PublishingAdapter:
    def __init__(self) -> None:
        self._service = PublishingService()

    def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
        return self._service.process_media(media_paths=media_paths, output_dir=output_dir)

    def register_platform(self, platform_name: str, platform_config: dict[str, Any]) -> None:
        self._service.register_platform(platform_name, platform_config)

    def publish(self, post: MediaPost, platforms: list[str] | None = None) -> dict[str, bool]:
        native_post = NativeMediaPost(
            media_paths=post.media_paths,
            caption=post.caption,
            hashtags=post.hashtags,
            additional_params=post.additional_params,
        )
        return self._service.publish_to_social_media(post=native_post, platforms=platforms)
