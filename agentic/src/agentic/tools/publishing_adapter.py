from __future__ import annotations

from typing import Any

from agentic.tools.social_native import MediaPost, PublishingService

FACEBOOK_PROFILE_HANDOFF_PLATFORM = "facebook_profile_handoff"


def build_dispatch_plan(
    media_paths: list[str],
    caption: str,
    hashtags: str,
    platforms: list[str],
    platform_bundle: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Normalize one platform-aware publish contract for skills and tools."""
    effective_platforms = platforms or [str(platform) for platform in platform_bundle.keys()]
    if not effective_platforms:
        effective_platforms = ["generic"]
    dispatch_plan: dict[str, dict[str, object]] = {}
    for platform in effective_platforms:
        bundle = platform_bundle.get(platform, {})
        if not isinstance(bundle, dict):
            bundle = {}
        platform_caption = str(bundle.get("caption") or caption)
        platform_hashtags = str(bundle.get("hashtags") or hashtags)
        platform_media_paths = [str(path) for path in bundle.get("media_paths", media_paths)]
        validation = bundle.get("validation", {})
        if not isinstance(validation, dict):
            validation = {}
        dispatch_plan[platform] = {
            "caption": platform_caption,
            "hashtags": platform_hashtags,
            "media_paths": platform_media_paths,
            "validation": {
                "has_caption": bool(validation.get("has_caption", platform_caption)),
                "has_media": bool(validation.get("has_media", platform_media_paths)),
                "is_publish_ready": bool(
                    validation.get("is_publish_ready", bool(platform_caption) and bool(platform_media_paths))
                ),
            },
        }
    return dispatch_plan


class PublishingAdapter:
    def __init__(self) -> None:
        self._service = PublishingService()

    def process_media(self, media_paths: list[str], output_dir: str) -> list[str]:
        return self._service.process_media(media_paths=media_paths, output_dir=output_dir)

    def register_platform(self, platform_name: str, platform_config: dict[str, Any]) -> None:
        self._service.register_platform(platform_name, platform_config)

    def publish(self, post: MediaPost, platforms: list[str] | None = None) -> dict[str, bool]:
        return self._service.publish_to_social_media(post=post, platforms=platforms)

    def get_publish_receipts(self) -> dict[str, dict[str, Any]]:
        return self._service.get_publish_receipts()
