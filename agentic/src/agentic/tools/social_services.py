from __future__ import annotations

from pathlib import Path

from agentic.runtime.registry import ToolRegistry
from agentic.tools.publishing_adapter import MediaPost, PublishingAdapter


class SocialServiceTools:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._publishing: PublishingAdapter | None = None

    def process_media(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._publishing_service()
        media_paths = [str(path) for path in payload.get("media_paths", [])]
        output_dir = str(payload.get("output_dir") or (self.output_root / "publish_ready"))
        processed = service.process_media(media_paths=media_paths, output_dir=output_dir)
        return {"media_paths": processed, "output_dir": output_dir}

    def publish_social(self, payload: dict[str, object]) -> dict[str, object]:
        media_paths = [str(path) for path in payload.get("media_paths", [])]
        caption = str(payload.get("caption", ""))
        hashtags = payload.get("hashtags")
        hashtags_str = str(hashtags) if isinstance(hashtags, str) else None
        platforms = [str(platform) for platform in payload.get("platforms", [])]
        platform_bundle = payload.get("platform_bundle", {}) or {}
        platform_configs = payload.get("platform_configs", {}) or {}
        dry_run = bool(payload.get("dry_run", False))
        dispatch_plan = self._build_dispatch_plan(
            media_paths=media_paths,
            caption=caption,
            hashtags=hashtags_str or "",
            platforms=platforms,
            platform_bundle=platform_bundle if isinstance(platform_bundle, dict) else {},
        )
        if dry_run:
            return {
                "status": "dry_run",
                "media_paths": media_paths,
                "caption": caption,
                "hashtags": hashtags_str,
                "platforms": platforms,
                "dispatch_plan": dispatch_plan,
            }

        service = self._publishing_service()
        for platform_name, platform_config in platform_configs.items():
            service.register_platform(str(platform_name), dict(platform_config))
        effective_platforms = platforms or list(dispatch_plan.keys())
        results: dict[str, bool] = {}
        errors: dict[str, str] = {}
        global_additional_params = dict(payload.get("additional_params", {}) or {})
        for platform in effective_platforms:
            platform_plan = dispatch_plan.get(platform, {})
            platform_media_paths = [str(path) for path in platform_plan.get("media_paths", media_paths)]
            platform_bundle_entry = platform_bundle.get(platform, {}) if isinstance(platform_bundle, dict) else {}
            platform_additional_params = {
                **global_additional_params,
                **dict(platform_bundle_entry.get("additional_params", {}) or {}),
            }
            platform_post = MediaPost(
                media_paths=platform_media_paths,
                caption=str(platform_plan.get("caption") or caption),
                hashtags=str(platform_plan.get("hashtags") or hashtags_str or ""),
                additional_params=platform_additional_params,
            )
            try:
                platform_result = service.publish(post=platform_post, platforms=[platform])
                results[platform] = bool(platform_result.get(platform))
                if not results[platform]:
                    errors[platform] = f"RuntimeError: Publishing returned false for {platform}"
            except Exception as exc:
                results[platform] = False
                errors[platform] = f"{type(exc).__name__}: {exc}"
        return {
            "status": "success" if not errors else "partial_failure",
            "results": results,
            "errors": errors,
            "media_paths": media_paths,
            "caption": caption,
            "hashtags": hashtags_str,
            "platforms": platforms or list(results.keys()),
            "dispatch_plan": dispatch_plan,
        }

    def _publishing_service(self) -> PublishingAdapter:
        if self._publishing is None:
            self._publishing = PublishingAdapter()
        return self._publishing

    @staticmethod
    def _build_dispatch_plan(
        media_paths: list[str],
        caption: str,
        hashtags: str,
        platforms: list[str],
        platform_bundle: dict[str, object],
    ) -> dict[str, dict[str, object]]:
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
                    "is_publish_ready": bool(validation.get("is_publish_ready", bool(platform_caption) and bool(platform_media_paths))),
                },
            }
        return dispatch_plan


def register_social_service_tools(tool_registry: ToolRegistry, output_root: Path) -> None:
    tools = SocialServiceTools(output_root=output_root)
    tool_registry.register("publish.process_media", tools.process_media, "Prepare media files for social publishing")
    tool_registry.register("publish.social", tools.publish_social, "Publish prepared media to configured social platforms")
