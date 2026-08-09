from __future__ import annotations

import json
from datetime import datetime, timezone
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
        publish_mode = str(payload.get("publish_mode") or "").strip().lower()
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
        effective_platforms = platforms or list(dispatch_plan.keys())
        results: dict[str, bool] = {}
        errors: dict[str, str] = {}
        publish_receipts: dict[str, dict[str, object]] = {}
        global_additional_params = dict(payload.get("additional_params", {}) or {})
        global_additional_params.update(self._safe_poc_params(publish_mode))
        for platform in effective_platforms:
            platform_plan = dispatch_plan.get(platform, {})
            platform_media_paths = [str(path) for path in platform_plan.get("media_paths", media_paths)]
            platform_bundle_entry = platform_bundle.get(platform, {}) if isinstance(platform_bundle, dict) else {}
            platform_additional_params = {
                **global_additional_params,
                **dict(platform_bundle_entry.get("additional_params", {}) or {}),
            }
            platform_additional_params.update(self._safe_poc_params(publish_mode))
            platform_post = MediaPost(
                media_paths=platform_media_paths,
                caption=str(platform_plan.get("caption") or caption),
                hashtags=str(platform_plan.get("hashtags") or hashtags_str or ""),
                additional_params=platform_additional_params,
            )
            try:
                platform_config = platform_configs.get(platform)
                if not isinstance(platform_config, dict):
                    raise RuntimeError(f"No configured publisher for platform: {platform}")
                service.register_platform(platform, dict(platform_config))
                platform_additional_params.update(self._safe_poc_platform_params(platform, publish_mode))
                platform_post.additional_params = platform_additional_params
                platform_result = service.publish(post=platform_post, platforms=[platform])
                results[platform] = bool(platform_result.get(platform))
                receipts = getattr(service, "get_publish_receipts", lambda: {})()
                receipt = receipts.get(platform) if isinstance(receipts, dict) else None
                if isinstance(receipt, dict):
                    publish_receipts[platform] = dict(receipt)
                if not results[platform]:
                    errors[platform] = f"RuntimeError: Publishing returned false for {platform}"
            except Exception as exc:
                results[platform] = False
                errors[platform] = f"{type(exc).__name__}: {exc}"
                receipts = getattr(service, "get_publish_receipts", lambda: {})()
                receipt = receipts.get(platform) if isinstance(receipts, dict) else None
                if isinstance(receipt, dict):
                    publish_receipts[platform] = dict(receipt)
        manifest_path = self._write_publish_manifest(
            payload=payload,
            media_paths=media_paths,
            platforms=platforms or list(results.keys()),
            publish_mode=publish_mode,
            results=results,
            errors=errors,
            receipts=publish_receipts,
        )
        return {
            "status": "success" if not errors else "partial_failure",
            "results": results,
            "errors": errors,
            "publish_receipts": publish_receipts,
            "manifest_path": manifest_path,
            "media_paths": media_paths,
            "caption": caption,
            "hashtags": hashtags_str,
            "platforms": platforms or list(results.keys()),
            "dispatch_plan": dispatch_plan,
            "publish_mode": publish_mode,
        }

    @staticmethod
    def _write_publish_manifest(
        *,
        payload: dict[str, object],
        media_paths: list[str],
        platforms: list[str],
        publish_mode: str,
        results: dict[str, bool],
        errors: dict[str, str],
        receipts: dict[str, dict[str, object]],
    ) -> str | None:
        manifest_dir = str(payload.get("manifest_dir") or "").strip()
        if not manifest_dir:
            return None
        output_dir = Path(manifest_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "publish_manifest.json"
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "publish_mode": publish_mode,
            "platforms": platforms,
            "media_paths": media_paths,
            "results": results,
            "errors": errors,
            "publish_receipts": receipts,
            "status": "success" if not errors else "partial_failure",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(manifest_path)

    @staticmethod
    def _safe_poc_params(publish_mode: str) -> dict[str, object]:
        if publish_mode != "safe_poc":
            return {}
        return {
            "youtube_privacy_status": "private",
            "facebook_video_state": "DRAFT",
            "facebook_use_reels": True,
            "instagram_publish_mode": "container_only",
        }

    @staticmethod
    def _safe_poc_platform_params(platform: str, publish_mode: str) -> dict[str, object]:
        if publish_mode != "safe_poc":
            return {}
        normalized = str(platform).lower().strip()
        if normalized in {"twitter", "x"}:
            raise RuntimeError("X publishing is disabled in safe_poc because the current X API requires paid credits")
        return {}

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
