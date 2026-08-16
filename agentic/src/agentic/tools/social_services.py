from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentic.runtime.registry import ToolRegistry
from agentic.tools.publishing_adapter import MediaPost, PublishingAdapter, build_dispatch_plan


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
        dispatch_plan = build_dispatch_plan(
            media_paths=media_paths,
            caption=caption,
            hashtags=hashtags_str or "",
            platforms=platforms,
            platform_bundle=platform_bundle if isinstance(platform_bundle, dict) else {},
        )
        if dry_run:
            return {
                "status": "dry_run",
                "publication_state": "not_attempted",
                "publicly_visible": False,
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
        effective_platforms = platforms or list(results.keys())
        publication_state, publicly_visible = self._publication_state(
            publish_mode=publish_mode,
            platforms=effective_platforms,
            results=results,
            errors=errors,
            receipts=publish_receipts,
        )
        manifest_path = self._write_publish_manifest(
            payload=payload,
            media_paths=media_paths,
            platforms=effective_platforms,
            publish_mode=publish_mode,
            results=results,
            errors=errors,
            receipts=publish_receipts,
            publication_state=publication_state,
            publicly_visible=publicly_visible,
        )
        return {
            "status": "success" if not errors else "partial_failure",
            "publication_state": publication_state,
            "publicly_visible": publicly_visible,
            "results": results,
            "errors": errors,
            "publish_receipts": publish_receipts,
            "manifest_path": manifest_path,
            "media_paths": media_paths,
            "caption": caption,
            "hashtags": hashtags_str,
            "platforms": effective_platforms,
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
        publication_state: str,
        publicly_visible: bool,
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
            "publication_state": publication_state,
            "publicly_visible": publicly_visible,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(manifest_path)

    @classmethod
    def _publication_state(
        cls,
        *,
        publish_mode: str,
        platforms: list[str],
        results: dict[str, bool],
        errors: dict[str, str],
        receipts: dict[str, dict[str, object]],
    ) -> tuple[str, bool]:
        """Separate transport success from actual public visibility.

        A platform API can accept an upload while leaving it private, as a
        draft, or as an unpublished Instagram container.  Those states must
        never be reported as a public publication.
        """
        public_count = sum(
            1 for platform in platforms if cls._receipt_is_public(receipts.get(platform))
        )
        if errors:
            if public_count:
                return "partially_published", False
            return "failed", False
        if not platforms or not all(results.get(platform, False) for platform in platforms):
            return "failed", False
        if publish_mode == "safe_poc":
            return "staged", False
        if len(receipts) < len(platforms):
            return "unknown", False
        if public_count == len(platforms):
            return "published", True
        if any(cls._receipt_is_non_public(receipts.get(platform)) for platform in platforms):
            return "staged", False
        return "uploaded_unverified", False

    @staticmethod
    def _receipt_is_public(receipt: dict[str, object] | None) -> bool:
        if not isinstance(receipt, dict) or not bool(receipt.get("verified")):
            return False
        visibility = str(receipt.get("visibility") or "").strip().lower()
        status = str(receipt.get("status") or "").strip().lower()
        return visibility in {"public", "published"} and status not in {
            "container_ready",
            "draft_ready",
            "uploaded_unverified",
        }

    @classmethod
    def _receipt_is_non_public(cls, receipt: dict[str, object] | None) -> bool:
        if not isinstance(receipt, dict):
            return False
        if cls._receipt_is_public(receipt):
            return False
        visibility = str(receipt.get("visibility") or "").strip().lower()
        status = str(receipt.get("status") or "").strip().lower()
        return visibility in {"private", "draft", "container_only", "container_ready"} or status in {
            "draft_ready",
            "container_ready",
        }

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

def register_social_service_tools(tool_registry: ToolRegistry, output_root: Path) -> None:
    tools = SocialServiceTools(output_root=output_root)
    tool_registry.register("publish.process_media", tools.process_media, "Prepare media files for social publishing")
    tool_registry.register("publish.social", tools.publish_social, "Publish prepared media to configured social platforms")
