from __future__ import annotations

import ipaddress
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlparse

from agentic.runtime.registry import ToolRegistry
from agentic.tools.publishing_adapter import (
    FACEBOOK_PROFILE_HANDOFF_PLATFORM,
    MediaPost,
    PublishingAdapter,
    build_dispatch_plan,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
HANDOFF_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", *VIDEO_EXTENSIONS}
MAX_HANDOFF_FILE_BYTES = 200 * 1024 * 1024


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
        platform_content = self._platform_content_evidence(platform_bundle)
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
                "platform_content": platform_content,
            }

        service = self._publishing_service()
        effective_platforms = platforms or list(dispatch_plan.keys())
        results: dict[str, bool] = {}
        errors: dict[str, str] = {}
        skipped_platforms: dict[str, str] = {}
        publish_receipts: dict[str, dict[str, object]] = {}
        handoff_receipts: dict[str, dict[str, object]] = {}
        handoffs: dict[str, dict[str, object]] = {}
        handoff_media_paths: list[str] = []
        handoff_attachment_paths: list[str] = []
        global_additional_params = dict(payload.get("additional_params", {}) or {})
        global_additional_params.update(self._safe_poc_params(publish_mode))
        for platform in effective_platforms:
            platform_plan = dispatch_plan.get(platform, {})
            platform_media_paths = [str(path) for path in platform_plan.get("media_paths", media_paths)]
            if platform == "youtube" and not any(
                Path(path).suffix.lower() in VIDEO_EXTENSIONS for path in platform_media_paths
            ):
                skipped_platforms[platform] = "requires_video_media"
                continue
            platform_validation = platform_plan.get("validation", {})
            if (
                isinstance(platform_validation, dict)
                and platform_validation.get("is_platform_publish_ready") is False
            ):
                validation_issues = platform_validation.get("issues", [])
                issue_text = (
                    ",".join(str(issue) for issue in validation_issues if str(issue).strip())
                    if isinstance(validation_issues, list)
                    else ""
                )
                skipped_platforms[platform] = (
                    "platform_validation_failed"
                    + (f":{issue_text}" if issue_text else "")
                )
                continue
            platform_bundle_entry = platform_bundle.get(platform, {}) if isinstance(platform_bundle, dict) else {}
            bundle_additional_params = dict(
                platform_bundle_entry.get("additional_params", {}) or {}
            )
            if platform_bundle_entry.get("metadata_source") == "derived":
                # Explicit node parameters override generated defaults.
                platform_additional_params = {
                    **bundle_additional_params,
                    **global_additional_params,
                }
            else:
                # Preserve the existing explicit platform-bundle override path.
                platform_additional_params = {
                    **global_additional_params,
                    **bundle_additional_params,
                }
            content_strategy = platform_bundle_entry.get("content_strategy", {})
            if (
                platform == "youtube"
                and platform_bundle_entry.get("metadata_source") == "derived"
                and isinstance(content_strategy, dict)
                and content_strategy.get("synthetic_media_disclosed") is True
            ):
                # Generated-media disclosure is an intentional safety invariant.
                platform_additional_params["youtube_contains_synthetic_media"] = True
            platform_additional_params.update(self._safe_poc_params(publish_mode))
            try:
                if platform == FACEBOOK_PROFILE_HANDOFF_PLATFORM:
                    handoff = self._create_facebook_profile_handoff(
                        media_paths=platform_media_paths,
                        caption=str(platform_plan.get("caption") or caption),
                        hashtags=str(platform_plan.get("hashtags") or hashtags_str or ""),
                        additional_params=platform_additional_params,
                        manifest_dir=str(payload.get("manifest_dir") or "").strip(),
                    )
                    handoffs[platform] = handoff
                    handoff_media_paths.extend(
                        str(path) for path in handoff.get("media_paths", []) if path
                    )
                    handoff_attachment_paths.extend(
                        str(path) for path in handoff.get("media_paths", []) if path
                    )
                    handoff_path = str(handoff.get("handoff_path") or "").strip()
                    if handoff_path:
                        handoff_attachment_paths.append(handoff_path)
                    receipt = handoff.get("receipt")
                    if not isinstance(receipt, dict):
                        raise RuntimeError("Facebook Profile handoff did not create a receipt record")
                    handoff_receipts[platform] = dict(receipt)
                    results[platform] = True
                    continue
                platform_post = MediaPost(
                    media_paths=platform_media_paths,
                    caption=str(platform_plan.get("caption") or caption),
                    hashtags=str(platform_plan.get("hashtags") or hashtags_str or ""),
                    additional_params=platform_additional_params,
                )
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
        effective_platforms = [platform for platform in effective_platforms if platform not in skipped_platforms]
        publication_state, publicly_visible = self._publication_state(
            publish_mode=publish_mode,
            platforms=effective_platforms,
            results=results,
            errors=errors,
            receipts=publish_receipts,
        )
        if handoffs and not errors:
            status = "awaiting_user_action"
        else:
            status = "success" if not errors else "partial_failure"
        dispatch_status = "skipped" if skipped_platforms and not effective_platforms and not errors else status
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
            skipped_platforms=skipped_platforms,
            status=status,
            handoffs=handoffs,
            handoff_receipts=handoff_receipts,
            handoff_media_paths=list(dict.fromkeys(handoff_media_paths)),
            handoff_attachment_paths=list(dict.fromkeys(handoff_attachment_paths)),
            platform_content=platform_content,
        )
        return {
            "status": status,
            "dispatch_status": dispatch_status,
            "publication_state": publication_state,
            "publicly_visible": publicly_visible,
            "results": results,
            "errors": errors,
            "skipped_platforms": skipped_platforms,
            "publish_receipts": publish_receipts,
            "handoff_receipts": handoff_receipts,
            "manifest_path": manifest_path,
            "media_paths": media_paths,
            "caption": caption,
            "hashtags": hashtags_str,
            "platforms": effective_platforms,
            "dispatch_plan": dispatch_plan,
            "platform_content": platform_content,
            "publish_mode": publish_mode,
            "handoffs": handoffs,
            "handoff_media_paths": list(dict.fromkeys(handoff_media_paths)),
            "handoff_attachment_paths": list(dict.fromkeys(handoff_attachment_paths)),
        }

    def _create_facebook_profile_handoff(
        self,
        *,
        media_paths: list[str],
        caption: str,
        hashtags: str,
        additional_params: dict[str, object],
        manifest_dir: str,
    ) -> dict[str, object]:
        output_root = self.output_root.expanduser().resolve()
        resolved_media_paths: list[str] = []
        rejected_paths: list[str] = []
        for raw_path in media_paths:
            try:
                resolved = Path(raw_path).expanduser().resolve(strict=True)
                resolved.relative_to(output_root)
                if (
                    not resolved.is_file()
                    or resolved.suffix.lower() not in HANDOFF_MEDIA_EXTENSIONS
                    or resolved.stat().st_size > MAX_HANDOFF_FILE_BYTES
                ):
                    raise ValueError("unsupported or oversized media")
            except (OSError, RuntimeError, ValueError):
                rejected_paths.append(str(raw_path))
                continue
            resolved_media_paths.append(str(resolved))
        if rejected_paths:
            raise FileNotFoundError(
                "Facebook Profile handoff media must be a supported file under the runtime output root: "
                + ", ".join(rejected_paths[:3])
            )
        if not resolved_media_paths:
            raise ValueError("Facebook Profile handoff requires at least one media file")

        share_url = str(additional_params.get("facebook_profile_share_url") or "").strip()
        share_dialog_url = ""
        if share_url:
            self._validate_public_share_url(share_url)
            share_dialog_url = (
                "https://www.facebook.com/sharer/sharer.php?u="
                + quote(share_url, safe="")
            )

        handoff_path = ""
        if manifest_dir:
            output_dir = Path(manifest_dir)
            try:
                output_dir = output_dir.expanduser().resolve()
                output_dir.relative_to(output_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError("manifest_dir must be under the runtime output root") from exc
            output_dir.mkdir(parents=True, exist_ok=True)
            handoff_path = str(output_dir / "facebook_profile_handoff.json")
        handoff = {
            "platform": FACEBOOK_PROFILE_HANDOFF_PLATFORM,
            "status": "awaiting_user_action",
            "publication_state": "awaiting_user_action",
            "publicly_visible": False,
            "requires_human_confirmation": True,
            "receipt_required": True,
            "media_paths": resolved_media_paths,
            "caption": caption,
            "hashtags": hashtags,
            "share_url": share_url,
            "share_dialog_url": share_dialog_url,
            "handoff_path": handoff_path,
            "instructions": [
                "在手機開啟這則 Discord 推播並下載附件。",
                "在 Facebook App 開啟個人 Profile 的建立貼文畫面，附加媒體並貼上文案。",
                "確認內容後按一次發布；完成後保留實際貼文網址作為 receipt。",
            ],
        }
        handoff["notification_text"] = SocialServiceTools._facebook_profile_notification_text(handoff)
        handoff["receipt"] = {
            "platform": FACEBOOK_PROFILE_HANDOFF_PLATFORM,
            "external_id": handoff_path,
            "status": "awaiting_user_action",
            "verified": False,
            "visibility": "not_published",
            "details": {
                "handoff_prepared": True,
                "handoff_delivered": False,
                "delivery_status": "pending",
                "receipt_required": True,
                "share_dialog_url": share_dialog_url,
            },
        }
        if handoff_path:
            Path(handoff_path).write_text(
                json.dumps(handoff, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return handoff

    @staticmethod
    def _validate_public_share_url(share_url: str) -> None:
        parsed = urlparse(share_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("facebook_profile_share_url must be an https URL")
        if parsed.username or parsed.password:
            raise ValueError("facebook_profile_share_url must not contain credentials")
        try:
            hostname = str(parsed.hostname or "").strip().lower().rstrip(".")
        except ValueError as exc:
            raise ValueError("facebook_profile_share_url must use a valid hostname") from exc
        try:
            ip_address = ipaddress.ip_address(hostname)
        except ValueError:
            ip_address = None
        if (
            not hostname
            or hostname in {"localhost", "localhost.localdomain"}
            or hostname.endswith((".local", ".internal"))
        ):
            raise ValueError("facebook_profile_share_url must use a public hostname")
        if ip_address is not None and (
            ip_address.is_private
            or ip_address.is_loopback
            or ip_address.is_link_local
            or ip_address.is_reserved
            or ip_address.is_multicast
            or ip_address.is_unspecified
        ):
            raise ValueError("facebook_profile_share_url must use a public host")
        sensitive_query_keys = {
            "access_token",
            "api_key",
            "client_secret",
            "password",
            "refresh_token",
            "secret",
            "token",
        }
        if any(
            str(key).strip().lower() in sensitive_query_keys
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            raise ValueError("facebook_profile_share_url must not contain credentials")

    @staticmethod
    def _facebook_profile_notification_text(handoff: dict[str, object]) -> str:
        lines = [
            "Facebook Profile handoff：等待一次人工確認，尚未發布。",
            *[str(item) for item in handoff.get("instructions", [])],
        ]
        share_dialog_url = str(handoff.get("share_dialog_url") or "").strip()
        if share_dialog_url:
            lines.append(f"分享對話框：{share_dialog_url}")
        caption = str(handoff.get("caption") or "").strip()
        hashtags = str(handoff.get("hashtags") or "").strip()
        if caption:
            lines.append(f"Caption:\n{caption}")
        if hashtags:
            lines.append(hashtags)
        return "\n".join(lines)

    @staticmethod
    def _platform_content_evidence(platform_bundle: object) -> dict[str, dict[str, object]]:
        if not isinstance(platform_bundle, dict):
            return {}
        evidence: dict[str, dict[str, object]] = {}
        for platform, raw_entry in platform_bundle.items():
            if not isinstance(raw_entry, dict):
                continue
            strategy = raw_entry.get("content_strategy", {})
            validation = raw_entry.get("validation", {})
            additional_params = raw_entry.get("additional_params", {})
            if not isinstance(strategy, dict):
                strategy = {}
            if not isinstance(validation, dict):
                validation = {}
            if not isinstance(additional_params, dict):
                additional_params = {}
            safe_metadata: dict[str, object] = {}
            for key in (
                "youtube_title",
                "youtube_description",
                "youtube_tags",
                "youtube_contains_synthetic_media",
                "facebook_use_reels",
                "facebook_title",
            ):
                if key in additional_params:
                    safe_metadata[key] = additional_params[key]
            evidence[str(platform)] = {
                "format": str(raw_entry.get("format") or ""),
                "caption": str(raw_entry.get("caption") or ""),
                "hashtags": str(raw_entry.get("hashtags") or ""),
                "content_strategy": dict(strategy),
                "validation": dict(validation),
                "derived_metadata": safe_metadata,
            }
        return evidence

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
        skipped_platforms: dict[str, str],
        status: str,
        handoffs: dict[str, dict[str, object]],
        handoff_receipts: dict[str, dict[str, object]],
        handoff_media_paths: list[str],
        handoff_attachment_paths: list[str],
        platform_content: dict[str, dict[str, object]],
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
            "status": status,
            "skipped_platforms": skipped_platforms,
            "publication_state": publication_state,
            "publicly_visible": publicly_visible,
            "handoffs": handoffs,
            "handoff_receipts": handoff_receipts,
            "handoff_media_paths": handoff_media_paths,
            "handoff_attachment_paths": handoff_attachment_paths,
            "platform_content": platform_content,
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
        if FACEBOOK_PROFILE_HANDOFF_PLATFORM in platforms and not errors:
            non_handoff_platforms = [
                platform for platform in platforms if platform != FACEBOOK_PROFILE_HANDOFF_PLATFORM
            ]
            if public_count and non_handoff_platforms:
                return "partially_published_awaiting_user_action", False
            return "awaiting_user_action", False
        if errors:
            if public_count:
                return "partially_published", False
            return "failed", False
        if not platforms:
            return "not_applicable", False
        if not all(results.get(platform, False) for platform in platforms):
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


def record_facebook_profile_handoff_delivery(
    handoff_path: str,
    notification: dict[str, object],
    *,
    expected_attachment_count: int,
) -> dict[str, object]:
    path = Path(handoff_path).expanduser().resolve(strict=True)
    handoff = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(handoff, dict) or handoff.get("platform") != FACEBOOK_PROFILE_HANDOFF_PLATFORM:
        raise ValueError("Not a Facebook Profile handoff artifact")
    notification_status = str(notification.get("status") or "unknown").strip().lower()
    message_id = str(notification.get("message_id") or "").strip()
    attachment_count = int(notification.get("attachment_count") or 0)
    delivered = bool(
        notification_status == "sent"
        and message_id
        and attachment_count >= max(0, int(expected_attachment_count))
    )
    delivery = {
        "status": "sent" if delivered else notification_status,
        "delivered": delivered,
        "message_id": message_id,
        "attachment_count": attachment_count,
        "expected_attachment_count": max(0, int(expected_attachment_count)),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    handoff["delivery"] = delivery
    receipt = handoff.get("receipt")
    if isinstance(receipt, dict):
        details = receipt.get("details")
        if not isinstance(details, dict):
            details = {}
        details.update(
            {
                "handoff_delivered": delivered,
                "delivery_status": delivery["status"],
                "message_id": message_id,
            }
        )
        receipt["details"] = details
        handoff["receipt"] = receipt
    path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")

    publish_manifest_path = path.parent / "publish_manifest.json"
    if publish_manifest_path.exists():
        manifest = json.loads(publish_manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            handoffs = manifest.get("handoffs")
            if not isinstance(handoffs, dict):
                handoffs = {}
            handoffs[FACEBOOK_PROFILE_HANDOFF_PLATFORM] = handoff
            manifest["handoffs"] = handoffs
            handoff_receipts = manifest.get("handoff_receipts")
            if not isinstance(handoff_receipts, dict):
                handoff_receipts = {}
            if isinstance(receipt, dict):
                handoff_receipts[FACEBOOK_PROFILE_HANDOFF_PLATFORM] = receipt
            manifest["handoff_receipts"] = handoff_receipts
            manifest["handoff_delivery"] = delivery
            publish_manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    return {
        "handoff": handoff,
        "delivery": delivery,
        "handoff_path": str(path),
        "publish_manifest_path": str(publish_manifest_path) if publish_manifest_path.exists() else "",
    }


def complete_facebook_profile_handoff(handoff_path: str, post_url: str) -> dict[str, object]:
    path = Path(handoff_path).expanduser().resolve(strict=True)
    normalized_post_url = str(post_url or "").strip()
    SocialServiceTools._validate_public_share_url(normalized_post_url)
    parsed = urlparse(normalized_post_url)
    hostname = str(parsed.hostname or "").lower().rstrip(".")
    if hostname not in {"facebook.com", "fb.watch"} and not hostname.endswith(".facebook.com"):
        raise ValueError("facebook_profile_post_url must be a Facebook URL")
    handoff = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(handoff, dict) or handoff.get("platform") != FACEBOOK_PROFILE_HANDOFF_PLATFORM:
        raise ValueError("Not a Facebook Profile handoff artifact")
    completed_at = datetime.now(timezone.utc).isoformat()
    handoff["status"] = "published"
    handoff["publication_state"] = "published"
    handoff["publicly_visible"] = True
    handoff["post_url"] = normalized_post_url
    handoff["completed_at"] = completed_at
    handoff["receipt"] = {
        "platform": FACEBOOK_PROFILE_HANDOFF_PLATFORM,
        "external_id": normalized_post_url,
        "status": "published",
        "verified": True,
        "visibility": "public",
        "details": {
            "source": "manual_profile_confirmation",
            "confirmed_at": completed_at,
        },
    }
    path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")

    publish_manifest_path = path.parent / "publish_manifest.json"
    if publish_manifest_path.exists():
        manifest = json.loads(publish_manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            handoffs = manifest.get("handoffs")
            if not isinstance(handoffs, dict):
                handoffs = {}
            handoffs[FACEBOOK_PROFILE_HANDOFF_PLATFORM] = handoff
            manifest["handoffs"] = handoffs
            handoff_receipts = manifest.get("handoff_receipts")
            if not isinstance(handoff_receipts, dict):
                handoff_receipts = {}
            handoff_receipts[FACEBOOK_PROFILE_HANDOFF_PLATFORM] = handoff["receipt"]
            manifest["handoff_receipts"] = handoff_receipts
            platforms = [str(item) for item in manifest.get("platforms", [])]
            non_handoff = [
                platform for platform in platforms if platform != FACEBOOK_PROFILE_HANDOFF_PLATFORM
            ]
            publish_receipts = manifest.get("publish_receipts")
            if not isinstance(publish_receipts, dict):
                publish_receipts = {}
            results = manifest.get("results")
            if not isinstance(results, dict):
                results = {}
            all_public = all(
                bool(results.get(platform, False))
                and SocialServiceTools._receipt_is_public(
                    publish_receipts.get(platform) if isinstance(publish_receipts, dict) else None
                )
                for platform in non_handoff
            )
            if not non_handoff or all_public:
                manifest["status"] = "success"
                manifest["dispatch_status"] = "success"
                manifest["publication_state"] = "published"
                manifest["publicly_visible"] = True
            else:
                manifest["status"] = "success"
                manifest["dispatch_status"] = "success"
                manifest["publication_state"] = "staged"
                manifest["publicly_visible"] = False
            publish_manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    return {
        "status": "published",
        "publication_state": "published",
        "publicly_visible": True,
        "handoff_path": str(path),
        "publish_manifest_path": str(publish_manifest_path) if publish_manifest_path.exists() else "",
        "post_url": normalized_post_url,
    }


def register_social_service_tools(tool_registry: ToolRegistry, output_root: Path) -> None:
    tools = SocialServiceTools(output_root=output_root)
    tool_registry.register("publish.process_media", tools.process_media, "Prepare media files for social publishing")
    tool_registry.register("publish.social", tools.publish_social, "Publish prepared media to configured social platforms")
