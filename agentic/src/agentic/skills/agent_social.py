from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import (
    DEFAULT_DISCORD_REVIEW_TIMEOUT_SECONDS,
    SkillContext,
    SkillResult,
)
from agentic.runtime.platform_content import (
    build_platform_bundle,
    enrich_publish_bundle,
    sanitize_hashtags,
)
from agentic.runtime.post_strategy import resolve_post_strategy
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.serial_video import store_from_constraints
from agentic.tools.context_services import DiscordHumanReviewService
from agentic.tools.publishing_adapter import build_dispatch_plan, merge_platform_delivery_variants

VIDEO_REVIEW_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_REVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class AgentSocialSkills:
    def __init__(
        self,
        tools: ToolRegistry,
        output_root: Path,
        prompt_engine: LLMPromptEngine | None = None,
        discord_review: Any | None = None,
    ) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.prompt_engine = prompt_engine or LLMPromptEngine(mode="template")
        self.discord_review = discord_review or DiscordHumanReviewService(output_root=self.output_root)

    def prepare_caption(self, context: SkillContext) -> SkillResult:
        prefix = str(context.node.inputs.get("prefix", "")).strip()
        serial_mode = bool(context.plan.goal.constraints.get("serial_mode"))
        if serial_mode:
            prefix = ""
        hashtags = context.node.inputs.get("hashtags") or context.plan.goal.constraints.get("hashtags") or []
        platforms = [str(platform) for platform in (context.plan.goal.constraints.get("platforms") or context.node.inputs.get("platforms") or [])]
        selected_media = self._collect_media_from_dependencies(context)
        review_select = context.state[context.node.depends_on[0]] if context.node.depends_on else {}
        approved_review_text = str(review_select.get("approved_review_text") or "").strip()
        edited_review_text = str(review_select.get("edited_review_text") or "").strip()
        review_text = approved_review_text or edited_review_text
        post_strategy = resolve_post_strategy(context.plan.goal, selected_media)
        if review_text:
            caption_override, hashtags_override = self._split_review_caption_and_hashtags(review_text)
            if not caption_override:
                raise ValueError("Approved review text did not contain a caption body.")
            normalized_hashtags = sanitize_hashtags(hashtags_override)
            platform_captions = {
                str(platform): caption_override
                for platform in platforms
            }
            bundle = {
                "caption": caption_override,
                "hashtags": normalized_hashtags,
                "platform_captions": platform_captions,
                "platform_bundle": build_platform_bundle(
                    goal=context.plan.goal,
                    caption=caption_override,
                    hashtags=normalized_hashtags,
                    platform_captions=platform_captions,
                    platforms=platforms,
                    media_paths=selected_media,
                    post_strategy=post_strategy,
                ),
                "caption_strategy": "human_approved_review",
                "prompt_mode": "human_approved_review",
                "dispatch_ready": bool(selected_media),
            }
        else:
            bundle = self.prompt_engine.prepare_publish_caption(
                context.plan.goal,
                prefix=prefix,
                hashtags=[str(hashtags)] if isinstance(hashtags, str) else [str(tag) for tag in hashtags],
                platforms=platforms,
                media_paths=selected_media,
                review_notes=str(context.plan.goal.constraints.get("review_notes", "") or ""),
                visual_paths=self._visual_evidence_paths(
                    selected_media,
                    visual_grounding=context.plan.goal.constraints.get("visual_grounding"),
                ),
            )
            bundle = enrich_publish_bundle(bundle, context.plan.goal, platforms, selected_media)
        return SkillResult(
            status="success",
            outputs={
                "caption": str(bundle["caption"]),
                "hashtags": str(bundle["hashtags"]),
                "platform_captions": dict(bundle.get("platform_captions", {})),
                "platform_bundle": dict(bundle.get("platform_bundle", {})),
                "post_strategy": dict(bundle.get("post_strategy") or post_strategy),
                "caption_strategy": str(bundle.get("caption_strategy", "generic")),
                "dispatch_ready": bool(bundle.get("dispatch_ready", False)),
                "prompt_mode": str(bundle.get("prompt_mode", "template")),
                "selected_assets": selected_media,
                "approved_review_text": approved_review_text,
                "edited_review_text": edited_review_text,
                "publication_attempt_number": review_select.get("publication_attempt_number"),
            },
            logs=["Prepared a reusable social caption bundle."],
        )

    def ingest_media(self, context: SkillContext) -> SkillResult:
        constraint_paths = context.node.inputs.get("media_paths") or context.plan.goal.constraints.get("media_paths") or []
        if isinstance(constraint_paths, str):
            media_paths = [constraint_paths]
        else:
            media_paths = [str(path) for path in constraint_paths if path]
        if not media_paths:
            input_dir = str(context.plan.goal.constraints.get("input_dir") or "").strip()
            if input_dir:
                root = Path(input_dir)
                allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov"}
                media_paths = [str(path) for path in sorted(root.iterdir()) if path.is_file() and path.suffix.lower() in allowed_suffixes]
        return SkillResult(
            status="success",
            outputs={"media_paths": media_paths, "media_count": len(media_paths)},
            metrics={"media_count": len(media_paths)},
            logs=["Ingested publish/review candidate media from goal constraints."],
        )

    def collect_media(self, context: SkillContext) -> SkillResult:
        media_paths: list[str] = []
        for dependency in context.node.depends_on:
            dependency_output = context.state[dependency]
            for key in ("saved_files", "video_path", "gif_path", "frame_path"):
                value = dependency_output.get(key)
                if isinstance(value, list):
                    media_paths.extend(str(item) for item in value if item)
                elif isinstance(value, str) and value:
                    media_paths.append(value)
        deduped = list(dict.fromkeys(media_paths))
        return SkillResult(
            status="success",
            outputs={"media_paths": deduped, "media_count": len(deduped)},
            metrics={"media_count": len(deduped)},
            logs=["Collected publishable media artifacts."],
        )

    def process_media(self, context: SkillContext) -> SkillResult:
        media_paths = context.node.inputs.get("media_paths") or context.state[context.node.depends_on[0]].get("media_paths", [])
        result = self.tools.call(
            "publish.process_media",
            {
                "media_paths": media_paths,
                "output_dir": context.node.inputs.get("output_dir"),
                "platforms": context.node.inputs.get("platforms")
                or context.plan.goal.constraints.get("platforms")
                or [],
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Prepared media artifacts for social publishing."])

    def publish_social(self, context: SkillContext) -> SkillResult:
        processed = context.state[context.node.depends_on[0]] if context.node.depends_on else {}
        caption_bundle = context.state[context.node.depends_on[1]] if len(context.node.depends_on) > 1 else {}
        platform_captions = caption_bundle.get("platform_captions", {})
        raw_platform_bundle = caption_bundle.get("platform_bundle", {})
        platform_bundle = merge_platform_delivery_variants(
            raw_platform_bundle if isinstance(raw_platform_bundle, dict) else {},
            processed.get("delivery_variants", {}),
        )
        requested_platforms = context.node.inputs.get("platforms") or context.plan.goal.constraints.get("platforms") or []
        platforms = [str(platform) for platform in requested_platforms]
        dispatch_plan = build_dispatch_plan(
            media_paths=context.node.inputs.get("media_paths") or processed.get("media_paths", []),
            caption=context.node.inputs.get("caption") or caption_bundle.get("caption", ""),
            hashtags=context.node.inputs.get("hashtags") or caption_bundle.get("hashtags", ""),
            platforms=platforms,
            platform_bundle=platform_bundle,
        )
        dry_run = bool(context.node.inputs.get("dry_run", False))
        blocked_platforms = [
            platform
            for platform, payload in dispatch_plan.items()
            if not bool(payload.get("validation", {}).get("is_publish_ready", False))
        ]
        platform_ineligible = [
            platform
            for platform, payload in dispatch_plan.items()
            if not bool(payload.get("validation", {}).get("is_platform_publish_ready", True))
        ]
        platform_validation_failures = [
            platform
            for platform in platform_ineligible
            if not (
                isinstance(dispatch_plan.get(platform, {}).get("validation", {}), dict)
                and "requires_video_media"
                in dispatch_plan.get(platform, {}).get("validation", {}).get("issues", [])
            )
        ]
        blocked_platforms = list(dict.fromkeys([*blocked_platforms, *platform_validation_failures]))
        serial_store = store_from_constraints(context.plan.goal.constraints)
        review_select = context.state[context.node.depends_on[1]] if len(context.node.depends_on) > 1 else {}
        publication_attempt_number = review_select.get("publication_attempt_number") or caption_bundle.get(
            "publication_attempt_number"
        )
        if blocked_platforms and not dry_run:
            if serial_store is not None and publication_attempt_number:
                serial_store.record_publication_dispatch(
                    str(context.plan.goal.constraints.get("serial_episode_id") or ""),
                    int(publication_attempt_number),
                    results={platform: False for platform in platforms},
                    receipts={},
                    errors={platform: "dispatch readiness validation failed" for platform in platforms},
                    dispatch_status="blocked",
                )
            return SkillResult(
                status="blocked",
                outputs={
                    "status": "blocked",
                    "media_paths": context.node.inputs.get("media_paths") or processed.get("media_paths", []),
                    "caption": context.node.inputs.get("caption") or caption_bundle.get("caption", ""),
                    "hashtags": context.node.inputs.get("hashtags") or caption_bundle.get("hashtags", ""),
                    "platforms": platforms or list(dispatch_plan.keys()),
                    "platform_bundle": platform_bundle,
                    "dispatch_plan": dispatch_plan,
                    "dispatch_ready": False,
                    "blocked_platforms": blocked_platforms,
                    "platform_ineligible": platform_ineligible,
                    "blocked_reason": "publish bundle is not dispatch-ready for every requested platform",
                },
                logs=["Blocked social dispatch because one or more platforms failed publish-readiness validation."],
            )
        try:
            if serial_store is not None:
                if not publication_attempt_number:
                    raise RuntimeError("Serial social dispatch has no approved publication attempt.")
                serial_store.assert_publication_media_current(
                    str(context.plan.goal.constraints.get("serial_episode_id") or ""),
                    int(publication_attempt_number),
                )
            result = self.tools.call(
                "publish.social",
                {
                "media_paths": context.node.inputs.get("media_paths") or processed.get("media_paths", []),
                "caption": context.node.inputs.get("caption") or caption_bundle.get("caption", ""),
                "hashtags": context.node.inputs.get("hashtags") or caption_bundle.get("hashtags", ""),
                "platform_captions": platform_captions if isinstance(platform_captions, dict) else {},
                "platforms": platforms,
                "platform_configs": context.node.inputs.get("platform_configs", {}),
                "additional_params": context.node.inputs.get("additional_params", {}),
                "publish_mode": str(context.node.inputs.get("publish_mode") or context.plan.goal.constraints.get("publish_mode") or ""),
                "dry_run": dry_run,
                "platform_bundle": platform_bundle,
                "delivery_variants": processed.get("delivery_variants", {}),
                "manifest_dir": str(self.output_root),
                },
            )
        except Exception as exc:
            if serial_store is not None and publication_attempt_number:
                serial_store.record_publication_dispatch(
                    str(context.plan.goal.constraints.get("serial_episode_id") or ""),
                    int(publication_attempt_number),
                    results={platform: False for platform in platforms},
                    receipts={},
                    errors={platform: f"{type(exc).__name__}: {exc}" for platform in platforms},
                    dispatch_status="failed",
                )
            raise
        outputs = dict(result)
        outputs["platform_bundle"] = platform_bundle
        outputs["delivery_variants"] = processed.get("delivery_variants", {})
        outputs["dispatch_plan"] = dispatch_plan
        outputs["platform_ineligible"] = platform_ineligible
        outputs["dispatch_ready"] = (
            bool(caption_bundle.get("dispatch_ready", False))
            and not blocked_platforms
            and not platform_ineligible
        )
        dispatch_status = str(outputs.get("status") or "").strip().lower()
        if serial_store is not None and publication_attempt_number:
            result_map = outputs.get("results") if isinstance(outputs.get("results"), dict) else {}
            receipt_map = (
                outputs.get("publish_receipts")
                if isinstance(outputs.get("publish_receipts"), dict)
                else {}
            )
            error_map = outputs.get("errors") if isinstance(outputs.get("errors"), dict) else {}
            serial_store.record_publication_dispatch(
                str(context.plan.goal.constraints.get("serial_episode_id") or ""),
                int(publication_attempt_number),
                results=result_map,
                receipts=receipt_map,
                errors=error_map,
                dispatch_status=dispatch_status,
            )
        skill_status = "success" if dispatch_status in {"success", "dry_run", "awaiting_user_action"} else "failed"
        return SkillResult(status=skill_status, outputs=outputs, logs=["Dispatched a social publishing action."])

    def select_best_assets(self, context: SkillContext) -> SkillResult:
        preferred_video_node = str(context.node.inputs.get("preferred_video_node") or "").strip()
        preferred_video_path = self._preferred_video_path(context, preferred_video_node)
        explicit_media_paths = context.node.inputs.get("media_paths") or []
        if isinstance(explicit_media_paths, str):
            explicit_media_paths = [explicit_media_paths]
        media_paths = [str(path) for path in explicit_media_paths if path]
        if not media_paths:
            media_paths = (
                [preferred_video_path]
                if preferred_video_path
                else self._collect_media_from_dependencies(context)
            )
        limit = int(context.node.inputs.get("limit", 10))
        review_scope = str(
            context.node.inputs.get("review_scope")
            or context.plan.goal.constraints.get("review_scope")
            or self._infer_review_scope(media_paths)
        ).strip().lower()
        review_phase = str(context.node.inputs.get("review_phase") or "").strip().lower()
        segment_acceptance_review = review_phase == "segment_acceptance"
        first_frame_review = review_scope == "first_frame" or review_phase == "opening_frame"
        last_frame_review = review_scope == "last_frame"
        anchor_set_review = review_scope in {"anchor_set", "anchors"} or review_phase == "anchor_set"
        reference_review = review_scope == "reference" or review_phase == "reference_selection"
        final_video_review = review_scope == "final_video"
        final_media_review = review_scope == "final_media"
        auto_select_for_probe = bool(context.node.inputs.get("auto_select_for_probe", False))
        if final_video_review:
            media_paths = [path for path in media_paths if Path(path).suffix.lower() in VIDEO_REVIEW_EXTENSIONS]
        elif final_media_review:
            media_paths = [
                path
                for path in media_paths
                if Path(path).suffix.lower() in (VIDEO_REVIEW_EXTENSIONS | IMAGE_REVIEW_EXTENSIONS)
            ]
        review_all_candidates = bool(context.node.inputs.get("review_all_candidates", False))
        # The publish boundary is an approval gate for every media type. A
        # caller may omit the flag when constructing a publish_review goal,
        # but that must never turn into automatic social dispatch.
        require_human_review = bool(
            context.plan.goal.media_type == "publish_review"
            or context.node.inputs.get("require_human_review", False)
            or context.plan.goal.constraints.get("require_human_review", False)
        )
        human_review_enabled = bool(
            not auto_select_for_probe
            and (
                context.plan.goal.constraints.get("enable_stage_review", False)
                or context.plan.goal.constraints.get("enable_review_loop", False)
                or require_human_review
            )
        )
        preferred_extensions = tuple(context.node.inputs.get("preferred_extensions", [".mp4", ".gif", ".png", ".jpg", ".jpeg", ".webp"]))
        ranked = sorted(
            media_paths,
            key=lambda path: (0 if path.lower().endswith(preferred_extensions) else 1, path),
        )
        heuristic_ranked = [
            {
                "media_path": path,
                "rationale": f"Preferred extensions {preferred_extensions} with deterministic path ordering.",
            }
            for index, path in enumerate(ranked)
        ]
        caption_review_notes = str(
            context.node.inputs.get("review_notes")
            or context.plan.goal.constraints.get("review_notes")
            or ""
        )
        review_notes = caption_review_notes
        if final_video_review and not review_notes.strip():
            review_notes = (
                "Final video review: inspect the rendered video for a clear opening conflict, visible escalation or reversal, "
                "and a concrete ending. Accept to approve, Edit to update the caption, or Reject to stop publishing."
            )
        if final_media_review and not review_notes.strip():
            review_notes = (
                "Final media review: inspect every attached image or video, choose the media to publish, and verify the "
                "prepared caption and hashtags. Accept to approve, Edit to update the caption, or Reject to stop publishing."
            )
        if (first_frame_review or last_frame_review or anchor_set_review) and not auto_select_for_probe:
            bundle = {
                "selected_assets": ranked[:limit],
                "ranked_candidates": heuristic_ranked,
                "selection_rationale": (
                    f"{len(ranked)} opening-frame candidates are waiting for mandatory human selection."
                    if first_frame_review
                    else "The generated ending frame is waiting for mandatory human approval."
                    if last_frame_review
                    else "The first/last anchor set is waiting for explicit paired selection."
                ),
                "prompt_mode": (
                    "human_first_frame"
                    if first_frame_review
                    else "human_last_frame"
                    if last_frame_review
                    else "human_anchor_set"
                ),
            }
        elif final_video_review:
            bundle = {
                "selected_assets": ranked[:limit],
                "ranked_candidates": heuristic_ranked,
                "selection_rationale": "Final-video review is restricted to playable video artifacts.",
                "prompt_mode": "final_video_deterministic",
            }
        elif final_media_review:
            bundle = {
                "selected_assets": ranked[:limit],
                "ranked_candidates": heuristic_ranked,
                "selection_rationale": "Final-media review keeps all generated media available for explicit Discord selection.",
                "prompt_mode": "final_media_deterministic",
            }
        else:
            bundle = {
                "selected_assets": ranked[:limit],
                "ranked_candidates": heuristic_ranked,
                "selection_rationale": "All available candidates are presented; Discord owns creative selection.",
                "prompt_mode": "human_review",
            }
        selected = [path for path in bundle.get("selected_assets", []) if path in ranked][:limit]
        if not selected:
            selected = ranked[:limit]
        review_media_paths = ranked if (review_all_candidates or final_video_review or final_media_review) else (selected or ranked[:limit])
        caption_media_paths = review_media_paths if (review_all_candidates or final_video_review or final_media_review) else (selected or ranked[:limit])
        hashtags = context.plan.goal.constraints.get("hashtags") or []
        platforms = [str(platform) for platform in (context.plan.goal.constraints.get("platforms") or [])]
        if auto_select_for_probe or segment_acceptance_review:
            caption_bundle = {"caption": context.plan.goal.prompt, "hashtags": ""}
        else:
            caption_bundle = self.prompt_engine.prepare_publish_caption(
                context.plan.goal,
                prefix=(
                    ""
                    if bool(context.plan.goal.constraints.get("serial_mode"))
                    else str(
                        context.node.inputs.get("caption_prefix")
                        or context.plan.goal.constraints.get("caption_prefix")
                        or ""
                    )
                ),
                hashtags=[str(hashtags)] if isinstance(hashtags, str) else [str(tag) for tag in hashtags],
                platforms=platforms,
                media_paths=caption_media_paths,
                review_notes=caption_review_notes,
                visual_paths=self._visual_evidence_paths(
                    caption_media_paths,
                    visual_grounding=context.plan.goal.constraints.get("visual_grounding"),
                ),
            )
        review_metadata = self._review_metadata(context)
        review_context_text = ""
        if review_phase != "serial_final_publish":
            review_context_text = self._build_review_context_text(
                context=context,
                strategy=review_metadata["strategy"],
                workflow=review_metadata["workflow"],
                review_phase=review_phase,
                review_scope=review_scope,
            )
        if first_frame_review or last_frame_review or anchor_set_review or reference_review:
            review_text = self._build_stage_review_text(strategy=review_metadata["strategy"])
        elif final_video_review or final_media_review:
            review_text = self._build_final_publish_review_text(
                draft_caption=str(caption_bundle.get("caption", "") or context.plan.goal.prompt),
                draft_hashtags=str(caption_bundle.get("hashtags", "") or ""),
                review_guidance=(
                    review_notes if review_phase == "segment_acceptance" else ""
                ),
            )
        else:
            review_text = self._build_review_text(
                strategy=review_metadata["strategy"],
                workflow=review_metadata["workflow"],
                prompt=context.plan.goal.prompt,
                review_notes=review_notes,
                ranked_candidates=bundle.get("ranked_candidates", heuristic_ranked),
                selection_limit=limit,
                draft_caption=str(caption_bundle.get("caption", "") or context.plan.goal.prompt),
                draft_hashtags=str(caption_bundle.get("hashtags", "") or ""),
                candidate_paths=review_media_paths,
                review_scope=review_scope,
            )
        decision = None
        if human_review_enabled:
            decision = self.discord_review.review_candidates(
                text=review_text,
                context_text=review_context_text,
                media_paths=review_media_paths,
                timeout_seconds=int(
                    context.plan.goal.constraints.get(
                        "discord_review_timeout_seconds"
                    )
                    or DEFAULT_DISCORD_REVIEW_TIMEOUT_SECONDS
                ),
                allow_asset_selection=first_frame_review or last_frame_review or anchor_set_review or reference_review or final_media_review,
                allow_text_edit=not (
                    first_frame_review
                    or last_frame_review
                    or anchor_set_review
                    or reference_review
                    or segment_acceptance_review
                ),
                selection_mode="single" if (first_frame_review or last_frame_review) else "multi",
                selection_required=first_frame_review or last_frame_review or anchor_set_review or reference_review or final_media_review,
                selection_limit=(1 if (first_frame_review or last_frame_review) else (limit if (anchor_set_review or reference_review) else None)),
                review_scope=review_scope,
                allow_regenerate=bool(context.plan.goal.constraints.get("allow_regenerate_review", False)),
                allow_series_lifecycle_choice=review_phase == "serial_final_publish",
            )
        if require_human_review and (decision is None or getattr(decision, "review_mode", "") != "discord"):
            fallback_reason = str(getattr(decision, "fallback_reason", "Discord human review did not start."))
            return SkillResult(
                status="blocked",
                outputs={
                    "media_paths": [],
                    "selected_assets": [],
                    "selected_count": 0,
                    "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                    "rejected_assets": ranked,
                    "selection_rationale": "Required Discord human review was unavailable; no candidate was selected automatically.",
                    "review_mode": str(getattr(decision, "review_mode", "none")),
                    "review_scope": review_scope,
                    "fallback_reason": fallback_reason,
                },
                logs=["Blocked workflow because required Discord human review was unavailable; automatic selection is disabled."],
            )
        if decision is not None and decision.review_mode == "discord":
            if decision.status == "regeneration_requested":
                return SkillResult(
                    status="blocked",
                    outputs={
                        "media_paths": [],
                        "selected_assets": [],
                        "selected_count": 0,
                        "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                        "rejected_assets": ranked,
                        "selection_rationale": "Human reviewer requested a full end-to-end rerun in Discord.",
                        "review_decision": "regenerate",
                        "regeneration_requested": True,
                        "review_mode": decision.review_mode,
                        "review_scope": review_scope,
                        "reviewer": decision.reviewer,
                        "review_session_id": decision.session_id,
                        "review_session_path": decision.session_path,
                    },
                    logs=["Stopped the current workflow because Discord requested a fresh end-to-end run."],
                )
            if decision.status == "rejected":
                return SkillResult(
                    status="blocked",
                    outputs={
                        "media_paths": [],
                        "selected_assets": [],
                        "selected_count": 0,
                        "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                        "rejected_assets": ranked,
                        "selection_rationale": "Human reviewer rejected all candidates in Discord.",
                        "regeneration_notes": decision.edited_text or str(context.node.inputs.get("review_notes") or ""),
                        "review_mode": decision.review_mode,
                        "reviewer": decision.reviewer,
                        "review_session_id": decision.session_id,
                        "review_session_path": decision.session_path,
                        "review_delivery": dict(getattr(decision, "delivery", {}) or {}),
                        "prompt_mode": str(bundle.get("prompt_mode", "template")),
                    },
                    logs=["Blocked workflow because the Discord reviewer rejected the candidate set."],
                )
            if decision.status == "failed":
                failure_policy = (
                    "block"
                    if segment_acceptance_review
                    else str(
                        context.plan.goal.constraints.get(
                            "pre_video_review_failure_policy"
                        )
                        or "block"
                    ).strip().lower()
                )
                fallback_selected = ranked[:limit]
                if failure_policy == "fallback_to_top" and fallback_selected:
                    fallback_reason = str(getattr(decision, "fallback_reason", "") or "Discord review did not complete")
                    return SkillResult(
                        status="success",
                        outputs={
                            "media_paths": fallback_selected,
                            "selected_assets": fallback_selected,
                            "selected_count": len(fallback_selected),
                            "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                            "rejected_assets": [],
                            "selection_rationale": "Discord review did not complete; used the deterministic top-ranked candidate.",
                            "regeneration_notes": decision.edited_text or str(context.node.inputs.get("review_notes") or ""),
                            "review_mode": decision.review_mode,
                            "reviewer": decision.reviewer,
                            "review_session_id": decision.session_id,
                            "review_session_path": decision.session_path,
                            "prompt_mode": str(bundle.get("prompt_mode", "template")),
                            "fallback_reason": fallback_reason,
                            "review_fallback_used": True,
                        },
                        logs=["Discord review did not complete; continued with the deterministic top-ranked candidate."],
                    )
                return SkillResult(
                    status="blocked",
                    outputs={
                        "media_paths": [],
                        "selected_assets": [],
                        "selected_count": 0,
                        "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                        "rejected_assets": [],
                        "selection_rationale": "Discord review did not complete.",
                        "regeneration_notes": decision.edited_text or str(context.node.inputs.get("review_notes") or ""),
                        "review_mode": decision.review_mode,
                        "reviewer": decision.reviewer,
                        "review_session_id": decision.session_id,
                        "review_session_path": decision.session_path,
                        "prompt_mode": str(bundle.get("prompt_mode", "template")),
                        "fallback_reason": getattr(decision, "fallback_reason", ""),
                    },
                    logs=["Blocked workflow because Discord review did not complete successfully."],
                )
            if first_frame_review or last_frame_review:
                decision_selected = [path for path in (decision.selected_paths or []) if path in ranked]
                if len(decision_selected) != 1:
                    frame_label = "opening" if first_frame_review else "ending"
                    return SkillResult(
                        status="blocked",
                        outputs={
                            "media_paths": [],
                            "selected_assets": [],
                            "selected_count": 0,
                            "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                            "rejected_assets": ranked,
                            "selection_rationale": f"{frame_label.capitalize()}-frame review must select exactly one candidate; automatic first-item selection is disabled.",
                            "review_mode": decision.review_mode,
                            "review_scope": review_scope,
                            "reviewer": decision.reviewer,
                            "review_session_id": decision.session_id,
                            "review_session_path": decision.session_path,
                            "fallback_reason": f"Select exactly one {frame_label} frame in Discord before pressing Accept.",
                        },
                        logs=[f"Blocked workflow because {frame_label}-frame Discord review did not select exactly one candidate."],
                    )
                selected = decision_selected[:limit]
            elif anchor_set_review:
                decision_selected = [path for path in (decision.selected_paths or []) if path in ranked]
                required_count = int(context.node.inputs.get("anchor_set_size") or 2)
                if len(decision_selected) != required_count:
                    return SkillResult(
                        status="blocked",
                        outputs={
                            "media_paths": [],
                            "selected_assets": [],
                            "selected_count": 0,
                            "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                            "rejected_assets": ranked,
                            "selection_rationale": f"Anchor-set review must select exactly {required_count} paired assets.",
                            "review_mode": decision.review_mode,
                            "review_scope": review_scope,
                            "reviewer": decision.reviewer,
                            "review_session_id": decision.session_id,
                            "review_session_path": decision.session_path,
                            "fallback_reason": f"Select exactly {required_count} paired anchor assets in Discord before pressing Accept.",
                        },
                        logs=["Blocked workflow because the anchor-set Discord review did not select the required pair."],
                    )
                selected = decision_selected
            elif reference_review:
                decision_selected = [path for path in (decision.selected_paths or []) if path in ranked]
                if not decision_selected or len(decision_selected) > limit:
                    return SkillResult(
                        status="blocked",
                        outputs={
                            "media_paths": [],
                            "selected_assets": [],
                            "selected_count": 0,
                            "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                            "rejected_assets": ranked,
                            "selection_rationale": (
                                f"Reference review must select between 1 and {limit} candidates; "
                                "the received selection was invalid."
                            ),
                            "review_mode": decision.review_mode,
                            "review_scope": review_scope,
                            "reviewer": decision.reviewer,
                            "review_session_id": decision.session_id,
                            "review_session_path": decision.session_path,
                            "fallback_reason": f"Select 1-{limit} reference assets in Discord before pressing Accept.",
                        },
                        logs=["Blocked workflow because the Discord reference selection exceeded its configured limit or was empty."],
                    )
                selected = decision_selected
            elif decision.selected_paths:
                selected = [path for path in decision.selected_paths if path in ranked][:limit]
        review_mode = str(getattr(decision, "review_mode", "automatic"))
        reviewer = str(getattr(decision, "reviewer", ""))
        review_session_id = str(getattr(decision, "session_id", ""))
        review_session_path = str(getattr(decision, "session_path", ""))
        edited_review_text = (
            str(getattr(decision, "edited_text", ""))
            if bool(getattr(decision, "edit_requested", False))
            else ""
        )
        approved_review_text = (
            str(getattr(decision, "edited_text", ""))
            if review_mode == "discord" and str(getattr(decision, "status", "")).strip().lower() == "approved"
            else ""
        )
        if (
            not approved_review_text
            and review_mode == "discord"
            and str(getattr(decision, "status", "")).strip().lower() == "approved"
        ):
            draft_caption = str(caption_bundle.get("caption", "") or context.plan.goal.prompt).strip()
            draft_hashtags = str(caption_bundle.get("hashtags", "") or "").strip()
            approved_review_text = "\n\n".join(
                part for part in (draft_caption, draft_hashtags) if part
            )
        fallback_reason = str(getattr(decision, "fallback_reason", ""))
        review_delivery = dict(getattr(decision, "delivery", {}) or {})
        rejected = [path for path in ranked if path not in selected]
        selection_logs = ["Presented candidates without automatic media-quality filtering."]
        if not human_review_enabled:
            selection_logs.append(
                "Discord review disabled for this run; selection was automatic. Use --no-review only when that is intentional."
            )
        return SkillResult(
            status="success",
            outputs={
                "media_paths": selected,
                "selected_assets": selected,
                "selected_count": len(selected),
                "media_inspection": {},
                "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                "rejected_assets": rejected,
                "rejected_asset_details": [],
                "selection_rationale": str(bundle.get("selection_rationale") or f"Preferred extensions {preferred_extensions} with deterministic path ordering."),
                "regeneration_notes": str(bundle.get("regeneration_notes") or context.node.inputs.get("review_notes") or context.plan.goal.constraints.get("review_notes", "")),
                "publish_ready": bool(bundle.get("publish_ready", bool(selected))),
                "review_notes": str(context.node.inputs.get("review_notes") or context.plan.goal.constraints.get("review_notes", "")),
                "prompt_mode": str(bundle.get("prompt_mode", "template")),
                "review_mode": review_mode,
                "reviewer": reviewer,
                "review_session_id": review_session_id,
                "review_session_path": review_session_path,
                "review_delivery": review_delivery,
                "approved_review_text": approved_review_text,
                "edited_review_text": edited_review_text,
                "review_draft_caption": str(caption_bundle.get("caption", "") or context.plan.goal.prompt),
                "review_draft_hashtags": str(caption_bundle.get("hashtags", "") or ""),
                "fallback_reason": fallback_reason,
                "auto_select_for_probe": auto_select_for_probe,
            },
            metrics={"selected_count": len(selected), "candidate_count": len(ranked), "auto_select_for_probe": auto_select_for_probe},
            logs=selection_logs,
        )

    @staticmethod
    def _story_review_summary(context: SkillContext) -> str:
        """Return the audience-facing story summary for frame review messages."""
        story_output = context.state.node_outputs.get("native-story-prompt", {})
        storyboard = story_output.get("generated_storyboard", {}) if isinstance(story_output, dict) else {}
        if not isinstance(storyboard, dict):
            storyboard = {}
        spine = storyboard.get("story_spine", {})
        if not isinstance(spine, dict):
            spine = {}
        name = str(storyboard.get("name") or "").strip()
        premise = str(spine.get("premise") or "").strip()
        objective = str(spine.get("objective") or "").strip()
        if name and premise:
            summary = f"{name}: {premise}"
        else:
            summary = premise or objective
        if not summary:
            segment_output = context.state.node_outputs.get("script-plan", {})
            segments = segment_output.get("segments", []) if isinstance(segment_output, dict) else []
            segments = [item for item in segments if isinstance(item, dict)]
            if segments:
                first = str(segments[0].get("visual") or segments[0].get("narration") or "").strip()
                last = str(segments[-1].get("end_state") or segments[-1].get("effect") or "").strip()
                if first and last:
                    summary = f"{first}；最後必須到達：{last}"
                else:
                    summary = first or last
        if not summary:
            goal_prompt = str(getattr(context.plan.goal, "prompt", "") or "").strip()
            summary = goal_prompt
        if not summary:
            summary = "未提供故事摘要"
        return textwrap.shorten(" ".join(summary.split()), width=280, placeholder="...")

    @staticmethod
    def _infer_review_scope(media_paths: list[str]) -> str:
        if media_paths and all(Path(path).suffix.lower() in VIDEO_REVIEW_EXTENSIONS for path in media_paths):
            return "final_video"
        return "final_media"

    def _visual_evidence_paths(
        self,
        media_paths: list[str],
        *,
        visual_grounding: object = None,
    ) -> list[str]:
        """Resolve image evidence for captioning, including sampled video frames.

        OpenRouter vision models receive images, not raw video containers. Reuse
        the H3 QA contact sheet when available; otherwise create one in the
        configured output root. A missing video evidence artifact is an error
        boundary, not a reason to silently switch to a text-only caption model.
        """
        grounding = visual_grounding if isinstance(visual_grounding, dict) else {}
        evidence: list[str] = []
        for media_path in media_paths:
            path = Path(media_path)
            if not path.exists():
                continue
            if path.suffix.lower() in IMAGE_REVIEW_EXTENSIONS:
                evidence.append(str(path))
                continue
            if path.suffix.lower() not in VIDEO_REVIEW_EXTENSIONS:
                continue

            candidates = []
            grounded_contact_sheet = str(grounding.get("contact_sheet_path") or "").strip()
            if grounded_contact_sheet:
                candidates.append(Path(grounded_contact_sheet))
            candidates.append(path.with_name(f"{path.stem}_contact_sheet.jpg"))
            contact_sheet = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.is_file() and self._is_safe_visual_evidence_path(candidate)
                ),
                None,
            )
            if contact_sheet is None:
                digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
                contact_sheet = self.output_root / "caption_visual_evidence" / f"{path.stem}_{digest}_contact_sheet.jpg"
                result = self.tools.call(
                    "media.video_qa",
                    {
                        "video_path": str(path),
                        "contact_sheet_path": str(contact_sheet),
                        "frame_count": 6,
                        "columns": 3,
                        "scale_width": 480,
                    },
                )
                generated = str(result.get("contact_sheet_path") or contact_sheet).strip()
                contact_sheet = Path(generated)
            if not contact_sheet.is_file() or not self._is_safe_visual_evidence_path(contact_sheet):
                raise RuntimeError(f"Visual evidence could not be created for video: {path}")
            evidence.append(str(contact_sheet))
        return list(dict.fromkeys(evidence))

    def _is_safe_visual_evidence_path(self, path: Path) -> bool:
        if path.suffix.lower() not in IMAGE_REVIEW_EXTENSIONS:
            return False
        try:
            path.resolve().relative_to(self.output_root.resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _review_metadata(context: SkillContext) -> dict[str, str]:
        constraints = context.plan.goal.constraints
        strategy = str(
            constraints.get("source_generation_type")
            or constraints.get("generation_type")
            or context.plan.goal.media_type
        ).strip()
        workflow = str(
            constraints.get("workflow_name")
            or constraints.get("selected_workflow")
            or context.plan.workflow_name
        ).strip()
        return {"strategy": strategy or "unknown", "workflow": workflow or "unknown"}

    @staticmethod
    def _build_review_context_text(
        *,
        context: SkillContext,
        strategy: str,
        workflow: str,
        review_phase: str,
        review_scope: str,
    ) -> str:
        constraints = context.plan.goal.constraints
        serial_mode = bool(constraints.get("serial_mode"))
        phase_labels = {
            "segment_acceptance": "分段驗收",
            "serial_final_publish": "最終成片・發布前審核",
            "opening_frame": "開場畫面選擇",
            "ending_frame": "結尾畫面審核",
            "reference_selection": "參考素材選擇",
            "anchor_set": "首尾畫面配對選擇",
        }
        scope_labels = {
            "first_frame": "開場畫面選擇",
            "last_frame": "結尾畫面審核",
            "anchor_set": "首尾畫面配對選擇",
            "anchors": "首尾畫面配對選擇",
            "reference": "參考素材選擇",
            "final_video": "成片審核",
            "final_media": "發布素材審核",
        }
        phase_label = phase_labels.get(
            review_phase,
            scope_labels.get(review_scope, "素材審核"),
        )
        lines = ["【審核任務】"]
        if serial_mode:
            lines.append(f"策略：連載影片（{strategy}）")
            episode_number = int(constraints.get("serial_episode_number") or 0)
            automation_mode = str(
                constraints.get("serial_automation_mode") or "manual"
            ).strip().lower()
            if automation_mode == "new":
                series_relation = "新開系列"
            elif automation_mode == "continue":
                series_relation = "接續上集"
            elif automation_mode == "resume":
                series_relation = "續跑未完成集數"
            elif episode_number > 1:
                series_relation = f"接續第 {episode_number - 1} 集"
            else:
                series_relation = "新系列首集"
            episode_label = f"第 {episode_number} 集" if episode_number > 0 else "集數未提供"
            lines.append(f"系列：{series_relation}｜{episode_label}")

            if review_phase == "segment_acceptance":
                segment_index = max(0, int(context.node.inputs.get("segment_index", 0)))
                segment_indices = {
                    int(node.inputs["segment_index"])
                    for node in context.plan.nodes
                    if str(node.inputs.get("review_phase") or "").strip().lower()
                    == "segment_acceptance"
                    and node.inputs.get("segment_index") is not None
                }
                segment_count = max(segment_indices, default=segment_index) + 1
                lines.append(f"段落：第 {segment_index + 1}/{segment_count} 段・連續性驗收")
            else:
                lines.append(f"審核：{phase_label}")

            topic = str(constraints.get("episode_topic") or "").strip()
            if topic:
                lines.append(f"主題：{textwrap.shorten(topic, width=90, placeholder='…')}")
            lines.append(f"工作流：{workflow}")
        else:
            lines.extend(
                [
                    f"策略：{strategy}",
                    f"工作流：{workflow}",
                    f"審核：{phase_label}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _build_stage_review_text(*, strategy: str) -> str:
        return f"Strategy: {strategy}\nstage: preview"

    @staticmethod
    def _build_review_text(
        *,
        strategy: str,
        workflow: str,
        prompt: str,
        review_notes: str,
        ranked_candidates: list[dict[str, object]],
        selection_limit: int,
        draft_caption: str,
        draft_hashtags: str,
        candidate_paths: list[str] | None = None,
        review_scope: str = "",
    ) -> str:
        del ranked_candidates, selection_limit, review_scope
        lines: list[str] = [f"Strategy: {strategy}", f"Workflow: {workflow}"]
        if review_notes.strip():
            lines.extend(["", review_notes.strip()])
        if candidate_paths:
            lines.extend(["", "Candidates attached in this order:"])
            lines.extend(
                f"Asset {index}: {AgentSocialSkills._candidate_label(path)}"
                for index, path in enumerate(candidate_paths, start=1)
            )
        review_body = AgentSocialSkills._format_review_post(
            caption=str(draft_caption or prompt),
            hashtags=str(draft_hashtags or ""),
        )
        if review_body:
            lines.extend(["", review_body])
        return textwrap.shorten("\n".join(lines).strip(), width=1900, placeholder="...")

    @staticmethod
    def _build_final_publish_review_text(
        *,
        draft_caption: str,
        draft_hashtags: str,
        review_guidance: str = "",
    ) -> str:
        """Build final-media review text with an optional instruction for the reviewer."""
        review_text = AgentSocialSkills._format_review_post(
            caption=draft_caption,
            hashtags=draft_hashtags,
        )
        guidance = str(review_guidance or "").strip()
        if not guidance:
            return review_text
        return f"{guidance}\n\n{review_text}".strip()

    @staticmethod
    def _candidate_label(media_path: str) -> str:
        if not media_path:
            return "unknown"
        path = Path(media_path)
        parts = [part for part in (path.parent.parent.name, path.name) if part]
        label = "/".join(parts) if parts else path.name
        safe_label = textwrap.shorten(label, width=80, placeholder="...").replace("`", "'")
        return f"`{safe_label}`"

    @staticmethod
    def _format_review_post(*, caption: str, hashtags: str) -> str:
        lines = [str(caption).strip()]
        if hashtags.strip():
            lines.extend(["", hashtags.strip()])
        return "\n".join(lines).strip()

    @staticmethod
    def _split_review_caption_and_hashtags(review_text: str) -> tuple[str, str]:
        lines = [line.rstrip() for line in review_text.splitlines()]
        content_lines: list[str] = []
        hashtag_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if content_lines and content_lines[-1] != "":
                    content_lines.append("")
                continue
            if stripped.lower().startswith("draft post:") or stripped.lower().startswith("platforms:"):
                continue
            lowered = stripped.lower()
            if lowered.startswith(("strategy:", "workflow:", "stage:", "selection:", "assets:")):
                continue
            if lowered.startswith("caption:"):
                stripped = stripped.split(":", 1)[1].strip()
            elif lowered.startswith("hashtags:"):
                stripped = stripped.split(":", 1)[1].strip()
            if stripped.startswith("#"):
                hashtag_lines.append(stripped)
                continue
            if stripped.lower().startswith("accept to publish with these assets"):
                continue
            content_lines.append(stripped)
        while content_lines and content_lines[-1] == "":
            content_lines.pop()
        return "\n".join(content_lines).strip(), "\n".join(hashtag_lines).strip()

    @staticmethod
    def _collect_media_from_dependencies(context: SkillContext) -> list[str]:
        media_paths: list[str] = []
        for dependency in context.node.depends_on:
            dependency_output = context.state[dependency]
            for key in ("media_paths", "saved_files", "video_path", "gif_path", "frame_path"):
                value = dependency_output.get(key)
                if isinstance(value, list):
                    media_paths.extend(str(item) for item in value if item)
                elif isinstance(value, str) and value:
                    media_paths.append(value)
        return list(dict.fromkeys(media_paths))

    @staticmethod
    def _preferred_video_path(context: SkillContext, node_id: str) -> str:
        if not node_id:
            return ""
        try:
            output = context.state[node_id]
        except (KeyError, TypeError):
            output = {}
        if not isinstance(output, dict):
            return ""
        values: list[str] = []
        for key in ("final_video_path", "video_path"):
            value = output.get(key)
            if isinstance(value, str) and value:
                values.append(value)
            elif isinstance(value, list):
                values.extend(str(item) for item in value if item)
        for value in values:
            if Path(value).suffix.lower() in VIDEO_REVIEW_EXTENSIONS:
                return value
        return ""

def register_agent_social_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    prompt_engine: LLMPromptEngine | None = None,
) -> None:
    skills = AgentSocialSkills(tool_registry, output_root, prompt_engine=prompt_engine)
    skill_registry.register("publish.media.ingest", skills.ingest_media, "Ingest candidate media for publish/review plans")
    skill_registry.register("publish.caption.prepare", skills.prepare_caption, "Prepare a social caption bundle")
    skill_registry.register("publish.media.collect", skills.collect_media, "Collect media artifacts for publishing")
    skill_registry.register("publish.media.process", skills.process_media, "Prepare media files for publishing")
    skill_registry.register("publish.social.dispatch", skills.publish_social, "Dispatch content to social platforms")

    def select_and_record_serial_review(context: SkillContext) -> SkillResult:
        serial_store = store_from_constraints(context.plan.goal.constraints)
        review_phase = str(context.node.inputs.get("review_phase") or "").strip().lower()
        serial_acceptance = (
            serial_store is not None
            and review_phase == "segment_acceptance"
        )
        serial_final_publish = serial_store is not None and review_phase == "serial_final_publish"
        publication_attempt = None
        if serial_final_publish:
            episode_id = str(context.plan.goal.constraints.get("serial_episode_id") or "").strip()
            video_path = skills._preferred_video_path(
                context, str(context.node.inputs.get("preferred_video_node") or "").strip()
            )
            if not video_path:
                explicit_paths = context.node.inputs.get("media_paths") or []
                if isinstance(explicit_paths, str):
                    explicit_paths = [explicit_paths]
                video_candidates = [
                    str(path)
                    for path in explicit_paths
                    if path and Path(str(path)).suffix.lower() in VIDEO_REVIEW_EXTENSIONS
                ]
                if not video_candidates:
                    video_candidates = [
                        path
                        for path in skills._collect_media_from_dependencies(context)
                        if Path(path).suffix.lower() in VIDEO_REVIEW_EXTENSIONS
                    ]
                if len(video_candidates) == 1:
                    video_path = video_candidates[0]
            if not episode_id or not video_path:
                raise RuntimeError("Final serial publication review requires its episode ID and QA-passed assembled MP4.")
            publication_attempt = serial_store.begin_publication_attempt(
                episode_id,
                platforms=context.node.inputs.get("platforms")
                or context.plan.goal.constraints.get("platforms")
                or [],
                episode_topic=str(context.plan.goal.constraints.get("episode_topic") or ""),
                caption="",
                hashtags=" ".join(str(tag) for tag in context.plan.goal.constraints.get("hashtags", []) or []),
                media_path=video_path,
            )
        try:
            result = skills.select_best_assets(context)
        except Exception as exc:
            if publication_attempt is not None:
                serial_store.record_publication_review(
                    str(context.plan.goal.constraints.get("serial_episode_id") or ""),
                    int(publication_attempt["attempt_number"]),
                    decision="failed",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            if serial_acceptance:
                episode_id = str(context.plan.goal.constraints.get("serial_episode_id") or "").strip()
                if episode_id:
                    serial_store.record_review(
                        episode_id,
                        int(context.node.inputs.get("segment_index", 0)) + 1,
                        "failed",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    )
            raise
        if serial_final_publish and publication_attempt is not None:
            outputs = dict(result.outputs)
            selected = outputs.get("selected_assets") or outputs.get("media_paths") or []
            selected_paths = [str(item) for item in selected if item]
            review_mode = str(outputs.get("review_mode") or "").strip().lower()
            reason = " ".join(
                str(outputs.get(key) or "")
                for key in ("selection_rationale", "fallback_reason")
            ).lower()
            if result.status == "success" and review_mode == "discord" and selected_paths:
                decision = "accepted"
            elif outputs.get("regeneration_requested") or outputs.get("review_decision") == "regenerate":
                decision = "rejected"
            elif review_mode != "discord":
                decision = "unavailable"
            else:
                decision = "timeout" if "timed out" in reason else "rejected" if "rejected" in reason else "failed"
            approved_text = str(outputs.get("approved_review_text") or "").strip()
            if not approved_text:
                approved_text = "\n\n".join(
                    part
                    for part in (
                        str(outputs.get("review_draft_caption") or "").strip(),
                        str(outputs.get("review_draft_hashtags") or "").strip(),
                    )
                    if part
                )
            caption, hashtags = skills._split_review_caption_and_hashtags(approved_text)
            serial_store.record_publication_review(
                str(context.plan.goal.constraints.get("serial_episode_id") or ""),
                int(publication_attempt["attempt_number"]),
                decision=decision,
                review_session_id=str(outputs.get("review_session_id") or ""),
                caption=caption or None,
                hashtags=hashtags or None,
                details={
                    "review_mode": review_mode,
                    "reviewer": outputs.get("reviewer", ""),
                    "review_session_path": outputs.get("review_session_path", ""),
                    "review_decision": outputs.get("review_decision", ""),
                    "selected_paths": selected_paths,
                    "result_status": result.status,
                    "series_disposition": str(
                        (outputs.get("review_delivery") or {}).get("series_disposition") or ""
                    ),
                },
            )
            series_disposition = str(
                (outputs.get("review_delivery") or {}).get("series_disposition") or ""
            ).strip().lower()
            if series_disposition in {"completed", "discarded"}:
                serial_store.set_series_lifecycle_status(
                    str(context.plan.goal.constraints.get("series_id") or ""),
                    series_disposition,
                    note=(
                        f"Creator selected {series_disposition} after reviewing episode "
                        f"{context.plan.goal.constraints.get('serial_episode_number')}"
                    ),
                )
            outputs["publication_attempt_number"] = publication_attempt["attempt_number"]
            result.outputs = outputs
            if not selected_paths:
                return result
        if not serial_acceptance:
            return result

        episode_id = str(context.plan.goal.constraints.get("serial_episode_id") or "").strip()
        if not episode_id:
            raise RuntimeError("Serial segment review requires a persisted episode ID")
        outputs = dict(result.outputs)
        selected = outputs.get("selected_assets") or outputs.get("media_paths") or []
        selected_paths = [str(item) for item in selected if item]
        review_mode = str(outputs.get("review_mode") or "").strip().lower()
        if result.status == "success" and review_mode == "discord" and selected_paths:
            decision = "accepted"
            reviewed_path = selected_paths[0]
        elif outputs.get("regeneration_requested") or outputs.get("review_decision") == "regenerate":
            decision = "regeneration_requested"
            reviewed_path = None
        elif review_mode != "discord":
            decision = "unavailable"
            reviewed_path = None
        else:
            reason = " ".join(
                str(outputs.get(key) or "") for key in ("selection_rationale", "fallback_reason")
            ).lower()
            decision = "rejected" if "rejected" in reason else "timeout" if "timed out" in reason else "failed"
            reviewed_path = None
        serial_store.record_review(
            episode_id,
            int(context.node.inputs.get("segment_index", 0)) + 1,
            decision,
            reviewed_path,
            {
                "review_mode": review_mode,
                "reviewer": outputs.get("reviewer", ""),
                "review_session_id": outputs.get("review_session_id", ""),
                "review_session_path": outputs.get("review_session_path", ""),
                "fallback_reason": outputs.get("fallback_reason", ""),
                "selected_paths": selected_paths,
                "result_status": result.status,
            },
        )
        return result

    skill_registry.register("review.assets.select", select_and_record_serial_review, "Select a shortlist of candidate assets")
