from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import textwrap

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.platform_content import (
    build_platform_bundle,
    sanitize_hashtags,
)
from agentic.runtime.post_strategy import resolve_post_strategy
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.tools.context_services import DiscordHumanReviewService
from agentic.tools.publishing_adapter import build_dispatch_plan

VIDEO_REVIEW_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_REVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class AgentSocialSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path, prompt_engine: PromptEngine | None = None) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.prompt_engine = prompt_engine or PromptEngine()
        self.discord_review = DiscordHumanReviewService(output_root=self.output_root)

    def prepare_caption(self, context: SkillContext) -> SkillResult:
        prefix = str(context.node.inputs.get("prefix", "")).strip()
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
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Prepared media artifacts for social publishing."])

    def publish_social(self, context: SkillContext) -> SkillResult:
        processed = context.state[context.node.depends_on[0]] if context.node.depends_on else {}
        caption_bundle = context.state[context.node.depends_on[1]] if len(context.node.depends_on) > 1 else {}
        platform_captions = caption_bundle.get("platform_captions", {})
        platform_bundle = caption_bundle.get("platform_bundle", {})
        requested_platforms = context.node.inputs.get("platforms") or context.plan.goal.constraints.get("platforms") or []
        platforms = [str(platform) for platform in requested_platforms]
        dispatch_plan = build_dispatch_plan(
            media_paths=context.node.inputs.get("media_paths") or processed.get("media_paths", []),
            caption=context.node.inputs.get("caption") or caption_bundle.get("caption", ""),
            hashtags=context.node.inputs.get("hashtags") or caption_bundle.get("hashtags", ""),
            platforms=platforms,
            platform_bundle=platform_bundle if isinstance(platform_bundle, dict) else {},
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
        if blocked_platforms and not dry_run:
            return SkillResult(
                status="blocked",
                outputs={
                    "status": "blocked",
                    "media_paths": context.node.inputs.get("media_paths") or processed.get("media_paths", []),
                    "caption": context.node.inputs.get("caption") or caption_bundle.get("caption", ""),
                    "hashtags": context.node.inputs.get("hashtags") or caption_bundle.get("hashtags", ""),
                    "platforms": platforms or list(dispatch_plan.keys()),
                    "platform_bundle": platform_bundle if isinstance(platform_bundle, dict) else {},
                    "dispatch_plan": dispatch_plan,
                    "dispatch_ready": False,
                    "blocked_platforms": blocked_platforms,
                    "platform_ineligible": platform_ineligible,
                    "blocked_reason": "publish bundle is not dispatch-ready for every requested platform",
                },
                logs=["Blocked social dispatch because one or more platforms failed publish-readiness validation."],
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
                "platform_bundle": platform_bundle if isinstance(platform_bundle, dict) else {},
                "manifest_dir": str(self.output_root),
            },
        )
        outputs = dict(result)
        outputs["platform_bundle"] = platform_bundle if isinstance(platform_bundle, dict) else {}
        outputs["dispatch_plan"] = dispatch_plan
        outputs["platform_ineligible"] = platform_ineligible
        outputs["dispatch_ready"] = (
            bool(caption_bundle.get("dispatch_ready", False))
            and not blocked_platforms
            and not platform_ineligible
        )
        dispatch_status = str(outputs.get("status") or "").strip().lower()
        skill_status = "success" if dispatch_status in {"success", "dry_run", "awaiting_user_action"} else "failed"
        return SkillResult(status=skill_status, outputs=outputs, logs=["Dispatched a social publishing action."])

    def select_best_assets(self, context: SkillContext) -> SkillResult:
        media_paths = self._collect_media_from_dependencies(context)
        limit = int(context.node.inputs.get("limit", 10))
        review_scope = str(
            context.node.inputs.get("review_scope")
            or context.plan.goal.constraints.get("review_scope")
            or self._infer_review_scope(media_paths)
        ).strip().lower()
        review_phase = str(context.node.inputs.get("review_phase") or "").strip().lower()
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
        preferred_extensions = tuple(context.node.inputs.get("preferred_extensions", [".mp4", ".gif", ".png", ".jpg", ".jpeg", ".webp"]))
        ranked = sorted(
            media_paths,
            key=lambda path: (0 if path.lower().endswith(preferred_extensions) else 1, path),
        )
        heuristic_ranked = [
            {
                "media_path": path,
                "score": max(1, 100 - (index * 5)),
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
            review_goal = context.plan.goal
            if auto_select_for_probe:
                story_output = context.state.node_outputs.get("native-story-prompt", {})
                story_prompt = ""
                if isinstance(story_output, dict):
                    story_prompt = str(
                        story_output.get("opening_keyframe_prompt")
                        or story_output.get("prompt")
                        or ""
                    ).strip()
                if story_prompt:
                    review_goal = replace(context.plan.goal, prompt=story_prompt)
            try:
                bundle = self.prompt_engine.review_asset_candidates(
                    review_goal,
                    media_paths=ranked,
                    review_notes=review_notes,
                    selection_limit=limit,
                )
            except Exception as exc:
                if not auto_select_for_probe:
                    raise
                if any(
                    marker in str(exc)
                    for marker in ("stage_probe_quality_gate:", "asset_review_hard_gate:")
                ):
                    raise
                bundle = {
                    "selected_assets": ranked[:limit],
                    "ranked_candidates": heuristic_ranked,
                    "selection_rationale": "Automatic candidate review timed out or was unavailable; used the deterministic ranked candidate.",
                    "regeneration_notes": review_notes or "Retry vision review before human-facing publication.",
                    "prompt_mode": "automatic_timeout_fallback",
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                }
        selected = [path for path in bundle.get("selected_assets", []) if path in ranked][:limit]
        if not selected:
            selected = ranked[:limit]
        review_media_paths = ranked if (review_all_candidates or final_video_review or final_media_review) else (selected or ranked[:limit])
        caption_media_paths = review_media_paths if (review_all_candidates or final_video_review or final_media_review) else (selected or ranked[:limit])
        hashtags = context.plan.goal.constraints.get("hashtags") or []
        platforms = [str(platform) for platform in (context.plan.goal.constraints.get("platforms") or [])]
        if first_frame_review or last_frame_review or anchor_set_review or reference_review:
            caption_bundle = {"caption": context.plan.goal.prompt, "hashtags": ""}
        else:
            caption_bundle = self.prompt_engine.prepare_publish_caption(
                context.plan.goal,
                prefix="",
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
        if first_frame_review or last_frame_review or anchor_set_review or reference_review:
            review_text = self._build_stage_review_text(strategy=review_metadata["strategy"])
        elif final_video_review or final_media_review:
            review_text = self._build_final_publish_review_text(
                draft_caption=str(caption_bundle.get("caption", "") or context.plan.goal.prompt),
                draft_hashtags=str(caption_bundle.get("hashtags", "") or ""),
                platforms=platforms,
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
                platforms=platforms,
                candidate_paths=review_media_paths,
                review_scope=review_scope,
            )
        human_review_enabled = bool(
            not auto_select_for_probe
            and (
                context.plan.goal.constraints.get("enable_stage_review", False)
                or context.plan.goal.constraints.get("enable_review_loop", False)
                or require_human_review
            )
        )
        decision = None
        if human_review_enabled:
            decision = self.discord_review.review_candidates(
                text=review_text,
                media_paths=review_media_paths,
                timeout_seconds=int(context.plan.goal.constraints.get("discord_review_timeout_seconds") or 3600),
                allow_asset_selection=first_frame_review or last_frame_review or anchor_set_review or reference_review or final_media_review,
                allow_text_edit=not (first_frame_review or last_frame_review or anchor_set_review or reference_review),
                selection_mode="single" if (first_frame_review or last_frame_review) else "multi",
                selection_required=first_frame_review or last_frame_review or anchor_set_review or reference_review or final_media_review,
                selection_limit=(1 if (first_frame_review or last_frame_review) else (limit if (anchor_set_review or reference_review) else None)),
                review_scope=review_scope,
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
                        "prompt_mode": str(bundle.get("prompt_mode", "template")),
                    },
                    logs=["Blocked workflow because the Discord reviewer rejected the candidate set."],
                )
            if decision.status == "failed":
                failure_policy = str(
                    context.plan.goal.constraints.get("pre_video_review_failure_policy") or "block"
                ).strip().lower()
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
        fallback_reason = str(getattr(decision, "fallback_reason", ""))
        review_delivery = dict(getattr(decision, "delivery", {}) or {})
        rejected = [path for path in ranked if path not in selected]
        selection_logs = ["Selected a best-effort shortlist of candidate assets."]
        if not human_review_enabled:
            selection_logs.append(
                "Discord review disabled for this run; selection was automatic. Use --no-review only when that is intentional."
            )
        if str(bundle.get("prompt_mode") or "") == "automatic_timeout_fallback":
            selection_logs.append(
                "Automatic vision/text review timed out; deterministic selection was recorded as a stage fallback."
            )
        return SkillResult(
            status="success",
            outputs={
                "media_paths": selected,
                "selected_assets": selected,
                "selected_count": len(selected),
                "ranked_candidates": bundle.get("ranked_candidates", heuristic_ranked),
                "rejected_assets": rejected,
                "rejected_asset_details": bundle.get("rejected_asset_details", []),
                "selection_rationale": str(bundle.get("selection_rationale") or f"Preferred extensions {preferred_extensions} with deterministic path ordering."),
                "regeneration_notes": str(bundle.get("regeneration_notes") or context.node.inputs.get("review_notes") or context.plan.goal.constraints.get("review_notes", "")),
                "failure_tags": list(bundle.get("failure_tags", [])),
                "retry_direction": str(bundle.get("retry_direction", "")),
                "retry_intensity": str(bundle.get("retry_intensity", "medium")),
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
        platforms: list[str],
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
            platforms=platforms,
        )
        if review_body:
            lines.extend(["", review_body])
        return textwrap.shorten("\n".join(lines).strip(), width=1900, placeholder="...")

    @staticmethod
    def _build_final_publish_review_text(*, draft_caption: str, draft_hashtags: str, platforms: list[str]) -> str:
        """Build only the final publish copy; editing is enabled at this stage."""
        return AgentSocialSkills._format_review_post(
            caption=draft_caption,
            hashtags=draft_hashtags,
            platforms=platforms,
        )

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
    def _format_review_post(*, caption: str, hashtags: str, platforms: list[str]) -> str:
        del platforms
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

def register_agent_social_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    prompt_engine: PromptEngine | None = None,
) -> None:
    skills = AgentSocialSkills(tool_registry, output_root, prompt_engine=prompt_engine)
    skill_registry.register("publish.media.ingest", skills.ingest_media, "Ingest candidate media for publish/review plans")
    skill_registry.register("publish.caption.prepare", skills.prepare_caption, "Prepare a social caption bundle")
    skill_registry.register("publish.media.collect", skills.collect_media, "Collect media artifacts for publishing")
    skill_registry.register("publish.media.process", skills.process_media, "Prepare media files for publishing")
    skill_registry.register("publish.social.dispatch", skills.publish_social, "Dispatch content to social platforms")
    skill_registry.register("review.assets.select", skills.select_best_assets, "Select a shortlist of candidate assets")
