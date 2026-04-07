from __future__ import annotations

from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.tools.context_services import DiscordHumanReviewService


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
        bundle = self.prompt_engine.prepare_publish_caption(
            context.plan.goal,
            prefix=prefix,
            hashtags=[str(hashtags)] if isinstance(hashtags, str) else [str(tag) for tag in hashtags],
            platforms=platforms,
            media_paths=selected_media,
            review_notes=str(context.plan.goal.constraints.get("review_notes", "") or ""),
        )
        return SkillResult(
            status="success",
            outputs={
                "caption": str(bundle["caption"]),
                "hashtags": str(bundle["hashtags"]),
                "platform_captions": dict(bundle.get("platform_captions", {})),
                "platform_bundle": dict(bundle.get("platform_bundle", {})),
                "caption_strategy": str(bundle.get("caption_strategy", "generic")),
                "dispatch_ready": bool(bundle.get("dispatch_ready", False)),
                "prompt_mode": str(bundle.get("prompt_mode", "template")),
                "selected_assets": selected_media,
            },
            logs=["Prepared a reusable social caption bundle."],
        )

    def ingest_media(self, context: SkillContext) -> SkillResult:
        constraint_paths = context.plan.goal.constraints.get("media_paths") or []
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
        dispatch_plan = self._build_dispatch_plan(
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
                "dry_run": dry_run,
                "platform_bundle": platform_bundle if isinstance(platform_bundle, dict) else {},
            },
        )
        outputs = dict(result)
        outputs["platform_bundle"] = platform_bundle if isinstance(platform_bundle, dict) else {}
        outputs["dispatch_plan"] = dispatch_plan
        outputs["dispatch_ready"] = bool(caption_bundle.get("dispatch_ready", False)) and not blocked_platforms
        return SkillResult(status="success", outputs=outputs, logs=["Dispatched a social publishing action."])

    def select_best_assets(self, context: SkillContext) -> SkillResult:
        media_paths = self._collect_media_from_dependencies(context)
        limit = int(context.node.inputs.get("limit", 10))
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
        bundle = self.prompt_engine.review_asset_candidates(
            context.plan.goal,
            media_paths=ranked,
            review_notes=str(context.node.inputs.get("review_notes") or context.plan.goal.constraints.get("review_notes", "")),
            selection_limit=limit,
        )
        selected = [path for path in bundle.get("selected_assets", []) if path in ranked][:limit]
        if not selected:
            selected = ranked[:limit]
        review_text = self._build_review_text(
            prompt=context.plan.goal.prompt,
            review_notes=str(context.node.inputs.get("review_notes") or context.plan.goal.constraints.get("review_notes", "")),
            ranked_candidates=bundle.get("ranked_candidates", heuristic_ranked),
            selection_limit=limit,
        )
        decision = self.discord_review.review_candidates(
            text=review_text,
            media_paths=selected or ranked[:limit],
            timeout_seconds=int(context.plan.goal.constraints.get("discord_review_timeout_seconds") or 3600),
        )
        if decision.review_mode == "discord":
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
            if decision.selected_paths:
                selected = [path for path in decision.selected_paths if path in ranked]
        rejected = [path for path in ranked if path not in selected]
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
                "review_mode": decision.review_mode,
                "reviewer": decision.reviewer,
                "review_session_id": decision.session_id,
                "review_session_path": decision.session_path,
                "edited_review_text": decision.edited_text,
            },
            metrics={"selected_count": len(selected)},
            logs=["Selected a best-effort shortlist of candidate assets."],
        )

    @staticmethod
    def _build_review_text(
        *,
        prompt: str,
        review_notes: str,
        ranked_candidates: list[dict[str, object]],
        selection_limit: int,
    ) -> str:
        lines = [
            f"Goal: {prompt}",
            f"Selection limit: {selection_limit}",
        ]
        if review_notes:
            lines.append(f"Review notes: {review_notes}")
        if ranked_candidates:
            lines.append("Suggested ranking:")
            for index, candidate in enumerate(ranked_candidates[:selection_limit], start=1):
                lines.append(
                    f"{index}. {candidate.get('media_path', '')} | score={candidate.get('score', '')} | {candidate.get('rationale', '')}"
                )
        lines.append("Accept to continue, Reject to stop, or Edit to change review notes and chosen assets.")
        return "\n".join(lines)

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
            bundle_validation = bundle.get("validation", {})
            if not isinstance(bundle_validation, dict):
                bundle_validation = {}
            platform_caption = str(bundle.get("caption") or caption)
            platform_hashtags = str(bundle.get("hashtags") or hashtags)
            validation = {
                "has_caption": bool(platform_caption),
                "has_media": bool(media_paths),
                "is_publish_ready": bool(bundle_validation.get("is_publish_ready", bool(platform_caption) and bool(media_paths))),
            }
            dispatch_plan[platform] = {
                "caption": platform_caption,
                "hashtags": platform_hashtags,
                "media_paths": [str(path) for path in media_paths],
                "validation": validation,
            }
        return dispatch_plan


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
