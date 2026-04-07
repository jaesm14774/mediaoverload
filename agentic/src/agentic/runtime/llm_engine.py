from __future__ import annotations

import json
import os
from typing import Any

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_manager_adapter import build_llm_manager
from agentic.runtime.prompting import (
    LONG_VIDEO_SYSTEM_PROMPT,
    STICKER_SYSTEM_PROMPT,
    build_animated_sticker_motion_prompt,
    build_autonomous_scene_prompt,
    build_segment_prompt,
    build_goal_brief,
    build_sticker_prompt,
    build_story_segments,
)

WORKFLOW_STAGE_KEYS = (
    "image_workflow_name",
    "video_workflow_name",
    "refine_workflow_name",
    "transition_workflow_name",
    "upscale_workflow_name",
)


class PromptGenerationError(RuntimeError):
    """Raised when a prompt-producing step cannot complete with an LLM."""


class LLMPromptEngine:
    def __init__(self, mode: str = "auto", manager: Any | None = None) -> None:
        self.mode = mode
        self._manager = manager
        self._backend_info: dict[str, Any] | None = None
        self._manager_error: str | None = None

    def _mark_llm_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["prompt_mode"] = "llm"
        enriched["llm_backend"] = self.backend_info()
        return enriched

    def backend_info(self) -> dict[str, Any]:
        if self._backend_info is None:
            self._backend_info = self._resolve_backend_info()
        enriched = dict(self._backend_info)
        if self._manager_error:
            enriched["manager_error"] = self._manager_error
        return enriched

    def route_generation_strategy(
        self,
        prompt: str,
        character: str,
        style: str,
        generation_type_candidates: list[str],
        workflow_stage_candidates: dict[str, dict[str, list[str]]],
        count_policies: dict[str, dict[str, Any]],
        routing_hints: dict[str, Any] | None = None,
        preferred_generation_type: str | None = None,
    ) -> dict[str, Any]:
        normalized_candidates = [str(item).strip() for item in generation_type_candidates if str(item).strip()]
        if not normalized_candidates:
            raise ValueError("generation_type_candidates cannot be empty for LLM routing.")
        manager = self._require_manager()
        all_workflow_candidates = sorted(
            {
                workflow_name
                for stages in workflow_stage_candidates.values()
                for items in stages.values()
                for workflow_name in items
                if str(workflow_name).strip()
            }
        )
        user_prompt = "\n".join(
            [
                f"Character: {character}",
                f"Prompt: {prompt}",
                f"Style: {style}",
                f"Preferred generation type override: {preferred_generation_type or ''}",
                f"Generation type candidates JSON: {json.dumps(normalized_candidates, ensure_ascii=False)}",
                f"Workflow stage candidates JSON: {json.dumps(workflow_stage_candidates, ensure_ascii=False)}",
                f"Count policies JSON: {json.dumps(count_policies, ensure_ascii=False)}",
                f"Routing hints JSON: {json.dumps(routing_hints or {}, ensure_ascii=False)}",
                "Pick the best generation_type, stage workflows, and count plan for the user's request.",
                "Only choose values from the provided candidate lists and policy ranges.",
                "You must populate the workflow stages needed by the chosen generation_type.",
                "Return JSON with keys: generation_type, workflow_plan, count_plan, reason.",
            ]
        )
        try:
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="generation_strategy_route",
                schema={
                    "type": "object",
                    "properties": {
                        "generation_type": {"type": "string", "enum": normalized_candidates},
                        "workflow_plan": {
                            "type": "object",
                            "properties": {
                                "image_workflow_name": {"type": "string", "enum": all_workflow_candidates + [""]},
                                "video_workflow_name": {"type": "string", "enum": all_workflow_candidates + [""]},
                                "refine_workflow_name": {"type": "string", "enum": all_workflow_candidates + [""]},
                                "transition_workflow_name": {"type": "string", "enum": all_workflow_candidates + [""]},
                                "upscale_workflow_name": {"type": "string", "enum": all_workflow_candidates + [""]},
                            },
                            "required": [
                                "image_workflow_name",
                                "video_workflow_name",
                                "refine_workflow_name",
                                "transition_workflow_name",
                                "upscale_workflow_name",
                            ],
                            "additionalProperties": False,
                        },
                        "count_plan": {
                            "type": "object",
                            "properties": {
                                "image_count": {"type": "integer", "minimum": 1, "maximum": 32},
                                "video_count": {"type": "integer", "minimum": 1, "maximum": 32},
                                "segment_count": {"type": "integer", "minimum": 1, "maximum": 32},
                                "review_selection_limit": {"type": "integer", "minimum": 1, "maximum": 32},
                                "sticker_expression_count": {"type": "integer", "minimum": 1, "maximum": 32},
                                "images_per_prompt": {"type": "integer", "minimum": 1, "maximum": 32},
                            },
                            "required": [
                                "image_count",
                                "video_count",
                                "segment_count",
                                "review_selection_limit",
                                "sticker_expression_count",
                                "images_per_prompt",
                            ],
                            "additionalProperties": False,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["generation_type", "workflow_plan", "count_plan", "reason"],
                    "additionalProperties": False,
                },
            )
            selected_generation_type = str(payload.get("generation_type") or "").strip()
            if selected_generation_type not in normalized_candidates:
                raise ValueError(f"LLM selected unsupported generation_type: {selected_generation_type}")
            workflow_plan = dict(payload.get("workflow_plan") or {})
            count_plan = dict(payload.get("count_plan") or {})
            allowed_stage_candidates = workflow_stage_candidates.get(selected_generation_type, {})
            normalized_workflow_plan: dict[str, str] = {}
            workflow_corrections: list[str] = []
            for stage_key in WORKFLOW_STAGE_KEYS:
                selected_workflow = str(workflow_plan.get(stage_key) or "").strip()
                allowed_workflows = [
                    str(item).strip()
                    for item in allowed_stage_candidates.get(stage_key, [])
                    if str(item).strip()
                ]
                if allowed_workflows:
                    if not selected_workflow:
                        selected_workflow = allowed_workflows[0]
                        workflow_corrections.append(f"{stage_key}=default:{selected_workflow}")
                    elif selected_workflow not in set(allowed_workflows):
                        selected_workflow = allowed_workflows[0]
                        workflow_corrections.append(f"{stage_key}=fallback:{selected_workflow}")
                elif selected_workflow:
                    raise ValueError(
                        f"LLM selected workflow '{selected_workflow}' for unavailable stage '{stage_key}' "
                        f"under generation_type '{selected_generation_type}'."
                    )
                normalized_workflow_plan[stage_key] = selected_workflow
            normalized_count_plan: dict[str, int] = {}
            allowed_count_policy = count_policies.get(selected_generation_type, {})
            for count_key in (
                "image_count",
                "video_count",
                "segment_count",
                "review_selection_limit",
                "sticker_expression_count",
                "images_per_prompt",
            ):
                if count_key not in count_plan:
                    raise ValueError(f"LLM routing omitted required count key: {count_key}")
                value = int(count_plan[count_key])
                policy = allowed_count_policy.get(count_key)
                if policy:
                    minimum = int(policy["min"])
                    maximum = int(policy["max"])
                    if value < minimum or value > maximum:
                        raise ValueError(
                            f"LLM selected out-of-range {count_key}={value} for generation_type '{selected_generation_type}'."
                        )
                normalized_count_plan[count_key] = value
            reason = str(payload.get("reason") or "").strip()
            if workflow_corrections:
                correction_note = f"workflow defaults applied ({', '.join(workflow_corrections)})"
                reason = f"{reason}; {correction_note}" if reason else correction_note
            return self._mark_llm_payload(
                {
                    "generation_type": selected_generation_type,
                    "workflow_plan": normalized_workflow_plan,
                    "count_plan": normalized_count_plan,
                    "reason": reason,
                }
            )
        except Exception as exc:
            raise self._generation_error("route_generation_strategy", exc) from exc

    def generate_autonomous_scene_prompt(
        self,
        *,
        character: str,
        style: str,
        media_type: str,
        news_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fallback = build_autonomous_scene_prompt(
            character=character,
            style=style,
            media_type=media_type,
            news_context=news_context,
        )
        try:
            manager = self._require_manager()
            user_prompt = "\n".join(
                [
                    f"Character: {character}",
                    f"Style: {style}",
                    f"Media type: {media_type}",
                    f"News context JSON: {json.dumps(news_context or {}, ensure_ascii=False)}",
                    "Generate one publishable and visually interesting scenario prompt.",
                    "If news exists, translate it into 2-4 visual motifs rather than reenacting the headline.",
                    "Keep the character as the clear protagonist.",
                    "Do not ask follow-up questions.",
                    "Return JSON with keys: prompt, creative_seed, source.",
                ]
            )
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="autonomous_scene_prompt",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "creative_seed": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["prompt", "creative_seed", "source"],
                    "additionalProperties": False,
                },
            )
            return self._mark_llm_payload(
                {
                    "prompt": str(payload.get("prompt") or fallback["prompt"]),
                    "creative_seed": str(payload.get("creative_seed") or fallback["creative_seed"]),
                    "source": str(payload.get("source") or "autonomous_llm"),
                    "news_context": dict(news_context or {}),
                }
            )
        except Exception as exc:
            fallback["prompt_mode"] = "template"
            fallback["fallback_reason"] = "manager_unavailable"
            fallback["manager_error"] = f"{type(exc).__name__}: {exc}"
            fallback["llm_backend"] = self.backend_info()
            fallback["news_context"] = dict(news_context or {})
            return fallback

    def expand_goal(self, goal: GoalRequest, selected_style: str, idea_variants: list[dict[str, Any]]) -> dict[str, Any]:
        fallback = build_goal_brief(goal, selected_style, idea_variants)
        try:
            manager = self._require_manager()
            user_prompt = "\n".join(
                [
                    f"Goal: {goal.prompt}",
                    f"Media type: {goal.media_type}",
                    f"Style: {selected_style}",
                    f"Character: {goal.constraints.get('character', '')}",
                    f"Duration seconds: {goal.duration_seconds}",
                    f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
                    "Return JSON with keys: creative_brief, prompt, negative_prompt.",
                    "The prompt must be generation-ready for diffusion and image-to-video models.",
                    "If news context exists, treat it as inspiration for props, tension, environment, or symbols only.",
                    "Do not make the output look like literal news coverage unless the user explicitly asked for that.",
                ]
            )
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="goal_brief",
                schema={
                    "type": "object",
                    "properties": {
                        "creative_brief": {"type": "string"},
                        "prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                    },
                    "required": ["creative_brief", "prompt", "negative_prompt"],
                    "additionalProperties": False,
                },
            )
            fallback.update(
                {
                    "creative_brief": str(payload.get("creative_brief") or fallback["creative_brief"]),
                    "prompt": str(payload.get("prompt") or fallback["prompt"]),
                    "negative_prompt": str(payload.get("negative_prompt") or fallback["negative_prompt"]),
                }
            )
            return self._mark_llm_payload(fallback)
        except Exception as exc:
            return self._template_fallback(fallback, exc)

    def compose_prompt(
        self,
        goal: GoalRequest,
        prompt: str,
        style: str,
        prefix: str = "",
        suffix: str = "",
        negative_prompt: str = "ugly, blurry, low quality, bad anatomy, deformed, duplicate, watermark, text",
    ) -> dict[str, Any]:
        fallback = {
            "prompt": ", ".join(part for part in (prefix, prompt, style, suffix) if part),
            "negative_prompt": negative_prompt,
        }
        manager = self._require_manager()
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Media type: {goal.media_type}",
                f"Base prompt: {prompt}",
                f"Style: {style}",
                f"Prefix: {prefix}",
                f"Suffix: {suffix}",
                f"Character: {goal.constraints.get('character', '')}",
                f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
                "Return JSON with keys: prompt, negative_prompt.",
                "Create a concise generation-ready prompt for the requested workflow.",
                "If news context exists, merge only a few concrete visual motifs into the scene instead of recreating the headline.",
            ]
        )
        try:
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="compose_prompt",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                    },
                    "required": ["prompt", "negative_prompt"],
                    "additionalProperties": False,
                },
            )
            return self._mark_llm_payload(
                {
                    "prompt": str(payload.get("prompt") or fallback["prompt"]),
                    "negative_prompt": str(payload.get("negative_prompt") or fallback["negative_prompt"]),
                }
            )
        except Exception as exc:
            raise self._generation_error("compose_prompt", exc) from exc

    def segment_story(
        self,
        goal: GoalRequest,
        creative_brief: str,
        segment_count: int,
        tone: str,
    ) -> list[dict[str, Any]]:
        fallback = build_story_segments(goal, creative_brief, segment_count, tone)
        manager = self._require_manager()

        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Media type: {goal.media_type}",
                f"Character: {goal.constraints.get('character', '')}",
                f"Style: {goal.style}",
                f"Creative brief: {creative_brief}",
                f"Tone: {tone}",
                f"Segment count: {segment_count}",
                f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
                "Return JSON object with key: segments.",
                "segments must be an array where each item has keys: segment_id, visual, narration.",
                "Every segment must preserve identity, escalate or progress action, and avoid static micro-motion.",
                "Use news only as symbolic or environmental inspiration when provided.",
            ]
        )
        try:
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="story_segments",
                schema={
                    "type": "object",
                    "properties": {
                        "segments": {
                            "type": "array",
                            "minItems": segment_count,
                            "maxItems": segment_count,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "segment_id": {"type": "string"},
                                    "visual": {"type": "string"},
                                    "narration": {"type": "string"},
                                },
                                "required": ["segment_id", "visual", "narration"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["segments"],
                    "additionalProperties": False,
                },
            )
            segments = payload.get("segments") if isinstance(payload, dict) else payload
            if isinstance(segments, list) and segments:
                normalized: list[dict[str, Any]] = []
                for index, item in enumerate(segments[:segment_count]):
                    normalized.append(
                        {
                            "segment_id": str(item.get("segment_id") or f"segment-{index + 1}"),
                            "visual": str(item.get("visual") or fallback[index]["visual"]),
                            "narration": str(item.get("narration") or fallback[index]["narration"]),
                            "stage": fallback[index].get("stage"),
                            "camera": fallback[index].get("camera"),
                            "creative_brief": creative_brief,
                        }
                    )
                while len(normalized) < segment_count:
                    normalized.append(fallback[len(normalized)])
                return normalized
        except Exception as exc:
            del exc
        return fallback

    def sticker_expressions(self, goal: GoalRequest, prompt: str, character: str, expression_count: int) -> list[str]:
        target_count = max(1, int(expression_count))

        user_prompt = "\n".join(
            [
                f"Character: {character}",
                f"Theme/Context: {prompt}",
                f"Generate exactly {target_count} unique sticker expressions.",
                'Return JSON object with key "expressions" only.',
            ]
        )
        try:
            manager = self._require_manager()
            payload = self._chat_json(
                manager,
                STICKER_SYSTEM_PROMPT,
                user_prompt,
                schema_name="sticker_expressions",
                schema={
                    "type": "object",
                    "properties": {
                        "expressions": {
                            "type": "array",
                            "minItems": target_count,
                            "maxItems": target_count,
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["expressions"],
                    "additionalProperties": False,
                },
            )
            expressions = payload.get("expressions") if isinstance(payload, dict) else payload
            if isinstance(expressions, list) and expressions:
                normalized = [str(item).strip() for item in expressions if str(item).strip()]
                if len(normalized) == target_count:
                    return normalized
        except Exception:
            pass
        fallback = self._fallback_sticker_expressions(prompt, target_count)
        return fallback

    def build_sticker_prompt_set(
        self,
        goal: GoalRequest,
        expressions: list[str],
        character: str,
        prompt_prefix: str,
        style: str,
    ) -> dict[str, Any]:
        fallback_prompt_sets = [
            {
                "label": f"sticker_{index + 1:02d}",
                "expression": expression,
                "prompt": build_sticker_prompt(character, expression, prompt_prefix, style),
            }
            for index, expression in enumerate(expressions)
        ]
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Character: {character}",
                f"Prompt prefix: {prompt_prefix}",
                f"Style: {style}",
                f"Expressions JSON: {json.dumps(expressions, ensure_ascii=False)}",
                "Return a JSON object with key prompt_sets.",
                "prompt_sets must be an array where each item has keys: label, expression, prompt.",
                "Each prompt must be image-generation ready, visually distinct, and preserve the same character identity.",
            ]
        )
        try:
            manager = self._require_manager()
            payload = self._chat_json(
                manager,
                STICKER_SYSTEM_PROMPT,
                user_prompt,
                schema_name="sticker_prompt_set",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt_sets": {
                            "type": "array",
                            "minItems": len(fallback_prompt_sets),
                            "maxItems": len(fallback_prompt_sets),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "expression": {"type": "string"},
                                    "prompt": {"type": "string"},
                                },
                                "required": ["label", "expression", "prompt"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["prompt_sets"],
                    "additionalProperties": False,
                },
            )
            prompt_items = payload.get("prompt_sets") if isinstance(payload, dict) else payload
            if isinstance(prompt_items, list) and prompt_items:
                prompt_sets: list[dict[str, Any]] = []
                for index, item in enumerate(prompt_items[: len(fallback_prompt_sets)]):
                    base_item = fallback_prompt_sets[index]
                    prompt_sets.append(
                        {
                            "label": str(item.get("label") or base_item["label"]),
                            "expression": str(item.get("expression") or base_item["expression"]),
                            "prompt": str(item.get("prompt") or base_item["prompt"]),
                        }
                    )
                while len(prompt_sets) < len(fallback_prompt_sets):
                    prompt_sets.append(fallback_prompt_sets[len(prompt_sets)])
                return self._mark_llm_payload(
                    {
                        "prompt_sets": prompt_sets,
                        "prompt_count": len(prompt_sets),
                    }
                )
        except Exception as exc:
            return self._template_fallback(
                {"prompt_sets": fallback_prompt_sets, "prompt_count": len(fallback_prompt_sets)},
                exc,
                fallback_reason="json_parse_failed",
            )
        return self._template_fallback(
            {"prompt_sets": fallback_prompt_sets, "prompt_count": len(fallback_prompt_sets)},
            fallback_reason="json_parse_failed",
        )

    def prepare_segment(
        self,
        goal: GoalRequest,
        segment: dict[str, Any],
        negative_prompt: str,
        previous_segment: dict[str, Any] | None = None,
        prior_frame: str | None = None,
    ) -> dict[str, Any]:
        fallback = build_segment_prompt(goal, segment, prior_frame)
        fallback["negative_prompt"] = negative_prompt
        manager = self._require_manager()

        continuity_lines = [
            f"Current segment id: {segment.get('segment_id', '')}",
            f"Current segment visual: {segment.get('visual', '')}",
            f"Current segment narration: {segment.get('narration', '')}",
            f"Style: {goal.style}",
            f"Character: {goal.constraints.get('character', '')}",
            f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
            f"Has prior frame path: {'yes' if prior_frame else 'no'}",
        ]
        if previous_segment:
            continuity_lines.extend(
                [
                    f"Previous segment visual: {previous_segment.get('visual', '')}",
                    f"Previous segment narration: {previous_segment.get('narration', '')}",
                ]
            )
        continuity_lines.extend(
                [
                    "Return JSON with keys: prompt, narration.",
                    "The prompt must preserve character identity, continue scene geography, and add meaningful motion.",
                    "If news exists, keep it as stylized motifs or atmosphere rather than literal reporting.",
                ]
            )
        try:
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                "\n".join(continuity_lines),
                schema_name="segment_prompt",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "narration": {"type": "string"},
                    },
                    "required": ["prompt", "narration"],
                    "additionalProperties": False,
                },
            )
            fallback["prompt"] = str(payload.get("prompt") or fallback["prompt"])
            fallback["narration"] = str(payload.get("narration") or fallback["narration"])
            return self._mark_llm_payload(fallback)
        except Exception as exc:
            raise self._generation_error("prepare_segment", exc) from exc

    def refine_prompt_from_review(
        self,
        goal: GoalRequest,
        original_prompt: str,
        review_notes: str,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        fallback = {
            "prompt": ", ".join(
                part
                for part in (
                    original_prompt,
                    f"revision notes: {review_notes}" if review_notes else "",
                    "improve composition, subject clarity, and action readability",
                )
                if part
            ),
            "negative_prompt": "blurry, low quality, identity drift, weak composition, text, watermark",
        }
        try:
            manager = self._require_manager()
            user_prompt = "\n".join(
                [
                    f"Goal: {goal.prompt}",
                    f"Media type: {goal.media_type}",
                    f"Original prompt: {original_prompt}",
                    f"Review notes: {review_notes}",
                    f"Selected media count: {len(media_paths or [])}",
                    "Return JSON with keys: prompt, negative_prompt.",
                    "Improve the prompt while preserving the original intent and character identity.",
                ]
            )
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="review_prompt_refinement",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                    },
                    "required": ["prompt", "negative_prompt"],
                    "additionalProperties": False,
                },
            )
            return self._mark_llm_payload(
                {
                    "prompt": str(payload.get("prompt") or fallback["prompt"]),
                    "negative_prompt": str(payload.get("negative_prompt") or fallback["negative_prompt"]),
                }
            )
        except Exception as exc:
            return self._template_fallback(fallback, exc, fallback_reason="json_parse_failed")

    def build_sticker_motion_prompt(
        self,
        goal: GoalRequest,
        base_prompt: str,
        character: str,
        selected_expression: str = "",
    ) -> dict[str, Any]:
        fallback = {
            "prompt": build_animated_sticker_motion_prompt(goal),
        }
        manager = self._require_manager()

        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Character: {character}",
                f"Base sticker prompt: {base_prompt}",
                f"Primary expression: {selected_expression}",
                f"Style: {goal.style}",
                "Return JSON with key: prompt.",
                "Create a short generation-ready animated sticker motion prompt that preserves silhouette clarity and loopability.",
            ]
        )
        try:
            payload = self._chat_json(
                manager,
                STICKER_SYSTEM_PROMPT,
                user_prompt,
                schema_name="animated_sticker_motion",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            )
            fallback["prompt"] = str(payload.get("prompt") or fallback["prompt"])
            return self._mark_llm_payload(fallback)
        except Exception as exc:
            raise self._generation_error("build_sticker_motion_prompt", exc) from exc

    def build_carousel_prompt_set(
        self,
        goal: GoalRequest,
        segments: list[dict[str, Any]],
        style: str,
    ) -> dict[str, Any]:
        fallback_prompt_sets = [
            {
                "label": f"slide_{index + 1:02d}",
                "prompt": ", ".join(
                    part
                    for part in (
                        str(segment.get("visual", "")),
                        style,
                        "carousel slide, strong composition, clean focal point",
                    )
                    if part
                ),
                "narration": str(segment.get("narration", "")),
            }
            for index, segment in enumerate(segments)
        ]
        fallback = {
            "prompt_sets": fallback_prompt_sets,
            "prompt_count": len(fallback_prompt_sets),
        }
        compact_segments = [
            {
                "segment_id": str(segment.get("segment_id", f"segment-{index + 1}")),
                "visual": str(segment.get("visual", "")),
                "narration": str(segment.get("narration", "")),
            }
            for index, segment in enumerate(segments)
        ]
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Style: {style}",
                f"Segment count: {len(compact_segments)}",
                f"Segments JSON: {json.dumps(compact_segments, ensure_ascii=False)}",
                "Return a JSON object with key prompt_sets.",
                "prompt_sets must be an array where each item has keys: label, prompt, narration.",
                "Each prompt should feel like a distinct carousel slide while preserving subject identity and sequence progression.",
            ]
        )
        try:
            manager = self._require_manager()
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="carousel_prompt_set",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt_sets": {
                            "type": "array",
                            "minItems": len(fallback_prompt_sets),
                            "maxItems": len(fallback_prompt_sets),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "prompt": {"type": "string"},
                                    "narration": {"type": "string"},
                                },
                                "required": ["label", "prompt", "narration"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["prompt_sets"],
                    "additionalProperties": False,
                },
            )
            prompt_items = payload.get("prompt_sets") if isinstance(payload, dict) else payload
            if isinstance(prompt_items, list) and prompt_items:
                prompt_sets: list[dict[str, Any]] = []
                for index, item in enumerate(prompt_items[: len(fallback_prompt_sets)]):
                    base_item = fallback_prompt_sets[index]
                    prompt_sets.append(
                        {
                            "label": str(item.get("label") or base_item["label"]),
                            "prompt": str(item.get("prompt") or base_item["prompt"]),
                            "narration": str(item.get("narration") or base_item["narration"]),
                        }
                    )
                while len(prompt_sets) < len(fallback_prompt_sets):
                    prompt_sets.append(fallback_prompt_sets[len(prompt_sets)])
                return self._mark_llm_payload(
                    {
                        "prompt_sets": prompt_sets,
                        "prompt_count": len(prompt_sets),
                    }
                )
        except Exception as exc:
            return self._template_fallback(fallback, exc)
        return self._template_fallback(fallback)

    def prepare_publish_caption(
        self,
        goal: GoalRequest,
        prefix: str,
        hashtags: list[str],
        platforms: list[str],
        media_paths: list[str] | None = None,
        review_notes: str = "",
    ) -> dict[str, Any]:
        normalized_hashtags = [tag if tag.startswith("#") else f"#{tag}" for tag in hashtags if tag]
        character = str(goal.constraints.get("character", "") or "").strip()
        parts = [part for part in (prefix, goal.prompt, f"style: {goal.style}") if part]
        if character:
            parts.append(f"character: {character}")
        fallback = {
            "caption": " | ".join(parts),
            "hashtags": " ".join(normalized_hashtags),
            "platform_captions": {platform: " | ".join(parts) for platform in platforms},
        }
        manager = self._require_manager()

        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Style: {goal.style}",
                f"Character: {character}",
                f"Platforms: {', '.join(platforms) if platforms else 'generic'}",
                f"Prefix: {prefix}",
                f"Review notes: {review_notes}",
                f"Media count: {len(media_paths or [])}",
                f"Requested hashtags: {', '.join(normalized_hashtags)}",
                "Return JSON with keys: caption, hashtags, platform_captions.",
                "Keep the caption concise, publish-ready, and adapted per platform when platforms are supplied.",
            ]
        )
        try:
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="publish_caption",
                schema={
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "hashtags": {"type": "string"},
                        "platform_captions": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["caption", "hashtags", "platform_captions"],
                    "additionalProperties": False,
                },
            )
            platform_captions = payload.get("platform_captions")
            if not isinstance(platform_captions, dict):
                platform_captions = fallback["platform_captions"]
            return self._mark_llm_payload(
                {
                "caption": str(payload.get("caption") or fallback["caption"]),
                "hashtags": str(payload.get("hashtags") or fallback["hashtags"]),
                "platform_captions": {str(key): str(value) for key, value in platform_captions.items()},
                }
            )
        except Exception as exc:
            raise self._generation_error("prepare_publish_caption", exc) from exc

    def review_asset_candidates(
        self,
        goal: GoalRequest,
        media_paths: list[str],
        review_notes: str,
        selection_limit: int,
    ) -> dict[str, Any]:
        fallback_candidates = []
        for index, media_path in enumerate(media_paths):
            fallback_candidates.append(
                {
                    "media_path": media_path,
                    "score": max(1, 100 - (index * 5)),
                    "rationale": f"Deterministic fallback ranking for candidate #{index + 1}.",
                }
            )
        fallback_candidates = fallback_candidates[:selection_limit]
        fallback = {
            "selected_assets": [item["media_path"] for item in fallback_candidates],
            "ranked_candidates": fallback_candidates,
            "selection_rationale": "Fallback ranking prefers earlier deterministic candidates and publish-friendly ordering.",
            "regeneration_notes": review_notes or "No review notes supplied.",
        }
        manager = self._require_manager()

        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Media type: {goal.media_type}",
                f"Style: {goal.style}",
                f"Review notes: {review_notes}",
                f"Selection limit: {selection_limit}",
                f"Candidate media paths: {json.dumps(media_paths, ensure_ascii=False)}",
                "Return JSON with keys: selected_assets, ranked_candidates, selection_rationale, regeneration_notes.",
                "Each ranked_candidates item must include: media_path, score, rationale.",
                "Prefer assets that look strongest for publishing and best satisfy the review notes based on filenames and ordering signals.",
            ]
        )
        try:
            payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                user_prompt,
                schema_name="asset_review_selection",
                schema={
                    "type": "object",
                    "properties": {
                        "selected_assets": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "ranked_candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "media_path": {"type": "string"},
                                    "score": {"type": "integer"},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["media_path", "score", "rationale"],
                                "additionalProperties": False,
                            },
                        },
                        "selection_rationale": {"type": "string"},
                        "regeneration_notes": {"type": "string"},
                    },
                    "required": [
                        "selected_assets",
                        "ranked_candidates",
                        "selection_rationale",
                        "regeneration_notes",
                    ],
                    "additionalProperties": False,
                },
            )
            ranked_candidates = payload.get("ranked_candidates")
            if not isinstance(ranked_candidates, list):
                ranked_candidates = fallback["ranked_candidates"]
            normalized_ranked: list[dict[str, Any]] = []
            valid_paths = {str(path) for path in media_paths}
            for item in ranked_candidates:
                if not isinstance(item, dict):
                    continue
                media_path = str(item.get("media_path", ""))
                if media_path not in valid_paths:
                    continue
                normalized_ranked.append(
                    {
                        "media_path": media_path,
                        "score": int(item.get("score", 0)),
                        "rationale": str(item.get("rationale", "")),
                    }
                )
            if not normalized_ranked:
                normalized_ranked = fallback_candidates
            selected_assets = payload.get("selected_assets")
            if not isinstance(selected_assets, list):
                selected_assets = [item["media_path"] for item in normalized_ranked]
            normalized_selected = [str(path) for path in selected_assets if str(path) in valid_paths][:selection_limit]
            if not normalized_selected:
                normalized_selected = [item["media_path"] for item in normalized_ranked[:selection_limit]]
            return self._mark_llm_payload(
                {
                "selected_assets": normalized_selected,
                "ranked_candidates": normalized_ranked[: max(selection_limit, len(normalized_selected))],
                "selection_rationale": str(payload.get("selection_rationale") or fallback["selection_rationale"]),
                "regeneration_notes": str(payload.get("regeneration_notes") or fallback["regeneration_notes"]),
                }
            )
        except Exception as exc:
            raise self._generation_error("review_asset_candidates", exc) from exc

    def _template_fallback(
        self,
        payload: dict[str, Any],
        exc: Exception | None = None,
        *,
        fallback_reason: str = "manager_unavailable",
    ) -> dict[str, Any]:
        fallback = dict(payload)
        fallback["prompt_mode"] = "template"
        fallback["fallback_reason"] = fallback_reason
        fallback["llm_backend"] = self.backend_info()
        if exc is not None:
            fallback["manager_error"] = f"{type(exc).__name__}: {exc}"
        elif self._manager_error:
            fallback["manager_error"] = self._manager_error
        return fallback

    @staticmethod
    def _fallback_sticker_expressions(prompt: str, target_count: int) -> list[str]:
        prompt_tail = prompt.split(":", 1)[-1]
        candidates = [item.strip() for item in prompt_tail.split(",") if item.strip()]
        fallback = candidates[:target_count]
        default_pool = [
            "happy",
            "angry",
            "crying",
            "sleepy",
            "surprised",
            "celebrating",
            "confused",
            "love struck",
            "thinking",
            "cheering",
        ]
        for item in default_pool:
            if len(fallback) >= target_count:
                break
            if item not in fallback:
                fallback.append(item)
        return fallback[:target_count]

    def _manager_or_none(self) -> Any | None:
        if self._manager is not None:
            return self._manager
        if self.mode == "template":
            self._manager_error = "LLM prompt generation is required but AGENTIC_LLM_MODE=template."
            return None
        try:
            backend = self.backend_info()
            self._manager = build_llm_manager(backend)
            self._manager_error = None
            return self._manager
        except Exception as exc:
            self._manager_error = f"{type(exc).__name__}: {exc}"
            return None

    def _resolve_backend_info(self) -> dict[str, Any]:
        text_provider = os.environ.get("AGENTIC_TEXT_MODEL_PROVIDER", "openrouter")
        text_model = os.environ.get("AGENTIC_TEXT_MODEL", "qwen/qwen3.6-plus:free")
        vision_provider = os.environ.get("AGENTIC_VISION_MODEL_PROVIDER", text_provider)
        vision_model = os.environ.get("AGENTIC_VISION_MODEL", "qwen/qwen3.6-plus:free")
        random_models = os.environ.get("AGENTIC_RANDOM_MODELS", "").lower() in {"1", "true", "yes"}
        return {
            "mode": self.mode,
            "text_provider": text_provider,
            "text_model": text_model,
            "vision_provider": vision_provider,
            "vision_model": vision_model,
            "random_models": random_models,
        }

    def _require_manager(self) -> Any:
        manager = self._manager_or_none()
        if manager is not None:
            return manager
        raise self._generation_error("manager_initialization")

    def _generation_error(self, operation: str, exc: Exception | None = None) -> PromptGenerationError:
        backend = self.backend_info()
        details = self._manager_error
        if exc is not None:
            details = f"{type(exc).__name__}: {exc}"
        message = f"LLM prompt generation failed during {operation}."
        if details:
            message = f"{message} {details}"
        message = f"{message} Backend={json.dumps(backend, ensure_ascii=False, sort_keys=True)}"
        return PromptGenerationError(message)

    @staticmethod
    def _chat_json(
        manager: Any,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> Any:
        response = manager.text_model.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        return LLMPromptEngine._parse_json(response)

    @staticmethod
    def _parse_json(response: str) -> Any:
        cleaned = response.strip()
        if not cleaned:
            raise json.JSONDecodeError("Expecting value", cleaned, 0)
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>")[-1]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            fragment = LLMPromptEngine._extract_json_fragment(cleaned)
            if fragment:
                return json.loads(fragment)
            raise

    @staticmethod
    def _extract_json_fragment(text: str) -> str | None:
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            stack: list[str] = []
            in_string = False
            escaped = False
            for end in range(index, len(text)):
                current = text[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == '"':
                        in_string = False
                    continue
                if current == '"':
                    in_string = True
                    continue
                if current in "{[":
                    stack.append(current)
                    continue
                if current == "}" and stack and stack[-1] == "{":
                    stack.pop()
                elif current == "]" and stack and stack[-1] == "[":
                    stack.pop()
                elif current in "}]":
                    break
                if not stack:
                    return text[index : end + 1]
        return None
