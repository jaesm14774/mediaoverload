from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_manager_adapter import build_llm_manager
from agentic.runtime.model_backends import _load_project_env
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
from agentic.storyboard import (
    evaluate_native_h3_story_quality,
    native_h3_duration_from_times,
    native_h3_shot_times,
    validate_native_h3_shot_timing,
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
        if self._manager is not None:
            tm = self._manager.text_model
            primary = getattr(tm, "_primary", tm)
            last = getattr(primary, "last_success_model", "") or getattr(tm, "last_success_model", "")
            if last:
                enriched["openrouter_last_text_model"] = last
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
        route_schema = self._build_generation_strategy_schema(
            generation_type_candidates=normalized_candidates,
            workflow_stage_candidates=workflow_stage_candidates,
            count_policies=count_policies,
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
                schema=route_schema,
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

    @staticmethod
    def _build_generation_strategy_schema(
        *,
        generation_type_candidates: list[str],
        workflow_stage_candidates: dict[str, dict[str, list[str]]],
        count_policies: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        branches: list[dict[str, Any]] = []
        for generation_type in generation_type_candidates:
            branches.append(
                {
                    "type": "object",
                    "properties": {
                        "generation_type": {"const": generation_type},
                        "workflow_plan": LLMPromptEngine._build_workflow_plan_schema(
                            workflow_stage_candidates.get(generation_type, {})
                        ),
                        "count_plan": LLMPromptEngine._build_count_plan_schema(
                            count_policies.get(generation_type, {})
                        ),
                        "reason": {"type": "string"},
                    },
                    "required": ["generation_type", "workflow_plan", "count_plan", "reason"],
                    "additionalProperties": False,
                }
            )
        return {"oneOf": branches}

    @staticmethod
    def _build_workflow_plan_schema(stage_candidates: dict[str, list[str]]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for stage_key in WORKFLOW_STAGE_KEYS:
            allowed_workflows = [
                str(item).strip()
                for item in stage_candidates.get(stage_key, [])
                if str(item).strip()
            ]
            if allowed_workflows:
                properties[stage_key] = {"type": "string", "enum": allowed_workflows + [""]}
            else:
                properties[stage_key] = {"const": ""}
        return {
            "type": "object",
            "properties": properties,
            "required": list(WORKFLOW_STAGE_KEYS),
            "additionalProperties": False,
        }

    @staticmethod
    def _build_count_plan_schema(count_policy: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required_keys = (
            "image_count",
            "video_count",
            "segment_count",
            "review_selection_limit",
            "sticker_expression_count",
            "images_per_prompt",
        )
        for count_key in required_keys:
            policy = count_policy.get(count_key)
            if isinstance(policy, dict):
                minimum = int(policy.get("min", 1))
                maximum = int(policy.get("max", minimum))
                properties[count_key] = {
                    "type": "integer",
                    "minimum": minimum,
                    "maximum": max(minimum, maximum),
                }
            else:
                properties[count_key] = {"type": "integer", "minimum": 1, "maximum": 32}
        return {
            "type": "object",
            "properties": properties,
            "required": list(required_keys),
            "additionalProperties": False,
        }

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

    def generate_native_h3_storyboard(
        self,
        *,
        character: str,
        style: str,
        duration_seconds: int,
        base_storyboard: dict[str, Any],
        news_context: dict[str, Any],
        creative_brief: str = "",
    ) -> dict[str, Any]:
        """Generate the complete causal story consumed by native H3.

        This intentionally has no template fallback. Native H3 must either
        receive a valid news-grounded storyboard from the configured LLM or
        fail before any keyframe/video workflow is submitted.
        """
        manager = self._require_manager()
        expected_times = native_h3_shot_times(base_storyboard)
        if int(duration_seconds) not in {15, 20}:
            raise PromptGenerationError("Native H3 storyboard generation currently supports duration_seconds=15 or 20.")
        world = dict(base_storyboard.get("world") or {})
        continuity_rules = world.get("continuity_rules") or []
        if not isinstance(continuity_rules, list) or not continuity_rules:
            raise PromptGenerationError(
                "Native H3 base storyboard must define non-empty world.continuity_rules."
            )
        for key, value in news_context.items():
            if isinstance(value, str) and len(value) > 2000:
                raise PromptGenerationError(f"Native H3 news_context.{key} exceeds 2000 characters.")
        if len(str(creative_brief or "")) > 2000:
            raise PromptGenerationError("Native H3 creative_brief exceeds 2000 characters.")
        schema = {
            "type": "object",
            "properties": {
                "story": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "base_prompt": {"type": "string"},
                        "opening_keyframe_prompt": {"type": "string"},
                        "ending_keyframe_prompt": {"type": "string"},
                        "negative_prompt": {"type": "string"},
                        "story_spine": {
                            "type": "object",
                            "properties": {
                                "premise": {"type": "string"},
                                "objective": {"type": "string"},
                                "obstacle": {"type": "string"},
                                "stakes": {"type": "string"},
                                "emotional_arc": {"type": "string"},
                                "climax": {"type": "string"},
                                "resolution": {"type": "string"},
                            },
                            "required": [
                                "premise",
                                "objective",
                                "obstacle",
                                "stakes",
                                "emotional_arc",
                                "climax",
                                "resolution",
                            ],
                            "additionalProperties": False,
                        },
                        "world": {
                            "type": "object",
                            "properties": {
                                "setting": {"type": "string"},
                                "visual_language": {"type": "string"},
                                "continuity_rules": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "maxItems": 12,
                                },
                            },
                            "required": ["setting", "visual_language", "continuity_rules"],
                            "additionalProperties": False,
                        },
                        "native_audio": {"type": "string"},
                        "native_shots": {
                            "type": "array",
                            "minItems": len(expected_times),
                            "maxItems": len(expected_times),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "time": {"type": "string"},
                                    "title": {"type": "string"},
                                    "action": {"type": "string"},
                                    "camera": {"type": "string"},
                                    "state_change": {"type": "string"},
                                },
                                "required": ["time", "title", "action", "camera", "state_change"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "name",
                        "base_prompt",
                        "opening_keyframe_prompt",
                        "ending_keyframe_prompt",
                        "negative_prompt",
                        "story_spine",
                        "world",
                        "native_audio",
                        "native_shots",
                    ],
                    "additionalProperties": False,
                },
                "creative_seed": {"type": "string"},
                "source": {"type": "string"},
            },
            "required": ["story", "creative_seed", "source"],
            "additionalProperties": False,
        }
        self._apply_native_h3_schema_limits(schema)
        safe_creative_brief = self._sanitize_native_h3_creative_brief(creative_brief)
        user_prompt = "\n".join(
            [
                f"Character: {character}",
                f"Style: {style}",
                f"Duration seconds: {int(duration_seconds)}",
                f"Creative brief: {safe_creative_brief}",
                f"News context JSON: {json.dumps(news_context, ensure_ascii=False)}",
                f"Generate a new, original, publishable short-form story for one continuous native H3 clip with {len(expected_times)} causal beats.",
                "The news context is mandatory creative input: translate it into concrete visual motifs, a tension, or a prop; do not recreate the headline literally.",
                "The character must remain the clear protagonist and the story must be complete within the requested duration.",
                "base_prompt is an identity-and-animation-style anchor only: keep it concise, do not describe a calm/peaceful opening, fixed camera, or a posed character, and do not let it conflict with the first shot's disruption.",
                "Write the positive video description in MiniMax H3's integrated multimodal order: each shot has a timestamp, visible action, camera movement, and state change; keep overall_soundscape and non_diegetic_music as separate audio directions.",
                "Attach the camera instruction to the action it controls. Prefer one concrete camera movement per beat (follow, push in, pan, tilt, pull out, or a deliberate static hold after motion), and use a readable physical handoff between beats.",
                f"Return exactly {len(expected_times)} causal native_shots. Use these recommended beat windows as a pacing guide: {', '.join(expected_times)}. You may adjust internal boundaries, but the numeric ranges must be contiguous from 0s to {int(duration_seconds)}s with no gaps or overlap. Use the beat order hook, promise, escalation, reversal, payoff when five beats are requested; each beat must change the mission state and hand off visibly to the next beat.",
                "Story quality contract: the hook must show a striking disruption within the first second and create one concrete question; the protagonist must visibly want one thing and risk losing something specific.",
                "The middle must contain a visible setback or reversal that costs the protagonist something and changes the plan; do not describe a smooth journey with no price.",
                "The final beat must resolve the same objective introduced in the hook and show physical evidence of the resolution; do not introduce a new quest or end on an unexplained spectacle.",
                "Short-video rhythm contract: every beat has one dominant physical action, one visible composition change, one visible state change, and one reason to keep watching; avoid an atmospheric opening with no problem.",
                "Prefer one simple, readable threat that can be seen in silhouette (for example a strong gust, collapsing surface, rolling object, or spreading shadow). Avoid complex machines, distant facilities, extra characters, or background lore that the video model may ignore.",
                "Write actions as visible cause-and-effect: the threat must displace the central prop, Kirby must visibly fail or lose ground, the plan must change, and the payoff must visibly restore the prop or world. Do not let the character merely look at, hold, or pose with the prop across multiple beats.",
                "Use concrete verbs and consequences in state_change. Avoid vague phrases such as Kirby reacts, the mood shifts, the scene becomes exciting, or the story progresses.",
                "Do not use readable words, letters, numbers, signs, labels, subtitles, headlines, or written symbols anywhere in the visuals; communicate the idea through shape, color, gesture, and physical props only.",
                "Do not use writing-bearing props or marked surfaces such as documents, reports, newspapers, ledgers, charts, screens, interfaces, glyphs, runes, or financial symbols. Translate news into unmarked physical shapes, color, light, motion, and environmental change.",
                "The first shot must show visible character or camera motion within the first second and must not hold the opening pose; every beat must change composition and mission state rather than repeating a setup.",
                "Do not reuse the base storyboard's plot, props, setting, or ending. The base storyboard only supplies character and continuity rules.",
                f"Character identity rule: one {character} only; preserve its identity, proportions, silhouette, and palette in every shot.",
                "Do not add humans, extra named characters, subtitles, logos, or written news text.",
            ]
        )
        payload = self._chat_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="native_h3_storyboard",
            schema=schema,
        )
        story: dict[str, Any] | None = None
        validation_error: PromptGenerationError | None = None
        for repair_round in range(4):
            try:
                story = self._validate_native_h3_story_payload(
                    payload,
                    expected_times=expected_times,
                    duration_seconds=duration_seconds,
                )
                break
            except PromptGenerationError as error:
                validation_error = error
                if repair_round >= 3:
                    raise
            repaired_payload = self._chat_json(
                manager,
                LONG_VIDEO_SYSTEM_PROMPT,
                self._build_native_h3_repair_prompt(
                    user_prompt,
                    validation_error,
                    expected_times=expected_times,
                    duration_seconds=duration_seconds,
                ),
                schema_name=f"native_h3_storyboard_repair_{repair_round + 1}",
                schema=schema,
            )
            payload = repaired_payload
        if story is None:
            raise validation_error or PromptGenerationError("Native H3 story validation did not produce a story.")
        return self._mark_llm_payload(
            {
                "story": story,
                "creative_seed": str(payload.get("creative_seed") or "").strip(),
                "source": str(payload.get("source") or "native_h3_llm").strip(),
                "news_context": dict(news_context),
                "story_quality": evaluate_native_h3_story_quality(story, expected_times=expected_times),
            }
        )

    @staticmethod
    def _sanitize_native_h3_creative_brief(creative_brief: str) -> str:
        text = " ".join(str(creative_brief or "").split()).strip()
        if not text:
            return ""
        risky_patterns = (
            r"\d",
            r"%",
            r"ticker",
            r"document",
            r"report",
            r"newspaper",
            r"ledger",
            r"chart",
            r"graph",
            r"screen",
            r"interface",
            r"symbol",
            r"數字",
            r"報表",
            r"圖表",
            r"股票",
            r"營收",
            r"台股",
        )
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in risky_patterns):
            return (
                "Use only abstract atmosphere, color, weather, and emotional emphasis from the supplied brief; "
                "do not copy its objects, text-bearing props, figures, named locations, or literal reporting."
            )
        return text[:600]

    @staticmethod
    def _build_native_h3_repair_prompt(
        user_prompt: str,
        validation_error: PromptGenerationError,
        *,
        expected_times: tuple[str, ...] | list[str] | None = None,
        duration_seconds: int | float | None = None,
    ) -> str:
        """Build a repair request without echoing forbidden cue vocabulary.

        The first implementation appended the complete validation error and the
        original prohibition list to the repair request. That gave the LLM the
        exact words the validator rejects and could make it copy them into
        positive visual fields again. Keep structural instructions, replace the
        text-cue rules with an abstract repair instruction, and preserve the
        no-fallback behavior if the replacement still fails validation.
        """
        error_text = str(validation_error)
        repair_times = tuple(expected_times or ("0-4s", "4-10s", "10-15s"))
        repair_duration = int(duration_seconds or native_h3_duration_from_times(repair_times))
        visual_repair = "forbidden readable-text visual cues" in error_text
        retained_lines = []
        for line in str(user_prompt).splitlines():
            if line.startswith("Do not use readable words"):
                continue
            if line.startswith("Do not use writing-bearing props"):
                continue
            if line.startswith("Do not add humans"):
                continue
            if visual_repair and (
                line.startswith("Creative brief:") or line.startswith("News context JSON:")
            ):
                continue
            retained_lines.append(line)
        if visual_repair:
            retained_lines.append(
                "Use the news only as an abstract emotional or environmental cue; do not reproduce headline wording, figures, names, marks, or text-bearing objects."
            )
        if visual_repair:
            issue = (
                "The positive visual fields contain a writing or signage cue, or a copied rule about such a cue. "
                "Rewrite those fields using only visible action, camera, environment, lighting, and physical props. "
                "Put all exclusions only in negative_prompt; do not mention the validation rule in any positive field."
            )
        elif "story_spine missing required values:" in error_text:
            missing_fields = error_text.split("story_spine missing required values:", 1)[1].strip()
            issue = (
                f"Fill these missing story_spine fields with concrete story content: {missing_fields}. "
                "They must be non-empty consequences or stakes tied to this story, not placeholders. "
                "Do not omit any other story_spine field."
            )
        elif "base_prompt must be an identity" in error_text:
            issue = (
                "Rewrite base_prompt as a concise character identity and animation-style anchor only. "
                "Remove calm, peaceful, serene, static, fixed-camera, or posed-opening language; the first shot must begin with visible disruption."
            )
        elif "native_shots item" in error_text and "missing values:" in error_text:
            issue = (
                f"Repair the incomplete shot described by this validation issue: {error_text}. "
                f"Return exactly {len(repair_times)} shots using {', '.join(repair_times)} as recommended windows. Their numeric ranges must be contiguous from 0s to {repair_duration}s. Every shot must contain a non-empty time, title, action, camera, and state_change."
            )
        elif "hook must contain a visible disruption or motion in the opening beat" in error_text:
            issue = (
                "Repair the opening beat specifically. Rewrite both native_shots[0].action and native_shots[0].camera, "
                "while preserving the same protagonist, objective, setting, and visual identity. The first second must show "
                "one unmistakable physical event that changes screen position or physical state: name the obstacle or central prop, "
                "show what moves or is displaced, and show Kirby reacting to it. Use a concrete visible motion verb such as "
                "forms, whips, tears loose, rushes, grabs, slides, falls, bursts, or flies. Do not open on a static pose, "
                "observation, waiting, or an atmosphere-only establishing frame. The camera must also describe motion or a motivated "
                "reframe that follows this event. Keep every later shot causal and make the second shot a consequential setback."
            )
        elif "story quality is insufficient:" in error_text:
            beat_contract = (
                "make the second shot a costly setback or reversal that changes the plan; "
                "make the final shot resolve the same objective"
            )
            issue = (
                "Rewrite the causal story, not just the wording. Make the first shot a visible disruption with a clear protagonist goal; "
                f"{beat_contract} with a concrete visible result. Use specific physical verbs and consequences in every state_change."
            )
        else:
            issue = f"Repair the structural validation issue: {error_text}"
        retained_lines.extend(
            [
                "QUALITY REPAIR: The previous JSON parsed but failed native H3 story validation.",
                issue,
                "Return a complete replacement JSON object, not a patch. The story_spine must contain non-empty premise, objective, obstacle, stakes, emotional_arc, climax, and resolution fields.",
                f"Also include exactly {len(repair_times)} distinct native_shots with contiguous numeric ranges covering 0s to {repair_duration}s; each shot needs non-empty time, title, action, camera, and state_change.",
                "Self-check every required field before returning JSON. If a field is not relevant, rewrite the story so it is relevant; never omit or leave it blank.",
            ]
        )
        return "\n".join(retained_lines)

    @staticmethod
    def _apply_native_h3_schema_limits(schema: dict[str, Any], max_length: int = 1600) -> None:
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "string":
                    node["maxLength"] = max_length
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(schema)

    @staticmethod
    def _validate_native_h3_story_payload(
        payload: Any,
        *,
        expected_times: tuple[str, ...] | list[str] | None = None,
        duration_seconds: int | float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict) or not isinstance(payload.get("story"), dict):
            raise PromptGenerationError("Native H3 LLM response did not contain a story object.")
        story = payload["story"]
        spine = story.get("story_spine")
        shots = story.get("native_shots")
        shot_times = tuple(expected_times or ("0-4s", "4-10s", "10-15s"))
        if not isinstance(spine, dict) or not isinstance(shots, list) or len(shots) != len(shot_times):
            raise PromptGenerationError(
                f"Native H3 LLM response must contain story_spine and exactly {len(shot_times)} native_shots."
            )
        required_spine = (
            "premise",
            "objective",
            "obstacle",
            "stakes",
            "emotional_arc",
            "climax",
            "resolution",
        )
        missing_spine = [key for key in required_spine if not str(spine.get(key) or "").strip()]
        if missing_spine:
            raise PromptGenerationError(
                "Native H3 LLM story_spine missing required values: " + ", ".join(missing_spine)
            )
        required_shot = ("time", "title", "action", "camera", "state_change")
        shot_titles: list[str] = []
        state_changes: list[str] = []
        for index, shot in enumerate(shots, start=1):
            if not isinstance(shot, dict):
                raise PromptGenerationError(f"Native H3 native_shots item {index} is not an object.")
            missing = [key for key in required_shot if not str(shot.get(key) or "").strip()]
            if missing:
                raise PromptGenerationError(
                    f"Native H3 native_shots item {index} missing values: " + ", ".join(missing)
                )
            shot_titles.append(str(shot["title"]).strip().lower())
            state_changes.append(str(shot["state_change"]).strip().lower())
        timing_ok, timing_error = validate_native_h3_shot_timing(
            shots,
            duration_seconds=float(duration_seconds or native_h3_duration_from_times(shot_times)),
        )
        if not timing_ok:
            raise PromptGenerationError("Native H3 native_shots timing is invalid: " + timing_error)
        if len(set(shot_titles)) != len(shot_times):
            raise PromptGenerationError("Native H3 native_shots titles must be distinct across all beats.")
        if len(set(state_changes)) != len(shot_times):
            raise PromptGenerationError("Native H3 native_shots must contain distinct state changes for every beat.")
        quality = evaluate_native_h3_story_quality(story, expected_times=shot_times)
        if not quality["passed"]:
            raise PromptGenerationError(
                "Native H3 story quality is insufficient: " + "; ".join(str(error) for error in quality["errors"])
            )
        visual_fields: list[str] = []
        for key in ("base_prompt", "opening_keyframe_prompt", "ending_keyframe_prompt"):
            value = story.get(key)
            if isinstance(value, str):
                visual_fields.append(value)
        base_prompt = str(story.get("base_prompt") or "")
        if re.search(r"\b(?:calm|peaceful|serene|no immediate threats|fixed camera|posed character)\b", base_prompt, flags=re.IGNORECASE):
            raise PromptGenerationError(
                "Native H3 base_prompt must be an identity-and-style anchor and must not establish a calm or posed opening."
            )
        world = story.get("world")
        if isinstance(world, dict):
            for key in ("setting", "visual_language"):
                value = world.get(key)
                if isinstance(value, str):
                    visual_fields.append(value)
        for shot in shots:
            for key in ("title", "action", "camera", "state_change"):
                value = shot.get(key)
                if isinstance(value, str):
                    visual_fields.append(value)
        visual_story_text = "\n".join(visual_fields).lower()
        forbidden_visual_patterns = (
            ("reads", r"\breads\b"),
            ("written", r"\bwritten\b"),
            ("readable word", r"\breadable\s+word\b"),
            ("readable text", r"\breadable\s+text\b"),
            ("letters", r"\bletters?\b"),
            ("numbers", r"\bnumbers?\b"),
            ("sign says", r"\bsign\s+says\b"),
            ("sign reads", r"\bsign\s+reads\b"),
            ("subtitle", r"\bsubtitles?\b"),
            ("headline", r"\bheadlines?\b"),
            ("ticker", r"\btickers?\b"),
            ("document", r"\bdocuments?\b"),
            ("report", r"\breports?\b"),
            ("newspaper", r"\bnewspapers?\b"),
            ("ledger", r"\bledgers?\b"),
            ("chart", r"\bcharts?\b"),
            ("graph", r"\bgraphs?\b"),
            ("screen", r"\bscreens?\b"),
            ("interface", r"\binterfaces?\b"),
            ("signage", r"\bsignage\b"),
            ("glyph", r"\bglyphs?\b"),
            ("rune", r"\brunes?\b"),
            ("symbol", r"\bsymbols?\b"),
        )
        violations = [label for label, pattern in forbidden_visual_patterns if re.search(pattern, visual_story_text)]
        if violations:
            raise PromptGenerationError(
                "Native H3 story contains forbidden readable-text visual cues: " + ", ".join(violations)
            )
        LLMPromptEngine._validate_native_h3_text_lengths(story)
        return story

    @staticmethod
    def _validate_native_h3_text_lengths(story: dict[str, Any], max_length: int = 1600) -> None:
        def visit(value: Any, path: str) -> None:
            if isinstance(value, str) and len(value) > max_length:
                raise PromptGenerationError(f"Native H3 {path} exceeds {max_length} characters.")
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(story, "story")

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
                    "Build the prompt in this order: Subject, Scene, Action, Environment, Camera, Style and lighting, Quality.",
                    "The prompt must be generation-ready for diffusion and image-to-video models; use concrete visible nouns and verbs rather than abstract mood words.",
                    "For image-to-video, describe how the supplied image starts moving and evolves; do not spend the prompt redrawing the static image.",
                    "For text-to-video, establish the subject inside the first moving action instead of opening on a character sheet or posed portrait.",
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
                    "Create a concise generation-ready prompt using this order: Subject, Scene, Action, Environment, Camera, Style and lighting, Quality.",
                    "Use one primary physical action with a visible beginning, change, and end; place the camera instruction next to the action it controls.",
                    "When a prior/first frame is supplied, treat it as authoritative and describe motion/evolution from that frame rather than restating its appearance.",
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
                "segments must be an array where each item has keys: segment_id, visual, narration; when possible also provide action, camera, start_state, end_state, cause, and effect.",
                "Every segment must preserve identity, use one primary physical action, include a concrete camera instruction beside that action, and visibly hand off its end_state to the next segment.",
                "The opening must create a question immediately; the middle must change the plan or cost the protagonist something; the final segment must visibly answer the opening question.",
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
                                    "action": {"type": "string"},
                                    "camera": {"type": "string"},
                                    "start_state": {"type": "string"},
                                    "end_state": {"type": "string"},
                                    "cause": {"type": "string"},
                                    "effect": {"type": "string"},
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
                            "action": str(item.get("action") or fallback[index].get("action") or ""),
                            "camera": str(item.get("camera") or fallback[index].get("camera") or ""),
                            "start_state": str(item.get("start_state") or fallback[index].get("start_state") or ""),
                            "end_state": str(item.get("end_state") or fallback[index].get("end_state") or ""),
                            "cause": str(item.get("cause") or fallback[index].get("cause") or ""),
                            "effect": str(item.get("effect") or fallback[index].get("effect") or ""),
                            "stage": fallback[index].get("stage"),
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
                    "Use the MiniMax temporal order: subject continuity, current scene state, one primary physical action, camera movement attached to that action, visible end state, then audio or style.",
                    "Preserve character identity and scene geography; make the first half-second active and make the next state visibly different from the previous segment.",
                    "If a prior frame exists, describe only how the frame comes alive and evolves; do not replace it with a new composition.",
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
                    "Rewrite it in the order Subject, Scene, Action, Environment, Camera, Style and lighting, Quality; convert review notes into concrete visible changes.",
                    "If motion was weak, add one primary physical action with a clear start-to-end change and attach an explicit camera movement to it.",
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

        include_youtube = "youtube" in [p.lower() for p in platforms]
        youtube_instructions = (
            [
                "For youtube: also return youtube_title (max 100 chars, catchy standalone video title) "
                "and youtube_tags (list of short keyword strings without #).",
            ]
            if include_youtube
            else []
        )
        schema_properties: dict[str, Any] = {
            "caption": {"type": "string"},
            "hashtags": {"type": "string"},
            "platform_captions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        }
        if include_youtube:
            schema_properties["youtube_title"] = {"type": "string"}
            schema_properties["youtube_tags"] = {"type": "array", "items": {"type": "string"}}
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Style: {goal.style}",
                f"Character: {character}",
                f"Platforms: {', '.join(platforms) if platforms else 'generic'}",
                f"Prefix: {prefix}",
                f"Review notes: {review_notes}",
                f"Media count: {len(media_paths or [])}",
                f"Required hashtags to include: {', '.join(normalized_hashtags)}",
                "Return JSON with keys: caption, hashtags, platform_captions.",
                "Write a concise publish-ready social caption, not a review note or metadata dump.",
                "Generate a fresh hashtag line that fits the scene and character instead of echoing only the required hashtags.",
                "If required hashtags are provided, include them naturally inside the final hashtag line.",
                "Keep hashtags space-separated and prefixed with #.",
                "Adapt platform_captions per platform when platforms are supplied.",
                *youtube_instructions,
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
                    "properties": schema_properties,
                    "required": ["caption", "hashtags", "platform_captions"],
                    "additionalProperties": False,
                },
            )
            platform_captions = payload.get("platform_captions")
            if not isinstance(platform_captions, dict):
                platform_captions = fallback["platform_captions"]
            normalized_caption = str(payload.get("caption") or fallback["caption"]).strip()
            normalized_hashtag_text = self._normalize_hashtag_text(
                str(payload.get("hashtags") or fallback["hashtags"]).strip(),
                required_hashtags=normalized_hashtags,
            )
            result: dict[str, Any] = {
                "caption": normalized_caption,
                "hashtags": normalized_hashtag_text,
                "platform_captions": {str(key): str(value).strip() for key, value in platform_captions.items()},
            }
            if include_youtube:
                raw_yt_title = payload.get("youtube_title")
                raw_yt_tags = payload.get("youtube_tags")
                result["youtube_title"] = str(raw_yt_title or "").strip()
                result["youtube_tags"] = [str(tag).strip() for tag in (raw_yt_tags or []) if str(tag).strip()]
            return self._mark_llm_payload(result)
        except Exception as exc:
            raise self._generation_error("prepare_publish_caption", exc) from exc

    @staticmethod
    def _normalize_hashtag_text(hashtags: str, *, required_hashtags: list[str]) -> str:
        seen: list[str] = []
        for token in str(hashtags or "").replace("\n", " ").split():
            cleaned = token.strip().rstrip(".,;")
            if not cleaned:
                continue
            if not cleaned.startswith("#"):
                cleaned = f"#{cleaned.lstrip('#')}"
            if cleaned not in seen:
                seen.append(cleaned)
        for tag in required_hashtags:
            cleaned = tag if str(tag).startswith("#") else f"#{str(tag).lstrip('#')}"
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return " ".join(seen)

    def review_asset_candidates(
        self,
        goal: GoalRequest,
        media_paths: list[str],
        review_notes: str,
        selection_limit: int,
    ) -> dict[str, Any]:
        ranked_media_paths = self._rank_media_by_prompt_match(goal, media_paths)
        candidate_pool = ranked_media_paths[: max(selection_limit, min(len(ranked_media_paths), 10))]
        fallback_candidates = []
        for index, media_path in enumerate(candidate_pool):
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
                f"Candidate media paths: {json.dumps(candidate_pool, ensure_ascii=False)}",
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
            valid_paths = {str(path) for path in candidate_pool}
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

    def _rank_media_by_prompt_match(self, goal: GoalRequest, media_paths: list[str]) -> list[str]:
        existing_paths = [str(path) for path in media_paths if Path(str(path)).exists()]
        missing_paths = [str(path) for path in media_paths if not Path(str(path)).exists()]
        if not existing_paths:
            return [str(path) for path in media_paths]

        fallback_ranked = existing_paths + missing_paths
        manager = self._manager_or_none()
        if manager is None:
            return fallback_ranked

        try:
            analyses: list[dict[str, Any]] = []
            character = str(goal.constraints.get("character", "") or "").strip()
            for media_path in existing_paths:
                payload = self._chat_json(
                    manager,
                    LONG_VIDEO_SYSTEM_PROMPT,
                    "\n".join(
                        [
                            f"Goal: {goal.prompt}",
                            f"Style: {goal.style}",
                            f"Character: {character}",
                            "Score how well this image matches the goal and character identity.",
                            "Return JSON with keys: score, rationale.",
                            "Score must be an integer from 0 to 100.",
                            "Penalize images that clearly mismatch the requested subject, action, setting, or character.",
                        ]
                    ),
                    schema_name="media_prompt_match",
                    schema={
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer", "minimum": 0, "maximum": 100},
                            "rationale": {"type": "string"},
                        },
                        "required": ["score", "rationale"],
                        "additionalProperties": False,
                    },
                    model="vision",
                    images=[media_path],
                )
                analyses.append(
                    {
                        "media_path": media_path,
                        "score": int(payload.get("score", 0)),
                        "rationale": str(payload.get("rationale", "")).strip(),
                    }
                )
            analyses.sort(key=lambda item: (-int(item["score"]), str(item["media_path"])))
            return [str(item["media_path"]) for item in analyses] + missing_paths
        except Exception:
            return fallback_ranked

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
        _load_project_env()
        text_provider = str(os.environ.get("AGENTIC_TEXT_MODEL_PROVIDER", "openrouter") or "openrouter").strip() or "openrouter"
        text_model_raw = str(os.environ.get("AGENTIC_TEXT_MODEL", "") or "").strip()
        vision_provider = str(os.environ.get("AGENTIC_VISION_MODEL_PROVIDER", text_provider) or text_provider).strip() or text_provider
        vision_model_raw = str(os.environ.get("AGENTIC_VISION_MODEL", "") or "").strip()
        random_models = os.environ.get("AGENTIC_RANDOM_MODELS", "true").lower() in {"1", "true", "yes"}

        text_strategy = os.environ.get("AGENTIC_OPENROUTER_TEXT_MODEL_STRATEGY", "").strip().lower()
        openrouter_text_pool_mode = text_strategy == "free_pool" or (
            text_provider.lower() == "openrouter" and text_model_raw.strip() == ""
        )
        vision_strategy = os.environ.get("AGENTIC_OPENROUTER_VISION_MODEL_STRATEGY", "").strip().lower()
        openrouter_vision_pool_mode = vision_strategy == "free_pool" or (
            vision_provider.lower() == "openrouter" and vision_model_raw.strip() == ""
        )

        if openrouter_text_pool_mode:
            text_model_display = "free_pool"
        else:
            text_model_display = text_model_raw.strip() or "free_pool"

        if openrouter_vision_pool_mode:
            vision_model_display = "free_pool"
        else:
            vision_model_display = vision_model_raw.strip() or "free_pool"

        rotate_text = os.environ.get("AGENTIC_OPENROUTER_ROTATE_TEXT_MODELS", "true").lower() in {"1", "true", "yes"}
        rotate_vision = os.environ.get("AGENTIC_OPENROUTER_ROTATE_VISION_MODELS", "true").lower() in {
            "1",
            "true",
            "yes",
        }

        max_text_s = os.environ.get("AGENTIC_OPENROUTER_MAX_TEXT_MODELS_PER_CALL", "").strip()
        max_vision_s = os.environ.get("AGENTIC_OPENROUTER_MAX_VISION_MODELS_PER_CALL", "").strip()
        max_text_models = int(max_text_s) if max_text_s.isdigit() else 0
        max_vision_models = int(max_vision_s) if max_vision_s.isdigit() else 0
        discover_models = os.environ.get("AGENTIC_OPENROUTER_DISCOVER_MODELS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        text_models = [
            item.strip()
            for item in os.environ.get("AGENTIC_OPENROUTER_TEXT_MODELS", "").split(",")
            if item.strip()
        ]
        vision_models = [
            item.strip()
            for item in os.environ.get("AGENTIC_OPENROUTER_VISION_MODELS", "").split(",")
            if item.strip()
        ]
        free_pool_s = os.environ.get("AGENTIC_OPENROUTER_FREE_POOL_SIZE", "5").strip()
        free_pool_size = int(free_pool_s) if free_pool_s.isdigit() else 5
        cache_ttl_s = os.environ.get("AGENTIC_OPENROUTER_MODEL_CACHE_TTL_SECONDS", "21600").strip()
        cache_ttl_seconds = int(cache_ttl_s) if cache_ttl_s.isdigit() else 21600

        text_fallback_provider = os.environ.get("AGENTIC_TEXT_FALLBACK_PROVIDER", "").strip()
        text_fallback_model = os.environ.get("AGENTIC_TEXT_FALLBACK_MODEL", "").strip()
        allow_text_fallback = os.environ.get("AGENTIC_TEXT_ALLOW_FALLBACK", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        return {
            "mode": self.mode,
            "text_provider": text_provider,
            "text_model": text_model_display,
            "text_model_raw": text_model_raw,
            "vision_provider": vision_provider,
            "vision_model": vision_model_display,
            "vision_model_raw": vision_model_raw,
            "random_models": random_models,
            "openrouter_text_pool_mode": openrouter_text_pool_mode,
            "openrouter_vision_pool_mode": openrouter_vision_pool_mode,
            "openrouter_rotate_text_models": rotate_text,
            "openrouter_rotate_vision_models": rotate_vision,
            "openrouter_max_text_models_per_call": max_text_models,
            "openrouter_max_vision_models_per_call": max_vision_models,
            "openrouter_discover_models": discover_models,
            "openrouter_text_models": text_models,
            "openrouter_vision_models": vision_models,
            "openrouter_free_pool_size": max(0, free_pool_size),
            "openrouter_model_cache_ttl_seconds": max(0, cache_ttl_seconds),
            "text_fallback_provider": text_fallback_provider,
            "text_fallback_model": text_fallback_model,
            "allow_text_fallback": allow_text_fallback,
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
        *,
        model: str = "text",
        images: list[str] | None = None,
    ) -> Any:
        chat_model = manager.vision_model if model == "vision" else manager.text_model
        expected_json = "JSON array" if schema.get("type") == "array" else "JSON object"
        opening = "[" if expected_json == "JSON array" else "{"
        closing = "]" if expected_json == "JSON array" else "}"
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    f"OUTPUT CONTRACT: Return exactly one valid {expected_json} that matches the supplied schema. "
                    "Do not return markdown, code fences, XML tags, explanations, analysis, or a second object. "
                    f"Put the JSON value directly in the final answer, starting with {opening} and ending with {closing}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{user_prompt}\n\n"
                    "FINAL FORMAT: output JSON only. Before sending, silently validate that it parses as one complete "
                    f"{expected_json} and that all required fields are present."
                ),
            },
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }
        first_error: Exception | None = None
        try:
            response = chat_model.chat_completion(
                messages=messages,
                images=images,
                response_format=response_format,
            )
            return LLMPromptEngine._parse_json(response)
        except Exception as exc:
            first_error = exc

        repair_errors: list[Exception] = []
        for repair_round in range(2):
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "JSON REPAIR MODE: Your previous answer did not satisfy the JSON parser. "
                        f"Return only one complete {expected_json} matching the schema. No markdown, no code fences, "
                        "no reasoning, no comments, no prose, and no trailing text. "
                        f"This is repair pass {repair_round + 1} of 2."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        "This is a strict parser repair attempt. Silently correct formatting and missing required "
                        f"fields, then output the complete {expected_json} only."
                    ),
                },
            ]
            try:
                repaired = chat_model.chat_completion(
                    messages=repair_messages,
                    images=images,
                )
                return LLMPromptEngine._parse_json(repaired)
            except Exception as repair_error:
                repair_errors.append(repair_error)

        if first_error is not None:
            for repair_error in repair_errors:
                try:
                    first_error.add_note(
                        f"JSON repair attempt also failed: {type(repair_error).__name__}: {repair_error}"
                    )
                except AttributeError:
                    pass
            raise first_error from (repair_errors[-1] if repair_errors else None)
        raise repair_errors[-1]

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
        if "<think>" in cleaned:
            cleaned = cleaned.split("<think>", 1)[-1]
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
