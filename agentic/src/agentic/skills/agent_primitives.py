from __future__ import annotations

from agentic.runtime.media_dq import expected_subject_count

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from agentic.assets.image_input import assert_image_input, inspect_image_input
from agentic.minimax_prompting import subject_identity_lock
from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.reference_video import format_reference_video_directive
from agentic.runtime.story_cards import (
    STORY_CARD_DEFAULT_HEIGHT,
    STORY_CARD_DEFAULT_WIDTH,
    new_story_card_visual_seed,
    render_story_card_images,
    story_card_visual_beat,
    validate_story_card_payload,
)
from agentic.runtime.prompting import (
    build_segment_prompt,
    validate_story_segments,
)
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import (
    asset_check_result,
    build_run_dir,
    collect_output_values,
    resolve_dependency_negative_prompt,
    resolve_dependency_prompt,
    resolve_dependency_value,
    safe_path_component,
    slug_path_component,
)


class AgentPlanningSkills:
    def __init__(self, prompt_engine: PromptEngine | None = None) -> None:
        self.prompt_engine = prompt_engine or PromptEngine()

    def expand_goal(self, context: SkillContext) -> SkillResult:
        prompt = str(context.node.inputs["prompt"])
        style = str(context.node.inputs.get("style", context.plan.goal.style))
        idea_variants = list(context.node.inputs.get("idea_variants", []))
        selected_variant = idea_variants[0] if idea_variants else {"style": style}
        selected_style = str(selected_variant.get("style", style))
        reference_analysis = context.state.node_outputs.get("reference-video-analysis")
        outputs = self.prompt_engine.expand_goal(
            context.plan.goal,
            selected_style,
            idea_variants,
            reference_analysis=reference_analysis if isinstance(reference_analysis, dict) else None,
        )
        reference_directive = format_reference_video_directive(reference_analysis, max_chars=2200)
        if reference_directive and reference_directive not in str(outputs.get("creative_brief") or ""):
            outputs["creative_brief"] = "\n".join(
                part for part in (str(outputs.get("creative_brief") or "").strip(), reference_directive) if part
            )
        return SkillResult(
            status="success",
            outputs=outputs,
            logs=["Expanded the goal into an agent-ready brief."],
        )

    def compose_prompt(self, context: SkillContext) -> SkillResult:
        prompt = str(context.node.inputs["prompt"])
        style = str(context.node.inputs.get("style", context.plan.goal.style))
        prefix = str(context.node.inputs.get("prefix", "")).strip()
        suffix = str(context.node.inputs.get("suffix", "")).strip()
        negative_prompt = str(
            context.node.inputs.get(
                "negative_prompt",
                "ugly, blurry, low quality, bad anatomy, deformed, duplicate, watermark, text",
            )
        )
        bundle = self.prompt_engine.compose_prompt(
            context.plan.goal,
            prompt=prompt,
            style=style,
            prefix=prefix,
            suffix=suffix,
            negative_prompt=negative_prompt,
        )
        return SkillResult(
            status="success",
            outputs=bundle,
            logs=["Composed a reusable prompt bundle."],
        )

    def segment_story(self, context: SkillContext) -> SkillResult:
        segment_count = int(context.node.inputs["segment_count"])
        brief_source = context.state[context.node.depends_on[0]] if context.node.depends_on else {}
        brief = str(brief_source.get("creative_brief", context.plan.goal.prompt))
        tone = str(context.node.inputs.get("tone", "coherent progression"))
        reference_analysis = context.state.node_outputs.get("reference-video-analysis")
        segments = self.prompt_engine.segment_story(
            context.plan.goal,
            brief,
            segment_count,
            tone,
            reference_analysis=reference_analysis if isinstance(reference_analysis, dict) else None,
            production_profile=str(context.node.inputs.get("production_profile") or ""),
        )
        validate_story_segments(segments, segment_count)
        return SkillResult(
            status="success",
            outputs={"segments": segments, "segment_count": segment_count},
            metrics={"segment_count": segment_count},
            logs=[f"Planned {segment_count} reusable story segments."],
        )

    def prepare_segment(self, context: SkillContext) -> SkillResult:
        segment_index = int(context.node.inputs["segment_index"])
        segment = context.state["script-plan"]["segments"][segment_index]
        prior_frame = self._resolve_first_value(context, ("frame_path", "saved_files"))
        previous_segment = context.state["script-plan"]["segments"][segment_index - 1] if segment_index > 0 else None
        # Only a review node may inject a revision direction.  The ordinary
        # idea-brief dependency also exposes ``prompt``; treating that as a
        # review direction duplicated the entire expanded long-video brief in
        # every keyframe prompt and caused image models to render contact
        # sheets/storyboards instead of a single frame.
        review_direction = self._resolve_first_value(context, ("revised_prompt",))
        outputs = self.prompt_engine.prepare_segment(
            context.plan.goal,
            segment,
            str(context.state["idea-brief"].get("negative_prompt", "")),
            previous_segment=previous_segment,
            prior_frame=prior_frame,
        )
        if review_direction and review_direction not in outputs["prompt"]:
            outputs["original_prompt"] = outputs["prompt"]
            outputs["prompt"] = ", ".join(part for part in (outputs["prompt"], f"revision direction: {review_direction}") if part)
            outputs["revised_prompt"] = outputs["prompt"]
        return SkillResult(status="success", outputs=outputs, logs=[f"Prepared prompt package for {segment['segment_id']}."])

    def generate_sticker_expressions(self, context: SkillContext) -> SkillResult:
        character = str(context.node.inputs.get("character", "")).strip()
        prompt = str(context.node.inputs.get("prompt", context.plan.goal.prompt)).strip()
        expression_count = int(
            context.node.inputs.get("expression_count")
            or context.plan.goal.constraints.get("sticker_expression_count")
            or 8
        )
        expressions = self.prompt_engine.sticker_expressions(context.plan.goal, prompt, character, expression_count)
        return SkillResult(
            status="success",
            outputs={"expressions": expressions, "expression_count": len(expressions)},
            metrics={"expression_count": len(expressions)},
            logs=["Generated a reusable sticker expression set for the agent."],
        )

    def build_sticker_prompt_set(self, context: SkillContext) -> SkillResult:
        expressions = list(context.state[context.node.depends_on[0]].get("expressions", []))
        style = str(
            context.node.inputs.get(
                "style",
                "LINE sticker style, chibi proportions, white outline, simple clean background, 2D flat shading",
            )
        )
        character = str(context.node.inputs.get("character") or context.plan.goal.constraints.get("character", "")).strip()
        prompt_prefix = str(context.node.inputs.get("prompt_prefix", context.plan.goal.prompt)).strip()
        bundle = self.prompt_engine.build_sticker_prompt_set(
            context.plan.goal,
            expressions=expressions,
            character=character,
            prompt_prefix=prompt_prefix,
            style=style,
        )
        return SkillResult(
            status="success",
            outputs=bundle,
            metrics={"prompt_count": int(bundle.get("prompt_count", 0))},
            logs=["Prepared a reusable sticker prompt set."],
        )

    def build_sticker_motion_prompt(self, context: SkillContext) -> SkillResult:
        sticker_batch = context.state[context.node.depends_on[0]]
        items = list(sticker_batch.get("items", []))
        selected_path = ""
        selected_expression = ""
        if items:
            selected_path = str(items[0].get("saved_file", ""))
            selected_expression = str(items[0].get("expression", ""))
        base_prompt = str(items[0].get("prompt", context.plan.goal.prompt)) if items else context.plan.goal.prompt
        character = str(context.plan.goal.constraints.get("character", "") or context.node.inputs.get("character", "")).strip()
        prompt_bundle = self.prompt_engine.build_sticker_motion_prompt(
            context.plan.goal,
            base_prompt=base_prompt,
            character=character,
            selected_expression=selected_expression,
        )
        return SkillResult(
            status="success",
            outputs={
                "image_path": selected_path,
                "prompt": str(prompt_bundle["prompt"]),
                "prompt_mode": str(prompt_bundle.get("prompt_mode", "template")),
            },
            logs=["Prepared an animated sticker motion prompt from the rendered sticker batch."],
        )

    def build_dynamic_sprite_motion_plan(self, context: SkillContext) -> SkillResult:
        plan = self.prompt_engine.build_dynamic_sprite_motion_plan(context.plan.goal)
        frame_map = list(plan.get("frame_map") or [])
        if len(frame_map) != 16:
            raise ValueError("Dynamic sprite motion plan must contain exactly sixteen frame-map entries")
        return SkillResult(
            status="success",
            outputs=plan,
            metrics={"frame_count": len(frame_map)},
            logs=[f"Generated open-ended sprite motion plan '{plan.get('motion_name', '')}'."],
        )

    def build_slide_prompt_set(self, context: SkillContext) -> SkillResult:
        segments = list(context.state[context.node.depends_on[0]].get("segments", []))
        style = str(context.node.inputs.get("style", context.plan.goal.style))
        bundle = self.prompt_engine.build_carousel_prompt_set(context.plan.goal, segments, style)
        return SkillResult(
            status="success",
            outputs=bundle,
            metrics={"prompt_count": int(bundle.get("prompt_count", 0))},
            logs=["Prepared a reusable carousel prompt set."],
        )

    def write_story_card(self, context: SkillContext) -> SkillResult:
        goal = context.plan.goal
        if "page_count" in context.node.inputs:
            goal = replace(
                goal,
                constraints={
                    **goal.constraints,
                    "story_card_page_count": context.node.inputs["page_count"],
                },
            )
        bundle = self.prompt_engine.build_story_card(goal)
        return SkillResult(
            status="success",
            outputs=bundle,
            metrics={"page_count": int(bundle.get("page_count", len(bundle.get("pages", []))))},
            logs=["Wrote a dedicated writing-first story-card draft."],
        )

    def refine_prompt_after_review(self, context: SkillContext) -> SkillResult:
        source = context.state[context.node.depends_on[0]] if context.node.depends_on else {}
        original_prompt = str(source.get("prompt") or context.node.inputs.get("prompt") or context.plan.goal.prompt)
        review_notes = str(context.node.inputs.get("review_notes") or context.plan.goal.constraints.get("review_notes", ""))
        media_paths = context.node.inputs.get("media_paths") or source.get("media_paths") or source.get("saved_files") or []
        selected_assets = self._resolve_many_values(context, ("media_paths", "saved_files", "video_path", "gif_path"))
        if selected_assets:
            media_paths = selected_assets
        outputs = self.prompt_engine.refine_prompt_from_review(
            context.plan.goal,
            original_prompt=original_prompt,
            review_notes=review_notes,
            media_paths=[str(path) for path in media_paths],
        )
        outputs["original_prompt"] = original_prompt
        outputs["revised_prompt"] = str(outputs.get("prompt", original_prompt))
        outputs["review_notes"] = review_notes
        outputs["selected_assets"] = [str(path) for path in media_paths]
        outputs["retry_count"] = int(context.node.inputs.get("retry_count", 1))
        return SkillResult(status="success", outputs=outputs, logs=["Refined prompt after review feedback."])

    @staticmethod
    def _resolve_first_value(context: SkillContext, candidate_keys: tuple[str, ...]) -> str | None:
        return resolve_dependency_value(context, candidate_keys)

    @staticmethod
    def _resolve_many_values(context: SkillContext, candidate_keys: tuple[str, ...]) -> list[str]:
        return list(dict.fromkeys(collect_output_values(context, candidate_keys)))


class AgentMediaSkills:
    def __init__(
        self,
        tools: ToolRegistry,
        output_root: Path,
        prompt_engine: PromptEngine | None = None,
    ) -> None:
        self.tools = tools
        self.output_root = output_root
        self.prompt_engine = prompt_engine or PromptEngine()
        self.output_root.mkdir(parents=True, exist_ok=True)

    def ensure_workflow(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "asset.ensure_workflow_ready",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "auto_download": context.node.inputs.get("auto_download", False),
            },
        )
        return asset_check_result(result, "Checked workflow assets for an agent step.")

    def refine_image(self, context: SkillContext) -> SkillResult:
        workflow_name = str(context.plan.goal.constraints.get("identity_refine_workflow_name") or "")
        prompt_key = str(context.node.inputs.get("prompt_key") or "").strip()
        prompt = ""
        if prompt_key:
            for dependency in reversed(context.node.depends_on):
                dependency_output = context.state[dependency]
                resolved_prompt = dependency_output.get(prompt_key)
                if isinstance(resolved_prompt, str) and resolved_prompt.strip():
                    prompt = resolved_prompt.strip()
                    break
        payload = {
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "img2img")),
            "image_path": self._resolve_image_path(context),
            "prompt": self._resolve_prompt_with_identity_lock(context, prompt or self._resolve_prompt(context)),
            "negative_prompt": self._resolve_negative_prompt(context),
        }
        if workflow_name:
            payload["workflow_name"] = workflow_name
        result = self.tools.call(
            "comfy.workflow.image_to_image",
            payload,
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result.get("saved_files", []))},
            logs=["Refined an input image with an agent media primitive."],
        )

    def generate_keyframe(self, context: SkillContext) -> SkillResult:
        use_prior_frame = bool(context.node.inputs.get("use_prior_frame", True))
        prior_frame_path = context.node.inputs.get("prior_frame_path")
        if use_prior_frame and not prior_frame_path:
            prior_frame_path = resolve_dependency_value(
                context,
                ("prior_frame_path", "frame_path", "selected_frame_path", "selected_assets", "saved_files"),
            )
        if isinstance(prior_frame_path, str) and prior_frame_path:
            workflow_name = str(
                context.node.inputs.get("identity_refine_workflow_name")
                or context.plan.goal.constraints.get("identity_refine_workflow_name")
                or "image_to_image"
            )
            result = self.tools.call(
                "comfy.workflow.image_to_image",
                {
                    "workflow_name": workflow_name,
                    "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "segment_keyframe")),
                    "image_path": prior_frame_path,
                    "prompt": self._resolve_prompt_with_identity_lock(context, self._resolve_prompt(context)),
                    "negative_prompt": self._resolve_negative_prompt(context),
                    # Continuity refinement is also the source of auto-generated
                    # Ref2VA candidates. Preserve the requested bundle size
                    # instead of collapsing every planned-anchor refinement to one
                    # image before reference validation.
                    "image_count": max(1, int(context.node.inputs.get("image_count", 1))),
                },
            )
            log = "Generated a continuity keyframe from a prior frame."
        else:
            workflow_name = str(
                context.plan.goal.constraints.get("keyframe_workflow_name")
                or context.node.inputs["workflow_name"]
            )
            character_label = str(
                context.plan.goal.constraints.get("character") or "the selected protagonist"
            ).strip()
            character = character_label.lower()
            subject_context = dict(context.plan.goal.constraints.get("subject_context") or {})
            profile = dict(
                subject_context.get("character_profile")
                or context.plan.goal.constraints.get("character_profile")
                or {}
            )
            profile_details = "; ".join(
                part
                for part in (
                    str(profile.get("role_description") or "").strip(),
                    str(profile.get("keywords") or "").strip(),
                )
                if part
            )
            resolved_subject = (
                f"{character_label} ({profile_details})"
                if profile_details
                else character_label
            )
            interaction_required = bool(
                dict(subject_context.get("interaction_contract") or {}).get("required", False)
            )
            subject_names = [
                str(item.get("name") or "").strip()
                for item in (subject_context.get("subjects") or [])
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ]
            prompt = self._resolve_prompt_with_identity_lock(context, self._resolve_prompt(context))
            negative_prompt = self._resolve_negative_prompt(context)
            if character == "kirby":
                anchor_position = str(context.node.inputs.get("anchor_position") or "").strip().lower()
                if anchor_position in {"first", "last"}:
                    script_output = context.state.node_outputs.get("script-plan", {})
                    segments = script_output.get("segments", []) if isinstance(script_output, dict) else []
                    segment_index = int(context.node.inputs.get("segment_index") or 0)
                    segment = segments[segment_index] if segment_index < len(segments) else {}
                    state_key = "start_state" if anchor_position == "first" else "end_state"
                    planned_state = str(segment.get(state_key) or "").strip()
                    if planned_state:
                        prompt = (
                            f"{'Opening' if anchor_position == 'first' else 'Landing'} keyframe only: {planned_state}. "
                            f"Freeze one single moment {'immediately before the physical action begins' if anchor_position == 'first' else 'immediately after the decisive outcome'}; "
                            "do not depict a sequence, a storyboard, or multiple copies of the character."
                        )
                        if anchor_position == "last":
                            prompt += (
                                " Render every named prop from the end state literally at a natural scale, "
                                f"grounded in a physically plausible location and clearly separate from {character_label}'s face, "
                                "eyes, and body; preserve one readable wide composition."
                            )
                prompt_key = str(context.node.inputs.get("prompt_key") or "").strip()
                if prompt_key == "opening_keyframe_prompt":
                    story_output = context.state.node_outputs.get("native-story-prompt", {})
                    generated_storyboard = (
                        story_output.get("generated_storyboard")
                        if isinstance(story_output, dict)
                        else None
                    )
                    news_trace = (
                        generated_storyboard.get("news_trace")
                        if isinstance(generated_storyboard, dict)
                        else None
                    )
                    mechanism = (
                        str(news_trace.get("news_mechanism") or "").strip().rstrip(".")
                        if isinstance(news_trace, dict)
                        else ""
                    )
                    if mechanism:
                        prompt = (
                            f"{prompt}. FIRST-FRAME NEWS-MECHANISM LOCK: show {mechanism} visibly operating "
                            "inside the opening composition now, not merely implied in the background; keep the "
                            "same mechanism and its dominant anchor large and readable from frame one."
                        )
                if interaction_required and len(subject_names) == 2:
                    prompt = (
                        f"{prompt}, single continuous animation frame, one composition, both declared subject slots "
                        f"({'; '.join(subject_names)}) large and clearly visible in the foreground or midground, "
                        "preserve both recognizable identities and show their visible mutual interaction, "
                        "no storyboard, no sequence, no contact sheet, no split composition, no unrequested third subject"
                    )
                else:
                    prompt = (
                        f"{prompt}, single continuous animation frame, one {resolved_subject} only, one composition, "
                        f"{character_label} large and clearly visible in the foreground or midground, "
                        "with the resolved character_profile appearance readable, "
                        "no storyboard, no sequence, no contact sheet, no split composition"
                    )
                negative_prompt = (
                    f"{negative_prompt}, storyboard, contact sheet, comic panels, multi-panel, split screen, collage, "
                    f"tiny distant subject, cropped character, unrecognizable character, duplicate {character_label}, "
                    f"second {character_label}, cloned subject, object on face, object on head, object covering eyes, "
                    "oversized prop, giant orb, floating unrelated object, humanoid silhouette, black figure, "
                    "light beam, lens flare"
                    + (", identity swap, unrequested third subject" if interaction_required else "")
                )
            attempts = 1 if character != "kirby" else max(1, int(context.node.inputs.get("max_regenerations", 2)) + 1)
            result: dict[str, object] = {}
            rejected_reasons: list[str] = []
            final_reports: list[tuple[str, object | None]] = []
            accepted_paths: list[str] = []
            for attempt in range(attempts):
                result = self.tools.call(
                    "comfy.workflow.text_to_image",
                    {
                        "run_dir": str(
                            self._build_run_dir(
                                context.plan.goal.prompt,
                                str(context.node.inputs.get("suffix") or "segment_keyframe"),
                            )
                        ),
                        "workflow_name": workflow_name,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": int(context.node.inputs.get("width", 1024)),
                        "height": int(context.node.inputs.get("height", 1024)),
                        "image_count": int(context.node.inputs.get("image_count", 1)),
                    },
                )
                if character != "kirby":
                    break
                candidate_paths = self._output_paths(result)
                reports = [
                    (
                        path,
                        inspect_image_input(
                            path,
                        ),
                    )
                    for path in candidate_paths
                ]
                final_reports = reports
                accepted_reports = [
                    (path, report)
                    for path, report in reports
                    if report is not None and report.passed
                ]
                failed_reports = [
                    (path, report)
                    for path, report in reports
                    if not report or not report.passed
                ]
                if accepted_reports:
                    accepted_paths = [path for path, _report in accepted_reports]
                    rejected_paths = [path for path, _report in failed_reports]
                    result = dict(result)
                    result["generated_files"] = list(candidate_paths)
                    result["saved_files"] = list(accepted_paths)
                    result["rejected_files"] = rejected_paths
                    result["rejected_asset_details"] = [
                        {
                            "path": path,
                            "reasons": list(getattr(report, "reasons", ())) if report else ["keyframe output is missing"],
                        }
                        for path, report in failed_reports
                    ]
                    break
                if candidate_paths and not failed_reports:
                    break
                if failed_reports:
                    rejected_reasons.extend(
                        f"{path}: {('; '.join(report.reasons) if report else 'keyframe output is missing')}"
                        for path, report in failed_reports
                    )
                else:
                    rejected_reasons.append("keyframe output is missing")
            if character == "kirby":
                if not accepted_paths:
                    details = " | ".join(rejected_reasons) or "unknown Kirby keyframe validation failure"
                    raise ValueError(f"Kirby keyframe generation failed after {attempts} attempts: {details}")
            log = (
                f"Generated {len(accepted_paths)} validated Kirby keyframe candidate(s) from text."
                if character == "kirby"
                else "Generated an opening keyframe from text."
            )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result.get("saved_files", []))},
            logs=[log],
        )

    def validate_character_frames(self, context: SkillContext) -> SkillResult:
        opening_node = str(context.node.inputs.get("opening_node") or context.node.depends_on[0])
        use_last_frame = bool(context.node.inputs.get("use_last_frame", False))
        ending_node = str(
            context.node.inputs.get("ending_node")
            or (context.node.depends_on[1] if len(context.node.depends_on) > 1 else "")
        )
        character = str(context.node.inputs.get("character") or context.plan.goal.constraints.get("character") or "").strip().lower()
        subject_context = dict(context.plan.goal.constraints.get("subject_context") or {})
        interaction_required = bool(
            dict(subject_context.get("interaction_contract") or {}).get("required", False)
        )
        story = context.state.node_outputs.get("native-story-prompt", {})
        frame_specs = [
            ("opening", opening_node, str(context.node.inputs.get("opening_prompt_key") or "opening_keyframe_prompt")),
        ]
        if use_last_frame:
            if not ending_node:
                raise ValueError("Native H3 use_last_frame=true requires an ending_node")
            frame_specs.append(
                ("ending", ending_node, str(context.node.inputs.get("ending_prompt_key") or "ending_keyframe_prompt"))
            )
        reports: list[dict[str, object]] = []
        regenerated_count = 0
        final_paths: dict[str, str] = {}
        preserve_opening_frame = bool(context.node.inputs.get("preserve_opening_frame", False))
        preserve_ending_frame = bool(context.node.inputs.get("preserve_ending_frame", False))
        for label, node_id, prompt_key in frame_specs:
            frame_path = self._first_output_path(context.state.node_outputs.get(node_id, {}))
            last_error = ""
            for attempt in range(max(0, int(context.node.inputs.get("max_regenerations", 0))) + 1):
                try:
                    if not frame_path:
                        raise ValueError("keyframe output is missing")
                    preserve_frame = preserve_opening_frame if label == "opening" else preserve_ending_frame
                    if preserve_frame:
                        if not Path(frame_path).is_file():
                            raise ValueError(f"human-selected {label} frame file is missing")
                        report = {
                            "path": frame_path,
                            "passed": True,
                            "validation": "human_selected_immutable",
                        }
                    elif character == "kirby":
                        report = assert_image_input(
                            frame_path,
                        ).to_dict()
                    else:
                        report = {"path": frame_path, "passed": Path(frame_path).is_file()}
                        if not report["passed"]:
                            raise ValueError("keyframe output file is missing")
                    reports.append(report)
                    final_paths[label] = frame_path
                    break
                except (OSError, ValueError) as exc:
                    last_error = str(exc)
                    if attempt >= int(context.node.inputs.get("max_regenerations", 0)):
                        raise ValueError(f"{label} image input gate failed after {attempt} regenerations: {last_error}") from exc
                    prompt = self._resolve_prompt_with_identity_lock(
                        context,
                        str(story.get(prompt_key) or context.plan.goal.prompt),
                    )
                    result = self.tools.call(
                        "comfy.workflow.text_to_image",
                        {
                            "workflow_name": str(context.node.inputs.get("workflow_name") or context.plan.goal.constraints.get("native_h3_keyframe_workflow_name") or context.plan.goal.constraints.get("keyframe_workflow_name") or ""),
                            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, f"identity_{label}_retry_{attempt + 1}")),
                            "prompt": prompt,
                            "negative_prompt": str(story.get("negative_prompt") or ""),
                            "width": int(context.node.inputs.get("width") or 608),
                            "height": int(context.node.inputs.get("height") or 352),
                            "image_count": 1,
                        },
                    )
                    frame_path = self._first_output_path(result)
                    regenerated_count += 1
        outputs: dict[str, object] = {
            "first_frame_path": final_paths["opening"],
            "character": character,
            "identity_reports": reports,
            "identity_gate": "passed",
            "use_last_frame": use_last_frame,
            "regenerated_count": regenerated_count,
        }
        if use_last_frame:
            outputs["last_frame_path"] = final_paths["ending"]
        return SkillResult(
            status="success",
            outputs=outputs,
            metrics={"validated_frame_count": len(frame_specs), "regenerated_count": regenerated_count},
            logs=[
                f"Validated {'opening and ending' if use_last_frame else 'opening'} continuity frame(s) for {character or 'the configured character'}."
            ]
        )

    def validate_last_frame(self, context: SkillContext) -> SkillResult:
        frame_node = str(context.node.inputs.get("frame_node") or context.node.depends_on[0])
        frame_path = self._first_output_path(context.state.node_outputs.get(frame_node, {}))
        if not frame_path:
            raise ValueError("Native H3 L2VA input gate requires a last-frame output")
        if not Path(frame_path).is_file():
            raise ValueError(f"Native H3 L2VA last-frame file is missing: {frame_path}")

        character = str(
            context.node.inputs.get("character")
            or context.plan.goal.constraints.get("character")
            or ""
        ).strip().lower()
        subject_context = dict(context.plan.goal.constraints.get("subject_context") or {})
        interaction_required = bool(
            dict(subject_context.get("interaction_contract") or {}).get("required", False)
        )
        preserve = bool(context.node.inputs.get("preserve_last_frame", True))
        if preserve:
            report = {
                "path": frame_path,
                "passed": True,
                "validation": "human_selected_immutable",
            }
        elif character == "kirby":
            report = assert_image_input(
                frame_path,
            ).to_dict()
        else:
            report = {"path": frame_path, "passed": True, "validation": "file_exists"}
        return SkillResult(
            status="success",
            outputs={
                "first_frame_path": "",
                "last_frame_path": frame_path,
                "character": character,
                "identity_reports": [report],
                "identity_gate": "passed",
                "preserve_last_frame": preserve,
                "regenerated_count": 0,
            },
            metrics={"validated_frame_count": 1, "regenerated_count": 0},
            logs=["Validated and preserved the selected last frame for native H3 L2VA."],
        )

    def upscale_image(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "comfy.workflow.image_upscale",
            {
                "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "upscale")),
                "image_path": self._resolve_image_path(context),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Upscaled an image with an agent media primitive."])

    def animate_image(self, context: SkillContext) -> SkillResult:
        workflow_name = str(context.node.inputs.get("workflow_name", ""))
        payload: dict[str, object] = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "i2v")),
            "image_path": self._resolve_image_path(context),
            "prompt": self._resolve_prompt(context),
            "model_profile": str(
                context.node.inputs.get("model_profile")
                or context.plan.goal.constraints.get("native_h3_model_profile")
                or "q4"
            ),
            "video_count": int(context.node.inputs.get("video_count") or context.plan.goal.constraints.get("video_count") or 1),
            "width": context.node.inputs.get("width"),
            "height": context.node.inputs.get("height"),
        }
        seed = context.node.inputs.get("seed", context.plan.goal.constraints.get("seed"))
        if seed is not None:
            payload["seed"] = int(seed)
        if context.plan.goal.media_type == "long_video" and workflow_name.startswith("minimax_h3_"):
            constraints = context.plan.goal.constraints
            payload.update(
                {
                    # Long-video runs queue multiple H3 jobs. Keep each draft
                    # below the standalone 5-second profile so an 8GB GPU can
                    # complete the segment and reload the next one.
                    "width": int(constraints.get("longvideo_h3_width", 512)),
                    "height": int(constraints.get("longvideo_h3_height", 288)),
                    "length": int(constraints.get("longvideo_h3_length", 81)),
                    "steps": int(constraints.get("longvideo_h3_steps", 16)),
                    "model_profile": str(constraints.get("longvideo_h3_model_profile", "q2")),
                }
            )
        elif workflow_name.startswith("minimax_h3_") and context.node.inputs.get("length") is not None:
            payload["length"] = int(context.node.inputs["length"])
        if workflow_name.startswith("minimax_h3_") and context.node.inputs.get("steps") is not None:
            payload["steps"] = int(context.node.inputs["steps"])
        result = self.tools.call(
            "comfy.workflow.image_to_video",
            payload,
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"video_count": len(result.get("saved_files", []))},
            logs=["Animated an image with an agent media primitive."],
        )

    def render_image_batch(self, context: SkillContext) -> SkillResult:
        prompt_sets = list(context.state[context.node.depends_on[0]].get("prompt_sets", []))
        workflow_name = str(context.node.inputs["workflow_name"])
        width = int(context.node.inputs.get("width", 1024))
        height = int(context.node.inputs.get("height", 1024))
        negative_prompt = str(context.node.inputs.get("negative_prompt") or self._resolve_negative_prompt(context))
        images_per_prompt = int(
            context.node.inputs.get("images_per_prompt")
            or context.plan.goal.constraints.get("images_per_prompt")
            or 1
        )
        run_dir = self._build_run_dir(context.plan.goal.prompt, str(context.node.inputs.get("suffix", "batch_images")))
        saved_files: list[str] = []
        item_runs: list[dict[str, str]] = []
        for index, prompt_set in enumerate(prompt_sets, start=1):
            label = slug_path_component(
                prompt_set.get("label", f"item_{index:02d}"),
                default=f"item-{index:02d}",
            )
            item_dir = run_dir / label
            result = self.tools.call(
                "comfy.workflow.text_to_image",
                {
                    "workflow_name": workflow_name,
                    "prompt": self._resolve_prompt_with_identity_lock(
                        context,
                        str(prompt_set["prompt"]),
                    ),
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "image_count": images_per_prompt,
                    "run_dir": str(item_dir),
                },
            )
            generated = [str(path) for path in result.get("saved_files", [])]
            saved_files.extend(generated)
            item_runs.append(
                {
                    "label": label,
                    "prompt": str(prompt_set["prompt"]),
                    "expression": str(prompt_set.get("expression", "")),
                    "run_dir": str(item_dir),
                    "saved_file": generated[0] if generated else "",
                }
            )
        return SkillResult(
            status="success",
            outputs={"run_dir": str(run_dir), "saved_files": saved_files, "items": item_runs},
            metrics={"image_count": len(saved_files)},
            logs=[f"Rendered {len(saved_files)} images as a reusable batch primitive."],
        )

    def render_story_card_backgrounds(self, context: SkillContext) -> SkillResult:
        """Render each page as an independent style-locked background."""

        story_output = context.state[context.node.inputs.get("story_node", "story-card-write")]
        story = validate_story_card_payload(dict(story_output))
        workflow_name = str(context.node.inputs["workflow_name"])
        render_tool = str(context.node.inputs.get("render_tool") or "comfy.workflow.text_to_image")
        width = int(context.node.inputs.get("width", STORY_CARD_DEFAULT_WIDTH))
        height = int(context.node.inputs.get("height", STORY_CARD_DEFAULT_HEIGHT))
        run_dir = self._build_run_dir(context.plan.goal.prompt, "story_card_backgrounds")
        background_paths: list[str] = []
        page_runs: list[dict[str, object]] = []
        background_hashes: set[str] = set()
        character = str(context.plan.goal.constraints.get("character") or "").strip()
        if not character:
            raise RuntimeError("Story-card background generation requires a resolved selected character")
        profile = context.plan.goal.constraints.get("character_profile")
        visual_config = context.plan.goal.constraints.get("story_card_visual")
        seed_value = context.plan.goal.constraints.get("seed")
        if seed_value is not None:
            seed_base = int(seed_value)
        else:
            seed_base = int(story.get("visual_seed") or new_story_card_visual_seed())
        for index, page in enumerate(story["pages"], start=1):
            page_dir = run_dir / f"page_{index:02d}"
            background_prompt = str(page["background_prompt"])
            page_seed = seed_base + (index * 1009)
            payload: dict[str, object] = {
                "workflow_name": workflow_name,
                "prompt": background_prompt,
                "negative_prompt": str(story["negative_prompt"]),
                "width": width,
                "height": height,
                "image_count": 1,
                "run_dir": str(page_dir),
                "seed": page_seed,
            }
            result = self.tools.call(render_tool, payload)
            generated = [str(path) for path in result.get("saved_files", []) if str(path)]
            if not generated:
                raise RuntimeError(f"Story-card background render produced no image for page {index}")
            generated_path = generated[0]
            digest = hashlib.sha256(Path(generated_path).read_bytes()).hexdigest()
            if digest in background_hashes:
                raise RuntimeError(
                    f"Story-card background page {index} duplicated an earlier page image; "
                    "page-specific motif and seed did not produce visual variation"
                )
            background_hashes.add(digest)
            background_paths.append(generated_path)
            page_runs.append(
                {
                    "page": str(index),
                    "style_anchor_prompt": story["anchor_prompt"],
                    "prompt": background_prompt,
                    "saved_file": generated_path,
                    "seed": str(page_seed),
                    "visual_beat": story_card_visual_beat(
                        index,
                        page_count=len(story["pages"]),
                        visual_config=visual_config,
                    ),
                    "sha256": digest,
                }
            )
        return SkillResult(
            status="success",
            outputs={
                "run_dir": str(run_dir),
                "anchor_path": "",
                "style_anchor_prompt": story["anchor_prompt"],
                "background_paths": background_paths,
                "page_runs": page_runs,
                "visual_seed": seed_base,
            },
            metrics={"background_count": len(background_paths)},
            logs=[f"Rendered {len(background_paths)} style-locked text-to-image page backgrounds."],
        )

    def compose_story_card(self, context: SkillContext) -> SkillResult:
        story_output = context.state[context.node.inputs.get("story_node", "story-card-write")]
        backgrounds_output = context.state[context.node.inputs.get("background_node", "story-card-backgrounds")]
        story = validate_story_card_payload(dict(story_output))
        background_paths = [
            str(path) for path in backgrounds_output.get("background_paths", []) if str(path)
        ]
        output_dir = Path(str(backgrounds_output.get("run_dir") or self._build_run_dir(context.plan.goal.prompt, "story_card"))) / "cards"
        render_result = render_story_card_images(
            background_paths=background_paths,
            story=story,
            output_dir=output_dir,
            width=int(context.node.inputs.get("width", STORY_CARD_DEFAULT_WIDTH)),
            height=int(context.node.inputs.get("height", STORY_CARD_DEFAULT_HEIGHT)),
            font_path=str(context.node.inputs.get("font_path") or context.plan.goal.constraints.get("story_card_font_path") or "") or None,
            overlay_opacity=int(context.node.inputs.get("overlay_opacity", 202)),
            brand_label=str(context.node.inputs.get("brand_label", "STORY NOTE")),
        )
        summary_path = output_dir / "story_card_summary.json"
        summary = {
            "goal": context.plan.goal.prompt,
            "media_type": context.plan.goal.media_type,
            "story": story,
            "anchor_path": backgrounds_output.get("anchor_path", ""),
            "background_paths": background_paths,
            "render": render_result,
            "prompt_lineage": self._collect_prompt_lineage(context),
            "node_prompt_modes": self._collect_node_prompt_modes(context),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_files = [*render_result["saved_files"], str(summary_path)]
        return SkillResult(
            status="success",
            outputs={
                **render_result,
                "summary_path": str(summary_path),
                "saved_files": saved_files,
                "story": story,
                "anchor_path": backgrounds_output.get("anchor_path", ""),
                "prompt_lineage": summary["prompt_lineage"],
                "node_prompt_modes": summary["node_prompt_modes"],
            },
            metrics={"page_count": len(render_result["saved_files"])},
            logs=["Composed exact Traditional-Chinese text over subdued story-card backgrounds."],
        )

    def narrate_text(self, context: SkillContext) -> SkillResult:
        output_name = str(context.node.inputs.get("output_name", "narration.mp3"))
        run_dir = self._build_run_dir(context.plan.goal.prompt, "tts")
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        text = str(context.node.inputs.get("text") or self._resolve_text(context))
        result = self.tools.call(
            "audio.generate_tts_real",
            {
                "text": text,
                "output_path": str(audio_dir / output_name),
                "voice": str(context.node.inputs.get("voice", "en-US-AriaNeural")),
                "rate": str(context.node.inputs.get("rate", "+0%")),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Generated narration audio for an agent step."])

    def concat_audio_tracks(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "audio_concat")
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        result = self.tools.call(
            "audio.concat_tracks",
            {
                "audio_paths": context.node.inputs.get("audio_paths") or self._resolve_many(context, ("audio_path",)),
                "output_path": str(audio_dir / "merged.mp3"),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Concatenated audio tracks for an agent step."])

    def concat_videos(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "concat")
        video_dir = run_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        result = self.tools.call(
            "media.concat_videos",
            {
                "video_paths": context.node.inputs.get("video_paths") or self._resolve_many(context, ("video_path", "saved_files")),
                "output_path": str(video_dir / "merged.mp4"),
                "method": str(context.node.inputs.get("method", "demuxer")),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Concatenated videos for an agent step."])

    def change_video_speed(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "video_speed")
        video_dir = run_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(
            context.node.inputs.get("video_path")
            or self._resolve_first(context, ("video_path", "saved_files", "media_paths"))
            or ""
        )
        if not video_path:
            raise RuntimeError(f"No video path available for node '{context.node.node_id}'")
        speed = float(context.node.inputs.get("speed", 1.0))
        result = self.tools.call(
            "media.change_video_speed",
            {
                "video_path": video_path,
                "output_path": str(video_dir / f"{Path(video_path).stem}_{speed:g}x.mp4"),
                "speed": speed,
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"speed": speed},
            logs=[f"Rendered final video at {speed:g}x playback speed."],
        )

    def normalize_video_canvas(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "video_canvas")
        video_dir = run_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(
            context.node.inputs.get("video_path")
            or self._resolve_first(context, ("video_path", "saved_files", "media_paths"))
            or ""
        )
        if not video_path:
            raise RuntimeError(f"No video path available for node '{context.node.node_id}'")
        width = int(context.node.inputs["target_width"])
        height = int(context.node.inputs["target_height"])
        result = self.tools.call(
            "media.normalize_video_canvas",
            {
                "video_path": video_path,
                "output_path": str(video_dir / f"{Path(video_path).stem}_{width}x{height}.mp4"),
                "target_width": width,
                "target_height": height,
                "background": str(context.node.inputs.get("background", "#15151f")),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"width": width, "height": height},
            logs=[f"Normalized final video canvas to {width}x{height} without changing its aspect ratio."],
        )

    def trim_video(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "video_trim")
        video_dir = run_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(
            context.node.inputs.get("video_path")
            or self._resolve_first(context, ("video_path", "saved_files", "media_paths"))
            or ""
        )
        if not video_path:
            raise RuntimeError(f"No video path available for node '{context.node.node_id}'")
        duration_seconds = float(
            context.node.inputs.get("duration_seconds")
            or context.plan.goal.duration_seconds
        )
        result = self.tools.call(
            "media.trim_video",
            {
                "video_path": video_path,
                "output_path": str(video_dir / f"{Path(video_path).stem}_trimmed.mp4"),
                "duration_seconds": duration_seconds,
                "normalize_audio": bool(context.node.inputs.get("normalize_audio", False)),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"duration_seconds": duration_seconds},
            logs=[f"Trimmed final video to {duration_seconds:g} seconds."],
        )

    def merge_audio_video(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "mux")
        video_dir = run_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        result = self.tools.call(
            "media.merge_audio_video",
            {
                "video_path": str(context.node.inputs.get("video_path") or self._resolve_first(context, ("video_path", "saved_files"))),
                "audio_path": str(context.node.inputs.get("audio_path") or self._resolve_first(context, ("audio_path",))),
                "output_path": str(video_dir / "muxed.mp4"),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Muxed audio and video for an agent step."])

    def video_to_gif(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "gif")
        gif_dir = run_dir / "gif"
        gif_dir.mkdir(parents=True, exist_ok=True)
        result = self.tools.call(
            "media.video_to_gif",
            {
                "video_path": str(context.node.inputs.get("video_path") or self._resolve_first(context, ("video_path", "saved_files"))),
                "output_path": str(gif_dir / "preview.gif"),
                "fps": int(context.node.inputs.get("fps", 12)),
                "scale_width": int(context.node.inputs.get("scale_width", 512)),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Created a GIF preview for an agent step."])

    def qa_video(self, context: SkillContext) -> SkillResult:
        video_path = str(
            context.node.inputs.get("video_path")
            or self._resolve_first(context, ("video_path", "saved_files", "media_paths"))
            or ""
        )
        if not video_path:
            raise RuntimeError(f"No video path available for node '{context.node.node_id}'")
        qa_dir = self._build_run_dir(context.plan.goal.prompt, "video_qa")
        payload = {
            "video_path": video_path,
            "target_duration": context.node.inputs.get("target_duration"),
            "duration_tolerance": float(context.node.inputs.get("duration_tolerance", 0.6)),
            "expected_width": context.node.inputs.get("expected_width"),
            "expected_height": context.node.inputs.get("expected_height"),
            "expected_fps": context.node.inputs.get("expected_fps"),
            "require_audio": bool(context.node.inputs.get("require_audio", False)),
            "require_stereo_audio": bool(context.node.inputs.get("require_stereo_audio", False)),
            "analyze_audio": bool(context.node.inputs.get("analyze_audio", False)),
            "contact_sheet_path": str(qa_dir / "contact_sheet.jpg"),
            "frame_count": int(context.node.inputs.get("frame_count", 6)),
            "columns": int(context.node.inputs.get("columns", 3)),
            "scale_width": int(context.node.inputs.get("scale_width", 480)),
        }
        result = self.tools.call("media.video_qa", payload)
        passed = bool(result.get("passed", False))
        subject_count = self.prompt_engine.evaluate_media_subjects(
            image_path=str(result.get("contact_sheet_path") or payload["contact_sheet_path"]),
            expected_count=expected_subject_count(context.plan.goal.constraints),
            frame_count=payload["frame_count"],
        ) if passed else {"required": False, "passed": False, "status": "not_run"}
        result["subject_count"] = subject_count
        passed = passed and subject_count["passed"] is True
        result["passed"] = passed
        return SkillResult(
            status="success" if passed else "failed",
            outputs=result,
            metrics={
                "passed": passed,
                "duration": result.get("duration", 0),
            },
            logs=["Hard video checks passed." if passed else "Hard video checks failed."],
        )

    def extract_last_frame(self, context: SkillContext) -> SkillResult:
        run_dir = self._build_run_dir(context.plan.goal.prompt, "frame")
        frame_dir = run_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        output_name = str(context.node.inputs.get("output_name", "last_frame.png"))
        result = self.tools.call(
            "media.extract_last_frame",
            {
                "video_path": str(context.node.inputs.get("video_path") or self._resolve_first(context, ("video_path", "saved_files"))),
                "output_path": str(frame_dir / output_name),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Extracted a tail frame for an agent step."])

    def collect_outputs(self, context: SkillContext) -> SkillResult:
        keys = tuple(context.node.inputs.get("keys", ["saved_files", "video_path", "audio_path", "gif_path", "frame_path"]))
        collected: dict[str, list[str]] = {}
        for dependency in context.node.depends_on:
            dependency_output = context.state[dependency]
            for key in keys:
                value = dependency_output.get(key)
                if isinstance(value, list) and value:
                    collected.setdefault(key, []).extend(str(item) for item in value)
                elif isinstance(value, str) and value:
                    collected.setdefault(key, []).append(value)
        preferred_video_node = str(context.node.inputs.get("preferred_video_node") or "").strip()
        if preferred_video_node:
            preferred_output = context.state.node_outputs.get(preferred_video_node, {})
            if isinstance(preferred_output, dict):
                preferred_video = preferred_output.get("video_path") or preferred_output.get("final_video_path")
                if isinstance(preferred_video, list):
                    preferred_video = next((item for item in preferred_video if item), "")
                if preferred_video:
                    preferred_path = str(preferred_video)
                    collected["final_video_path"] = [preferred_path]
                    collected["video_path"] = [preferred_path]
        collected = {key: list(dict.fromkeys(values)) for key, values in collected.items()}
        collected["prompt_lineage"] = self._collect_prompt_lineage(context)
        collected["node_prompt_modes"] = self._collect_node_prompt_modes(context)
        return SkillResult(status="success", outputs=collected, logs=["Collected upstream artifacts for an agent step."])

    def persist_workflow_summary(self, context: SkillContext) -> SkillResult:
        summary_name = str(context.node.inputs.get("summary_name") or f"{context.plan.workflow_name}_summary.json")
        summary_scope = str(context.node.inputs.get("summary_scope") or context.plan.workflow_name)
        preferred_dir = self._resolve_first(context, ("run_dir",))
        if preferred_dir:
            run_dir = Path(preferred_dir)
        else:
            run_dir = self._build_run_dir(context.plan.goal.prompt, "summary")
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / summary_name

        dependencies = {dependency: dict(context.state[dependency]) for dependency in context.node.depends_on}
        review_outputs = {}
        for dependency in reversed(context.node.depends_on):
            if dependency.startswith("review-"):
                review_outputs = dependencies[dependency]
                break

        summary = {
            "goal": context.plan.goal.prompt,
            "media_type": context.plan.goal.media_type,
            "style": context.plan.goal.style,
            "workflow_name": context.plan.workflow_name,
            "summary_scope": summary_scope,
            "metadata": dict(context.node.inputs.get("summary_metadata") or {}),
            "dependency_outputs": dependencies,
            "final_outputs": self._flatten_summary_outputs(dependencies),
            "review_summary": {
                "selected_assets": review_outputs.get("selected_assets", review_outputs.get("media_paths", [])),
                "rejected_assets": review_outputs.get("rejected_assets", []),
                "rejected_asset_details": review_outputs.get("rejected_asset_details", []),
                "selection_rationale": review_outputs.get("selection_rationale", ""),
            },
            "prompt_lineage": self._collect_prompt_lineage(context),
            "node_prompt_modes": self._collect_node_prompt_modes(context),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_files = [str(summary_path)]
        final_outputs = summary["final_outputs"]
        for key in ("saved_files", "media_paths"):
            for item in final_outputs.get(key, []):
                if item not in saved_files:
                    saved_files.append(str(item))
        return SkillResult(
            status="success",
            outputs={
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "saved_files": saved_files,
                "prompt_lineage": summary["prompt_lineage"],
                "node_prompt_modes": summary["node_prompt_modes"],
                "review_summary": summary["review_summary"],
            },
            logs=["completed"],
        )

    def package_sticker_outputs(self, context: SkillContext) -> SkillResult:
        sticker_batch = context.state[context.node.depends_on[0]]
        run_dir = Path(str(sticker_batch["run_dir"]))
        summary_path = run_dir / "sticker_pack_summary.json"
        items = list(sticker_batch.get("items", []))
        summary = {
            "goal": context.plan.goal.prompt,
            "media_type": context.plan.goal.media_type,
            "style": context.plan.goal.style,
            "item_count": len(items),
            "items": items,
            "prompt_lineage": self._collect_prompt_lineage(context),
            "node_prompt_modes": self._collect_node_prompt_modes(context),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return SkillResult(
            status="success",
            outputs={
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "saved_files": [item["saved_file"] for item in items if item.get("saved_file")],
                "items": items,
                "prompt_lineage": summary["prompt_lineage"],
                "node_prompt_modes": summary["node_prompt_modes"],
            },
            metrics={"sticker_count": len(items)},
            logs=["Packaged sticker outputs for downstream agent use."],
        )

    def package_carousel_outputs(self, context: SkillContext) -> SkillResult:
        rendered = context.state[context.node.depends_on[0]]
        run_dir = Path(str(rendered["run_dir"]))
        summary_path = run_dir / "carousel_summary.json"
        items = list(rendered.get("items", []))
        summary = {
            "goal": context.plan.goal.prompt,
            "media_type": context.plan.goal.media_type,
            "style": context.plan.goal.style,
            "slide_count": len(items),
            "slides": items,
            "prompt_lineage": self._collect_prompt_lineage(context),
            "node_prompt_modes": self._collect_node_prompt_modes(context),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return SkillResult(
            status="success",
            outputs={
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "saved_files": [item["saved_file"] for item in items if item.get("saved_file")],
                "slides": items,
                "prompt_lineage": summary["prompt_lineage"],
                "node_prompt_modes": summary["node_prompt_modes"],
            },
            metrics={"slide_count": len(items)},
            logs=["Packaged carousel outputs for downstream agent use."],
        )

    def package_game_sprite_outputs(self, context: SkillContext) -> SkillResult:
        motion_node = str(context.node.inputs.get("motion_node") or "sprite-motion-plan")
        video_node = str(context.node.inputs.get("video_node") or "sprite-motion-video")
        image_node = str(context.node.inputs.get("image_node") or "sprite-master-image")
        motion_plan = context.state.node_outputs.get(motion_node, {})
        video_output = context.state.node_outputs.get(video_node, {})
        image_output = context.state.node_outputs.get(image_node, {})
        if not isinstance(motion_plan, dict) or not isinstance(video_output, dict):
            raise RuntimeError("Game sprite packaging requires motion-plan and video outputs")

        video_paths = [
            str(path)
            for path in video_output.get("saved_files", [])
            if Path(str(path)).suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
        ]
        video_path = video_paths[0] if video_paths else str(video_output.get("video_path") or "")
        if not video_path:
            raise RuntimeError("Game sprite packaging could not resolve a generated motion video")

        run_dir = self._build_run_dir(context.plan.goal.prompt, "game_sprite")
        result = self.tools.call(
            "media.video_to_sprite",
            {
                "video_path": video_path,
                "output_dir": str(run_dir),
                "cell_width": int(context.node.inputs.get("cell_width", 64)),
                "cell_height": int(context.node.inputs.get("cell_height", 64)),
                "fps": float(context.node.inputs.get("fps", motion_plan.get("fps", 12))),
                "loop": str(motion_plan.get("animation_kind") or "periodic") == "periodic",
                "key_color": motion_plan.get("chroma_color", "magenta"),
                "chroma_threshold": int(motion_plan.get("chroma_threshold", 52)),
                "frame_map": motion_plan.get("frame_map", []),
            },
        )
        image_paths = [
            str(path)
            for path in image_output.get("saved_files", [])
            if Path(str(path)).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        ] if isinstance(image_output, dict) else []
        summary_path = run_dir / "game_sprite_summary.json"
        summary = {
            "asset_kind": "game_sprite",
            "goal": context.plan.goal.prompt,
            "style": context.plan.goal.style,
            "motion_plan": motion_plan,
            "master_image_path": image_paths[0] if image_paths else "",
            "source_video": video_path,
            "outputs": result,
            "prompt_lineage": self._collect_prompt_lineage(context),
            "node_prompt_modes": self._collect_node_prompt_modes(context),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        saved_files = [str(summary_path), video_path, *[str(item) for item in result.get("saved_files", [])]]
        if image_paths:
            saved_files.insert(1, image_paths[0])
        saved_files = list(dict.fromkeys(path for path in saved_files if path))
        return SkillResult(
            status="success",
            outputs={
                **result,
                "video_path": video_path,
                "master_image_path": image_paths[0] if image_paths else "",
                "summary_path": str(summary_path),
                "saved_files": saved_files,
                "motion_plan": motion_plan,
                "prompt_lineage": summary["prompt_lineage"],
                "node_prompt_modes": summary["node_prompt_modes"],
            },
            metrics={"frame_count": len(result.get("frame_paths", []))},
            logs=["Packaged an automatically generated dynamic motion into a transparent 4x4 game sprite atlas."],
        )

    def package_animated_sticker_outputs(self, context: SkillContext) -> SkillResult:
        rendered = context.state[context.node.depends_on[0]]
        gif_preview = context.state[context.node.depends_on[1]]
        saved_files = list(rendered.get("saved_files", []))
        run_dir = self._build_run_dir(context.plan.goal.prompt, "animated_sticker_package")
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "animated_sticker_summary.json"
        summary = {
            "goal": context.plan.goal.prompt,
            "media_type": context.plan.goal.media_type,
            "style": context.plan.goal.style,
            "video_path": rendered.get("video_path") or (saved_files[0] if saved_files else ""),
            "gif_path": gif_preview.get("gif_path", ""),
            "saved_files": [item for item in (gif_preview.get("gif_path", ""), *saved_files) if item],
            "prompt_lineage": self._collect_prompt_lineage(context),
            "node_prompt_modes": self._collect_node_prompt_modes(context),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return SkillResult(
            status="success",
            outputs={
                "video_path": summary["video_path"],
                "gif_path": summary["gif_path"],
                "saved_files": summary["saved_files"],
                "summary_path": str(summary_path),
                "prompt_lineage": summary["prompt_lineage"],
                "node_prompt_modes": summary["node_prompt_modes"],
            },
            metrics={"artifact_count": len(saved_files) + (1 if gif_preview.get("gif_path") else 0)},
            logs=["Packaged animated sticker outputs for downstream agent use."],
        )

    def _build_run_dir(self, prompt: str, suffix: str) -> Path:
        return build_run_dir(self.output_root, prompt, suffix, default_slug="agent")

    def _resolve_prompt(self, context: SkillContext) -> str:
        prompt_key = str(context.node.inputs.get("prompt_key") or "").strip()
        if prompt_key:
            resolved = resolve_dependency_value(context, (prompt_key,))
            if resolved is not None:
                return resolved
        return resolve_dependency_prompt(context)

    @staticmethod
    def _resolve_prompt_with_identity_lock(context: SkillContext, prompt: str) -> str:
        constraints = dict(getattr(context.plan.goal, "constraints", {}) or {})
        character = str(constraints.get("character") or "").strip()
        subject_context = dict(constraints.get("subject_context") or {})
        profile = constraints.get("character_profile")
        if isinstance(profile, dict) and profile:
            subject_context.setdefault("character_profile", profile)
        if not character and not subject_context:
            return str(prompt or "").strip()
        identity = subject_identity_lock(character, subject_context)
        if not identity:
            return str(prompt or "").strip()
        base_prompt = str(prompt or "").strip().rstrip(".")
        if base_prompt:
            return f"{base_prompt}. Character identity lock: {identity}."
        return f"Character identity lock: {identity}."

    @staticmethod
    def _first_output_path(outputs: dict[str, object]) -> str | None:
        for key in ("image_path", "frame_path", "selected_assets", "saved_files", "media_paths"):
            value = outputs.get(key)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _output_paths(outputs: dict[str, object]) -> list[str]:
        """Return every image output so batch candidates are validated individually."""
        for key in ("saved_files", "selected_assets", "media_paths"):
            value = outputs.get(key)
            if isinstance(value, list):
                return list(dict.fromkeys(str(item) for item in value if str(item)))
        first = AgentMediaSkills._first_output_path(outputs)
        return [first] if first else []

    def _resolve_negative_prompt(self, context: SkillContext) -> str:
        return resolve_dependency_negative_prompt(context)

    def _resolve_image_path(self, context: SkillContext) -> str:
        image_path = context.node.inputs.get("image_path") or context.node.inputs.get("input_image_path")
        if isinstance(image_path, str) and image_path:
            return image_path
        resolved = self._resolve_first(
            context, ("image_path", "frame_path", "saved_files", "selected_assets", "media_paths")
        )
        if not resolved:
            raise RuntimeError(f"No image path available for node '{context.node.node_id}'")
        return resolved

    def _resolve_text(self, context: SkillContext) -> str:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            narration = dependency_output.get("narration")
            if isinstance(narration, str) and narration:
                return narration
            text = dependency_output.get("text")
            if isinstance(text, str) and text:
                return text
        return context.plan.goal.constraints.get("text", "") or context.plan.goal.prompt

    @staticmethod
    def _resolve_first(context: SkillContext, candidate_keys: tuple[str, ...]) -> str | None:
        return resolve_dependency_value(context, candidate_keys)

    @staticmethod
    def _resolve_many(context: SkillContext, candidate_keys: tuple[str, ...]) -> list[str]:
        values = collect_output_values(context, candidate_keys, first_key_only=True)
        if not values:
            raise RuntimeError(f"No dependency outputs found for node '{context.node.node_id}'")
        return values

    @staticmethod
    def _collect_prompt_lineage(context: SkillContext) -> list[dict[str, object]]:
        relevant_node_ids = {context.node.node_id, *context.node.depends_on}
        return [
            dict(entry)
            for entry in context.state.prompt_lineage
            if str(entry.get("node_id", "")) in relevant_node_ids
        ]

    @staticmethod
    def _collect_node_prompt_modes(context: SkillContext) -> dict[str, str]:
        relevant_node_ids = {context.node.node_id, *context.node.depends_on}
        return {
            node_id: prompt_mode
            for node_id, prompt_mode in context.state.node_prompt_modes.items()
            if node_id in relevant_node_ids
        }

    @staticmethod
    def _flatten_summary_outputs(dependencies: dict[str, dict[str, object]]) -> dict[str, list[str]]:
        tracked_keys = ("saved_files", "media_paths", "video_path", "audio_path", "gif_path", "frame_path")
        flattened: dict[str, list[str]] = {}
        for outputs in dependencies.values():
            for key in tracked_keys:
                value = outputs.get(key)
                if isinstance(value, list):
                    flattened.setdefault(key, []).extend(str(item) for item in value if item)
                elif isinstance(value, str) and value:
                    flattened.setdefault(key, []).append(value)
        return {key: list(dict.fromkeys(values)) for key, values in flattened.items()}


def register_agent_primitive_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    prompt_engine: PromptEngine | None = None,
) -> None:
    planning = AgentPlanningSkills(prompt_engine=prompt_engine)
    media = AgentMediaSkills(tool_registry, output_root, prompt_engine=prompt_engine)
    skill_registry.register("agent.goal.expand", planning.expand_goal, "Expand a goal into an agent-ready brief")
    skill_registry.register("agent.prompt.compose", planning.compose_prompt, "Compose a reusable prompt bundle")
    skill_registry.register("agent.story.segment", planning.segment_story, "Break a story into reusable segments")
    skill_registry.register("agent.segment.prepare", planning.prepare_segment, "Prepare one segment prompt bundle")
    skill_registry.register("agent.review.refine_prompt", planning.refine_prompt_after_review, "Refine a prompt using review feedback")
    skill_registry.register("agent.sticker.expressions", planning.generate_sticker_expressions, "Generate sticker expression ideas")
    skill_registry.register("agent.sticker.prompt_set", planning.build_sticker_prompt_set, "Build sticker prompt sets")
    skill_registry.register("agent.sticker.motion_prompt", planning.build_sticker_motion_prompt, "Build an animated sticker motion prompt")
    skill_registry.register("agent.sprite.motion_plan", planning.build_dynamic_sprite_motion_plan, "Generate an open-ended game sprite motion plan")
    skill_registry.register("agent.carousel.prompt_set", planning.build_slide_prompt_set, "Build carousel slide prompt sets")
    skill_registry.register("agent.story_card.write", planning.write_story_card, "Write a dedicated story-card article")
    skill_registry.register("media.ensure_workflow", media.ensure_workflow, "Check workflow assets for any agent step")
    skill_registry.register("media.image.refine", media.refine_image, "Refine an image as an agent media primitive")
    skill_registry.register("media.image.generate_keyframe", media.generate_keyframe, "Generate a keyframe from text or a prior frame")
    skill_registry.register("media.image.validate_character", media.validate_character_frames, "Validate configured character continuity keyframes")
    skill_registry.register("media.image.validate_last_frame", media.validate_last_frame, "Validate and preserve a native H3 L2VA last frame")
    skill_registry.register("media.image.upscale", media.upscale_image, "Upscale an image as an agent media primitive")
    skill_registry.register("media.image.animate", media.animate_image, "Animate an image as an agent media primitive")
    skill_registry.register("media.image.render_batch", media.render_image_batch, "Render a batch of images as an agent media primitive")
    skill_registry.register("media.story_card.backgrounds", media.render_story_card_backgrounds, "Render style-locked story-card backgrounds per page")
    skill_registry.register("media.story_card.compose", media.compose_story_card, "Compose exact story-card text over generated backgrounds")
    skill_registry.register("media.audio.narrate", media.narrate_text, "Generate narration audio as an agent media primitive")
    skill_registry.register("media.audio.concat", media.concat_audio_tracks, "Concatenate audio tracks as an agent media primitive")
    skill_registry.register("media.video.concat", media.concat_videos, "Concatenate videos as an agent media primitive")
    skill_registry.register("media.video.change_speed", media.change_video_speed, "Change final video playback speed")
    skill_registry.register("media.video.normalize_canvas", media.normalize_video_canvas, "Normalize a final video to the requested canvas")
    skill_registry.register("media.video.trim", media.trim_video, "Trim final video to the requested duration")
    skill_registry.register("media.video.merge_audio", media.merge_audio_video, "Mux audio and video as an agent media primitive")
    skill_registry.register("media.video.gif_preview", media.video_to_gif, "Create a GIF preview as an agent media primitive")
    skill_registry.register("media.video.qa", media.qa_video, "Run technical video QA and create a contact sheet")
    skill_registry.register("media.video.extract_last_frame", media.extract_last_frame, "Extract the last frame as an agent media primitive")
    skill_registry.register("agent.sticker.package", media.package_sticker_outputs, "Package sticker artifacts for downstream agent use")
    skill_registry.register("agent.sticker.animate.package", media.package_animated_sticker_outputs, "Package animated sticker artifacts for downstream agent use")
    skill_registry.register("agent.sprite.package", media.package_game_sprite_outputs, "Package dynamic game sprite artifacts")
    skill_registry.register("agent.carousel.package", media.package_carousel_outputs, "Package carousel artifacts for downstream agent use")
    skill_registry.register("agent.output.collect", media.collect_outputs, "Collect upstream artifacts for downstream agent steps")
    skill_registry.register("agent.summary.persist", media.persist_workflow_summary, "Persist a structured workflow summary artifact")
