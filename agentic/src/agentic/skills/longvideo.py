from __future__ import annotations

from pathlib import Path

from agentic.h3_reference import build_reference_lineage, format_ref2va_prompt, normalize_reference_manifest
from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.minimax_prompting import structured_visual_prompt
from agentic.runtime.prompting import build_minimax_h3_prompt, build_story_segments
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.runtime.story_service import NativeH3StoryService
from agentic.skills.shared import asset_check_result, build_run_dir, resolve_dependency_value
from agentic.storyboard import format_native_h3_prompt, load_storyboard


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Native H3 {name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise RuntimeError(f"Native H3 {name} must be between {minimum} and {maximum}; received {parsed}")
    return parsed


class LongVideoSkills:
    def __init__(
        self,
        tools: ToolRegistry,
        output_root: Path,
        story_service: NativeH3StoryService | None = None,
        prompt_engine: PromptEngine | None = None,
    ) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Direct construction remains compatible for tests and standalone callers;
        # the application composition root injects the shared service instance.
        self.story_service = story_service or NativeH3StoryService()
        self.prompt_engine = prompt_engine or PromptEngine()

    def expand_idea(self, context: SkillContext) -> SkillResult:
        prompt = str(context.node.inputs["prompt"])
        style = str(context.node.inputs["style"])
        idea_variants = list(context.node.inputs.get("idea_variants", []))
        selected_variant = idea_variants[0] if idea_variants else {"style": style}
        selected_style = str(selected_variant.get("style", style))
        expanded_prompt = structured_visual_prompt(
            subject=str(context.plan.goal.constraints.get("character") or "the main subject"),
            scene=prompt,
            action="one meaningful physical action with a visible beginning, escalation, and end",
            environment="a coherent world that changes with the action",
            camera="camera framing and movement follow the action with a clear change in depth",
            style=selected_style,
            quality="cinematic lighting, strong silhouette, spatial depth, coherent continuity",
        )
        return SkillResult(
            status="success",
            outputs={
                "creative_brief": f"{prompt} rendered as a {selected_style} long-form video narrative",
                "tone": "playful cinematic escalation",
                "idea_variants": idea_variants,
                "prompt": expanded_prompt,
                "negative_prompt": "ugly, blurry, low quality, bad anatomy, deformed, duplicate, watermark, text",
            },
            logs=["Expanded goal into a creative brief."],
        )

    def segment_story(self, context: SkillContext) -> SkillResult:
        segment_count = int(context.node.inputs["segment_count"])
        brief = str(context.state["idea-brief"]["creative_brief"])
        segments = build_story_segments(context.plan.goal, brief, segment_count, "playful cinematic escalation")
        return SkillResult(
            status="success",
            outputs={"segments": segments, "segment_count": segment_count},
            metrics={"segment_count": segment_count},
            logs=[f"Planned {segment_count} story segments."],
        )

    def ensure_workflow(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "asset.ensure_workflow_ready",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "auto_download": context.node.inputs.get("auto_download", False),
            },
        )
        return asset_check_result(result, "Checked workflow assets.")

    def prepare_native_h3_story(self, context: SkillContext) -> SkillResult:
        storyboard_path = str(
            context.node.inputs.get("storyboard_path")
            or context.plan.goal.constraints.get("native_h3_storyboard_path")
            or context.plan.goal.constraints.get("storyboard_path")
            or ""
        )
        storyboard = load_storyboard(storyboard_path)
        duration_seconds = int(
            context.node.inputs.get("duration_seconds")
            or context.plan.goal.constraints.get("native_h3_duration_seconds")
            or storyboard.get("native_duration_seconds")
            or 15
        )
        style = str(context.node.inputs.get("style") or context.plan.goal.style)
        creative_brief = str(context.plan.goal.prompt or "").strip()
        # When no user prompt was supplied, the character workflow has already
        # selected news and may have produced an autonomous scene brief. That
        # brief is optional inspiration, not a second user objective. Passing
        # it into the news-grounding validator makes harmless scene details
        # look like mandatory constraints and can reject a valid news story.
        if str(context.plan.goal.constraints.get("prompt_source") or "").strip().lower() != "user":
            creative_brief = ""
        news_context = context.plan.goal.constraints.get("news_context") or {}
        if not isinstance(news_context, dict):
            raise RuntimeError("Native H3 news_context must be a mapping")
        storyboard, story_payload = self.story_service.resolve(
            storyboard,
            character=str(context.plan.goal.constraints.get("character") or "Kirby"),
            style=style,
            duration_seconds=duration_seconds,
            news_context=news_context,
            creative_brief=creative_brief,
        )
        render_mode = str(context.node.inputs.get("render_mode") or "").strip()
        if render_mode:
            storyboard["render_mode"] = render_mode
        prompt = format_native_h3_prompt(
            storyboard,
            style=style,
            duration_seconds=duration_seconds,
        )
        opening_prompt = str(storyboard.get("opening_keyframe_prompt") or "").strip()
        ending_prompt = str(storyboard.get("ending_keyframe_prompt") or "").strip()
        if not opening_prompt or not ending_prompt:
            raise ValueError("Native H3 storyboard must define opening_keyframe_prompt and ending_keyframe_prompt")
        negative_prompt = str(storyboard["negative_prompt"]).strip()
        return SkillResult(
            status="success",
            outputs={
                "storyboard_path": str(storyboard.get("_path") or storyboard_path),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "opening_keyframe_prompt": opening_prompt,
                "ending_keyframe_prompt": ending_prompt,
                "story_spine": dict(storyboard.get("story_spine") or {}),
                "native_audio": str(storyboard.get("native_audio") or ""),
                "duration_seconds": duration_seconds,
                "prompt_mode": str(story_payload["prompt_mode"]),
                "story_source": str(story_payload["source"]),
                "creative_seed": str(story_payload["creative_seed"]),
                "news_context": dict(story_payload["news_context"]),
                "story_quality": dict(storyboard.get("story_quality") or story_payload.get("story_quality") or {}),
                "news_grounding": dict(story_payload.get("news_grounding") or {}),
                "generated_storyboard": storyboard,
                "creative_brief": creative_brief,
            },
            metrics={"duration_seconds": duration_seconds, "native_shot_count": len(storyboard.get("native_shots") or [])},
            logs=[
                f"Prepared news-grounded native H3 story '{storyboard.get('name')}' from '{storyboard_path}'.",
                f"News source: {story_payload['news_context'].get('title', '')}",
            ],
        )

    def render_initial_frame(self, context: SkillContext) -> SkillResult:
        segment = self._segment(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt, f"{segment['segment_id']}_image")
        workflow_name = str(
            context.plan.goal.constraints.get("keyframe_workflow_name")
            or context.node.inputs["workflow_name"]
        )
        result = self.tools.call(
            "comfy.render_image",
            {
                "workflow_name": workflow_name,
                "run_dir": str(run_dir),
                "prompt": f"{segment['visual']}, {context.plan.goal.style}, cinematic lighting, consistent character design",
                "negative_prompt": str(context.state["idea-brief"].get("negative_prompt", "")),
                "width": context.node.inputs.get("width", 1024),
                "height": context.node.inputs.get("height", 1024),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result.get("saved_files", []))},
            logs=[f"Rendered initial keyframe for {segment['segment_id']}."],
        )

    def render_segment_video(self, context: SkillContext) -> SkillResult:
        segment = self._segment(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt, f"{segment['segment_id']}_video")
        image_path = self._optional_dependency_value(context, ("frame_path", "saved_files"))
        if not image_path:
            raise RuntimeError(f"Missing frame input for node '{context.node.node_id}'")

        workflow_name = str(
            context.plan.goal.constraints.get("video_workflow_name")
            or context.node.inputs["workflow_name"]
        )
        prompt = f"{segment['visual']}, {context.plan.goal.style}, motion continuity, coherent action"
        if workflow_name.startswith("minimax_h3_"):
            prompt = build_minimax_h3_prompt(
                context.plan.goal,
                segment,
                prior_frame=image_path,
            )["prompt"]
        width = context.node.inputs.get("width")
        height = context.node.inputs.get("height")
        length = context.node.inputs.get("length")
        steps = context.node.inputs.get("steps")
        if workflow_name.startswith("minimax_h3_"):
            # Long-video renders queue several H3 jobs in one run. Use a
            # smaller per-segment draft than the standalone H3 workflows so
            # an 8GB GPU can release/reload the graph between segments without
            # an allocation failure. Callers can override these through the
            # goal constraints when more VRAM is available.
            constraints = context.plan.goal.constraints
            width = constraints.get("longvideo_h3_width", width or 512)
            height = constraints.get("longvideo_h3_height", height or 288)
            length = constraints.get("longvideo_h3_length", length or 81)
            steps = constraints.get("longvideo_h3_steps", steps or 16)
        result = self.tools.call(
            "comfy.render_image_to_video",
            {
                "workflow_name": workflow_name,
                "run_dir": str(run_dir),
                "image_path": image_path,
                "prompt": prompt,
                "width": width if width is not None else (608 if workflow_name.startswith("minimax_h3_") else None),
                "height": height if height is not None else (352 if workflow_name.startswith("minimax_h3_") else None),
                "length": length,
                "steps": steps,
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"video_count": len(result.get("saved_files", []))},
            logs=[f"Rendered {segment['segment_id']} clip."],
        )

    def render_native_h3(self, context: SkillContext) -> SkillResult:
        story = context.state["native-story-prompt"]
        gate = context.state["native-keyframe-gate"]
        if int(context.plan.goal.duration_seconds) > 15:
            raise RuntimeError(
                "Direct MiniMax H3 rendering supports at most 15 seconds (362 frames) in the local trained range; "
                "use an explicit continuation workflow for a longer story."
            )
        workflow_name = str(context.node.inputs.get("workflow_name") or context.plan.goal.constraints.get("video_workflow_name") or "")
        first_frame = str(gate.get("first_frame_path") or "")
        use_last_frame = bool(
            context.node.inputs.get(
                "use_last_frame",
                context.plan.goal.constraints.get("native_h3_use_last_frame", False),
            )
        )
        last_frame = str(gate.get("last_frame_path") or "") if use_last_frame else ""
        if not workflow_name or not first_frame:
            raise RuntimeError("Native H3 render requires workflow_name and first_frame_path")
        if use_last_frame and not last_frame:
            raise RuntimeError("Native H3 render with use_last_frame=true requires last_frame_path")
        default_model_profile = "q2" if workflow_name.startswith("minimax_h3_lowvram_15s") else "q4"
        payload = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "native_h3")),
            "image_path": first_frame,
            "use_last_frame": use_last_frame,
            "h3_mode": "fl2va" if use_last_frame else "i2va",
            "prompt": str(story["prompt"]),
            "negative_prompt": str(story.get("negative_prompt") or ""),
            "character": str(context.plan.goal.constraints.get("character") or ""),
            "width": _bounded_int(
                context.node.inputs.get("width") or context.plan.goal.constraints.get("native_h3_width") or 608,
                name="width",
                minimum=256,
                maximum=1024,
            ),
            "height": _bounded_int(
                context.node.inputs.get("height") or context.plan.goal.constraints.get("native_h3_height") or 352,
                name="height",
                minimum=256,
                maximum=1024,
            ),
            "length": _bounded_int(
                context.node.inputs.get("length") or context.plan.goal.constraints.get("native_h3_length") or 362,
                name="length",
                minimum=17,
                maximum=362,
            ),
            "steps": _bounded_int(
                context.node.inputs.get("steps") or context.plan.goal.constraints.get("native_h3_steps") or 16,
                name="steps",
                minimum=1,
                maximum=32,
            ),
            "video_count": _bounded_int(
                context.node.inputs.get("video_count") or 1,
                name="video_count",
                minimum=1,
                maximum=4,
            ),
            "model_profile": str(
                context.node.inputs.get("model_profile")
                or context.plan.goal.constraints.get("native_h3_model_profile")
                or default_model_profile
            ),
        }
        if use_last_frame:
            payload["last_image_path"] = last_frame
        result = self.tools.call("comfy.workflow.image_to_video", payload)
        outputs: dict[str, object] = {
            **result,
            "first_frame_path": first_frame,
            "last_frame_path": last_frame,
            "use_last_frame": use_last_frame,
            "native_h3_prompt": story["prompt"],
            "story_source": story.get("story_source", "news_llm"),
            "creative_seed": story.get("creative_seed", ""),
            "news_context": story.get("news_context", {}),
            "story_quality": dict(story.get("story_quality") or {}),
            "news_grounding": dict(story.get("news_grounding") or {}),
            "generated_storyboard": story.get("generated_storyboard", {}),
        }
        return SkillResult(
            status="success",
            outputs=outputs,
            metrics={"video_count": len(result.get("saved_files", [])), "length": payload["length"], "steps": payload["steps"]},
            logs=[
                f"Rendered one continuous native H3 story with '{workflow_name}' ({'first + last frame' if use_last_frame else 'first frame only'})."
            ],
        )

    def validate_native_h3_references(self, context: SkillContext) -> SkillResult:
        constraints = dict(getattr(context.plan.goal, "constraints", {}) or {})
        manifest = context.node.inputs.get("reference_manifest")
        selected_assets = self._collect_accepted_reference_assets(context)
        manifest_entries: list[object] = []
        if isinstance(manifest, (list, tuple)):
            manifest_entries.extend(manifest)
        elif manifest:
            manifest_entries.append(manifest)
        existing_paths: set[str] = set()
        for item in manifest_entries:
            if isinstance(item, dict):
                raw_path = item.get("path", item.get("source_path"))
            else:
                raw_path = item if isinstance(item, str) else None
            if raw_path:
                existing_paths.add(str(Path(str(raw_path)).expanduser().resolve()))
        for path in selected_assets:
            canonical = str(Path(path).expanduser().resolve())
            if canonical not in existing_paths:
                manifest_entries.append({"path": path})
                existing_paths.add(canonical)
        references = normalize_reference_manifest(
            manifest_entries or None,
            image_paths=context.node.inputs.get("reference_image_paths") or constraints.get("native_h3_reference_image_paths"),
            video_paths=context.node.inputs.get("reference_video_paths") or constraints.get("native_h3_reference_video_paths"),
            require_files=True,
            max_images=int(
                context.node.inputs.get("max_images")
                or constraints.get("native_h3_reference_max_images")
                or 3
            ),
            max_videos=int(
                context.node.inputs.get("max_videos")
                or constraints.get("native_h3_reference_max_videos")
                or 1
            ),
        )
        return SkillResult(
            status="success",
            outputs={
                "reference_manifest": references,
                "reference_lineage": build_reference_lineage(references),
                "reference_audio_enabled": False,
            },
            metrics={
                "reference_image_count": sum(ref["type"] == "image" for ref in references),
                "reference_video_count": sum(ref["type"] == "video" for ref in references),
            },
            logs=[
                f"Validated H3 Ref2VA references ({len(references)} accepted image/video assets); reference audio is disabled."
            ],
        )

    @staticmethod
    def _collect_accepted_reference_assets(context: SkillContext) -> list[str]:
        """Collect only assets selected by the preceding review node.

        Ref2VA is the multi-reference route. Other video workflows resolve a
        single first image elsewhere; this helper is intentionally local to
        the Ref2VA validation step so accepted review media cannot silently
        widen ordinary I2V inputs.
        """

        selected: list[str] = []
        for dependency in context.node.depends_on:
            output = context.state[dependency]
            if not isinstance(output, dict):
                continue
            for key in ("selected_assets", "media_paths", "saved_files"):
                value = output.get(key)
                if isinstance(value, list):
                    selected.extend(str(path) for path in value if path)
                elif isinstance(value, str) and value:
                    selected.append(value)
        deduplicated = list(dict.fromkeys(selected))
        node_inputs = getattr(context.node, "inputs", {}) or {}
        if bool(node_inputs.get("auto_reference_generation", False)):
            limit = int(node_inputs.get("selection_limit") or 0)
            if limit > 0:
                return deduplicated[:limit]
        return deduplicated

    def render_native_h3_l2va(self, context: SkillContext) -> SkillResult:
        story = context.state["native-story-prompt"]
        gate = context.state["native-l2va-frame-gate"]
        if int(context.plan.goal.duration_seconds) > 15:
            raise RuntimeError(
                "Direct MiniMax H3 last-frame-to-video rendering supports at most 15 seconds (362 frames); "
                "use an explicit continuation workflow for a longer story."
            )
        workflow_name = str(
            context.node.inputs.get("workflow_name")
            or context.plan.goal.constraints.get("video_workflow_name")
            or ""
        )
        last_frame = str(gate.get("last_frame_path") or "")
        if not workflow_name or not last_frame:
            raise RuntimeError("Native H3 L2VA render requires workflow_name and last_frame_path")
        payload = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "native_h3_l2va")),
            "last_image_path": last_frame,
            "use_first_frame": False,
            "use_last_frame": True,
            "h3_mode": "l2va",
            "prompt": str(story["prompt"]),
            "negative_prompt": str(story.get("negative_prompt") or ""),
            "character": str(context.plan.goal.constraints.get("character") or ""),
            "width": _bounded_int(
                context.node.inputs.get("width") or context.plan.goal.constraints.get("native_h3_width") or 608,
                name="width",
                minimum=256,
                maximum=1024,
            ),
            "height": _bounded_int(
                context.node.inputs.get("height") or context.plan.goal.constraints.get("native_h3_height") or 352,
                name="height",
                minimum=256,
                maximum=1024,
            ),
            "length": _bounded_int(
                context.node.inputs.get("length") or context.plan.goal.constraints.get("native_h3_length") or 362,
                name="length",
                minimum=17,
                maximum=362,
            ),
            "steps": _bounded_int(
                context.node.inputs.get("steps") or context.plan.goal.constraints.get("native_h3_steps") or 16,
                name="steps",
                minimum=1,
                maximum=32,
            ),
            "video_count": _bounded_int(context.node.inputs.get("video_count") or 1, name="video_count", minimum=1, maximum=4),
            "model_profile": str(
                context.node.inputs.get("model_profile")
                or context.plan.goal.constraints.get("native_h3_model_profile")
                or "q4"
            ),
        }
        result = self.tools.call("comfy.workflow.image_to_video", payload)
        return SkillResult(
            status="success",
            outputs={
                **result,
                "first_frame_path": "",
                "last_frame_path": last_frame,
                "use_first_frame": False,
                "use_last_frame": True,
                "native_h3_prompt": story["prompt"],
                "story_source": story.get("story_source", "news_llm"),
                "creative_seed": story.get("creative_seed", ""),
                "news_context": story.get("news_context", {}),
                "story_quality": dict(story.get("story_quality") or {}),
                "news_grounding": dict(story.get("news_grounding") or {}),
                "generated_storyboard": story.get("generated_storyboard", {}),
                "render_mode": "last_frame_to_video",
            },
            metrics={"video_count": len(result.get("saved_files", [])), "length": payload["length"], "steps": payload["steps"]},
            logs=["Rendered native H3 L2VA with an approved last frame and no first-frame connection."],
        )

    def render_native_h3_ref2va(self, context: SkillContext) -> SkillResult:
        story = context.state["native-story-prompt"]
        validation = context.state["native-ref2va-reference-check"]
        references = list(validation.get("reference_manifest") or [])
        if int(context.plan.goal.duration_seconds) > 15:
            raise RuntimeError(
                "Direct MiniMax H3 Ref2VA rendering supports at most 15 seconds (362 frames); "
                "use an explicit continuation workflow for a longer story."
            )
        workflow_name = str(
            context.node.inputs.get("workflow_name")
            or context.plan.goal.constraints.get("video_workflow_name")
            or ""
        )
        if not workflow_name:
            raise RuntimeError("Native H3 Ref2VA render requires workflow_name")
        prompt = format_ref2va_prompt(
            str(story["prompt"]),
            references,
            soundscape=str(story.get("native_audio") or "Generate native H3 audio from the scene; do not use reference audio."),
        )
        payload = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "native_h3_ref2va")),
            "prompt": prompt,
            "h3_mode": "ref2va",
            "negative_prompt": str(story.get("negative_prompt") or ""),
            "reference_manifest": references,
            "ref_image_size": str(
                context.node.inputs.get("ref_image_size")
                or context.plan.goal.constraints.get("native_h3_reference_image_size")
                or "match"
            ),
            "model_profile": str(
                context.node.inputs.get("model_profile")
                or context.plan.goal.constraints.get("native_h3_model_profile")
                or "q4"
            ),
            "width": _bounded_int(
                context.node.inputs.get("width") or context.plan.goal.constraints.get("native_h3_width") or 608,
                name="width",
                minimum=256,
                maximum=1024,
            ),
            "height": _bounded_int(
                context.node.inputs.get("height") or context.plan.goal.constraints.get("native_h3_height") or 352,
                name="height",
                minimum=256,
                maximum=1024,
            ),
            "length": _bounded_int(
                context.node.inputs.get("length") or context.plan.goal.constraints.get("native_h3_length") or 124,
                name="length",
                minimum=17,
                maximum=362,
            ),
            "steps": _bounded_int(
                context.node.inputs.get("steps") or context.plan.goal.constraints.get("native_h3_steps") or 20,
                name="steps",
                minimum=1,
                maximum=32,
            ),
            "video_count": _bounded_int(context.node.inputs.get("video_count") or 1, name="video_count", minimum=1, maximum=4),
        }
        result = self.tools.call("comfy.workflow.reference_to_video", payload)
        return SkillResult(
            status="success",
            outputs={
                **result,
                "reference_manifest": references,
                "reference_lineage": list(validation.get("reference_lineage") or build_reference_lineage(references)),
                "reference_audio_enabled": False,
                "native_h3_prompt": prompt,
                "story_source": story.get("story_source", "news_llm"),
                "creative_seed": story.get("creative_seed", ""),
                "news_context": story.get("news_context", {}),
                "story_quality": dict(story.get("story_quality") or {}),
                "news_grounding": dict(story.get("news_grounding") or {}),
                "generated_storyboard": story.get("generated_storyboard", {}),
                "render_mode": "reference_to_video",
            },
            metrics={
                "video_count": len(result.get("saved_files", [])),
                "reference_count": len(references),
                "length": payload["length"],
                "steps": payload["steps"],
            },
            logs=[f"Rendered native H3 Ref2VA with {len(references)} image/video references and native generated audio only."],
        )

    def render_native_h3_t2v(self, context: SkillContext) -> SkillResult:
        story = context.state["native-story-prompt"]
        if int(context.plan.goal.duration_seconds) > 15:
            raise RuntimeError(
                "Direct MiniMax H3 text-to-video supports at most 15 seconds (362 frames) in the local trained range; "
                "use an explicit continuation workflow for a longer story."
            )
        workflow_name = str(
            context.node.inputs.get("workflow_name")
            or context.plan.goal.constraints.get("video_workflow_name")
            or ""
        )
        if not workflow_name:
            raise RuntimeError("Native H3 T2V render requires workflow_name")
        lowvram_t2v = workflow_name.startswith("minimax_h3_lowvram_")
        constraints = context.plan.goal.constraints
        payload = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "native_h3_t2v")),
            "prompt": str(story["prompt"]),
            "h3_mode": "t2va",
            "negative_prompt": str(story.get("negative_prompt") or ""),
            "width": _bounded_int(
                context.node.inputs.get("width")
                or constraints.get("native_h3_t2v_width")
                or (512 if lowvram_t2v else constraints.get("native_h3_width") or 608),
                name="width",
                minimum=256,
                maximum=1024,
            ),
            "height": _bounded_int(
                context.node.inputs.get("height")
                or constraints.get("native_h3_t2v_height")
                or (288 if lowvram_t2v else constraints.get("native_h3_height") or 352),
                name="height",
                minimum=256,
                maximum=1024,
            ),
            "length": _bounded_int(
                context.node.inputs.get("length") or context.plan.goal.constraints.get("native_h3_length") or 362,
                name="length",
                minimum=17,
                maximum=362,
            ),
            "steps": _bounded_int(
                context.node.inputs.get("steps") or context.plan.goal.constraints.get("native_h3_steps") or 20,
                name="steps",
                minimum=1,
                maximum=32,
            ),
            "video_count": _bounded_int(
                context.node.inputs.get("video_count") or 1,
                name="video_count",
                minimum=1,
                maximum=4,
            ),
            "model_profile": str(
                context.node.inputs.get("model_profile")
                or constraints.get("native_h3_t2v_model_profile")
                or constraints.get("native_h3_model_profile")
                or ("q2" if lowvram_t2v else "q4")
            ),
        }
        result = self.tools.call("comfy.workflow.text_to_video", payload)
        return SkillResult(
            status="success",
            outputs={
                **result,
                "native_h3_prompt": story["prompt"],
                "story_source": story.get("story_source", "news_llm"),
                "creative_seed": story.get("creative_seed", ""),
                "news_context": story.get("news_context", {}),
                "story_quality": dict(story.get("story_quality") or {}),
                "generated_storyboard": story.get("generated_storyboard", {}),
                "render_mode": "text_to_video",
            },
            metrics={"video_count": len(result.get("saved_files", [])), "length": payload["length"], "steps": payload["steps"]},
            logs=[f"Rendered one continuous native H3 text-to-video story with '{workflow_name}'."],
        )

    def qa_native_h3(self, context: SkillContext) -> SkillResult:
        render = context.state["native-h3-render"]
        saved_files = [str(path) for path in render.get("saved_files", []) if path]
        if not saved_files:
            raise RuntimeError("Native H3 QA requires at least one generated video")
        video_path = saved_files[0]
        story_quality = dict((context.state["native-story-prompt"] or {}).get("story_quality") or {})
        constraints = dict(getattr(context.plan.goal, "constraints", {}) or {})
        native_recipe = dict(constraints.get("native_h3_recipe") or {})
        h3_mode = str(render.get("h3_mode") or constraints.get("native_h3_mode") or "")
        target_duration = float(
            context.node.inputs.get("target_duration")
            or getattr(context.plan.goal, "duration_seconds", 15)
            or 15
        )
        duration_tolerance = float(context.node.inputs.get("duration_tolerance") or 0.75)
        expected_width = int(
            context.node.inputs.get("expected_width")
            or constraints.get("native_h3_width")
            or native_recipe.get("width")
            or 608
        )
        expected_height = int(
            context.node.inputs.get("expected_height")
            or constraints.get("native_h3_height")
            or native_recipe.get("height")
            or 352
        )
        expected_fps = float(
            context.node.inputs.get("expected_fps")
            or constraints.get("native_h3_frame_rate")
            or native_recipe.get("frame_rate")
            or 24
        )
        render_dir = Path(str(render.get("run_dir") or Path(video_path).parent))
        contact_sheet_path = render_dir / "qa" / "contact_sheet.jpg"
        technical_qa = self.tools.call(
            "media.video_qa",
            {
                "video_path": video_path,
                "target_duration": target_duration,
                "duration_tolerance": duration_tolerance,
                "warn_if_no_audio": True,
                "expected_width": expected_width,
                "expected_height": expected_height,
                "expected_fps": expected_fps,
                "require_audio": bool(
                    context.node.inputs.get("require_audio", native_recipe.get("require_audio", False))
                ),
                "require_stereo_audio": bool(
                    context.node.inputs.get(
                        "require_stereo_audio", native_recipe.get("require_stereo_audio", False)
                    )
                ),
                "analyze_audio": bool(
                    context.node.inputs.get("analyze_audio", native_recipe.get("analyze_audio", False))
                ),
                "contact_sheet_path": str(contact_sheet_path),
                "frame_count": int(context.node.inputs.get("frame_count") or 6),
                "columns": int(context.node.inputs.get("columns") or 3),
                "scale_width": int(context.node.inputs.get("scale_width") or 320),
            },
        )
        story = dict(context.state["native-story-prompt"] or {})
        semantic_qa_required = bool(
            context.node.inputs.get(
                "semantic_qa_required",
                constraints.get("native_h3_semantic_qa_required", native_recipe.get("semantic_qa_required", False)),
            )
        )
        require_human_review = bool(constraints.get("require_human_review", False))
        semantic_qa: dict[str, object]
        if not semantic_qa_required:
            semantic_qa = {
                "enabled": False,
                "required": False,
                "status": "disabled",
                "passed": None,
                "reason": "semantic QA is not enabled by the native H3 recipe",
                "contact_sheet_path": str(technical_qa.get("contact_sheet_path") or contact_sheet_path),
            }
        else:
            storyboard = dict(story.get("generated_storyboard") or {})
            semantic_qa = self.prompt_engine.evaluate_video_contact_sheet(
                contact_sheet_path=str(technical_qa.get("contact_sheet_path") or contact_sheet_path),
                character=str(constraints.get("character") or "Kirby"),
                story_spine=dict(story.get("story_spine") or storyboard.get("story_spine") or {}),
                native_shots=[item for item in (storyboard.get("native_shots") or []) if isinstance(item, dict)],
                news_context=dict(story.get("news_context") or {}),
                rendered_prompt=str(story.get("prompt") or render.get("native_h3_prompt") or ""),
                news_anchor_terms=[
                    str(item)
                    for item in (dict(storyboard.get("news_trace") or {}).get("visual_anchors") or [])
                    if str(item).strip()
                ],
            )
            semantic_qa["enabled"] = True
            semantic_qa["required"] = semantic_qa_required
            semantic_qa["blocking"] = semantic_qa_required and not require_human_review

        technical_passed = bool(technical_qa.get("passed"))
        semantic_blocked = bool(semantic_qa.get("blocking")) and semantic_qa.get("passed") is not True
        passed = technical_passed and not semantic_blocked
        failures = [str(item) for item in (technical_qa.get("errors") or []) if str(item)]
        if semantic_blocked:
            failures.append(f"semantic QA {semantic_qa.get('status', 'unknown')}")
        log_message = (
            "Native H3 technical and semantic media QA completed before publication."
            if passed
            else "Native H3 QA failed: " + ", ".join(failures)
        )
        if semantic_qa_required and require_human_review and semantic_qa.get("passed") is not True:
            log_message += " Semantic result is advisory because Discord human review remains authoritative."
        return SkillResult(
            status="success" if passed else "failed",
            outputs={
                "passed": passed,
                "video_path": video_path,
                "h3_mode": h3_mode,
                "story_quality": story_quality,
                "technical_qa": {**technical_qa, "bypassed": False, "failures": failures},
                "semantic_qa": semantic_qa,
                "contact_sheet_path": str(technical_qa.get("contact_sheet_path") or contact_sheet_path),
            },
            metrics={
                "video_count": len(saved_files),
                "technical_qa_passed": int(technical_passed),
                "semantic_qa_enabled": int(bool(semantic_qa.get("enabled"))),
                "semantic_qa_passed": int(semantic_qa.get("passed") is True),
            },
            logs=[log_message],
        )

    def package_native_h3(self, context: SkillContext) -> SkillResult:
        render = context.state[str(context.node.inputs.get("render_node") or "native-h3-render")]
        qa = context.state[str(context.node.inputs.get("qa_node") or "native-h3-qa")]
        preview = context.state[str(context.node.inputs.get("preview_node") or "native-h3-preview")]
        saved_files = [str(path) for path in render.get("saved_files", []) if path]
        video_path = str(qa.get("video_path") or (saved_files[0] if saved_files else ""))
        if not video_path:
            raise RuntimeError("Native H3 package has no final video path")
        return SkillResult(
            status="success",
            outputs={
                "video_path": video_path,
                "saved_files": saved_files,
                "gif_path": str(preview.get("gif_path") or ""),
                "contact_sheet_path": str(qa.get("contact_sheet_path") or ""),
                "qa": qa,
                "first_frame_path": str(render.get("first_frame_path") or ""),
                "last_frame_path": str(render.get("last_frame_path") or ""),
                "workflow_name": str(render.get("workflow_name") or ""),
                "story_source": str(render.get("story_source") or ""),
                "creative_seed": str(render.get("creative_seed") or ""),
                "news_context": dict(render.get("news_context") or {}),
                "story_quality": dict(render.get("story_quality") or qa.get("story_quality") or {}),
                "news_grounding": dict(render.get("news_grounding") or {}),
                "generated_storyboard": dict(render.get("generated_storyboard") or {}),
                "reference_manifest": list(render.get("reference_manifest") or []),
                "reference_lineage": list(render.get("reference_lineage") or []),
                "reference_audio_enabled": bool(render.get("reference_audio_enabled", False)),
                "render_mode": str(render.get("render_mode") or "image_to_video"),
                "native_h3": True,
            },
            metrics={"artifact_count": len(saved_files) + 2},
            logs=["Packaged native H3 video, GIF preview, and manual-review metadata."],
        )

    def generate_segment_tts(self, context: SkillContext) -> SkillResult:
        segment = self._segment(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt, f"{segment['segment_id']}_tts")
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        result = self.tools.call(
            "audio.generate_tts_real",
            {
                "text": segment["narration"],
                "output_path": str(audio_dir / f"{segment['segment_id']}.mp3"),
                "voice": context.node.inputs.get("voice", "en-US-AriaNeural"),
                "rate": context.node.inputs.get("rate", "+0%"),
            },
        )
        return SkillResult(status="success", outputs=result, logs=[f"Generated TTS for {segment['segment_id']}."])

    def package_outputs(self, context: SkillContext) -> SkillResult:
        final_node = str(context.node.inputs["final_video_node"])
        preview_node = str(context.node.inputs["preview_node"])
        segment_videos = []
        for dependency in context.node.depends_on:
            if dependency.startswith("segment-video-"):
                saved_files = context.state[dependency].get("saved_files", [])
                if saved_files:
                    segment_videos.append(str(saved_files[0]))

        return SkillResult(
            status="success",
            outputs={
                "video_path": str(context.state[final_node]["video_path"]),
                "preview_gif_path": str(context.state[preview_node]["gif_path"]),
                "segment_videos": segment_videos,
                "segment_count": len(segment_videos),
            },
            metrics={"segment_count": len(segment_videos)},
            logs=["Packaged final long-video artifacts."],
        )

    def _segment(self, context: SkillContext) -> dict[str, object]:
        segment_index = int(context.node.inputs["segment_index"])
        return context.state["script-plan"]["segments"][segment_index]

    def _build_run_dir(self, prompt: str, suffix: str) -> Path:
        return build_run_dir(self.output_root, prompt, suffix, default_slug="longvideo")

    @staticmethod
    def _optional_dependency_value(context: SkillContext, keys: tuple[str, ...]) -> str | None:
        return resolve_dependency_value(context, keys)


def register_longvideo_skills(
    skill_registry: SkillRegistry,
    tool_registry: ToolRegistry,
    output_root: Path,
    story_service: NativeH3StoryService | None = None,
    prompt_engine: PromptEngine | None = None,
) -> None:
    skills = LongVideoSkills(
        tool_registry,
        output_root,
        story_service=story_service,
        prompt_engine=prompt_engine,
    )
    skill_registry.register("idea.expand", skills.expand_idea, "Expand the goal into a creative brief")
    skill_registry.register("script.segment_story", skills.segment_story, "Create long-video story segments")
    skill_registry.register("asset.ensure_workflow", skills.ensure_workflow, "Verify workflow assets")
    skill_registry.register("longvideo.prepare_native_h3_story", skills.prepare_native_h3_story, "Prepare one causal native H3 storyboard prompt")
    skill_registry.register("longvideo.render_initial_frame", skills.render_initial_frame, "Render the opening seed frame")
    skill_registry.register("longvideo.render_segment_video", skills.render_segment_video, "Render one long-video segment")
    skill_registry.register("longvideo.render_native_h3", skills.render_native_h3, "Render one continuous native H3 story clip")
    skill_registry.register("longvideo.render_native_h3_t2v", skills.render_native_h3_t2v, "Render one continuous native H3 text-to-video story clip")
    skill_registry.register("longvideo.validate_native_h3_references", skills.validate_native_h3_references, "Validate H3 Ref2VA image/video references and provenance")
    skill_registry.register("longvideo.render_native_h3_l2va", skills.render_native_h3_l2va, "Render one continuous native H3 last-frame-to-video story clip")
    skill_registry.register("longvideo.render_native_h3_ref2va", skills.render_native_h3_ref2va, "Render one continuous native H3 reference-image/video story clip")
    skill_registry.register("longvideo.qa_native_h3", skills.qa_native_h3, "Run shared technical and semantic native H3 media QA")
    skill_registry.register("longvideo.package_native_h3", skills.package_native_h3, "Package native H3 artifacts and QA evidence")
    skill_registry.register("longvideo.generate_segment_tts", skills.generate_segment_tts, "Generate one narration segment")
    skill_registry.register("longvideo.package_outputs", skills.package_outputs, "Package long-video outputs")
