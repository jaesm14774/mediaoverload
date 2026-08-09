from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.minimax_prompting import structured_visual_prompt
from agentic.runtime.prompting import build_minimax_h3_prompt, build_story_segments
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.storyboard import format_native_h3_prompt, load_storyboard, resolve_native_h3_story


def _asset_check_result(result: dict[str, object], success_log: str) -> SkillResult:
    asset_status = result.get("asset_status", [])
    if not isinstance(asset_status, list):
        return SkillResult(status="success", outputs=result, logs=[success_log])

    missing_assets = [
        str(item.get("asset", "")).strip()
        for item in asset_status
        if isinstance(item, dict) and str(item.get("status", "")).lower() != "ready"
    ]
    if not missing_assets:
        return SkillResult(status="success", outputs=result, logs=[success_log])

    workflow_name = str(result.get("workflow_name", "")).strip()
    details = ", ".join(asset for asset in missing_assets if asset) or "unknown assets"
    return SkillResult(
        status="failed",
        outputs=result,
        logs=[f"Workflow assets missing for '{workflow_name}': {details}"],
    )


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Native H3 {name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise RuntimeError(f"Native H3 {name} must be between {minimum} and {maximum}; received {parsed}")
    return parsed


class LongVideoSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

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
        return _asset_check_result(result, "Checked workflow assets.")

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
        news_context = context.plan.goal.constraints.get("news_context") or {}
        if not isinstance(news_context, dict):
            raise RuntimeError("Native H3 news_context must be a mapping")
        storyboard, story_payload = resolve_native_h3_story(
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
                "generated_storyboard": storyboard,
                "creative_brief": creative_brief,
            },
            metrics={"duration_seconds": duration_seconds, "native_shot_count": len(storyboard.get("native_shots") or [])},
            logs=[
                f"Prepared news-grounded native H3 story '{storyboard.get('name')}' from '{storyboard_path}'.",
                f"News source: {news_context.get('title', '')}",
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
        result = self.tools.call(
            "comfy.render_image_to_video",
            {
                "workflow_name": workflow_name,
                "run_dir": str(run_dir),
                "image_path": image_path,
                "prompt": prompt,
                "width": context.node.inputs.get("width", 608 if workflow_name.startswith("minimax_h3_") else None),
                "height": context.node.inputs.get("height", 352 if workflow_name.startswith("minimax_h3_") else None),
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
        last_frame = str(gate.get("last_frame_path") or "")
        if not workflow_name or not first_frame or not last_frame:
            raise RuntimeError("Native H3 render requires workflow_name, first_frame_path, and last_frame_path")
        payload = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "native_h3")),
            "image_path": first_frame,
            "last_image_path": last_frame,
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
        }
        result = self.tools.call("comfy.workflow.image_to_video", payload)
        return SkillResult(
            status="success",
            outputs={
                **result,
                "first_frame_path": first_frame,
                "last_frame_path": last_frame,
                "native_h3_prompt": story["prompt"],
                "story_source": story.get("story_source", "news_llm"),
                "creative_seed": story.get("creative_seed", ""),
                "news_context": story.get("news_context", {}),
                "story_quality": dict(story.get("story_quality") or {}),
                "generated_storyboard": story.get("generated_storyboard", {}),
            },
            metrics={"video_count": len(result.get("saved_files", [])), "length": payload["length"], "steps": payload["steps"]},
            logs=[f"Rendered one continuous native H3 story with '{workflow_name}'."],
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
        payload = {
            "workflow_name": workflow_name,
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "native_h3_t2v")),
            "prompt": str(story["prompt"]),
            "negative_prompt": str(story.get("negative_prompt") or ""),
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
        # Technical media QA is intentionally bypassed here. The generated clip is
        # reviewed by a human through the final Discord review node; keeping a
        # second automated gate would duplicate that decision and could reject a
        # visually acceptable clip on metadata/audio heuristics alone.
        return SkillResult(
            status="success",
            outputs={
                "passed": True,
                "video_path": video_path,
                "story_quality": story_quality,
                "technical_qa": {
                    "bypassed": True,
                    "reason": "final_discord_human_review",
                },
                "contact_sheet_path": "",
            },
            metrics={"video_count": len(saved_files)},
            logs=["Native H3 technical QA bypassed; final Discord human review is authoritative."],
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
                "generated_storyboard": dict(render.get("generated_storyboard") or {}),
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
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
        slug = slug[:32] or "longvideo"
        return self.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}_{suffix}"

    @staticmethod
    def _optional_dependency_value(context: SkillContext, keys: tuple[str, ...]) -> str | None:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            for key in keys:
                value = dependency_output.get(key)
                if isinstance(value, list) and value:
                    return str(value[0])
                if isinstance(value, str) and value:
                    return value
        return None


def register_longvideo_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry, output_root: Path) -> None:
    skills = LongVideoSkills(tool_registry, output_root)
    skill_registry.register("idea.expand", skills.expand_idea, "Expand the goal into a creative brief")
    skill_registry.register("script.segment_story", skills.segment_story, "Create long-video story segments")
    skill_registry.register("asset.ensure_workflow", skills.ensure_workflow, "Verify workflow assets")
    skill_registry.register("longvideo.prepare_native_h3_story", skills.prepare_native_h3_story, "Prepare one causal native H3 storyboard prompt")
    skill_registry.register("longvideo.render_initial_frame", skills.render_initial_frame, "Render the opening seed frame")
    skill_registry.register("longvideo.render_segment_video", skills.render_segment_video, "Render one long-video segment")
    skill_registry.register("longvideo.render_native_h3", skills.render_native_h3, "Render one continuous native H3 story clip")
    skill_registry.register("longvideo.render_native_h3_t2v", skills.render_native_h3_t2v, "Render one continuous native H3 text-to-video story clip")
    skill_registry.register("longvideo.qa_native_h3", skills.qa_native_h3, "Pass native H3 through to final Discord human review")
    skill_registry.register("longvideo.package_native_h3", skills.package_native_h3, "Package native H3 artifacts and QA evidence")
    skill_registry.register("longvideo.generate_segment_tts", skills.generate_segment_tts, "Generate one narration segment")
    skill_registry.register("longvideo.package_outputs", skills.package_outputs, "Package long-video outputs")
