from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.prompt_engine import PromptEngine
from agentic.runtime.prompting import (
    build_segment_prompt,
)
from agentic.runtime.registry import SkillRegistry, ToolRegistry


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


class AgentPlanningSkills:
    def __init__(self, prompt_engine: PromptEngine | None = None) -> None:
        self.prompt_engine = prompt_engine or PromptEngine()

    def expand_goal(self, context: SkillContext) -> SkillResult:
        prompt = str(context.node.inputs["prompt"])
        style = str(context.node.inputs.get("style", context.plan.goal.style))
        idea_variants = list(context.node.inputs.get("idea_variants", []))
        selected_variant = idea_variants[0] if idea_variants else {"style": style}
        selected_style = str(selected_variant.get("style", style))
        outputs = self.prompt_engine.expand_goal(context.plan.goal, selected_style, idea_variants)
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
        segments = self.prompt_engine.segment_story(context.plan.goal, brief, segment_count, tone)
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
        review_direction = self._resolve_first_value(context, ("revised_prompt", "prompt"))
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
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            for key in candidate_keys:
                value = dependency_output.get(key)
                if isinstance(value, list) and value:
                    return str(value[0])
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _resolve_many_values(context: SkillContext, candidate_keys: tuple[str, ...]) -> list[str]:
        resolved: list[str] = []
        for dependency in context.node.depends_on:
            dependency_output = context.state[dependency]
            for key in candidate_keys:
                value = dependency_output.get(key)
                if isinstance(value, list):
                    resolved.extend(str(item) for item in value if item)
                elif isinstance(value, str) and value:
                    resolved.append(value)
        return list(dict.fromkeys(resolved))


class AgentMediaSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def ensure_workflow(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "asset.ensure_workflow_ready",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "auto_download": context.node.inputs.get("auto_download", False),
            },
        )
        return _asset_check_result(result, "Checked workflow assets for an agent step.")

    def refine_image(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "comfy.workflow.image_to_image",
            {
                "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "img2img")),
                "image_path": self._resolve_image_path(context),
                "prompt": self._resolve_prompt(context),
                "negative_prompt": self._resolve_negative_prompt(context),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result.get("saved_files", []))},
            logs=["Refined an input image with an agent media primitive."],
        )

    def generate_keyframe(self, context: SkillContext) -> SkillResult:
        prior_frame_path = context.node.inputs.get("prior_frame_path")
        if not prior_frame_path:
            for dependency in reversed(context.node.depends_on):
                dependency_output = context.state[dependency]
                candidate = dependency_output.get("prior_frame_path") or dependency_output.get("frame_path")
                if isinstance(candidate, str) and candidate:
                    prior_frame_path = candidate
                    break
        if isinstance(prior_frame_path, str) and prior_frame_path:
            result = self.tools.call(
                "comfy.workflow.image_to_image",
                {
                    "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "segment_keyframe")),
                    "image_path": prior_frame_path,
                    "prompt": self._resolve_prompt(context),
                    "negative_prompt": self._resolve_negative_prompt(context),
                },
            )
            log = "Generated a continuity keyframe from a prior frame."
        else:
            result = self.tools.call(
                "comfy.workflow.text_to_image",
                {
                    "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "segment_keyframe")),
                    "workflow_name": context.node.inputs["workflow_name"],
                    "prompt": self._resolve_prompt(context),
                    "negative_prompt": self._resolve_negative_prompt(context),
                    "width": int(context.node.inputs.get("width", 1024)),
                    "height": int(context.node.inputs.get("height", 1024)),
                    "image_count": int(context.node.inputs.get("image_count", 1)),
                },
            )
            log = "Generated an opening keyframe from text."
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result.get("saved_files", []))},
            logs=[log],
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
        result = self.tools.call(
            "comfy.workflow.image_to_video",
            {
                "workflow_name": str(context.node.inputs.get("workflow_name", "")),
                "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "i2v")),
                "image_path": self._resolve_image_path(context),
                "prompt": self._resolve_prompt(context),
                "video_count": int(context.node.inputs.get("video_count") or context.plan.goal.constraints.get("video_count") or 1),
            },
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
            label = str(prompt_set.get("label", f"item_{index:02d}"))
            item_dir = run_dir / label
            result = self.tools.call(
                "comfy.workflow.text_to_image",
                {
                    "workflow_name": workflow_name,
                    "prompt": str(prompt_set["prompt"]),
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
            "dependency_outputs": dependencies,
            "final_outputs": self._flatten_summary_outputs(dependencies),
            "review_summary": {
                "selected_assets": review_outputs.get("selected_assets", review_outputs.get("media_paths", [])),
                "rejected_assets": review_outputs.get("rejected_assets", []),
                "rejected_asset_details": review_outputs.get("rejected_asset_details", []),
                "selection_rationale": review_outputs.get("selection_rationale", ""),
                "failure_tags": review_outputs.get("failure_tags", []),
                "retry_direction": review_outputs.get("retry_direction", ""),
                "retry_intensity": review_outputs.get("retry_intensity", ""),
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
            logs=[f"Persisted workflow summary for {summary_scope}."],
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
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
        slug = slug[:32] or "agent"
        return self.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}_{suffix}"

    def _resolve_prompt(self, context: SkillContext) -> str:
        prompt = context.node.inputs.get("prompt")
        if isinstance(prompt, str) and prompt:
            return prompt
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            resolved = dependency_output.get("prompt")
            if isinstance(resolved, str) and resolved:
                return resolved
        return context.plan.goal.prompt

    def _resolve_negative_prompt(self, context: SkillContext) -> str:
        negative_prompt = context.node.inputs.get("negative_prompt")
        if isinstance(negative_prompt, str):
            return negative_prompt
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            resolved = dependency_output.get("negative_prompt")
            if isinstance(resolved, str):
                return resolved
        return ""

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
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            for key in candidate_keys:
                value = dependency_output.get(key)
                if isinstance(value, list) and value:
                    return str(value[0])
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _resolve_many(context: SkillContext, candidate_keys: tuple[str, ...]) -> list[str]:
        values: list[str] = []
        for dependency in context.node.depends_on:
            dependency_output = context.state[dependency]
            for key in candidate_keys:
                value = dependency_output.get(key)
                if isinstance(value, list):
                    values.extend(str(item) for item in value)
                    break
                if isinstance(value, str) and value:
                    values.append(value)
                    break
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
    media = AgentMediaSkills(tool_registry, output_root)
    skill_registry.register("agent.goal.expand", planning.expand_goal, "Expand a goal into an agent-ready brief")
    skill_registry.register("agent.prompt.compose", planning.compose_prompt, "Compose a reusable prompt bundle")
    skill_registry.register("agent.story.segment", planning.segment_story, "Break a story into reusable segments")
    skill_registry.register("agent.segment.prepare", planning.prepare_segment, "Prepare one segment prompt bundle")
    skill_registry.register("agent.review.refine_prompt", planning.refine_prompt_after_review, "Refine a prompt using review feedback")
    skill_registry.register("agent.sticker.expressions", planning.generate_sticker_expressions, "Generate sticker expression ideas")
    skill_registry.register("agent.sticker.prompt_set", planning.build_sticker_prompt_set, "Build sticker prompt sets")
    skill_registry.register("agent.sticker.motion_prompt", planning.build_sticker_motion_prompt, "Build an animated sticker motion prompt")
    skill_registry.register("agent.carousel.prompt_set", planning.build_slide_prompt_set, "Build carousel slide prompt sets")
    skill_registry.register("media.ensure_workflow", media.ensure_workflow, "Check workflow assets for any agent step")
    skill_registry.register("media.image.refine", media.refine_image, "Refine an image as an agent media primitive")
    skill_registry.register("media.image.generate_keyframe", media.generate_keyframe, "Generate a keyframe from text or a prior frame")
    skill_registry.register("media.image.upscale", media.upscale_image, "Upscale an image as an agent media primitive")
    skill_registry.register("media.image.animate", media.animate_image, "Animate an image as an agent media primitive")
    skill_registry.register("media.image.render_batch", media.render_image_batch, "Render a batch of images as an agent media primitive")
    skill_registry.register("media.audio.narrate", media.narrate_text, "Generate narration audio as an agent media primitive")
    skill_registry.register("media.audio.concat", media.concat_audio_tracks, "Concatenate audio tracks as an agent media primitive")
    skill_registry.register("media.video.concat", media.concat_videos, "Concatenate videos as an agent media primitive")
    skill_registry.register("media.video.merge_audio", media.merge_audio_video, "Mux audio and video as an agent media primitive")
    skill_registry.register("media.video.gif_preview", media.video_to_gif, "Create a GIF preview as an agent media primitive")
    skill_registry.register("media.video.extract_last_frame", media.extract_last_frame, "Extract the last frame as an agent media primitive")
    skill_registry.register("agent.sticker.package", media.package_sticker_outputs, "Package sticker artifacts for downstream agent use")
    skill_registry.register("agent.sticker.animate.package", media.package_animated_sticker_outputs, "Package animated sticker artifacts for downstream agent use")
    skill_registry.register("agent.carousel.package", media.package_carousel_outputs, "Package carousel artifacts for downstream agent use")
    skill_registry.register("agent.output.collect", media.collect_outputs, "Collect upstream artifacts for downstream agent steps")
    skill_registry.register("agent.summary.persist", media.persist_workflow_summary, "Persist a structured workflow summary artifact")
