from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest
from agentic.runtime.drama import DramaPlan, DramaPlanError, compile_drama_plan
from agentic.runtime.editing import EDIT_PROFILES, IMAGE_SUFFIXES, EditPlan
from agentic.assets.registry import AssetRegistry, WorkflowManifest
from agentic.runtime.creativity import IdeaDirector

PUBLISH_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def _infer_publish_review_scope(goal: GoalRequest) -> str:
    explicit_scope = str(goal.constraints.get("review_scope") or "").strip().lower()
    if explicit_scope:
        return explicit_scope
    media_paths = goal.constraints.get("media_paths") or []
    if isinstance(media_paths, str):
        media_paths = [media_paths]
    normalized_paths = [str(path) for path in media_paths if str(path).strip()]
    if not normalized_paths:
        input_dir = str(goal.constraints.get("input_dir") or "").strip()
        if input_dir:
            root = Path(input_dir)
            if root.is_dir():
                normalized_paths = [str(path) for path in root.iterdir() if path.is_file()]
    if normalized_paths and all(Path(path).suffix.lower() in PUBLISH_VIDEO_EXTENSIONS for path in normalized_paths):
        return "final_video"
    return "final_media"


class TaskPlanner:
    DEFAULT_IMAGE_WORKFLOWS = ("krea2_turbo",)
    DEFAULT_REFINE_WORKFLOWS = ("krea2_turbo_img2img",)
    DEFAULT_UPSCALE_WORKFLOWS = ("Tile Upscaler SDXL",)
    DEFAULT_T2V_WORKFLOWS = ("minimax_h3_lowvram_t2v", "minimax_h3_native_t2v")
    DEFAULT_I2V_WORKFLOWS = ("minimax_h3_lowvram_i2v",)

    def __init__(self, asset_registry: AssetRegistry, idea_director: IdeaDirector | None = None) -> None:
        self.asset_registry = asset_registry
        self.idea_director = idea_director or IdeaDirector()

    def _preferred_manifest(self, *workflow_names: str) -> WorkflowManifest:
        for workflow_name in workflow_names:
            try:
                return self.asset_registry.get_manifest(workflow_name)
            except KeyError:
                continue
        requested = ", ".join(workflow_names)
        raise KeyError(f"None of the preferred workflows are available (configs/workflow): {requested}")

    def _manifest_from_goal_constraints(
        self,
        goal: GoalRequest,
        *workflow_names: str,
        constraint_keys: tuple[str, ...] = ("workflow_name",),
        allowed_media_types: set[str] | None = None,
    ) -> WorkflowManifest:
        for key in constraint_keys:
            selected_name = str(goal.constraints.get(key) or "").strip()
            if not selected_name:
                continue
            try:
                manifest = self.asset_registry.get_manifest(selected_name)
            except KeyError:
                continue
            if allowed_media_types and not set(manifest.media_types).intersection(allowed_media_types):
                continue
            return manifest
        return self._preferred_manifest(*workflow_names)

    def _pick_goal_workflow(self, goal: GoalRequest) -> WorkflowManifest:
        selected_name = str(goal.constraints.get("workflow_name") or "").strip()
        if selected_name:
            try:
                manifest = self.asset_registry.get_manifest(selected_name)
            except KeyError:
                manifest = None
            if manifest and goal.media_type in manifest.media_types:
                return manifest
        return self.asset_registry.pick_workflow(goal.media_type)

    @staticmethod
    def _constraint_int(goal: GoalRequest, key: str, default: int) -> int:
        value = goal.constraints.get(key)
        if value in {None, ""}:
            return default
        return max(1, int(value))

    @staticmethod
    def _longvideo_int(goal: GoalRequest, key: str, default: int, maximum: int) -> int:
        value = goal.constraints.get(key)
        if value in {None, ""}:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer between 1 and {maximum}") from exc
        if parsed < 1 or parsed > maximum:
            raise ValueError(f"{key} must be an integer between 1 and {maximum}")
        return parsed

    @staticmethod
    def _native_h3_render_config(goal: GoalRequest, manifest: WorkflowManifest, *, default_steps: int = 16) -> dict[str, int]:
        defaults = dict(manifest.recommended_defaults or {})
        return {
            "width": int(goal.constraints.get("native_h3_width") or defaults.get("width", 608)),
            "height": int(goal.constraints.get("native_h3_height") or defaults.get("height", 352)),
            "length": int(goal.constraints.get("native_h3_length") or defaults.get("length", 362)),
            "steps": int(goal.constraints.get("native_h3_steps") or defaults.get("steps", default_steps)),
            "video_count": TaskPlanner._constraint_int(goal, "video_count", 1),
        }

    @staticmethod
    def _video_qa_inputs(goal: GoalRequest, manifest: WorkflowManifest) -> dict[str, object]:
        defaults = dict(manifest.recommended_defaults or {})
        fps = float(defaults.get("frame_rate") or goal.constraints.get("video_frame_rate") or 24)
        length = defaults.get("length")
        target_duration = goal.constraints.get("duration_override_seconds")
        if target_duration in {None, ""} and length not in {None, ""} and fps > 0:
            target_duration = float(length) / fps
        return {
            "expected_width": defaults.get("width"),
            "expected_height": defaults.get("height"),
            "expected_fps": fps,
            "target_duration": target_duration,
            "duration_tolerance": 0.6,
            "require_audio": manifest.name.startswith("minimax_h3_"),
            "require_stereo_audio": manifest.name.startswith("minimax_h3_"),
            "analyze_audio": manifest.name.startswith("minimax_h3_"),
            "frame_count": 6,
            "columns": 3,
            "scale_width": 480,
        }

    @staticmethod
    def _video_speed_config(goal: GoalRequest) -> dict[str, object]:
        raw = goal.constraints.get("video_speed")
        if raw is None:
            config: dict[str, object] = {}
        elif isinstance(raw, dict):
            config = dict(raw)
        else:
            raise ValueError("video_speed must be a mapping")
        enabled = bool(config.get("enabled", False))
        try:
            factor = float(config.get("factor", 2.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("video_speed.factor must be a number greater than zero") from exc
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("video_speed.factor must be a finite number greater than zero")
        return {"enabled": enabled, "factor": factor}

    @classmethod
    def _scaled_video_qa_inputs(
        cls,
        goal: GoalRequest,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        result = dict(inputs)
        speed = cls._video_speed_config(goal)
        target_duration = result.get("target_duration")
        if bool(speed["enabled"]):
            source_duration = target_duration
            if source_duration in {None, ""} and goal.duration_seconds > 0:
                source_duration = goal.duration_seconds
            if source_duration not in {None, ""}:
                result["target_duration"] = float(source_duration) / float(speed["factor"])
        return result

    def _append_video_speed_node(
        self,
        goal: GoalRequest,
        nodes: list[ExecutionNode],
        *,
        source_node: str,
        node_id: str,
        retry: bool = False,
    ) -> str:
        speed = self._video_speed_config(goal)
        if not bool(speed["enabled"]):
            return source_node
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                skill_name="media.video.change_speed",
                inputs={"speed": float(speed["factor"])},
                depends_on=[source_node],
                tags=["package", "video", "speed"] + (["retry"] if retry else []),
                tool_name="media.change_video_speed",
                stage="package",
            )
        )
        return node_id

    @staticmethod
    def _native_h3_storyboard_path(goal: GoalRequest, media_type: str) -> str:
        storyboard_path = str(
            goal.constraints.get("native_h3_storyboard_path")
            or goal.constraints.get("storyboard_path")
            or ""
        )
        if not storyboard_path:
            raise ValueError(f"{media_type} requires native_h3_storyboard_path or storyboard_path")
        return storyboard_path

    @staticmethod
    def _reference_video_source(goal: GoalRequest) -> str:
        return str(goal.constraints.get("reference_video_source") or "").strip()

    @classmethod
    def _reference_video_analysis_node(cls, goal: GoalRequest) -> ExecutionNode | None:
        source = cls._reference_video_source(goal)
        if not source:
            return None
        try:
            max_keyframes = int(goal.constraints.get("reference_video_max_keyframes") or 12)
        except (TypeError, ValueError) as exc:
            raise ValueError("reference_video_max_keyframes must be an integer between 2 and 20") from exc
        if not 2 <= max_keyframes <= 20:
            raise ValueError("reference_video_max_keyframes must be an integer between 2 and 20")
        depth = str(goal.constraints.get("reference_video_depth") or "standard").strip().lower()
        if depth not in {"standard", "deep"}:
            raise ValueError("reference_video_depth must be 'standard' or 'deep'")
        return ExecutionNode(
            node_id="reference-video-analysis",
            skill_name="reference.video.analyze",
            inputs={
                "source": source,
                "max_keyframes": max_keyframes,
                "analysis_depth": depth,
            },
            tags=["reference-video", "analysis", "evidence"],
            stage="analysis",
        )

    @classmethod
    def _reference_video_metadata(cls, goal: GoalRequest) -> dict[str, object] | None:
        source = cls._reference_video_source(goal)
        if not source:
            return None
        return {
            "source": source,
            "analysis_node": "reference-video-analysis",
            "analysis_depth": str(goal.constraints.get("reference_video_depth") or "standard"),
            "max_keyframes": int(goal.constraints.get("reference_video_max_keyframes") or 12),
            "policy": "borrow_timing_framing_motion_grammar_not_source_assets",
        }

    def _native_h3_finalize_nodes(
        self,
        goal: GoalRequest,
        *,
        tags: list[str],
        include_keyframe_gate: bool,
        frame_gate_node_id: str = "native-keyframe-gate",
        qa_inputs: dict[str, object] | None = None,
    ) -> list[ExecutionNode]:
        nodes: list[ExecutionNode] = []
        video_node = self._append_video_speed_node(
            goal,
            nodes,
            source_node="native-h3-render",
            node_id="native-h3-speed",
        )
        final_qa_inputs = dict(qa_inputs or {})
        if final_qa_inputs.get("expected_fps") in {None, ""}:
            final_qa_inputs["expected_fps"] = float(
                goal.constraints.get("native_h3_frame_rate") or 24
            )
        final_qa_inputs = self._scaled_video_qa_inputs(goal, final_qa_inputs)
        final_qa_inputs.update({"render_node": "native-h3-render", "video_node": video_node})
        qa_tags = ["technical-qa", "semantic-qa", "manual-review", *tags]
        preview_tags = ["preview", *tags]
        package_dependencies = ["native-h3-render", "native-h3-qa", "native-h3-preview"]
        if include_keyframe_gate:
            package_dependencies.append(frame_gate_node_id)
        nodes.extend(
            [
                ExecutionNode(
                    node_id="native-h3-qa",
                    skill_name="longvideo.qa_native_h3",
                    inputs={
                        "mode": "technical_and_semantic_qa_before_optional_discord_review",
                        **final_qa_inputs,
                    },
                    depends_on=[video_node],
                    tags=qa_tags,
                    stage="quality",
                ),
                ExecutionNode(
                    node_id="native-h3-preview",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 8, "scale_width": 512},
                    depends_on=[video_node],
                    tags=preview_tags,
                    tool_name="media.video_to_gif",
                    stage="package",
                ),
                ExecutionNode(
                    node_id="native-h3-package",
                    skill_name="longvideo.package_native_h3",
                    inputs={
                        "render_node": "native-h3-render",
                        "video_node": video_node,
                        "qa_node": "native-h3-qa",
                        "preview_node": "native-h3-preview",
                    },
                    depends_on=package_dependencies,
                    tags=["artifact", "summary", *tags],
                    stage="package",
                ),
            ]
        )
        return nodes

    def create_goal(
        self,
        prompt: str,
        media_type: str,
        duration_seconds: int,
        style: str,
        auto_download_assets: bool,
        constraints: dict | None = None,
    ) -> GoalRequest:
        return GoalRequest(
            prompt=prompt,
            media_type=media_type,
            duration_seconds=duration_seconds,
            style=style,
            auto_download_assets=auto_download_assets,
            constraints=constraints or {},
        )

    def build_plan(self, goal: GoalRequest) -> ExecutionPlan:
        if goal.media_type == "publish_review":
            return self._build_publish_review_plan(goal)
        if goal.media_type == "animated_sticker":
            return self._build_animated_sticker_plan(goal)
        if goal.media_type == "carousel":
            return self._build_carousel_plan(goal)
        if goal.media_type == "sticker_pack":
            return self._build_sticker_pack_plan(goal)
        if goal.media_type == "text2img2img":
            return self._build_text2img2img_plan(goal)
        if goal.media_type == "text2video":
            return self._build_text2video_plan(goal)
        if goal.media_type == "text2img2video":
            return self._build_text2img2video_plan(goal)
        if goal.media_type == "native_h3_story":
            return self._build_native_h3_story_plan(goal)
        if goal.media_type == "native_h3_t2v_story":
            return self._build_native_h3_t2v_story_plan(goal)
        if goal.media_type == "native_h3_fl2va_story":
            return self._build_native_h3_story_plan(goal)
        if goal.media_type == "native_h3_l2va_story":
            return self._build_native_h3_l2va_story_plan(goal)
        if goal.media_type == "native_h3_ref2va":
            return self._build_native_h3_ref2va_plan(
                goal,
                auto_reference_generation=bool(goal.constraints.get("auto_reference_generation", False)),
            )
        if goal.media_type == "text2image2native_h3_ref2va":
            return self._build_native_h3_ref2va_plan(goal, auto_reference_generation=True)
        if goal.media_type == "video_narrate":
            return self._build_video_narrate_plan(goal)
        if goal.media_type == "image_sequence_edit":
            return self._build_image_sequence_edit_plan(goal)
        workflow_manifest = self._pick_goal_workflow(goal)
        if goal.media_type == "storyboard":
            return self._build_storyboard_plan(goal, workflow_manifest)
        if goal.media_type == "image":
            return self._build_image_plan(goal, workflow_manifest)
        if goal.media_type in {"image_refine", "image_upscale", "image_to_video"}:
            return self._build_comfy_primitive_plan(goal, workflow_manifest)
        return self._build_long_video_plan(goal, workflow_manifest)


    def _build_native_h3_story_plan(self, goal: GoalRequest) -> ExecutionPlan:
        """Build one causal native H3 clip through the normal skill graph.

        The story prompt and both continuity anchors are generated as graph nodes,
        so scheduled runs can swap the storyboard or topic without introducing a
        second execution mechanism.
        """
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("native_h3_keyframe_workflow_name", "keyframe_workflow_name", "image_workflow_name"),
            allowed_media_types={"image"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=(
                "video_workflow_name",
                "native_h3_workflow_name",
                "workflow_name",
            ),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        render_config = self._native_h3_render_config(goal, video_manifest)
        lowvram_fl2va = (
            goal.media_type == "native_h3_fl2va_story"
            and video_manifest.name.startswith("minimax_h3_lowvram_")
        )
        width = int(goal.constraints.get("native_h3_fl2va_width") or (512 if lowvram_fl2va else render_config["width"]))
        height = int(goal.constraints.get("native_h3_fl2va_height") or (288 if lowvram_fl2va else render_config["height"]))
        length = int(goal.constraints.get("native_h3_fl2va_length") or (124 if lowvram_fl2va else render_config["length"]))
        steps = int(goal.constraints.get("native_h3_fl2va_steps") or (16 if lowvram_fl2va else render_config["steps"]))
        video_count = render_config["video_count"]
        model_profile = str(
            goal.constraints.get("native_h3_model_profile") or ("q2" if lowvram_fl2va else "q4")
        )
        pre_video_review = self._pre_video_review_enabled(goal)
        pre_video_requires_human = self._pre_video_review_requires_human(goal)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        keyframe_candidate_count = (
            self._pre_video_candidate_count(goal)
            if pre_video_review and (pre_video_requires_human or stage_probe_auto_select)
            else self._constraint_int(goal, "native_h3_keyframe_candidate_count", 1)
        )
        require_human_review = bool(goal.constraints.get("require_human_review", False)) or (
            pre_video_review and pre_video_requires_human
        )
        use_last_frame = bool(goal.constraints.get("native_h3_use_last_frame", False))
        if keyframe_candidate_count > 1 and not require_human_review and not stage_probe_auto_select:
            raise ValueError(
                "Native H3 multiple keyframe candidates require require_human_review=true; refusing to select one automatically."
            )
        if use_last_frame and not require_human_review and not stage_probe_auto_select and keyframe_candidate_count > 1:
            raise ValueError(
                "Native H3 use_last_frame=true with multiple ending candidates requires require_human_review=true; refusing to select one automatically."
            )
        storyboard_path = self._native_h3_storyboard_path(goal, "native_h3_story")
        reference_node = self._reference_video_analysis_node(goal)
        story_prompt_dependencies = [reference_node.node_id] if reference_node else []

        nodes = [
            *([reference_node] if reference_node else []),
            ExecutionNode(
                node_id="native-story-prompt",
                skill_name="longvideo.prepare_native_h3_story",
                inputs={
                    "storyboard_path": storyboard_path,
                    "duration_seconds": int(goal.duration_seconds),
                    "style": goal.style,
                    "render_mode": (
                        "first_last_frame_to_video"
                        if goal.media_type == "native_h3_fl2va_story"
                        else "image_to_video"
                    ),
                },
                depends_on=story_prompt_dependencies,
                tags=["creative", "story", "native-h3"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="native-image-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": image_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["native-story-prompt"],
                tags=["assets", "image", "native-h3"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="native-video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": video_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["native-story-prompt"],
                tags=["assets", "video", "native-h3"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="native-opening-keyframe",
                skill_name="media.image.generate_keyframe",
                inputs={
                    "workflow_name": image_manifest.name,
                    "prompt_key": "opening_keyframe_prompt",
                    "width": width,
                    "height": height,
                    "image_count": keyframe_candidate_count,
                    "suffix": "native_h3_opening",
                },
                depends_on=["native-story-prompt", "native-image-asset-check"],
                tags=["render", "image", "continuity", "native-h3"],
                tool_name="comfy.workflow.text_to_image",
                stage="render",
            ),
        ]
        opening_source_node = "native-opening-keyframe"
        if require_human_review or stage_probe_auto_select:
            opening_review_node = "native-opening-review"
            nodes.append(
                ExecutionNode(
                    node_id=opening_review_node,
                    skill_name="review.assets.select",
                    inputs={
                        "limit": 1,
                        "review_all_candidates": True,
                        "review_scope": "first_frame",
                        "review_phase": "opening_frame",
                        "require_human_review": pre_video_requires_human,
                        "auto_select_for_probe": stage_probe_auto_select,
                        "review_notes": "stage: preview",
                    },
                    depends_on=["native-opening-keyframe"],
                    tags=["review", "first-frame", "native-h3"],
                    stage="review",
                )
            )
            opening_source_node = opening_review_node
        ending_source_node = ""
        if use_last_frame:
            nodes.append(
                ExecutionNode(
                    node_id="native-ending-keyframe",
                    skill_name="media.image.generate_keyframe",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "prompt_key": "ending_keyframe_prompt",
                        "width": width,
                        "height": height,
                        "image_count": 1,
                        "suffix": "native_h3_ending",
                    },
                    depends_on=[opening_source_node, "native-story-prompt", "native-image-asset-check"],
                    tags=["render", "image", "continuity", "native-h3"],
                    tool_name="comfy.workflow.text_to_image",
                    stage="render",
                )
            )
            ending_source_node = "native-ending-keyframe"
            if require_human_review:
                nodes.append(
                    ExecutionNode(
                        node_id="native-ending-review",
                        skill_name="review.assets.select",
                        inputs={
                            "limit": 1,
                            "review_all_candidates": True,
                            "review_scope": "last_frame",
                            "review_phase": "ending_frame",
                            "review_notes": "stage: preview",
                        },
                        depends_on=["native-ending-keyframe"],
                        tags=["review", "last-frame", "native-h3"],
                        stage="review",
                    )
                )
                ending_source_node = "native-ending-review"

        gate_depends_on = [opening_source_node]
        if ending_source_node:
            gate_depends_on.append(ending_source_node)
        gate_inputs = {
            "character": str(goal.constraints.get("character") or ""),
            "opening_node": opening_source_node,
            "ending_node": ending_source_node,
            "opening_prompt_key": "opening_keyframe_prompt",
            "ending_prompt_key": "ending_keyframe_prompt",
            "preserve_opening_frame": require_human_review,
            "preserve_ending_frame": bool(use_last_frame and require_human_review),
            "use_last_frame": use_last_frame,
            "workflow_name": image_manifest.name,
            "width": width,
            "height": height,
            "max_regenerations": 0,
        }
        nodes.extend(
            [
                ExecutionNode(
                    node_id="native-keyframe-gate",
                    skill_name="media.image.validate_character",
                    inputs=gate_inputs,
                    depends_on=gate_depends_on,
                    tags=["quality", "identity", "native-h3"],
                    stage="quality",
                ),
                ExecutionNode(
                    node_id="native-h3-render",
                    skill_name="longvideo.render_native_h3",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "width": width,
                        "height": height,
                        "length": length,
                        "steps": steps,
                        "video_count": video_count,
                        "model_profile": model_profile,
                        "use_last_frame": use_last_frame,
                        "h3_mode": "fl2va" if goal.media_type == "native_h3_fl2va_story" else "i2va",
                    },
                    depends_on=["native-story-prompt", "native-video-asset-check", "native-keyframe-gate"],
                    tags=["render", "video", "native-h3"],
                    tool_name="comfy.workflow.image_to_video",
                    stage="render",
                ),
                *self._native_h3_finalize_nodes(
                    goal,
                    tags=["native-h3"],
                    include_keyframe_gate=True,
                    qa_inputs=(
                        {
                            "target_duration": length / float(goal.constraints.get("native_h3_frame_rate") or 24),
                            "expected_width": width,
                            "expected_height": height,
                            "expected_fps": float(goal.constraints.get("native_h3_frame_rate") or 24),
                        }
                        if lowvram_fl2va
                        else None
                    ),
                ),
            ]
        )
        metadata = {
            "recipe": "native_h3_fl2va_story" if goal.media_type == "native_h3_fl2va_story" else "native_h3_story",
            "storyboard_path": storyboard_path,
            "selected_workflow": video_manifest.name,
            "keyframe_workflow": image_manifest.name,
            "required_assets": [asset.to_dict() for asset in (*image_manifest.required_assets, *video_manifest.required_assets)],
            "native_h3": {"width": width, "height": height, "length": length, "steps": steps, "target_duration": int(goal.duration_seconds), "keyframe_candidate_count": keyframe_candidate_count, "require_human_review": require_human_review, "stage_probe_auto_select": stage_probe_auto_select, "use_last_frame": use_last_frame, "lowvram_preview": lowvram_fl2va},
            **({"reference_video": self._reference_video_metadata(goal)} if reference_node else {}),
            "graph_overview": [node.node_id for node in nodes],
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name=video_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=(
                f"Native H3 first/last-frame causal story from storyboard '{storyboard_path}'"
                if goal.media_type == "native_h3_fl2va_story"
                else f"Native H3 causal story from storyboard '{storyboard_path}'"
            ),
        )

    def _build_native_h3_t2v_story_plan(self, goal: GoalRequest) -> ExecutionPlan:
        """Build one continuous native H3 text-to-video story without keyframes."""
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            "minimax_h3_lowvram_t2v",
            "minimax_h3_native_t2v",
            constraint_keys=("video_workflow_name", "native_h3_workflow_name", "workflow_name"),
            allowed_media_types={"text2video", "long_video"},
        )
        render_config = self._native_h3_render_config(goal, video_manifest)
        lowvram_t2v = video_manifest.name.startswith("minimax_h3_lowvram_")
        width = int(
            goal.constraints.get("native_h3_t2v_width")
            or goal.constraints.get("native_h3_width")
            or (512 if lowvram_t2v else render_config["width"])
        )
        height = int(
            goal.constraints.get("native_h3_t2v_height")
            or goal.constraints.get("native_h3_height")
            or (288 if lowvram_t2v else render_config["height"])
        )
        length = int(
            goal.constraints.get("native_h3_t2v_length")
            or goal.constraints.get("native_h3_length")
            or (124 if lowvram_t2v else render_config["length"])
        )
        steps = int(
            goal.constraints.get("native_h3_t2v_steps")
            or goal.constraints.get("native_h3_steps")
            or (16 if lowvram_t2v else render_config["steps"])
        )
        video_count = render_config["video_count"]
        frame_rate = float(goal.constraints.get("native_h3_frame_rate") or 24)
        render_duration = round(length / frame_rate, 3)
        storyboard_path = self._native_h3_storyboard_path(goal, "native_h3_t2v_story")
        reference_node = self._reference_video_analysis_node(goal)
        story_prompt_dependencies = [reference_node.node_id] if reference_node else []

        nodes = [
            *([reference_node] if reference_node else []),
            ExecutionNode(
                node_id="native-story-prompt",
                skill_name="longvideo.prepare_native_h3_story",
                inputs={
                    "storyboard_path": storyboard_path,
                    "duration_seconds": int(goal.duration_seconds),
                    "style": goal.style,
                    "render_mode": "text_to_video",
                },
                depends_on=story_prompt_dependencies,
                tags=["creative", "story", "native-h3", "t2v"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="native-video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={
                    "workflow_name": video_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["native-story-prompt"],
                tags=["assets", "video", "native-h3", "t2v"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="native-h3-render",
                skill_name="longvideo.render_native_h3_t2v",
                inputs={
                    "workflow_name": video_manifest.name,
                    "width": width,
                    "height": height,
                    "length": length,
                    "steps": steps,
                    "video_count": video_count,
                    "model_profile": str(
                        goal.constraints.get("native_h3_t2v_model_profile")
                        or goal.constraints.get("native_h3_model_profile")
                        or ("q2" if lowvram_t2v else "q4")
                    ),
                },
                depends_on=["native-story-prompt", "native-video-asset-check"],
                tags=["render", "video", "native-h3", "t2v"],
                tool_name="comfy.workflow.text_to_video",
                stage="render",
            ),
            *self._native_h3_finalize_nodes(
                goal,
                tags=["native-h3", "t2v"],
                include_keyframe_gate=False,
                qa_inputs={
                    "target_duration": render_duration,
                    "expected_width": width,
                    "expected_height": height,
                    "expected_fps": frame_rate,
                },
            ),
        ]
        metadata = {
            "recipe": "native_h3_t2v_story",
            "storyboard_path": storyboard_path,
            "selected_workflow": video_manifest.name,
            "required_assets": [asset.to_dict() for asset in video_manifest.required_assets],
            "native_h3": {
                "width": width,
                "height": height,
                "length": length,
                "steps": steps,
                "target_duration": render_duration,
                "requested_duration": int(goal.duration_seconds),
                "lowvram_preview": lowvram_t2v,
            },
            "render_mode": "text_to_video",
            **({"reference_video": self._reference_video_metadata(goal)} if reference_node else {}),
            "graph_overview": [node.node_id for node in nodes],
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name=video_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=f"Native H3 text-to-video causal story from storyboard '{storyboard_path}'",
        )

    def _build_native_h3_l2va_story_plan(self, goal: GoalRequest) -> ExecutionPlan:
        """Build a last-frame-only H3 causal clip.

        L2VA is deliberately a separate route so an approved ending frame can
        be treated as the immutable conditioning source rather than being
        accidentally paired with the opening-frame connection in the FL2VA
        template.
        """
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("native_h3_keyframe_workflow_name", "keyframe_workflow_name", "image_workflow_name"),
            allowed_media_types={"image"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            "minimax_h3_lowvram_15s_fl2va_i2v",
            constraint_keys=("video_workflow_name", "native_h3_workflow_name", "workflow_name"),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        render_config = self._native_h3_render_config(goal, video_manifest)
        lowvram_l2va = video_manifest.name.startswith("minimax_h3_lowvram_")
        width = int(goal.constraints.get("native_h3_l2va_width") or (512 if lowvram_l2va else render_config["width"]))
        height = int(goal.constraints.get("native_h3_l2va_height") or (288 if lowvram_l2va else render_config["height"]))
        length = int(goal.constraints.get("native_h3_l2va_length") or (124 if lowvram_l2va else render_config["length"]))
        steps = int(goal.constraints.get("native_h3_l2va_steps") or (16 if lowvram_l2va else render_config["steps"]))
        video_count = render_config["video_count"]
        model_profile = str(
            goal.constraints.get("native_h3_model_profile") or ("q2" if lowvram_l2va else "q4")
        )
        storyboard_path = self._native_h3_storyboard_path(goal, "native_h3_l2va_story")
        reference_node = self._reference_video_analysis_node(goal)
        story_prompt_dependencies = [reference_node.node_id] if reference_node else []
        pre_video_review = self._pre_video_review_enabled(goal)
        pre_video_requires_human = self._pre_video_review_requires_human(goal)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        require_human_review = bool(goal.constraints.get("require_human_review", True)) or (
            pre_video_review and pre_video_requires_human
        )
        if stage_probe_auto_select:
            require_human_review = False
        candidate_count = (
            self._pre_video_candidate_count(goal)
            if pre_video_review and (pre_video_requires_human or stage_probe_auto_select)
            else 1
        )
        ending_source_node = "native-l2va-ending-keyframe"
        nodes = [
            *([reference_node] if reference_node else []),
            ExecutionNode(
                node_id="native-story-prompt",
                skill_name="longvideo.prepare_native_h3_story",
                inputs={"storyboard_path": storyboard_path, "duration_seconds": int(goal.duration_seconds), "style": goal.style, "render_mode": "last_frame_to_video"},
                depends_on=story_prompt_dependencies,
                tags=["creative", "story", "native-h3", "l2va"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="native-image-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["native-story-prompt"],
                tags=["assets", "image", "native-h3", "l2va"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="native-video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["native-story-prompt"],
                tags=["assets", "video", "native-h3", "l2va"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id=ending_source_node,
                skill_name="media.image.generate_keyframe",
                inputs={"workflow_name": image_manifest.name, "prompt_key": "ending_keyframe_prompt", "width": width, "height": height, "image_count": candidate_count, "suffix": "native_h3_l2va_last"},
                depends_on=["native-story-prompt", "native-image-asset-check"],
                tags=["render", "image", "last-frame", "native-h3", "l2va"],
                tool_name="comfy.workflow.text_to_image",
                stage="render",
            ),
        ]
        frame_node = ending_source_node
        if require_human_review or stage_probe_auto_select:
            nodes.append(
                ExecutionNode(
                    node_id="native-l2va-ending-review",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": 1,
                        "review_all_candidates": True,
                        "review_scope": "last_frame",
                        "require_human_review": pre_video_requires_human,
                        "auto_select_for_probe": stage_probe_auto_select,
                        "review_notes": "Select the final conditioning frame. This frame is immutable after approval; reject identity drift, wrong pose, or unusable composition.",
                    },
                    depends_on=[ending_source_node],
                    tags=["review", "last-frame", "native-h3", "l2va"],
                    stage="review",
                )
            )
            frame_node = "native-l2va-ending-review"
        nodes.extend(
            [
                ExecutionNode(
                    node_id="native-l2va-frame-gate",
                    skill_name="media.image.validate_last_frame",
                    inputs={
                        "frame_node": frame_node,
                        "character": str(goal.constraints.get("character") or ""),
                        "preserve_last_frame": bool(require_human_review or stage_probe_auto_select),
                        "max_regenerations": 0,
                    },
                    depends_on=[frame_node],
                    tags=["quality", "identity", "last-frame", "native-h3", "l2va"],
                    stage="quality",
                ),
                ExecutionNode(
                    node_id="native-h3-render",
                    skill_name="longvideo.render_native_h3_l2va",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "width": width,
                        "height": height,
                        "length": length,
                        "steps": steps,
                        "video_count": video_count,
                        "model_profile": model_profile,
                    },
                    depends_on=["native-story-prompt", "native-video-asset-check", "native-l2va-frame-gate"],
                    tags=["render", "video", "native-h3", "l2va"],
                    tool_name="comfy.workflow.image_to_video",
                    stage="render",
                ),
                *self._native_h3_finalize_nodes(
                    goal,
                    tags=["native-h3", "l2va"],
                    include_keyframe_gate=True,
                    frame_gate_node_id="native-l2va-frame-gate",
                    qa_inputs=(
                        {
                            "target_duration": length / float(goal.constraints.get("native_h3_frame_rate") or 24),
                            "expected_width": width,
                            "expected_height": height,
                            "expected_fps": float(goal.constraints.get("native_h3_frame_rate") or 24),
                        }
                        if lowvram_l2va
                        else None
                    ),
                ),
            ]
        )
        metadata = {
            "recipe": "native_h3_l2va_story",
            "storyboard_path": storyboard_path,
            "selected_workflow": video_manifest.name,
            "keyframe_workflow": image_manifest.name,
            "required_assets": [asset.to_dict() for asset in (*image_manifest.required_assets, *video_manifest.required_assets)],
            "native_h3": {"width": width, "height": height, "length": length, "steps": steps, "target_duration": int(goal.duration_seconds), "require_human_review": require_human_review, "stage_probe_auto_select": stage_probe_auto_select, "lowvram_preview": lowvram_l2va},
            "render_mode": "last_frame_to_video",
            **({"reference_video": self._reference_video_metadata(goal)} if reference_node else {}),
            "graph_overview": [node.node_id for node in nodes],
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name=video_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=f"Native H3 L2VA story from storyboard '{storyboard_path}'",
        )

    def _build_native_h3_ref2va_plan(
        self,
        goal: GoalRequest,
        *,
        auto_reference_generation: bool = False,
    ) -> ExecutionPlan:
        """Build the multi-reference H3 route without audio refs.

        ``native_h3_ref2va`` uses a validated manifest directly when one is
        configured.  With an empty manifest it uses the same six-candidate
        T2I plus Discord selection gate as the composed
        ``text2image2native_h3_ref2va`` route.  The composed route keeps the
        candidate stage explicit in its strategy name.
        """
        image_manifest = None
        if auto_reference_generation:
            image_manifest = self._manifest_from_goal_constraints(
                goal,
                *self.DEFAULT_IMAGE_WORKFLOWS,
                constraint_keys=(
                    "native_h3_reference_workflow_name",
                    "keyframe_workflow_name",
                    "image_workflow_name",
                ),
                allowed_media_types={"image"},
            )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            "minimax_h3_ref2va",
            constraint_keys=("video_workflow_name", "native_h3_ref2va_workflow_name", "workflow_name"),
            allowed_media_types={"native_h3_ref2va", "long_video"},
        )
        render_config = self._native_h3_render_config(goal, video_manifest, default_steps=20)
        width, height = render_config["width"], render_config["height"]
        length, steps, video_count = render_config["length"], render_config["steps"], render_config["video_count"]
        storyboard_path = self._native_h3_storyboard_path(goal, "native_h3_ref2va")
        reference_manifest = list(goal.constraints.get("native_h3_reference_manifest") or goal.constraints.get("reference_manifest") or [])
        reference_image_paths = list(goal.constraints.get("native_h3_reference_image_paths") or [])
        reference_video_paths = list(goal.constraints.get("native_h3_reference_video_paths") or [])
        reference_candidates = list(
            goal.constraints.get("native_h3_reference_candidates")
            or goal.constraints.get("media_paths")
            or []
        )
        raw_selection_limit = int(goal.constraints.get("native_h3_reference_selection_limit") or 4)
        reference_selection_limit = max(
            1,
            min(4 if auto_reference_generation else 12, raw_selection_limit),
        )
        raw_candidate_count = int(
            (
                goal.constraints.get("image_count")
                if auto_reference_generation
                else goal.constraints.get("native_h3_reference_candidate_count")
            )
            or goal.constraints.get("native_h3_reference_candidate_count")
            or 4
        )
        reference_candidate_count = max(
            4 if auto_reference_generation else 1,
            min(6 if auto_reference_generation else 6, raw_candidate_count),
        )
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        if "require_human_review" in goal.constraints:
            require_human_review = bool(goal.constraints.get("require_human_review"))
        else:
            # Generated references must never silently become Ref2VA inputs.
            # Only an explicit --no-review path may disable this gate.
            require_human_review = auto_reference_generation or bool(reference_candidates)
        if stage_probe_auto_select:
            require_human_review = False
        nodes = [
            ExecutionNode(
                node_id="native-story-prompt",
                skill_name="longvideo.prepare_native_h3_story",
                inputs={"storyboard_path": storyboard_path, "duration_seconds": int(goal.duration_seconds), "style": goal.style, "render_mode": "reference_to_video"},
                tags=["creative", "story", "native-h3", "ref2va"],
                stage="prompting",
            ),
        ]
        if auto_reference_generation:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="native-image-asset-check",
                        skill_name="media.ensure_workflow",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "auto_download": goal.auto_download_assets,
                        },
                        depends_on=["native-story-prompt"],
                        tags=["assets", "image", "native-h3", "ref2va"],
                        tool_name="asset.ensure_workflow_ready",
                        stage="assets",
                    ),
                    ExecutionNode(
                        node_id="native-video-asset-check",
                        skill_name="media.ensure_workflow",
                        inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                        depends_on=["native-story-prompt"],
                        tags=["assets", "video", "native-h3", "ref2va"],
                        tool_name="asset.ensure_workflow_ready",
                        stage="assets",
                    ),
                    ExecutionNode(
                        node_id="native-ref2va-reference-candidates",
                        skill_name="media.image.generate_keyframe",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "prompt_key": "opening_keyframe_prompt",
                            "width": width,
                            "height": height,
                            "image_count": reference_candidate_count,
                            "suffix": "native_h3_ref2va_reference",
                            "max_regenerations": 0,
                        },
                        depends_on=["native-story-prompt", "native-image-asset-check"],
                        tags=["render", "image", "references", "native-h3", "ref2va"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    ),
                ]
            )
        else:
            nodes.append(
                ExecutionNode(
                    node_id="native-video-asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                    depends_on=["native-story-prompt"],
                    tags=["assets", "video", "native-h3", "ref2va"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                )
            )
        reference_check_dependencies = ["native-story-prompt"]
        if auto_reference_generation:
            candidate_node_id = "native-ref2va-reference-candidates"
            if require_human_review or stage_probe_auto_select:
                nodes.append(
                    ExecutionNode(
                        node_id="native-ref2va-reference-review",
                        skill_name="review.assets.select",
                        inputs={
                            "limit": reference_selection_limit,
                            "review_all_candidates": True,
                            "review_scope": "reference",
                            "review_phase": "reference_selection",
                            "require_human_review": require_human_review,
                            "auto_select_for_probe": stage_probe_auto_select,
                            "review_notes": str(goal.constraints.get("review_notes") or ""),
                        },
                        depends_on=[candidate_node_id],
                        tags=["review", "references", "native-h3", "ref2va"],
                        stage="review",
                    )
                )
                reference_check_dependencies.append("native-ref2va-reference-review")
            else:
                reference_check_dependencies.append(candidate_node_id)
        elif reference_candidates:
            candidate_node_id = "native-ref2va-reference-candidates"
            nodes.append(
                ExecutionNode(
                    node_id=candidate_node_id,
                    skill_name="publish.media.ingest",
                    inputs={"media_paths": reference_candidates},
                    depends_on=["native-story-prompt"],
                    tags=["assets", "references", "native-h3", "ref2va"],
                    stage="assets",
                )
            )
            if require_human_review or stage_probe_auto_select:
                nodes.append(
                    ExecutionNode(
                        node_id="native-ref2va-reference-review",
                        skill_name="review.assets.select",
                        inputs={
                            "limit": reference_selection_limit,
                            "review_all_candidates": True,
                            "review_scope": "reference",
                            "review_phase": "reference_selection",
                            "require_human_review": require_human_review,
                            "auto_select_for_probe": stage_probe_auto_select,
                            "review_notes": str(goal.constraints.get("review_notes") or ""),
                        },
                        depends_on=[candidate_node_id],
                        tags=["review", "references", "native-h3", "ref2va"],
                        stage="review",
                    )
                )
                reference_check_dependencies.append("native-ref2va-reference-review")
            else:
                reference_check_dependencies.append(candidate_node_id)
        nodes.extend(
            [
                ExecutionNode(
                    node_id="native-ref2va-reference-check",
                    skill_name="longvideo.validate_native_h3_references",
                    inputs={
                        "reference_manifest": reference_manifest,
                        "reference_image_paths": reference_image_paths,
                        "reference_video_paths": reference_video_paths,
                        "max_images": int(goal.constraints.get("native_h3_reference_max_images") or 9),
                        "max_videos": int(goal.constraints.get("native_h3_reference_max_videos") or 3),
                        "selection_limit": reference_selection_limit,
                        "auto_reference_generation": auto_reference_generation,
                    },
                    depends_on=reference_check_dependencies,
                    tags=["quality", "references", "native-h3", "ref2va"],
                    stage="quality",
                ),
                ExecutionNode(
                    node_id="native-h3-render",
                    skill_name="longvideo.render_native_h3_ref2va",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "width": width,
                        "height": height,
                        "length": length,
                        "steps": steps,
                        "video_count": video_count,
                        "ref_image_size": str(goal.constraints.get("native_h3_reference_image_size") or "match"),
                        "model_profile": str(goal.constraints.get("native_h3_model_profile") or "q4"),
                    },
                    depends_on=["native-story-prompt", "native-video-asset-check", "native-ref2va-reference-check"],
                    tags=["render", "video", "native-h3", "ref2va"],
                    tool_name="comfy.workflow.reference_to_video",
                    stage="render",
                ),
            ]
        )
        nodes.extend(
            self._native_h3_finalize_nodes(
                goal,
                tags=["native-h3", "ref2va"],
                include_keyframe_gate=False,
                qa_inputs={"frame_count": 12, "columns": 4},
            )
        )
        required_assets = [asset.to_dict() for asset in video_manifest.required_assets]
        if image_manifest is not None:
            required_assets = [asset.to_dict() for asset in image_manifest.required_assets] + required_assets
        metadata = {
            "recipe": "text2image2native_h3_ref2va" if auto_reference_generation else "native_h3_ref2va",
            "storyboard_path": storyboard_path,
            "selected_workflow": video_manifest.name,
            "selected_workflows": {
                "image": image_manifest.name if image_manifest is not None else "",
                "video": video_manifest.name,
            },
            "required_assets": required_assets,
            "image_required_assets": (
                [asset.to_dict() for asset in image_manifest.required_assets]
                if image_manifest is not None
                else []
            ),
            "video_required_assets": [asset.to_dict() for asset in video_manifest.required_assets],
            "reference_manifest": reference_manifest,
            "reference_candidates": reference_candidates,
            "reference_candidate_count": reference_candidate_count if auto_reference_generation else 0,
            "reference_selection_limit": reference_selection_limit,
            "reference_audio_enabled": False,
            "native_h3": {"width": width, "height": height, "length": length, "steps": steps, "target_duration": int(goal.duration_seconds), "reference_image_size": str(goal.constraints.get("native_h3_reference_image_size") or "match"), "stage_probe_auto_select": stage_probe_auto_select},
            "render_mode": "reference_to_video",
            "graph_overview": [node.node_id for node in nodes],
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name=video_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=(
                f"Native H3 T2I-to-Ref2VA story from storyboard '{storyboard_path}' with reviewed generated references"
                if auto_reference_generation
                else f"Native H3 Ref2VA story from storyboard '{storyboard_path}' with image/video references only"
            ),
        )

    @staticmethod
    def _review_loop_enabled(goal: GoalRequest) -> bool:
        if bool(goal.constraints.get("enable_review_loop", False)):
            return True
        review_notes = str(goal.constraints.get("review_notes", "") or "").strip()
        return bool(review_notes)

    @staticmethod
    def _review_selection_limit(goal: GoalRequest, default: int = 3) -> int:
        return int(goal.constraints.get("selection_limit") or goal.constraints.get("review_selection_limit") or default)

    @staticmethod
    def _stage_review_enabled(goal: GoalRequest) -> bool:
        return bool(goal.constraints.get("enable_stage_review", False))

    @staticmethod
    def _pre_video_review_enabled(goal: GoalRequest) -> bool:
        # Pure T2V routes intentionally keep their prompt-only conditioning.
        # A caller may still request an explicit stage review through
        # ``enable_stage_review``; the shared automatic image gate must not
        # silently change a T2V strategy into I2V.
        if goal.media_type in {"text2video", "native_h3_t2v_story"}:
            return False
        return bool(goal.constraints.get("pre_video_review_enabled", False))

    @staticmethod
    def _pre_video_review_requires_human(goal: GoalRequest) -> bool:
        return bool(
            goal.constraints.get(
                "pre_video_review_require_human",
                TaskPlanner._pre_video_review_enabled(goal),
            )
        )

    @staticmethod
    def _pre_video_candidate_count(goal: GoalRequest, default: int = 6) -> int:
        value = goal.constraints.get("pre_video_candidate_count")
        return max(1, int(value or default))

    def _build_long_video_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        segment_count = self._constraint_int(goal, "segment_count", max(2, (goal.duration_seconds + 4) // 5))
        use_tts = bool(goal.constraints.get("use_tts", False))
        review_loop_enabled = self._review_loop_enabled(goal)
        pre_video_review = self._pre_video_review_enabled(goal)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 3))
        production_profile = str(
            goal.constraints.get("longvideo_production_profile") or "text2longvideo"
        ).strip().lower()
        if production_profile != "text2longvideo":
            raise ValueError("longvideo_production_profile must be 'text2longvideo'")
        review_policy = str(
            goal.constraints.get("longvideo_review_policy")
            or goal.constraints.get("longvideo_frame_review_policy")
            or "opening_only"
        ).strip().lower()
        if review_policy not in {"opening_only", "anchors", "every_segment"}:
            raise ValueError("longvideo_review_policy must be 'opening_only', 'anchors', or 'every_segment'")
        continuity_mode = str(goal.constraints.get("longvideo_continuity_mode") or "rendered_tail").strip().lower()
        if continuity_mode != "rendered_tail":
            raise ValueError("longvideo_continuity_mode must be 'rendered_tail'")

        from agentic.runtime.video_conditioning import (
            ConditioningPlan,
            capabilities_from_manifests,
            production_recipe_sequence,
            recipe_candidates,
        )

        capabilities = capabilities_from_manifests(self.asset_registry.all_manifests())
        preferred_workflows = goal.constraints.get("longvideo_workflow_names") or []
        if isinstance(preferred_workflows, str):
            preferred_workflows = [preferred_workflows]
        candidates = recipe_candidates(capabilities, preferred_workflows=preferred_workflows)
        variant_seed = int(goal.constraints.get("edit_variant_seed") or 17)
        recipe_sequence = production_recipe_sequence(
            segment_count,
            candidates,
            use_reference_bundle=bool(
                goal.constraints.get("reference_manifest")
                or goal.constraints.get("native_h3_reference_manifest")
                or goal.constraints.get("reference_image_paths")
                or goal.constraints.get("native_h3_reference_image_paths")
                or goal.constraints.get("reference_video_paths")
                or goal.constraints.get("native_h3_reference_video_paths")
            ),
        )
        selected_capabilities = {
            recipe_name: candidates[recipe_name][0]
            for recipe_name in dict.fromkeys(recipe_sequence)
        }
        selected_workflows = {
            capability.workflow_name
            for capability in selected_capabilities.values()
        }
        recipe_contracts = {
            recipe_name: selected_capabilities[recipe_name].recipes[recipe_name]
            for recipe_name in selected_capabilities
        }
        requires_image_conditioning = any(
            recipe.requires_first or recipe.requires_last or recipe.requires_references
            for recipe in recipe_contracts.values()
        )
        image_manifest = (
            self._manifest_from_goal_constraints(
                goal,
                *self.DEFAULT_IMAGE_WORKFLOWS,
                constraint_keys=("image_workflow_name", "workflow_name"),
                allowed_media_types={"image"},
            )
            if requires_image_conditioning
            else None
        )
        transition_manifest = (
            self._manifest_from_goal_constraints(
                goal,
                *self.DEFAULT_REFINE_WORKFLOWS,
                constraint_keys=("transition_workflow_name", "refine_workflow_name"),
                allowed_media_types={"image_refine"},
            )
            if requires_image_conditioning
            else None
        )

        image_defaults = image_manifest.recommended_defaults if image_manifest else {}
        frame_width = int(goal.constraints.get("longvideo_frame_width") or image_defaults.get("width", 1024))
        frame_height = int(goal.constraints.get("longvideo_frame_height") or image_defaults.get("height", 1024))
        frame_candidate_count = max(
            1,
            int(
                goal.constraints.get("longvideo_frame_candidate_count")
                or goal.constraints.get("pre_video_candidate_count")
                or (4 if review_policy != "opening_only" else 1)
            ),
        )
        reference_candidate_count = max(
            1,
            int(
                goal.constraints.get("longvideo_reference_candidate_count")
                or goal.constraints.get("native_h3_reference_candidate_count")
                or 4
            ),
        )
        require_human_review = bool(
            goal.constraints.get("require_human_review", False)
            or goal.constraints.get("enable_stage_review", False)
            or pre_video_review
        )
        if stage_probe_auto_select:
            require_human_review = False
        review_all_anchors = review_policy in {"anchors", "every_segment"}
        human_anchor_review = require_human_review and review_all_anchors
        opening_human_review = require_human_review and (
            pre_video_review
            or bool(goal.constraints.get("enable_stage_review", False))
            or review_policy == "opening_only"
        )

        configured_reference_manifest = list(
            goal.constraints.get("reference_manifest")
            or goal.constraints.get("native_h3_reference_manifest")
            or []
        )
        configured_reference_images = list(
            goal.constraints.get("reference_image_paths")
            or goal.constraints.get("native_h3_reference_image_paths")
            or []
        )
        configured_reference_videos = list(
            goal.constraints.get("reference_video_paths")
            or goal.constraints.get("native_h3_reference_video_paths")
            or []
        )
        has_configured_references = bool(
            configured_reference_manifest or configured_reference_images or configured_reference_videos
        )
        max_reference_images = int(
            goal.constraints.get("longvideo_reference_max_images")
            or goal.constraints.get("native_h3_reference_max_images")
            or 9
        )
        max_reference_videos = int(
            goal.constraints.get("longvideo_reference_max_videos")
            or goal.constraints.get("native_h3_reference_max_videos")
            or 3
        )
        reference_selection_limit = max(
            1,
            int(
                goal.constraints.get("longvideo_reference_selection_limit")
                or goal.constraints.get("native_h3_reference_selection_limit")
                or 4
            ),
        )
        segment_frame_rate = float(
            goal.constraints.get("longvideo_frame_rate")
            or goal.constraints.get("video_frame_rate")
            or 24
        )
        segment_default_length = max(
            1,
            round((float(goal.duration_seconds) / max(1, segment_count)) * segment_frame_rate),
        )

        idea_variants = self.idea_director.generate_variations(goal)
        reference_node = self._reference_video_analysis_node(goal)
        planning_dependencies = [reference_node.node_id] if reference_node else []
        nodes: list[ExecutionNode] = [
            *([reference_node] if reference_node else []),
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={"prompt": goal.prompt, "style": goal.style, "idea_variants": idea_variants},
                depends_on=planning_dependencies,
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="script-plan",
                skill_name="agent.story.segment",
                inputs={
                    "segment_count": segment_count,
                    "duration_seconds": goal.duration_seconds,
                    "tone": "playful cinematic escalation",
                    "production_profile": production_profile,
                },
                depends_on=["idea-brief"],
                tags=["story"],
            ),
        ]
        if requires_image_conditioning:
            assert image_manifest is not None
            assert transition_manifest is not None
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="image-asset-check",
                        skill_name="media.ensure_workflow",
                        inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                        depends_on=["script-plan"],
                        tags=["assets", "image"],
                        tool_name="asset.ensure_workflow_ready",
                        stage="assets",
                    ),
                    ExecutionNode(
                        node_id="transition-asset-check",
                        skill_name="media.ensure_workflow",
                        inputs={"workflow_name": transition_manifest.name, "auto_download": goal.auto_download_assets},
                        depends_on=["script-plan"],
                        tags=["assets", "transition"],
                        tool_name="asset.ensure_workflow_ready",
                        stage="assets",
                    ),
                ]
            )

        def video_asset_check_for(workflow_name: str) -> str:
            node_id = f"video-asset-check-{workflow_name}"
            if not any(node.node_id == node_id for node in nodes):
                manifest = self.asset_registry.get_manifest(workflow_name)
                nodes.append(
                    ExecutionNode(
                        node_id=node_id,
                        skill_name="media.ensure_workflow",
                        inputs={"workflow_name": workflow_name, "auto_download": goal.auto_download_assets},
                        depends_on=["script-plan"],
                        tags=["assets", "video", "longvideo"],
                        tool_name="asset.ensure_workflow_ready",
                        stage="assets",
                    )
                )
            return node_id

        for workflow_name in sorted(selected_workflows):
            video_asset_check_for(workflow_name)

        segment_video_nodes: list[str] = []
        tts_nodes: list[str] = []

        def append_segment(
            prefix: str,
            index: int,
            previous_rendered_tail_node: str | None,
            *,
            retry: bool = False,
        ) -> tuple[str, str, str | None, str | None, str | None]:
            suffix = f"{index + 1:02d}"
            prompt_node = f"{prefix}-prompt-{suffix}"
            video_node = f"{prefix}-video-{suffix}"
            recipe_name = recipe_sequence[index]
            capability = selected_capabilities[recipe_name]
            recipe = recipe_contracts[recipe_name]
            prompt_dependencies = ["script-plan", "idea-brief"]
            if retry:
                prompt_dependencies.append("review-refine-prompt")
            nodes.append(
                ExecutionNode(
                    node_id=prompt_node,
                    skill_name="agent.segment.prepare",
                    inputs={"segment_index": index, "recipe": recipe_name},
                    depends_on=prompt_dependencies,
                    tags=["story", "segment", "retry"] if retry else ["story", "segment"],
                    stage="prompting",
                )
            )

            input_dependencies = [prompt_node, video_asset_check_for(capability.workflow_name)]
            anchor_nodes: dict[str, str] = {}
            anchor_review_nodes: list[str] = []

            if recipe.requires_first:
                if previous_rendered_tail_node:
                    anchor_nodes["first"] = previous_rendered_tail_node
                    input_dependencies.append(previous_rendered_tail_node)
                else:
                    # Keep the original segment-frame id stable for callers and
                    # persisted plans while the payload now carries a generic
                    # anchor contract.
                    first_candidate = f"{prefix}-frame-{suffix}"
                    first_dependencies = [prompt_node, "image-asset-check", "transition-asset-check"]
                    first_count = frame_candidate_count if (
                        (index == 0 and pre_video_review) or human_anchor_review
                    ) else 1
                    nodes.append(
                        ExecutionNode(
                            node_id=first_candidate,
                            skill_name="media.image.generate_keyframe",
                            inputs={
                                "workflow_name": image_manifest.name,
                                "segment_index": index,
                                "anchor_position": "first",
                                "width": frame_width,
                                "height": frame_height,
                                "image_count": first_count,
                                "suffix": f"{prefix}_anchor_first_{suffix}",
                            },
                            depends_on=first_dependencies,
                            tags=["render", "image", "anchor", "first", "segment"],
                            tool_name="comfy.workflow.text_to_image",
                            stage="render",
                        )
                    )
                    first_source = first_candidate
                    first_review_enabled = (
                        first_count > 1
                        or (opening_human_review and index == 0)
                        or (require_human_review and review_policy == "every_segment")
                        or (require_human_review and review_policy == "anchors")
                    )
                    if first_review_enabled:
                        first_review = (
                            f"stage-review-{suffix}"
                            if prefix == "segment" and index == 0
                            else f"{prefix}-anchor-first-select-{suffix}"
                        )
                        nodes.append(
                            ExecutionNode(
                                node_id=first_review,
                                skill_name="review.assets.select",
                                inputs={
                                    "limit": 1,
                                    "review_all_candidates": True,
                                    "review_scope": "first_frame",
                                    "review_phase": "opening_frame",
                                    "require_human_review": opening_human_review or human_anchor_review,
                                    "auto_select_for_probe": stage_probe_auto_select,
                                    "review_notes": review_notes,
                                },
                                depends_on=[first_candidate],
                                tags=["review", "anchor", "first", "segment"],
                                stage="review",
                            )
                        )
                        first_source = first_review
                        anchor_review_nodes.append(first_review)
                    anchor_nodes["first"] = first_source
                    input_dependencies.append(first_source)

            if recipe.requires_last:
                last_candidate = f"{prefix}-anchor-last-candidates-{suffix}"
                last_dependencies = [prompt_node, "image-asset-check", "transition-asset-check"]
                # Landing anchors are deliberate T2I state targets for FL2V;
                # the next segment still starts from its rendered tail.
                nodes.append(
                    ExecutionNode(
                        node_id=last_candidate,
                        skill_name="media.image.generate_keyframe",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "identity_refine_workflow_name": transition_manifest.name,
                            "segment_index": index,
                            "anchor_position": "last",
                            "width": frame_width,
                            "height": frame_height,
                            "image_count": frame_candidate_count if review_all_anchors else 1,
                            "suffix": f"{prefix}_anchor_last_{suffix}",
                        },
                        depends_on=last_dependencies,
                        tags=["render", "image", "anchor", "last", "segment"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    )
                )
                last_source = last_candidate
                last_count = frame_candidate_count if review_all_anchors else 1
                if last_count > 1 or (human_anchor_review and recipe.requires_last) or (
                    opening_human_review and index == 0
                ):
                    last_review = f"{prefix}-anchor-last-select-{suffix}"
                    nodes.append(
                        ExecutionNode(
                            node_id=last_review,
                            skill_name="review.assets.select",
                            inputs={
                                "limit": 1,
                                "review_all_candidates": True,
                                "review_scope": "last_frame",
                                "review_phase": "ending_frame",
                                "require_human_review": human_anchor_review,
                                "auto_select_for_probe": stage_probe_auto_select,
                                "review_notes": review_notes,
                            },
                            depends_on=[last_candidate],
                            tags=["review", "anchor", "last", "segment"],
                            stage="review",
                        )
                    )
                    last_source = last_review
                    anchor_review_nodes.append(last_review)
                anchor_nodes["last"] = last_source
                input_dependencies.append(last_source)

            reference_node = ""
            if recipe.requires_references:
                if has_configured_references:
                    reference_check = f"{prefix}-reference-check-{suffix}"
                    nodes.append(
                        ExecutionNode(
                            node_id=reference_check,
                            skill_name="longvideo.validate_references",
                            inputs={
                                "reference_manifest": configured_reference_manifest,
                                "reference_image_paths": configured_reference_images,
                                "reference_video_paths": configured_reference_videos,
                                "max_images": max_reference_images,
                                "max_videos": max_reference_videos,
                                "selection_limit": min(reference_selection_limit, recipe.reference_selection_limit or reference_selection_limit),
                            },
                            depends_on=[prompt_node],
                            tags=["quality", "reference", "segment"],
                            stage="quality",
                        )
                    )
                    reference_node = reference_check
                else:
                    reference_candidate = f"{prefix}-reference-candidates-{suffix}"
                    reference_dependencies = [prompt_node, "image-asset-check", "transition-asset-check"]
                    nodes.append(
                        ExecutionNode(
                            node_id=reference_candidate,
                            skill_name="media.image.generate_keyframe",
                            inputs={
                                "workflow_name": image_manifest.name,
                                "segment_index": index,
                                "anchor_position": "reference",
                                "width": frame_width,
                                "height": frame_height,
                                "image_count": reference_candidate_count,
                                "suffix": f"{prefix}_references_{suffix}",
                            },
                            depends_on=reference_dependencies,
                            tags=["render", "image", "reference", "segment"],
                            tool_name="comfy.workflow.text_to_image",
                            stage="render",
                        )
                    )
                    reference_source = reference_candidate
                    reference_review_enabled = (require_human_review or stage_probe_auto_select) and (
                        review_policy == "every_segment" or index == 0
                    )
                    if reference_review_enabled:
                        reference_review = f"{prefix}-reference-select-{suffix}"
                        nodes.append(
                            ExecutionNode(
                                node_id=reference_review,
                                skill_name="review.assets.select",
                                inputs={
                                    "limit": min(reference_selection_limit, recipe.reference_selection_limit or reference_selection_limit),
                                    "review_all_candidates": True,
                                    "review_scope": "reference",
                                    "review_phase": "reference_selection",
                                    "require_human_review": require_human_review,
                                    "auto_select_for_probe": stage_probe_auto_select,
                                    "review_notes": review_notes,
                                },
                                depends_on=[reference_candidate],
                                tags=["review", "reference", "segment"],
                                stage="review",
                            )
                        )
                        reference_source = reference_review
                    reference_check = f"{prefix}-reference-check-{suffix}"
                    nodes.append(
                        ExecutionNode(
                            node_id=reference_check,
                            skill_name="longvideo.validate_references",
                            inputs={
                                "auto_reference_generation": True,
                                "selection_limit": min(reference_selection_limit, recipe.reference_selection_limit or reference_selection_limit),
                                "max_images": max_reference_images,
                                "max_videos": max_reference_videos,
                            },
                            depends_on=[prompt_node, reference_source],
                            tags=["quality", "reference", "segment"],
                            stage="quality",
                        )
                    )
                    reference_node = reference_check
                input_dependencies.append(reference_node)

            conditioning_plan = ConditioningPlan(
                recipe=recipe_name,
                workflow_name=capability.workflow_name,
                anchor_nodes=anchor_nodes,
                reference_node=reference_node,
                continuation_node=previous_rendered_tail_node or "",
            )
            render_inputs = {
                "recipe": recipe_name,
                "workflow_name": capability.workflow_name,
                "render_tool": recipe.render_tool,
                "segment_index": index,
                "video_count": self._longvideo_int(goal, "video_count", 1, 8),
                "conditioning_plan": conditioning_plan.to_dict(),
                "anchor_nodes": dict(anchor_nodes),
                "reference_node": reference_node,
                "continuation": recipe.continuation,
                "continuity_mode": continuity_mode,
                "production_profile": production_profile,
                "width": self._longvideo_int(goal, "longvideo_width", int(goal.constraints.get("longvideo_h3_width") or 512), 2048),
                "height": self._longvideo_int(goal, "longvideo_height", int(goal.constraints.get("longvideo_h3_height") or 288), 2048),
                "length": self._longvideo_int(
                    goal,
                    "longvideo_length",
                    int(goal.constraints.get("longvideo_h3_length") or segment_default_length),
                    1000,
                ),
                "steps": self._longvideo_int(goal, "longvideo_steps", int(goal.constraints.get("longvideo_h3_steps") or 16), 100),
                "model_profile": str(
                    goal.constraints.get("longvideo_model_profile")
                    or goal.constraints.get("native_h3_model_profile")
                    or goal.constraints.get("longvideo_h3_model_profile")
                    or "q2"
                ),
            }
            nodes.append(
                ExecutionNode(
                    node_id=video_node,
                    skill_name="longvideo.render_segment_video",
                    inputs=render_inputs,
                    depends_on=list(dict.fromkeys(input_dependencies)),
                    tags=["render", "video", "segment", recipe_name] + (["retry"] if retry else []),
                    tool_name=recipe.render_tool,
                    stage="render",
                )
            )
            tail_node: str | None = None
            segment_qa_node: str | None = None
            if continuity_mode == "rendered_tail":
                tail_node = f"{prefix}-tail-{suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=tail_node,
                        skill_name="media.video.extract_last_frame",
                        inputs={
                            "output_name": f"{prefix}_tail_{suffix}.png",
                            "source_video_node": video_node,
                        },
                        depends_on=[video_node],
                        tags=["quality", "continuity", "tail", "segment"] + (["retry"] if retry else []),
                        tool_name="media.extract_last_frame",
                        stage="quality",
                    )
                )
                segment_qa_node = f"{prefix}-qa-{suffix}"
                segment_duration = max(1.0, float(render_inputs["length"]) / max(segment_frame_rate, 1.0))
                nodes.append(
                    ExecutionNode(
                        node_id=segment_qa_node,
                        skill_name="media.video.qa",
                        inputs={
                            "expected_width": render_inputs["width"],
                            "expected_height": render_inputs["height"],
                            "expected_fps": segment_frame_rate,
                            "target_duration": segment_duration,
                            "duration_tolerance": 0.75,
                            "require_audio": True,
                            "require_stereo_audio": True,
                            "analyze_audio": True,
                            "frame_count": 8,
                            "columns": 4,
                            "scale_width": 320,
                            "source_video_node": video_node,
                        },
                        depends_on=[video_node],
                        tags=["quality", "technical-qa", "segment"] + (["retry"] if retry else []),
                        tool_name="media.video_qa",
                        stage="quality",
                    )
                )
            tts_node: str | None = None
            if use_tts:
                tts_node = f"{prefix}-tts-audio-{suffix}"
                nodes.append(
                    ExecutionNode(
                        node_id=tts_node,
                        skill_name="media.audio.narrate",
                        inputs={"segment_index": index, "voice": "en-US-AriaNeural", "output_name": f"{prefix}_segment_{suffix}.mp3"},
                        depends_on=[prompt_node],
                        tags=["audio", "segment"] + (["retry"] if retry else []),
                        tool_name="audio.generate_tts_real",
                        stage="audio",
                    )
                )
            return video_node, prompt_node, tts_node, tail_node, segment_qa_node

        previous_rendered_tail_node: str | None = None
        segment_tail_nodes: list[str] = []
        segment_qa_nodes: list[str] = []
        for index in range(segment_count):
            video_node, _, tts_node, segment_tail_node, segment_qa_node = append_segment(
                "segment",
                index,
                previous_rendered_tail_node,
            )
            segment_video_nodes.append(video_node)
            if tts_node:
                tts_nodes.append(tts_node)
            if segment_tail_node:
                segment_tail_nodes.append(segment_tail_node)
                previous_rendered_tail_node = segment_tail_node
            if segment_qa_node:
                segment_qa_nodes.append(segment_qa_node)

        longvideo_edit_profile = str(goal.constraints.get("longvideo_edit_profile") or "").strip()
        if longvideo_edit_profile and longvideo_edit_profile not in EDIT_PROFILES:
            raise ValueError(f"Unsupported longvideo_edit_profile: {longvideo_edit_profile}")
        if longvideo_edit_profile and use_tts:
            raise ValueError("longvideo_edit_profile currently requires use_tts=false so the edit owns segment audio")
        if longvideo_edit_profile:
            final_video_node = "edit-segment-timeline"
            nodes.append(
                ExecutionNode(
                    node_id=final_video_node,
                    skill_name="media.video.compose_timeline",
                    inputs={
                        "profile": longvideo_edit_profile,
                        "output_width": int(goal.constraints.get("longvideo_width") or goal.constraints.get("longvideo_h3_width") or 512),
                        "output_height": int(goal.constraints.get("longvideo_height") or goal.constraints.get("longvideo_h3_height") or 288),
                        "fps": segment_frame_rate,
                        "target_duration_seconds": float(goal.duration_seconds),
                        "variant_seed": int(goal.constraints.get("edit_variant_seed") or variant_seed),
                        "transition_duration_seconds": float(goal.constraints.get("edit_transition_duration") or 0.1),
                    },
                    depends_on=segment_video_nodes,
                    tags=["package", "video", "editing", "timeline"],
                    tool_name="media.compose_edit",
                    stage="package",
                )
            )
        else:
            final_video_node = "concat-final-video"
            nodes.append(
                ExecutionNode(
                    node_id=final_video_node,
                    skill_name="media.video.concat",
                    inputs={"method": "demuxer"},
                    depends_on=segment_video_nodes,
                    tags=["package", "video"],
                    tool_name="media.concat_videos",
                    stage="package",
                )
            )
        preview_dependency = final_video_node
        if use_tts:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="concat-final-audio",
                        skill_name="media.audio.concat",
                        inputs={},
                        depends_on=tts_nodes,
                        tags=["package", "audio"],
                        tool_name="audio.concat_tracks",
                        stage="audio",
                    ),
                    ExecutionNode(
                        node_id="mux-final-video",
                        skill_name="media.video.merge_audio",
                        inputs={},
                        depends_on=["concat-final-video", "concat-final-audio"],
                        tags=["package", "mux"],
                        tool_name="media.merge_audio_video",
                        stage="package",
                    ),
                ]
            )
            preview_dependency = "mux-final-video"

        preview_dependency = self._append_video_speed_node(
            goal,
            nodes,
            source_node=preview_dependency,
            node_id="video-speed",
        )
        final_video_duration = float(goal.duration_seconds)
        if goal.duration_seconds > 0:
            scaled_duration = self._scaled_video_qa_inputs(
                goal,
                {"target_duration": float(goal.duration_seconds)},
            ).get("target_duration")
            if scaled_duration not in {None, ""}:
                final_video_duration = float(scaled_duration)
        if goal.duration_seconds > 0:
            nodes.append(
                ExecutionNode(
                    node_id="video-trim",
                    skill_name="media.video.trim",
                    inputs={"duration_seconds": final_video_duration},
                    depends_on=[preview_dependency],
                    tags=["package", "video", "duration"],
                    tool_name="media.trim_video",
                    stage="package",
                )
            )
            preview_dependency = "video-trim"
        final_video_node = preview_dependency

        longvideo_qa_inputs = self._scaled_video_qa_inputs(goal, {
            "expected_width": int(goal.constraints.get("longvideo_width") or goal.constraints.get("longvideo_h3_width") or 512),
            "expected_height": int(goal.constraints.get("longvideo_height") or goal.constraints.get("longvideo_h3_height") or 288),
            "expected_fps": segment_frame_rate,
            "target_duration": float(goal.duration_seconds) if goal.duration_seconds > 0 else None,
            "duration_tolerance": 0.6,
            "require_audio": use_tts,
            "require_stereo_audio": False,
            "analyze_audio": use_tts,
            "frame_count": 8,
            "columns": 4,
            "scale_width": 480,
        })
        nodes.append(
            ExecutionNode(
                node_id="longvideo-video-qa",
                skill_name="media.video.qa",
                inputs=longvideo_qa_inputs,
                depends_on=[preview_dependency],
                tags=["quality", "technical-qa", "video", "longvideo"],
                tool_name="media.video_qa",
                stage="quality",
            )
        )
        nodes.append(
            ExecutionNode(
                node_id="preview-gif",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=[preview_dependency, "longvideo-video-qa"],
                tags=["preview", "gif"],
                tool_name="media.video_to_gif",
                stage="package",
            )
        )
        collect_node = "collect-longvideo-outputs"
        nodes.append(
            ExecutionNode(
                node_id=collect_node,
                skill_name="agent.output.collect",
                inputs={
                    "keys": ["saved_files", "video_path", "audio_path", "gif_path", "frame_path"],
                    "preferred_video_node": final_video_node,
                },
                depends_on=[
                    preview_dependency,
                    "longvideo-video-qa",
                    "preview-gif",
                    *segment_video_nodes,
                    *segment_tail_nodes,
                    *segment_qa_nodes,
                    *tts_nodes,
                ],
                tags=["artifact", "summary"],
                stage="package",
            )
        )

        review_final_node = ""
        if review_loop_enabled:
            nodes.append(
                ExecutionNode(
                    node_id="review-select",
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_scope": "final_video", "review_notes": review_notes},
                    depends_on=[collect_node],
                    tags=["review", "longvideo"],
                    stage="review",
                )
            )
            nodes.append(
                ExecutionNode(
                    node_id="review-refine-prompt",
                    skill_name="agent.review.refine_prompt",
                    inputs={"review_notes": review_notes, "retry_count": 1},
                    depends_on=["idea-brief", "review-select"],
                    tags=["review", "creative", "retry"],
                    stage="review",
                )
            )
            retry_video_nodes: list[str] = []
            retry_tts_nodes: list[str] = []
            retry_tail_nodes: list[str] = []
            retry_qa_nodes: list[str] = []
            retry_rendered_tail: str | None = None
            for index in range(segment_count):
                retry_video, _, retry_tts, retry_tail, retry_qa = append_segment(
                    "review-segment",
                    index,
                    retry_rendered_tail,
                    retry=True,
                )
                retry_video_nodes.append(retry_video)
                if retry_tts:
                    retry_tts_nodes.append(retry_tts)
                if retry_tail:
                    retry_tail_nodes.append(retry_tail)
                    retry_rendered_tail = retry_tail
                if retry_qa:
                    retry_qa_nodes.append(retry_qa)
            if longvideo_edit_profile:
                retry_preview_dependency = "review-edit-segment-timeline"
                nodes.append(
                    ExecutionNode(
                        node_id=retry_preview_dependency,
                        skill_name="media.video.compose_timeline",
                        inputs={
                            "profile": longvideo_edit_profile,
                            "output_width": int(goal.constraints.get("longvideo_width") or goal.constraints.get("longvideo_h3_width") or 512),
                            "output_height": int(goal.constraints.get("longvideo_height") or goal.constraints.get("longvideo_h3_height") or 288),
                            "fps": segment_frame_rate,
                            "target_duration_seconds": float(goal.duration_seconds),
                            "variant_seed": int(goal.constraints.get("edit_variant_seed") or variant_seed) + 1,
                            "transition_duration_seconds": float(goal.constraints.get("edit_transition_duration") or 0.1),
                        },
                        depends_on=retry_video_nodes,
                        tags=["package", "video", "editing", "timeline", "retry"],
                        tool_name="media.compose_edit",
                        stage="package",
                    )
                )
            else:
                retry_preview_dependency = "review-concat-final-video"
                nodes.append(
                    ExecutionNode(
                        node_id=retry_preview_dependency,
                        skill_name="media.video.concat",
                        inputs={"method": "demuxer"},
                        depends_on=retry_video_nodes,
                        tags=["package", "video", "retry"],
                        tool_name="media.concat_videos",
                        stage="package",
                    )
                )
            if use_tts:
                nodes.extend(
                    [
                        ExecutionNode(
                            node_id="review-concat-final-audio",
                            skill_name="media.audio.concat",
                            inputs={},
                            depends_on=retry_tts_nodes,
                            tags=["package", "audio", "retry"],
                            tool_name="audio.concat_tracks",
                            stage="audio",
                        ),
                        ExecutionNode(
                            node_id="review-mux-final-video",
                            skill_name="media.video.merge_audio",
                            inputs={},
                            depends_on=["review-concat-final-video", "review-concat-final-audio"],
                            tags=["package", "mux", "retry"],
                            tool_name="media.merge_audio_video",
                            stage="package",
                        ),
                    ]
                )
                retry_preview_dependency = "review-mux-final-video"
            retry_preview_dependency = self._append_video_speed_node(
                goal,
                nodes,
                source_node=retry_preview_dependency,
                node_id="review-video-speed",
                retry=True,
            )
            if goal.duration_seconds > 0:
                nodes.append(
                    ExecutionNode(
                        node_id="review-video-trim",
                        skill_name="media.video.trim",
                        inputs={"duration_seconds": final_video_duration},
                        depends_on=[retry_preview_dependency],
                        tags=["package", "video", "duration", "retry"],
                        tool_name="media.trim_video",
                        stage="package",
                    )
                )
                retry_preview_dependency = "review-video-trim"
            nodes.append(
                ExecutionNode(
                    node_id="review-longvideo-video-qa",
                    skill_name="media.video.qa",
                    inputs=longvideo_qa_inputs,
                    depends_on=[retry_preview_dependency],
                    tags=["quality", "technical-qa", "video", "longvideo", "retry"],
                    tool_name="media.video_qa",
                    stage="quality",
                )
            )
            nodes.append(
                ExecutionNode(
                    node_id="review-preview-gif",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 12, "scale_width": 512},
                    depends_on=[retry_preview_dependency, "review-longvideo-video-qa"],
                    tags=["preview", "gif", "retry"],
                    tool_name="media.video_to_gif",
                    stage="package",
                )
            )
            retry_collect = "review-collect-longvideo-outputs"
            nodes.append(
                ExecutionNode(
                    node_id=retry_collect,
                    skill_name="agent.output.collect",
                    inputs={
                        "keys": ["saved_files", "video_path", "audio_path", "gif_path", "frame_path"],
                        "preferred_video_node": retry_preview_dependency,
                    },
                    depends_on=[
                        retry_preview_dependency,
                        "review-longvideo-video-qa",
                        "review-preview-gif",
                        *retry_video_nodes,
                        *retry_tail_nodes,
                        *retry_qa_nodes,
                        *retry_tts_nodes,
                    ],
                    tags=["artifact", "summary", "retry"],
                    stage="package",
                )
            )
            review_final_node = "review-final-select"
            final_video_node = retry_preview_dependency
            nodes.append(
                ExecutionNode(
                    node_id=review_final_node,
                    skill_name="review.assets.select",
                    inputs={"limit": selection_limit, "review_scope": "final_video", "review_notes": review_notes},
                    depends_on=[retry_collect],
                    tags=["review", "longvideo", "retry"],
                    stage="review",
                )
            )

        summary_dependencies = [review_final_node] if review_final_node else [collect_node]
        nodes.append(
            ExecutionNode(
                node_id="persist-longvideo-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "long_video_summary.json", "summary_scope": "long_video"},
                depends_on=summary_dependencies,
                tags=["artifact", "summary"],
                stage="package",
            )
        )
        publish_package_node = ""
        if production_profile == "text2longvideo":
            publish_package_node = "persist-publish-ready-longvideo"
            nodes.append(
                ExecutionNode(
                    node_id=publish_package_node,
                    skill_name="agent.summary.persist",
                    inputs={
                        "summary_name": "publish_ready_longvideo.json",
                        "summary_scope": "publish_ready_longvideo",
                        "summary_metadata": {
                            "production_profile": production_profile,
                            "target_duration_seconds": float(goal.duration_seconds),
                            "segment_count": segment_count,
                            "recipe_sequence": list(recipe_sequence),
                            "continuity_mode": continuity_mode,
                            "segment_qa_nodes": list(segment_qa_nodes),
                            "tail_frame_nodes": list(segment_tail_nodes),
                            "final_video_node": final_video_node,
                            "publish_contract": {
                                "human_review_required": True,
                                "technical_qa_required": True,
                                "public_dispatch_requires_platform_receipt": True,
                            },
                        },
                    },
                    depends_on=["persist-longvideo-summary", *segment_qa_nodes],
                    tags=["artifact", "summary", "publish", "longvideo"],
                    stage="package",
                )
            )
        metadata = {
            "segment_count": segment_count,
            "continuity_mode": continuity_mode,
            "selected_workflow": workflow_manifest.name,
            "required_assets": [
                *([asset.to_dict() for asset in image_manifest.required_assets] if image_manifest else []),
                *([asset.to_dict() for asset in transition_manifest.required_assets] if transition_manifest else []),
            ],
            "graph_overview": [node.node_id for node in nodes],
            "idea_variants": idea_variants,
            "variant_seed": variant_seed,
            "recipe_sequence": recipe_sequence,
            "recipe_workflows": {name: selected_capabilities[name].workflow_name for name in recipe_contracts},
            "conditioning_contracts": {name: recipe_contracts[name].to_dict() for name in recipe_contracts},
            "review_policy": review_policy,
            "stage_probe_auto_select": stage_probe_auto_select,
            "use_tts": use_tts,
            "longvideo_edit_profile": longvideo_edit_profile,
            "production_profile": production_profile,
            "segment_qa_nodes": segment_qa_nodes,
            "segment_tail_nodes": segment_tail_nodes,
            "publish_package_node": publish_package_node,
            "video_speed": self._video_speed_config(goal),
            "final_video_node": final_video_node,
            "review_loop_enabled": review_loop_enabled,
            "review_notes": review_notes,
            **({"reference_video": self._reference_video_metadata(goal)} if reference_node else {}),
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="longvideo_real_v2",
            nodes=nodes,
            metadata=metadata,
            description=f"Capability-driven long-video workflow for goal '{goal.prompt}'",
        )


    def _build_video_narrate_plan(self, goal: GoalRequest) -> ExecutionPlan:
        input_video_path = goal.constraints.get("input_video_path")
        narration_text = goal.constraints.get("text")
        if not input_video_path:
            raise ValueError("media_type 'video_narrate' requires --input-video")
        if not narration_text:
            raise ValueError("media_type 'video_narrate' requires --text")

        nodes = [
            ExecutionNode(
                node_id="tts-audio",
                skill_name="media.audio.narrate",
                inputs={"text": narration_text, "voice": "en-US-AriaNeural", "rate": "+0%"},
                tags=["audio"],
                tool_name="audio.generate_tts_real",
                stage="audio",
            ),
            ExecutionNode(
                node_id="mux-video",
                skill_name="media.video.merge_audio",
                inputs={"video_path": input_video_path},
                depends_on=["tts-audio"],
                tags=["video", "audio", "package"],
                tool_name="media.merge_audio_video",
                stage="package",
            ),
            ExecutionNode(
                node_id="gif-preview",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["mux-video"],
                tags=["preview", "gif"],
                tool_name="media.video_to_gif",
                stage="package",
            ),
        ]
        metadata = {
            "input_video_path": input_video_path,
            "graph_overview": [node.node_id for node in nodes],
            "use_case": "narrate_existing_video",
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="video_narrate_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Narrate existing video '{input_video_path}'",
        )

    def _build_image_sequence_edit_plan(self, goal: GoalRequest) -> ExecutionPlan:
        raw_paths = goal.constraints.get("edit_input_paths") or goal.constraints.get("media_paths") or []
        if isinstance(raw_paths, str):
            input_paths = [raw_paths]
        else:
            input_paths = [str(path) for path in raw_paths if str(path).strip()]
        raw_drama_plan = goal.constraints.get("drama_plan")
        drama_plan_source = str(goal.constraints.get("drama_plan_source") or "").strip()
        if raw_drama_plan is None and drama_plan_source:
            source_path = Path(drama_plan_source).expanduser().resolve()
            if not source_path.is_file():
                raise DramaPlanError(f"DramaPlan file does not exist: {source_path}")
            raw_drama_plan = json.loads(source_path.read_text(encoding="utf-8"))
        if not input_paths and not isinstance(goal.constraints.get("edit_plan"), dict) and raw_drama_plan is None:
            raise ValueError("media_type 'image_sequence_edit' requires --edit-input, an edit_plan, or a drama_plan")

        raw_edit_plan = goal.constraints.get("edit_plan")
        if raw_edit_plan is not None and raw_drama_plan is not None:
            raise ValueError("image_sequence_edit accepts either edit_plan or drama_plan, not both")
        explicit_plan = EditPlan.from_dict(raw_edit_plan) if isinstance(raw_edit_plan, dict) else None
        drama_plan = None
        compiled_drama_plan = None
        if raw_drama_plan is not None:
            if not isinstance(raw_drama_plan, dict):
                raise DramaPlanError("drama_plan must be an object or a JSON file path")
            drama_plan = DramaPlan.from_dict(raw_drama_plan).validate(require_assets=True)
            compiled_drama_plan = compile_drama_plan(drama_plan)
        if explicit_plan:
            effective_profile = explicit_plan.profile
        elif compiled_drama_plan:
            effective_profile = compiled_drama_plan.profile
        else:
            requested_profile = str(goal.constraints.get("edit_profile") or "").strip()
            if requested_profile:
                effective_profile = requested_profile
            elif input_paths and all(Path(path).suffix.lower() in IMAGE_SUFFIXES for path in input_paths):
                effective_profile = "motion_cut_v1"
            else:
                effective_profile = "baseline_concat"
        if effective_profile not in EDIT_PROFILES:
            raise ValueError(f"Unsupported edit_profile: {effective_profile}")

        output_path = str(goal.constraints.get("edit_output_path") or "").strip()
        if explicit_plan:
            qa_target_duration = explicit_plan.target_duration_seconds
        elif compiled_drama_plan:
            natural_duration = sum(scene.duration_seconds for scene in drama_plan.scenes) - sum(
                transition.duration_seconds for transition in compiled_drama_plan.transitions
            )
            qa_target_duration = compiled_drama_plan.target_duration_seconds or round(natural_duration, 3)
        else:
            qa_target_duration = float(goal.duration_seconds) if goal.duration_seconds > 0 else None
        creative_review_max_attempts = int(goal.constraints.get("edit_creative_review_max_attempts") or 3)
        if creative_review_max_attempts < 1 or creative_review_max_attempts > 4:
            raise ValueError("edit_creative_review_max_attempts must be between 1 and 4")
        compose_inputs: dict[str, Any] = {
            "profile": effective_profile,
            "output_width": explicit_plan.output_width if explicit_plan else compiled_drama_plan.output_width if compiled_drama_plan else int(goal.constraints.get("edit_width") or goal.constraints.get("width") or 576),
            "output_height": explicit_plan.output_height if explicit_plan else compiled_drama_plan.output_height if compiled_drama_plan else int(goal.constraints.get("edit_height") or goal.constraints.get("height") or 1024),
            "fps": explicit_plan.fps if explicit_plan else compiled_drama_plan.fps if compiled_drama_plan else float(goal.constraints.get("edit_fps") or goal.constraints.get("video_frame_rate") or 24),
            "target_duration_seconds": qa_target_duration,
            "variant_seed": explicit_plan.variant_seed if explicit_plan else compiled_drama_plan.variant_seed if compiled_drama_plan else int(goal.constraints.get("edit_variant_seed") or 0),
            "transition_duration_seconds": float(goal.constraints.get("edit_transition_duration") or 0.10),
            "creative_review": bool(goal.constraints.get("edit_creative_review", False))
            or effective_profile == "editorial_kinetic_v1",
            "creative_review_max_attempts": creative_review_max_attempts,
            "require_audio": bool(goal.constraints.get("edit_require_audio", False)) or effective_profile != "baseline_concat",
            "require_stereo_audio": bool(goal.constraints.get("edit_require_audio", False)) or effective_profile != "baseline_concat",
            "analyze_audio": bool(goal.constraints.get("edit_analyze_audio", False)),
        }
        if input_paths:
            compose_inputs["input_paths"] = input_paths
        if explicit_plan:
            compose_inputs["edit_plan"] = explicit_plan.to_dict()
        if drama_plan:
            compose_inputs["drama_plan"] = drama_plan.to_dict()
        if output_path:
            compose_inputs["output_path"] = output_path

        nodes = [
            ExecutionNode(
                node_id="compose-edit",
                skill_name="media.video.compose_timeline",
                inputs=compose_inputs,
                tags=["render", "editing", "timeline"],
                tool_name="media.compose_edit",
                stage="package",
            ),
            ExecutionNode(
                node_id="edit-video-qa",
                skill_name="media.video.qa",
                inputs={
                    "target_duration": qa_target_duration,
                    "duration_tolerance": 0.35,
                    "expected_width": compose_inputs["output_width"],
                    "expected_height": compose_inputs["output_height"],
                    "expected_fps": compose_inputs["fps"],
                    "require_audio": bool(goal.constraints.get("edit_require_audio", False)) or effective_profile != "baseline_concat",
                    "require_stereo_audio": bool(goal.constraints.get("edit_require_audio", False)) or effective_profile != "baseline_concat",
                    "analyze_audio": bool(goal.constraints.get("edit_analyze_audio", False)),
                    "frame_count": 12,
                    "columns": 4,
                    "scale_width": 360,
                },
                depends_on=["compose-edit"],
                tags=["quality", "editing", "video"],
                tool_name="media.video_qa",
                stage="quality",
            ),
            ExecutionNode(
                node_id="edit-preview-gif",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["compose-edit", "edit-video-qa"],
                tags=["preview", "editing"],
                tool_name="media.video_to_gif",
                stage="package",
            ),
            ExecutionNode(
                node_id="collect-edit-outputs",
                skill_name="agent.output.collect",
                inputs={"keys": ["video_path", "manifest_path", "contact_sheet_path", "gif_path", "saved_files"]},
                depends_on=["compose-edit", "edit-video-qa", "edit-preview-gif"],
                tags=["artifact", "editing"],
                stage="package",
            ),
            ExecutionNode(
                node_id="persist-edit-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "image_sequence_edit_summary.json", "summary_scope": "image_sequence_edit"},
                depends_on=["collect-edit-outputs"],
                tags=["artifact", "summary", "editing"],
                stage="package",
            ),
        ]
        return ExecutionPlan(
            goal=goal,
            workflow_name="image_sequence_edit_v1",
            nodes=nodes,
            metadata={
                "graph_overview": [node.node_id for node in nodes],
                "input_paths": input_paths,
                "profile": compose_inputs.get("profile"),
                "variant_seed": compose_inputs.get("variant_seed"),
                "output_path": output_path,
                "qa_target_duration": qa_target_duration,
                "explicit_edit_plan": explicit_plan is not None,
                "explicit_drama_plan": drama_plan is not None,
            },
            description=(
                "Compile a short-drama plan into the deterministic timeline editor"
                if drama_plan
                else "Compose an agent-controlled timeline from generated images or video segments"
            ),
        )

    def _build_text2img2video_plan(self, goal: GoalRequest) -> ExecutionPlan:
        pre_video_review = self._pre_video_review_enabled(goal)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        reference_node = self._reference_video_analysis_node(goal)
        reference_micro_gag_profile = str(
            goal.constraints.get("reference_micro_gag_profile") or ""
        ).strip()
        if reference_micro_gag_profile and reference_node is None:
            raise ValueError("reference_micro_gag_profile requires reference_video_source")
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        use_upscale_for_i2v = not bool(goal.constraints.get("skip_upscale_for_i2v", False))
        upscale_manifest = (
            self._manifest_from_goal_constraints(
                goal,
                *self.DEFAULT_UPSCALE_WORKFLOWS,
                constraint_keys=("upscale_workflow_name",),
                allowed_media_types={"image_upscale"},
            )
            if use_upscale_for_i2v
            else None
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=("video_workflow_name",),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        review_loop_enabled = self._review_loop_enabled(goal)
        stage_review_enabled = self._stage_review_enabled(goal) or pre_video_review
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 2))
        image_count = (
            self._pre_video_candidate_count(goal)
            if pre_video_review
            else self._constraint_int(goal, "image_count", 1)
        )
        nodes: list[ExecutionNode] = []
        if reference_node:
            nodes.append(reference_node)
        nodes.extend(
            [
                ExecutionNode(
                    node_id="idea-brief",
                    skill_name="agent.goal.expand",
                    inputs={
                        "prompt": goal.prompt,
                        "style": goal.style,
                        "idea_variants": self.idea_director.generate_variations(goal),
                    },
                    depends_on=[reference_node.node_id] if reference_node else [],
                    tags=["creative"],
                    stage="prompting",
                ),
                ExecutionNode(
                    node_id="image-asset-check",
                    skill_name="image.ensure_workflow",
                    inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                    depends_on=["idea-brief"],
                    tags=["assets"],
                    tool_name="asset.ensure_workflow_ready",
                    stage="assets",
                ),
                ExecutionNode(
                    node_id="render-image",
                    skill_name="image.render",
                    inputs={
                        "workflow_name": image_manifest.name,
                        "width": image_manifest.recommended_defaults.get("width", 1024),
                        "height": image_manifest.recommended_defaults.get("height", 1024),
                        "image_count": image_count,
                    },
                    depends_on=["idea-brief", "image-asset-check"],
                    tags=["render", "image"],
                    tool_name="comfy.workflow.text_to_image",
                    stage="render",
                ),
            ]
        )
        image_render_node = nodes[-1]
        if goal.constraints.get("seed") is not None:
            image_render_node.inputs["seed"] = int(goal.constraints["seed"])
        if use_upscale_for_i2v:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="upscale-asset-check",
                        skill_name="media.ensure_workflow",
                        inputs={"workflow_name": upscale_manifest.name, "auto_download": goal.auto_download_assets},
                        depends_on=["render-image"],
                        tags=["assets"],
                        tool_name="asset.ensure_workflow_ready",
                        stage="assets",
                    ),
                    ExecutionNode(
                        node_id="upscale-image",
                        skill_name="image.upscale",
                        inputs={},
                        depends_on=["render-image", "upscale-asset-check"],
                        tags=["render", "upscale"],
                        tool_name="comfy.workflow.image_upscale",
                        stage="render",
                    ),
                ]
            )
        nodes.append(
            ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["upscale-image" if use_upscale_for_i2v else "render-image"],
                tags=["assets"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            )
        )
        if stage_review_enabled:
            nodes.append(
                ExecutionNode(
                    node_id="stage-review-select",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": 1,
                        "review_all_candidates": True,
                        "review_scope": "first_frame",
                        "review_phase": "opening_frame",
                        "require_human_review": self._pre_video_review_requires_human(goal),
                        "auto_select_for_probe": stage_probe_auto_select,
                        "review_notes": review_notes,
                    },
                    depends_on=[
                        "render-image"
                        if pre_video_review or not use_upscale_for_i2v
                        else "upscale-image"
                    ],
                    tags=["review", "stage"],
                    stage="review",
                )
            )
        animate_dependencies = ["idea-brief", "render-image", "video-asset-check"]
        if use_upscale_for_i2v and not stage_review_enabled:
            animate_dependencies.insert(2, "upscale-image")
        if stage_review_enabled:
            animate_dependencies = ["idea-brief", "render-image", "stage-review-select", "video-asset-check"]
        nodes.extend(
            [
                ExecutionNode(
                    node_id="animate-video",
                    skill_name="image.animate",
                    inputs={
                        "workflow_name": video_manifest.name,
                        "video_count": self._constraint_int(goal, "video_count", 1),
                        **(
                            {
                                "length": round(
                                    int(goal.constraints["duration_override_seconds"])
                                    * float(goal.constraints.get("video_frame_rate") or 24)
                                )
                            }
                            if goal.constraints.get("duration_override_seconds") is not None
                            else {}
                        ),
                        **(
                            {"seed": int(goal.constraints["seed"])}
                            if goal.constraints.get("seed") is not None
                            else {}
                        ),
                    },
                    depends_on=animate_dependencies,
                    tags=["render", "video"],
                    tool_name="comfy.workflow.image_to_video",
                    stage="render",
                ),
                ExecutionNode(
                    node_id="video-qa",
                    skill_name="media.video.qa",
                    inputs={
                        **self._video_qa_inputs(goal, video_manifest),
                        "semantic_qa_required": bool(reference_micro_gag_profile),
                        "semantic_qa_profile": reference_micro_gag_profile,
                        "character": str(goal.constraints.get("character") or ""),
                        "subject_context": dict(goal.constraints.get("subject_context") or {}),
                    },
                    depends_on=[
                        "animate-video",
                        "idea-brief",
                        *([reference_node.node_id] if reference_node else []),
                    ],
                    tags=["quality", "technical-qa", "video"],
                    tool_name="media.video_qa",
                    stage="quality",
                ),
                ExecutionNode(
                    node_id="gif-preview",
                    skill_name="media.video.gif_preview",
                    inputs={"fps": 12, "scale_width": 512},
                    depends_on=["animate-video"],
                    tags=["preview", "gif"],
                    tool_name="media.video_to_gif",
                    stage="package",
                ),
            ]
        )
        video_output_node = self._append_video_speed_node(
            goal,
            nodes,
            source_node="animate-video",
            node_id="video-speed",
        )
        video_qa_node = next(node for node in nodes if node.node_id == "video-qa")
        video_qa_node.inputs = self._scaled_video_qa_inputs(goal, video_qa_node.inputs)
        video_qa_node.depends_on = [
            video_output_node,
            "idea-brief",
            *([reference_node.node_id] if reference_node else []),
        ]
        gif_preview_node = next(node for node in nodes if node.node_id == "gif-preview")
        gif_preview_node.depends_on = [video_output_node]
        if review_loop_enabled:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="review-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["render-image", "upscale-image", video_output_node, "gif-preview"],
                        tags=["review", "retry"],
                        stage="review",
                    ),
                    ExecutionNode(
                        node_id="review-refine-prompt",
                        skill_name="agent.review.refine_prompt",
                        inputs={"review_notes": review_notes, "retry_count": 1},
                        depends_on=["idea-brief", "review-select"],
                        tags=["review", "creative", "retry"],
                        stage="review",
                    ),
                    ExecutionNode(
                        node_id="review-render-image",
                        skill_name="image.render",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "width": image_manifest.recommended_defaults.get("width", 1024),
                            "height": image_manifest.recommended_defaults.get("height", 1024),
                            "image_count": image_count,
                            **(
                                {"seed": int(goal.constraints["seed"])}
                                if goal.constraints.get("seed") is not None
                                else {}
                            ),
                        },
                        depends_on=["review-refine-prompt", "image-asset-check"],
                        tags=["render", "image", "retry"],
                        tool_name="comfy.workflow.text_to_image",
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-upscale-image",
                        skill_name="image.upscale",
                        inputs={},
                        depends_on=["review-render-image", "upscale-asset-check"],
                        tags=["render", "upscale", "retry"],
                        tool_name="comfy.workflow.image_upscale",
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-animate-video",
                        skill_name="image.animate",
                        inputs={
                            "workflow_name": video_manifest.name,
                            "video_count": self._constraint_int(goal, "video_count", 1),
                            **(
                                {
                                    "length": round(
                                        int(goal.constraints["duration_override_seconds"])
                                        * float(goal.constraints.get("video_frame_rate") or 24)
                                    )
                                }
                                if goal.constraints.get("duration_override_seconds") is not None
                                else {}
                            ),
                            **(
                                {"seed": int(goal.constraints["seed"])}
                                if goal.constraints.get("seed") is not None
                                else {}
                            ),
                        },
                        depends_on=["review-refine-prompt", "review-upscale-image", "video-asset-check"],
                        tags=["render", "video", "retry"],
                        tool_name="comfy.workflow.image_to_video",
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-gif-preview",
                        skill_name="media.video.gif_preview",
                        inputs={"fps": 12, "scale_width": 512},
                        depends_on=["review-animate-video"],
                        tags=["preview", "gif", "retry"],
                        tool_name="media.video_to_gif",
                        stage="package",
                    ),
                    ExecutionNode(
                        node_id="review-final-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["review-render-image", "review-upscale-image", "review-animate-video", "review-gif-preview"],
                        tags=["review", "retry"],
                        stage="review",
                    ),
                ]
            )
            review_video_output_node = self._append_video_speed_node(
                goal,
                nodes,
                source_node="review-animate-video",
                node_id="review-video-speed",
                retry=True,
            )
            review_gif_preview_node = next(node for node in nodes if node.node_id == "review-gif-preview")
            review_gif_preview_node.depends_on = [review_video_output_node]
            review_final_select_node = next(node for node in nodes if node.node_id == "review-final-select")
            review_final_select_node.depends_on = [
                "review-render-image",
                "review-upscale-image",
                review_video_output_node,
                "review-gif-preview",
            ]
        summary_dependencies = ["render-image", "animate-video", "video-qa", "gif-preview"]
        if use_upscale_for_i2v:
            summary_dependencies.insert(1, "upscale-image")
        if review_loop_enabled:
            summary_dependencies.append("review-final-select")
        nodes.append(
            ExecutionNode(
                node_id="persist-text2img2video-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "text2img2video_summary.json", "summary_scope": "text2img2video"},
                depends_on=summary_dependencies,
                tags=["artifact", "summary"],
                stage="package",
            )
        )
        metadata = {
            "selected_workflows": [
                image_manifest.name,
                *([upscale_manifest.name] if use_upscale_for_i2v else []),
                video_manifest.name,
            ],
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *(
                    [asset.to_dict() for asset in upscale_manifest.required_assets]
                    if use_upscale_for_i2v
                    else []
                ),
                *[asset.to_dict() for asset in video_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
            "review_loop_enabled": review_loop_enabled,
            "video_speed": self._video_speed_config(goal),
            "review_notes": review_notes,
            "reference_micro_gag_profile": reference_micro_gag_profile,
            **({"reference_video": self._reference_video_metadata(goal)} if reference_node else {}),
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="text2img2video_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Composed text2img2video chain for goal '{goal.prompt}'",
        )

    def _build_storyboard_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        segment_count = self._constraint_int(
            goal,
            "segment_count",
            int(workflow_manifest.recommended_defaults.get("segment_count", 4)),
        )
        frame_width = workflow_manifest.recommended_defaults.get("frame_width", 1280)
        frame_height = workflow_manifest.recommended_defaults.get("frame_height", 720)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="script-plan",
                skill_name="agent.story.segment",
                inputs={"segment_count": segment_count, "tone": "clear visual beats"},
                depends_on=["idea-brief"],
                tags=["story"],
            ),
            ExecutionNode(
                node_id="asset-check",
                skill_name="story.ensure_workflow",
                inputs={"workflow_name": workflow_manifest.name},
                depends_on=["script-plan"],
                tags=["assets"],
            ),
            ExecutionNode(
                node_id="storyboard-frames",
                skill_name="story.render_frames",
                inputs={"frame_width": frame_width, "frame_height": frame_height},
                depends_on=["asset-check"],
                tags=["render"],
            ),
            ExecutionNode(
                node_id="storyboard-package",
                skill_name="story.package_outputs",
                inputs={},
                depends_on=["storyboard-frames"],
                tags=["artifact"],
            ),
        ]
        metadata = {
            "segment_count": segment_count,
            "selected_workflow": workflow_manifest.name,
            "required_assets": [],
            "graph_overview": [node.node_id for node in nodes],
            "frame_width": frame_width,
            "frame_height": frame_height,
        }
        description = f"Storyboard workflow '{workflow_manifest.name}' for goal '{goal.prompt}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name=workflow_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_image_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        width = workflow_manifest.recommended_defaults.get("width", 1024)
        height = workflow_manifest.recommended_defaults.get("height", 1024)
        image_count = self._constraint_int(goal, "image_count", 1)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
            ),
            ExecutionNode(
                node_id="asset-check",
                skill_name="image.ensure_workflow",
                inputs={
                    "workflow_name": workflow_manifest.name,
                    "auto_download": goal.auto_download_assets,
                },
                depends_on=["idea-brief"],
                tags=["assets"],
            ),
            ExecutionNode(
                node_id="render-image",
                skill_name="image.render",
                inputs={
                    "workflow_name": workflow_manifest.name,
                    "width": width,
                    "height": height,
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "asset-check"],
                tags=["render"],
            ),
        ]
        if bool(goal.constraints.get("stage_probe_auto_select", False)):
            nodes.append(
                ExecutionNode(
                    node_id="stage-preview-select",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": self._review_selection_limit(goal, default=1),
                        "review_all_candidates": True,
                        "review_scope": "reference",
                        "review_phase": "reference_selection",
                        "require_human_review": False,
                        "auto_select_for_probe": True,
                        "review_notes": "",
                    },
                    depends_on=["render-image"],
                    tags=["review", "stage", "probe"],
                    stage="review",
                )
            )
        metadata = {
            "selected_workflow": workflow_manifest.name,
            "required_assets": [asset.to_dict() for asset in workflow_manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "width": width,
            "height": height,
            "stage_probe_auto_select": bool(goal.constraints.get("stage_probe_auto_select", False)),
        }
        description = f"Image workflow '{workflow_manifest.name}' for goal '{goal.prompt}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name=workflow_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_comfy_primitive_plan(self, goal: GoalRequest, workflow_manifest: WorkflowManifest) -> ExecutionPlan:
        constraints = goal.constraints
        input_image_path = constraints.get("input_image_path")
        if goal.media_type in {"image_refine", "image_upscale", "image_to_video"} and not input_image_path:
            raise ValueError(f"media_type '{goal.media_type}' requires --input-image")

        if goal.media_type == "image_refine":
            nodes = [
                ExecutionNode(
                    node_id="prompt-prepare",
                    skill_name="agent.prompt.compose",
                    inputs={"prompt": goal.prompt, "style": goal.style},
                    tags=["creative"],
                ),
                ExecutionNode(
                    node_id="asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={"workflow_name": workflow_manifest.name, "auto_download": goal.auto_download_assets},
                    depends_on=["prompt-prepare"],
                    tags=["assets"],
                ),
                ExecutionNode(
                    node_id="img2img-render",
                    skill_name="media.image.refine",
                    inputs={
                        "image_path": input_image_path,
                    },
                    depends_on=["prompt-prepare", "asset-check"],
                    tags=["render"],
                ),
            ]
        elif goal.media_type == "image_upscale":
            nodes = [
                ExecutionNode(
                    node_id="asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={"workflow_name": workflow_manifest.name, "auto_download": goal.auto_download_assets},
                    tags=["assets"],
                ),
                ExecutionNode(
                    node_id="upscale-render",
                    skill_name="media.image.upscale",
                    inputs={"image_path": input_image_path},
                    depends_on=["asset-check"],
                    tags=["render"],
                ),
            ]
        else:
            nodes = [
                ExecutionNode(
                    node_id="prompt-prepare",
                    skill_name="agent.prompt.compose",
                    inputs={"prompt": goal.prompt, "style": goal.style},
                    tags=["creative"],
                ),
                ExecutionNode(
                    node_id="asset-check",
                    skill_name="media.ensure_workflow",
                    inputs={"workflow_name": workflow_manifest.name, "auto_download": goal.auto_download_assets},
                    depends_on=["prompt-prepare"],
                    tags=["assets"],
                ),
                ExecutionNode(
                    node_id="i2v-render",
                    skill_name="media.image.animate",
                    inputs={
                        "image_path": input_image_path,
                        "prompt": constraints.get("text", "") or goal.prompt,
                        "width": constraints.get("width"),
                        "height": constraints.get("height"),
                    },
                    depends_on=["prompt-prepare", "asset-check"],
                    tags=["render", "video"],
                ),
            ]

        metadata = {
            "selected_workflow": workflow_manifest.name,
            "required_assets": [asset.to_dict() for asset in workflow_manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "input_image_path": input_image_path,
        }
        description = f"Workflow '{workflow_manifest.name}' for media_type '{goal.media_type}'"
        return ExecutionPlan(
            goal=goal,
            workflow_name=workflow_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=description,
        )

    def _build_text2img2img_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        refine_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_REFINE_WORKFLOWS,
            constraint_keys=("refine_workflow_name", "transition_workflow_name"),
            allowed_media_types={"image_refine"},
        )
        image_count = self._constraint_int(goal, "image_count", 1)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["idea-brief"],
                tags=["assets"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-image",
                skill_name="image.render",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "image-asset-check"],
                tags=["render", "image"],
                stage="render",
            ),
            ExecutionNode(
                node_id="refine-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": refine_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["render-image"],
                tags=["assets"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="refine-image",
                skill_name="media.image.refine",
                inputs={},
                depends_on=["idea-brief", "render-image", "refine-asset-check"],
                tags=["render", "image", "refine"],
                stage="render",
            ),
        ]
        if stage_probe_auto_select:
            nodes.insert(
                3,
                ExecutionNode(
                    node_id="stage-preview-select",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": self._review_selection_limit(goal, default=1),
                        "review_all_candidates": True,
                        "review_scope": "reference",
                        "review_phase": "reference_selection",
                        "require_human_review": False,
                        "auto_select_for_probe": True,
                        "review_notes": "",
                    },
                    depends_on=["render-image"],
                    tags=["review", "stage", "probe"],
                    stage="review",
                ),
            )
            refine_node = next(node for node in nodes if node.node_id == "refine-image")
            refine_node.depends_on = ["idea-brief", "stage-preview-select", "refine-asset-check"]
        metadata = {
            "selected_workflows": [image_manifest.name, refine_manifest.name],
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *[asset.to_dict() for asset in refine_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
            "stage_probe_auto_select": stage_probe_auto_select,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="text2img2img_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic text2img2img chain for goal '{goal.prompt}'",
        )

    def _build_text2video_plan(self, goal: GoalRequest) -> ExecutionPlan:
        pre_video_review = self._pre_video_review_enabled(goal)
        stage_probe_auto_select = bool(goal.constraints.get("stage_probe_auto_select", False))
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *(self.DEFAULT_I2V_WORKFLOWS if pre_video_review else self.DEFAULT_T2V_WORKFLOWS),
            # A T2V manifest may advertise ``long_video`` as well, so merely
            # filtering by media type is not enough to guarantee that the
            # approved image becomes conditioning. Ignore the T2V override
            # whenever the shared image gate is active.
            constraint_keys=("video_workflow_name",) if not pre_video_review else (),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        review_loop_enabled = self._review_loop_enabled(goal)
        stage_review_enabled = self._stage_review_enabled(goal) or pre_video_review
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 2))
        image_count = (
            self._pre_video_candidate_count(goal)
            if pre_video_review
            else self._constraint_int(goal, "image_count", 1)
        )
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["idea-brief"],
                tags=["assets"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-image",
                skill_name="image.render",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "image_count": image_count,
                },
                depends_on=["idea-brief", "image-asset-check"],
                tags=["render", "image"],
                stage="render",
            ),
            ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["render-image"],
                tags=["assets"],
                stage="assets",
            ),
        ]
        if stage_review_enabled:
            nodes.append(
                ExecutionNode(
                    node_id="stage-review-select",
                    skill_name="review.assets.select",
                    inputs={
                        "limit": 1,
                        "review_all_candidates": True,
                        "review_scope": "first_frame",
                        "review_phase": "opening_frame",
                        "require_human_review": self._pre_video_review_requires_human(goal),
                        "auto_select_for_probe": stage_probe_auto_select,
                        "review_notes": review_notes,
                    },
                    depends_on=["render-image"],
                    tags=["review", "stage"],
                    stage="review",
                )
            )
        animate_dependencies = ["idea-brief", "render-image", "video-asset-check"]
        if stage_review_enabled:
            animate_dependencies = ["idea-brief", "stage-review-select", "video-asset-check"]
        nodes.extend(
            [
            ExecutionNode(
                node_id="animate-video",
                skill_name="media.image.animate",
                inputs={
                    "workflow_name": video_manifest.name,
                    "video_count": self._constraint_int(goal, "video_count", 1),
                },
                depends_on=animate_dependencies,
                tags=["render", "video"],
                stage="render",
            ),
            ExecutionNode(
                node_id="video-qa",
                skill_name="media.video.qa",
                inputs=self._video_qa_inputs(goal, video_manifest),
                depends_on=["animate-video"],
                tags=["quality", "technical-qa", "video"],
                tool_name="media.video_qa",
                stage="quality",
            ),
            ExecutionNode(
                node_id="gif-preview",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["animate-video"],
                tags=["preview", "gif"],
                stage="package",
            ),
        ]
        )
        video_output_node = self._append_video_speed_node(
            goal,
            nodes,
            source_node="animate-video",
            node_id="video-speed",
        )
        video_qa_node = next(node for node in nodes if node.node_id == "video-qa")
        video_qa_node.inputs = self._scaled_video_qa_inputs(goal, video_qa_node.inputs)
        video_qa_node.depends_on = [video_output_node]
        gif_preview_node = next(node for node in nodes if node.node_id == "gif-preview")
        gif_preview_node.depends_on = [video_output_node]
        if review_loop_enabled:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="review-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["render-image", video_output_node, "gif-preview"],
                        tags=["review", "retry"],
                        stage="review",
                    ),
                    ExecutionNode(
                        node_id="review-refine-prompt",
                        skill_name="agent.review.refine_prompt",
                        inputs={"review_notes": review_notes, "retry_count": 1},
                        depends_on=["idea-brief", "review-select"],
                        tags=["review", "creative", "retry"],
                        stage="review",
                    ),
                    ExecutionNode(
                        node_id="review-render-image",
                        skill_name="image.render",
                        inputs={
                            "workflow_name": image_manifest.name,
                            "width": image_manifest.recommended_defaults.get("width", 1024),
                            "height": image_manifest.recommended_defaults.get("height", 1024),
                            "image_count": image_count,
                        },
                        depends_on=["review-refine-prompt", "image-asset-check"],
                        tags=["render", "image", "retry"],
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-animate-video",
                        skill_name="media.image.animate",
                        inputs={
                            "workflow_name": video_manifest.name,
                            "video_count": self._constraint_int(goal, "video_count", 1),
                        },
                        depends_on=["review-refine-prompt", "review-render-image", "video-asset-check"],
                        tags=["render", "video", "retry"],
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-gif-preview",
                        skill_name="media.video.gif_preview",
                        inputs={"fps": 12, "scale_width": 512},
                        depends_on=["review-animate-video"],
                        tags=["preview", "gif", "retry"],
                        stage="package",
                    ),
                    ExecutionNode(
                        node_id="review-final-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["review-render-image", "review-animate-video", "review-gif-preview"],
                        tags=["review", "retry"],
                        stage="review",
                    ),
                ]
            )
            review_video_output_node = self._append_video_speed_node(
                goal,
                nodes,
                source_node="review-animate-video",
                node_id="review-video-speed",
                retry=True,
            )
            review_gif_preview_node = next(node for node in nodes if node.node_id == "review-gif-preview")
            review_gif_preview_node.depends_on = [review_video_output_node]
            review_final_select_node = next(node for node in nodes if node.node_id == "review-final-select")
            review_final_select_node.depends_on = [
                "review-render-image",
                review_video_output_node,
                "review-gif-preview",
            ]
        summary_dependencies = ["render-image", "animate-video", "video-qa", "gif-preview"]
        if review_loop_enabled:
            summary_dependencies.append("review-final-select")
        nodes.append(
            ExecutionNode(
                node_id="persist-text2video-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "text2video_summary.json", "summary_scope": "text2video"},
                depends_on=summary_dependencies,
                tags=["artifact", "summary"],
                stage="package",
            )
        )
        metadata = {
            "selected_workflows": [image_manifest.name, video_manifest.name],
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *[asset.to_dict() for asset in video_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
            "review_loop_enabled": review_loop_enabled,
            "video_speed": self._video_speed_config(goal),
            "review_notes": review_notes,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="text2video_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic text2video chain for goal '{goal.prompt}'",
        )

    def _build_sticker_pack_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        character = goal.constraints.get("character", "")
        expression_count = self._constraint_int(goal, "sticker_expression_count", 8)
        images_per_prompt = self._constraint_int(goal, "images_per_prompt", 1)
        nodes = [
            ExecutionNode(
                node_id="sticker-expressions",
                skill_name="agent.sticker.expressions",
                inputs={"prompt": goal.prompt, "character": character, "expression_count": expression_count},
                tags=["creative", "sticker"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="sticker-prompts",
                skill_name="agent.sticker.prompt_set",
                inputs={"character": character},
                depends_on=["sticker-expressions"],
                tags=["creative", "sticker"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["sticker-prompts"],
                tags=["assets", "sticker"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-stickers",
                skill_name="media.image.render_batch",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "images_per_prompt": images_per_prompt,
                    "suffix": "stickers",
                },
                depends_on=["sticker-prompts", "image-asset-check"],
                tags=["render", "sticker", "batch"],
                stage="render",
            ),
            ExecutionNode(
                node_id="package-stickers",
                skill_name="agent.sticker.package",
                inputs={},
                depends_on=["render-stickers"],
                tags=["artifact", "sticker"],
                stage="package",
            ),
        ]
        metadata = {
            "selected_workflow": image_manifest.name,
            "required_assets": [asset.to_dict() for asset in image_manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "character": character,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name=image_manifest.name,
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic sticker pack chain for goal '{goal.prompt}'",
        )

    def _build_animated_sticker_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        video_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_I2V_WORKFLOWS,
            constraint_keys=("video_workflow_name",),
            allowed_media_types={"image_to_video", "image_to_video_audio", "long_video"},
        )
        character = goal.constraints.get("character", "")
        review_loop_enabled = self._review_loop_enabled(goal)
        review_notes = str(goal.constraints.get("review_notes", "") or "")
        selection_limit = self._review_selection_limit(goal, default=self._constraint_int(goal, "review_selection_limit", 2))
        expression_count = self._constraint_int(goal, "sticker_expression_count", 8)
        images_per_prompt = self._constraint_int(goal, "images_per_prompt", 1)
        nodes = [
            ExecutionNode(
                node_id="sticker-expressions",
                skill_name="agent.sticker.expressions",
                inputs={"prompt": goal.prompt, "character": character, "expression_count": expression_count},
                tags=["creative", "sticker", "animated"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="sticker-prompts",
                skill_name="agent.sticker.prompt_set",
                inputs={"character": character},
                depends_on=["sticker-expressions"],
                tags=["creative", "sticker"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["sticker-prompts"],
                tags=["assets", "sticker"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-stickers",
                skill_name="media.image.render_batch",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "images_per_prompt": images_per_prompt,
                    "suffix": "animated_stickers",
                },
                depends_on=["sticker-prompts", "image-asset-check"],
                tags=["render", "sticker", "batch"],
                stage="render",
            ),
            ExecutionNode(
                node_id="motion-prompt",
                skill_name="agent.sticker.motion_prompt",
                inputs={},
                depends_on=["render-stickers"],
                tags=["creative", "sticker", "motion"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": video_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["motion-prompt"],
                tags=["assets", "sticker", "video"],
                tool_name="asset.ensure_workflow_ready",
                stage="assets",
            ),
            ExecutionNode(
                node_id="animate-sticker",
                skill_name="media.image.animate",
                inputs={
                    "workflow_name": video_manifest.name,
                    "video_count": self._constraint_int(goal, "video_count", 1),
                },
                depends_on=["motion-prompt", "video-asset-check"],
                tags=["render", "video", "sticker"],
                tool_name="comfy.workflow.image_to_video",
                stage="render",
            ),
            ExecutionNode(
                node_id="gif-preview",
                skill_name="media.video.gif_preview",
                inputs={"fps": 12, "scale_width": 512},
                depends_on=["animate-sticker"],
                tags=["preview", "gif", "sticker"],
                tool_name="media.video_to_gif",
                stage="package",
            ),
            ExecutionNode(
                node_id="package-animated-sticker",
                skill_name="agent.sticker.animate.package",
                inputs={},
                depends_on=["animate-sticker", "gif-preview"],
                tags=["artifact", "sticker", "animated"],
                stage="package",
            ),
        ]
        if review_loop_enabled:
            nodes.extend(
                [
                    ExecutionNode(
                        node_id="review-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["render-stickers", "animate-sticker", "gif-preview", "package-animated-sticker"],
                        tags=["review", "sticker", "retry"],
                        stage="review",
                    ),
                    ExecutionNode(
                        node_id="review-refine-prompt",
                        skill_name="agent.review.refine_prompt",
                        inputs={"review_notes": review_notes, "retry_count": 1},
                        depends_on=["motion-prompt", "review-select"],
                        tags=["review", "creative", "retry"],
                        stage="review",
                    ),
                    ExecutionNode(
                        node_id="review-animate-sticker",
                        skill_name="media.image.animate",
                        inputs={
                            "workflow_name": video_manifest.name,
                            "video_count": self._constraint_int(goal, "video_count", 1),
                        },
                        depends_on=["review-refine-prompt", "render-stickers", "video-asset-check"],
                        tags=["render", "video", "sticker", "retry"],
                        tool_name="comfy.workflow.image_to_video",
                        stage="render",
                    ),
                    ExecutionNode(
                        node_id="review-gif-preview",
                        skill_name="media.video.gif_preview",
                        inputs={"fps": 12, "scale_width": 512},
                        depends_on=["review-animate-sticker"],
                        tags=["preview", "gif", "sticker", "retry"],
                        tool_name="media.video_to_gif",
                        stage="package",
                    ),
                    ExecutionNode(
                        node_id="review-package-animated-sticker",
                        skill_name="agent.sticker.animate.package",
                        inputs={},
                        depends_on=["review-animate-sticker", "review-gif-preview"],
                        tags=["artifact", "sticker", "animated", "retry"],
                        stage="package",
                    ),
                    ExecutionNode(
                        node_id="review-final-select",
                        skill_name="review.assets.select",
                        inputs={"limit": selection_limit, "review_notes": review_notes},
                        depends_on=["review-package-animated-sticker"],
                        tags=["review", "sticker", "retry"],
                        stage="review",
                    ),
                ]
            )
        metadata = {
            "selected_workflows": [image_manifest.name, video_manifest.name],
            "required_assets": [
                *[asset.to_dict() for asset in image_manifest.required_assets],
                *[asset.to_dict() for asset in video_manifest.required_assets],
            ],
            "graph_overview": [node.node_id for node in nodes],
            "character": character,
            "review_loop_enabled": review_loop_enabled,
            "review_notes": review_notes,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="animated_sticker_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic animated sticker chain for goal '{goal.prompt}'",
        )

    def _build_carousel_plan(self, goal: GoalRequest) -> ExecutionPlan:
        image_manifest = self._manifest_from_goal_constraints(
            goal,
            *self.DEFAULT_IMAGE_WORKFLOWS,
            constraint_keys=("image_workflow_name", "workflow_name"),
            allowed_media_types={"image"},
        )
        slide_count = self._constraint_int(goal, "segment_count", max(3, goal.duration_seconds // 10))
        images_per_prompt = self._constraint_int(goal, "images_per_prompt", 1)
        nodes = [
            ExecutionNode(
                node_id="idea-brief",
                skill_name="agent.goal.expand",
                inputs={
                    "prompt": goal.prompt,
                    "style": goal.style,
                    "idea_variants": self.idea_director.generate_variations(goal),
                },
                tags=["creative", "carousel"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="script-plan",
                skill_name="agent.story.segment",
                inputs={"segment_count": slide_count, "tone": "distinct carousel beat"},
                depends_on=["idea-brief"],
                tags=["story", "carousel"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="carousel-prompts",
                skill_name="agent.carousel.prompt_set",
                inputs={"style": goal.style},
                depends_on=["script-plan"],
                tags=["creative", "carousel"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="image-asset-check",
                skill_name="image.ensure_workflow",
                inputs={"workflow_name": image_manifest.name, "auto_download": goal.auto_download_assets},
                depends_on=["carousel-prompts"],
                tags=["assets", "carousel"],
                stage="assets",
            ),
            ExecutionNode(
                node_id="render-carousel",
                skill_name="media.image.render_batch",
                inputs={
                    "workflow_name": image_manifest.name,
                    "width": image_manifest.recommended_defaults.get("width", 1024),
                    "height": image_manifest.recommended_defaults.get("height", 1024),
                    "images_per_prompt": images_per_prompt,
                    "suffix": "carousel",
                },
                depends_on=["carousel-prompts", "image-asset-check"],
                tags=["render", "carousel", "batch"],
                stage="render",
            ),
            ExecutionNode(
                node_id="package-carousel",
                skill_name="agent.carousel.package",
                inputs={},
                depends_on=["render-carousel"],
                tags=["artifact", "carousel"],
                stage="package",
            ),
        ]
        metadata = {
            "selected_workflow": image_manifest.name,
            "required_assets": [asset.to_dict() for asset in image_manifest.required_assets],
            "graph_overview": [node.node_id for node in nodes],
            "slide_count": slide_count,
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="carousel_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic carousel chain for goal '{goal.prompt}'",
        )

    def _build_publish_review_plan(self, goal: GoalRequest) -> ExecutionPlan:
        platforms = goal.constraints.get("platforms") or []
        dry_run = bool(goal.constraints.get("dry_run", True))
        selection_limit = int(goal.constraints.get("selection_limit") or 10)
        nodes = [
            ExecutionNode(
                node_id="ingest-media",
                skill_name="publish.media.ingest",
                inputs={},
                tags=["publish", "review"],
                stage="ingest",
            ),
            ExecutionNode(
                node_id="review-select",
                skill_name="review.assets.select",
                inputs={
                    "limit": selection_limit,
                    "review_scope": _infer_publish_review_scope(goal),
                    "review_all_candidates": bool(goal.constraints.get("review_all_candidates", False)),
                },
                depends_on=["ingest-media"],
                tags=["review", "publish"],
                stage="review",
            ),
            ExecutionNode(
                node_id="process-media",
                skill_name="publish.media.process",
                inputs={"output_dir": goal.constraints.get("output_dir")},
                depends_on=["review-select"],
                tags=["publish", "media"],
                tool_name="publish.process_media",
                stage="package",
            ),
            ExecutionNode(
                node_id="prepare-caption",
                skill_name="publish.caption.prepare",
                inputs={
                    "prefix": goal.constraints.get("caption_prefix", ""),
                    "hashtags": goal.constraints.get("hashtags", []),
                    "platforms": platforms,
                },
                depends_on=["review-select", "process-media"],
                tags=["publish", "caption"],
                stage="prompting",
            ),
            ExecutionNode(
                node_id="dispatch-publish",
                skill_name="publish.social.dispatch",
                inputs={
                    "platforms": platforms,
                    "platform_configs": goal.constraints.get("platform_configs", {}),
                    "additional_params": goal.constraints.get("additional_params", {}),
                    "publish_mode": goal.constraints.get("publish_mode", ""),
                    "dry_run": dry_run,
                },
                depends_on=["process-media", "prepare-caption"],
                tags=["publish", "dispatch"],
                tool_name="publish.social",
                stage="publish",
            ),
            ExecutionNode(
                node_id="persist-publish-review-summary",
                skill_name="agent.summary.persist",
                inputs={"summary_name": "publish_review_summary.json", "summary_scope": "publish_review"},
                depends_on=["review-select", "process-media", "prepare-caption", "dispatch-publish"],
                tags=["artifact", "summary", "publish"],
                stage="package",
            ),
        ]
        metadata = {
            "graph_overview": [node.node_id for node in nodes],
            "platforms": list(platforms),
            "dry_run": dry_run,
            "publish_mode": str(goal.constraints.get("publish_mode") or ""),
        }
        return ExecutionPlan(
            goal=goal,
            workflow_name="publish_review_v1",
            nodes=nodes,
            metadata=metadata,
            description=f"Agentic publish/review chain for goal '{goal.prompt}'",
        )
