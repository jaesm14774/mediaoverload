"""Shared asset selection, review and video delivery policies."""

from __future__ import annotations

import math
from agentic.runtime.contracts import (
    MIN_VIDEO_DURATION_SECONDS,
    ExecutionNode,
    GoalRequest,
)
from agentic.assets.registry import AssetRegistry, WorkflowManifest
from agentic.runtime.creativity import IdeaDirector


class PlanningContext:
    DEFAULT_IMAGE_WORKFLOWS = ("krea2_turbo",)
    DEFAULT_REFINE_WORKFLOWS = ("flux3_image_klein_img2img",)
    DEFAULT_UPSCALE_WORKFLOWS = ("Tile Upscaler SDXL",)
    DEFAULT_T2V_WORKFLOWS = ("minimax_h3_t2va",)
    DEFAULT_I2V_WORKFLOWS = ("minimax_h3_i2va",)

    def __init__(
        self, asset_registry: AssetRegistry, idea_director: IdeaDirector | None = None
    ) -> None:
        self.asset_registry = asset_registry
        self.idea_director = idea_director or IdeaDirector()

    def preferred_manifest(self, *workflow_names: str) -> WorkflowManifest:
        for workflow_name in workflow_names:
            try:
                return self.asset_registry.get_manifest(workflow_name)
            except KeyError:
                continue
        requested = ", ".join(workflow_names)
        raise KeyError(
            f"None of the preferred workflows are available (configs/workflow): {requested}"
        )

    def selected_manifest(
        self, name: str, allowed_media_types: set[str] | None = None
    ) -> WorkflowManifest:
        try:
            manifest = self.asset_registry.get_manifest(name)
        except KeyError as exc:
            raise ValueError(f"Unknown workflow: {name}") from exc
        if allowed_media_types and not set(manifest.media_types).intersection(
            allowed_media_types
        ):
            raise ValueError(
                f"Workflow {name} does not support {sorted(allowed_media_types)}"
            )
        return manifest

    def manifest_from_goal_constraints(
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
            return self.selected_manifest(selected_name, allowed_media_types)
        return self.preferred_manifest(*workflow_names)

    def pick_goal_workflow(self, goal: GoalRequest) -> WorkflowManifest:
        selected_name = str(goal.constraints.get("workflow_name") or "").strip()
        if selected_name:
            return self.selected_manifest(selected_name, {goal.media_type})
        return self.asset_registry.pick_workflow(goal.media_type)

    @staticmethod
    def constraint_int(goal: GoalRequest, key: str, default: int) -> int:
        value = goal.constraints.get(key)
        if value in {None, ""}:
            return default
        return max(1, int(value))

    @staticmethod
    def longvideo_int(goal: GoalRequest, key: str, default: int, maximum: int) -> int:
        value = goal.constraints.get(key)
        if value in {None, ""}:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{key} must be an integer between 1 and {maximum}"
            ) from exc
        if parsed < 1 or parsed > maximum:
            raise ValueError(f"{key} must be an integer between 1 and {maximum}")
        return parsed

    @staticmethod
    def native_h3_render_config(
        goal: GoalRequest, manifest: WorkflowManifest, *, default_steps: int = 16
    ) -> dict[str, int]:
        defaults = dict(manifest.recommended_defaults or {})
        minimum_length = round(MIN_VIDEO_DURATION_SECONDS * 24)
        configured_length = int(
            goal.constraints.get("native_h3_length")
            or max(
                107,
                math.ceil((min(15, goal.duration_seconds) * 24 - 5) / 17) * 17 + 5,
            )
        )
        return {
            "width": int(
                goal.constraints.get("canvas_width") or defaults.get("width", 608)
            ),
            "height": int(
                goal.constraints.get("canvas_height") or defaults.get("height", 352)
            ),
            "length": max(minimum_length, configured_length),
            "steps": int(
                goal.constraints.get("native_h3_steps")
                or defaults.get("steps", default_steps)
            ),
            "video_count": PlanningContext.constraint_int(goal, "video_count", 1),
        }

    @staticmethod
    def canvas_dimensions(
        goal: GoalRequest, manifest: WorkflowManifest
    ) -> tuple[int, int]:
        defaults = dict(manifest.recommended_defaults or {})
        return (
            int(goal.constraints.get("canvas_width") or defaults.get("width", 1024)),
            int(goal.constraints.get("canvas_height") or defaults.get("height", 1024)),
        )

    @staticmethod
    def video_qa_inputs(
        goal: GoalRequest, manifest: WorkflowManifest
    ) -> dict[str, object]:
        defaults = dict(manifest.recommended_defaults or {})
        frame_rate = max(
            1.0,
            float(defaults.get("frame_rate") or goal.constraints.get("video_frame_rate") or 24),
        )
        fps = frame_rate
        length = defaults.get("length")
        target_duration = goal.constraints.get("duration_override_seconds")
        if target_duration in {None, ""} and length not in {None, ""} and fps > 0:
            target_duration = float(length) / fps
        if target_duration not in {None, ""}:
            target_duration = max(
                MIN_VIDEO_DURATION_SECONDS, float(target_duration)
            )
        return {
            "expected_width": goal.constraints.get("canvas_width")
            or defaults.get("width"),
            "expected_height": goal.constraints.get("canvas_height")
            or defaults.get("height"),
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
    def text2img2video_length(goal: GoalRequest) -> int | None:
        duration = goal.constraints.get("duration_override_seconds")
        if duration in {None, ""}:
            if str(goal.media_type or "").strip().lower() == "text2img2video":
                duration = goal.duration_seconds
            else:
                return None
        frame_rate = max(1.0, float(goal.constraints.get("video_frame_rate") or 24))
        frames = round(int(duration) * frame_rate)
        if str(goal.media_type or "").strip().lower() == "text2img2video":
            try:
                safe_limit = int(goal.constraints.get("max_i2v_frames") or 240)
            except (TypeError, ValueError) as exc:
                raise ValueError("max_i2v_frames must be a positive integer") from exc
            if safe_limit <= 0:
                raise ValueError("max_i2v_frames must be a positive integer")
            minimum_frames = round(
                MIN_VIDEO_DURATION_SECONDS
                * frame_rate
            )
            frames = min(max(frames, minimum_frames), max(safe_limit, minimum_frames))
        return frames

    @staticmethod
    def video_speed_config(goal: GoalRequest) -> dict[str, object]:
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
            raise ValueError(
                "video_speed.factor must be a number greater than zero"
            ) from exc
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(
                "video_speed.factor must be a finite number greater than zero"
            )
        return {"enabled": enabled, "factor": factor}

    @staticmethod
    def nvidia_vfx_postprocess_enabled(goal: GoalRequest) -> bool:
        enabled = goal.constraints.get("enable_nvidia_vfx_postprocess", True)
        if not isinstance(enabled, bool):
            raise ValueError("enable_nvidia_vfx_postprocess must be a boolean")
        return enabled

    @classmethod
    def scaled_video_qa_inputs(
        cls,
        goal: GoalRequest,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        result = dict(inputs)
        speed = cls.video_speed_config(goal)
        target_duration = result.get("target_duration")
        if target_duration in {None, ""} and goal.duration_seconds > 0:
            target_duration = goal.duration_seconds
        if (
            target_duration not in {None, ""}
            and str(goal.media_type or "").strip().lower() == "text2img2video"
        ):
            source_length = cls.text2img2video_length(goal)
            frame_rate = max(
                1.0, float(goal.constraints.get("video_frame_rate") or 24)
            )
            if source_length is not None:
                target_duration = min(
                    float(target_duration), float(source_length) / frame_rate
                )
        if bool(speed["enabled"]):
            if target_duration not in {None, ""}:
                target_duration = float(target_duration) / float(speed["factor"])
        if target_duration not in {None, ""}:
            result["target_duration"] = float(target_duration)
        result["minimum_duration"] = (
            min(MIN_VIDEO_DURATION_SECONDS, float(target_duration))
            if bool(speed["enabled"]) and target_duration not in {None, ""}
            else MIN_VIDEO_DURATION_SECONDS
        )
        if cls.nvidia_vfx_postprocess_enabled(goal):
            for key in ("expected_width", "expected_height"):
                value = result.get(key)
                if value not in {None, ""}:
                    result[key] = int(value) * 2
            expected_fps = result.get("expected_fps")
            if expected_fps not in {None, ""}:
                result["expected_fps"] = float(expected_fps) * 2
        return result

    def append_video_speed_node(
        self,
        goal: GoalRequest,
        nodes: list[ExecutionNode],
        *,
        source_node: str,
        node_id: str,
        retry: bool = False,
    ) -> str:
        speed = self.video_speed_config(goal)
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

    def append_nvidia_vfx_postprocess_node(
        self,
        goal: GoalRequest,
        nodes: list[ExecutionNode],
        *,
        source_node: str,
        node_id: str,
        retry: bool = False,
    ) -> str:
        if not self.nvidia_vfx_postprocess_enabled(goal):
            return source_node
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                skill_name="media.video.nvidia_vfx_postprocess",
                inputs={},
                depends_on=[source_node],
                tags=["package", "video", "postprocess", "nvidia-vfx"]
                + (["retry"] if retry else []),
                tool_name="comfy.nvidia_vfx_postprocess",
                stage="package",
            )
        )
        return node_id

    @staticmethod
    def append_video_canvas_node(
        nodes: list[ExecutionNode],
        *,
        source_node: str,
        node_id: str,
        width: int,
        height: int,
        retry: bool = False,
    ) -> str:
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                skill_name="media.video.normalize_canvas",
                inputs={
                    "target_width": width,
                    "target_height": height,
                    "background": "#f8f5ef",
                },
                depends_on=[source_node],
                tags=["package", "video", "canvas"] + (["retry"] if retry else []),
                tool_name="media.normalize_video_canvas",
                stage="package",
            )
        )
        return node_id

    @staticmethod
    def native_h3_canvas_dimensions(goal: GoalRequest) -> tuple[int, int]:
        recipe = dict(goal.constraints.get("native_h3_recipe") or {})
        return (
            int(goal.constraints.get("canvas_width") or recipe.get("width") or 608),
            int(goal.constraints.get("canvas_height") or recipe.get("height") or 352),
        )

    @classmethod
    def native_h3_needs_canvas_normalization(cls, goal: GoalRequest) -> bool:
        width, height = cls.native_h3_canvas_dimensions(goal)
        return width % 32 != 0 or height % 32 != 0

    @classmethod
    def append_native_h3_canvas_node(
        cls,
        goal: GoalRequest,
        nodes: list[ExecutionNode],
        *,
        source_node: str,
        node_id: str,
    ) -> str:
        width, height = cls.native_h3_canvas_dimensions(goal)
        nodes.append(
            ExecutionNode(
                node_id=node_id,
                skill_name="media.video.normalize_canvas",
                inputs={
                    "target_width": width,
                    "target_height": height,
                    "background": "#15151f",
                },
                depends_on=[source_node],
                tags=["package", "video", "canvas", "native-h3"],
                tool_name="media.normalize_video_canvas",
                stage="package",
            )
        )
        return node_id

    @staticmethod
    def native_h3_storyboard_path(goal: GoalRequest, media_type: str) -> str:
        storyboard_path = str(
            goal.constraints.get("native_h3_storyboard_path")
            or goal.constraints.get("storyboard_path")
            or ""
        )
        if not storyboard_path:
            raise ValueError(
                f"{media_type} requires native_h3_storyboard_path or storyboard_path"
            )
        return storyboard_path

    @staticmethod
    def reference_video_source(goal: GoalRequest) -> str:
        return str(goal.constraints.get("reference_video_source") or "").strip()

    @classmethod
    def reference_video_analysis_node(cls, goal: GoalRequest) -> ExecutionNode | None:
        source = cls.reference_video_source(goal)
        if not source:
            return None
        try:
            max_keyframes = int(
                goal.constraints.get("reference_video_max_keyframes") or 12
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "reference_video_max_keyframes must be an integer between 2 and 20"
            ) from exc
        if not 2 <= max_keyframes <= 20:
            raise ValueError(
                "reference_video_max_keyframes must be an integer between 2 and 20"
            )
        depth = (
            str(goal.constraints.get("reference_video_depth") or "standard")
            .strip()
            .lower()
        )
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
    def reference_video_metadata(cls, goal: GoalRequest) -> dict[str, object] | None:
        source = cls.reference_video_source(goal)
        if not source:
            return None
        return {
            "source": source,
            "analysis_node": "reference-video-analysis",
            "analysis_depth": str(
                goal.constraints.get("reference_video_depth") or "standard"
            ),
            "max_keyframes": int(
                goal.constraints.get("reference_video_max_keyframes") or 12
            ),
            "policy": "borrow_timing_framing_motion_grammar_not_source_assets",
        }

    def native_h3_finalize_nodes(
        self,
        goal: GoalRequest,
        *,
        tags: list[str],
        include_keyframe_source_check: bool,
        frame_source_check_node_id: str = "native-keyframe-source-check",
        qa_inputs: dict[str, object] | None = None,
    ) -> list[ExecutionNode]:
        nodes: list[ExecutionNode] = []
        video_node = self.append_video_speed_node(
            goal,
            nodes,
            source_node="native-h3-render",
            node_id="native-h3-speed",
        )
        if self.native_h3_needs_canvas_normalization(goal):
            video_node = self.append_native_h3_canvas_node(
                goal,
                nodes,
                source_node=video_node,
                node_id="native-h3-canvas",
            )
        video_node = self.append_nvidia_vfx_postprocess_node(
            goal,
            nodes,
            source_node=video_node,
            node_id="native-h3-nvidia-vfx",
        )
        final_qa_inputs = {
            "require_audio": True,
            "require_stereo_audio": True,
            "analyze_audio": True,
            **(qa_inputs or {}),
        }
        expected_width, expected_height = self.native_h3_canvas_dimensions(goal)
        final_qa_inputs.setdefault("expected_width", expected_width)
        final_qa_inputs.setdefault("expected_height", expected_height)
        if final_qa_inputs.get("expected_fps") in {None, ""}:
            final_qa_inputs["expected_fps"] = float(
                goal.constraints.get("native_h3_frame_rate") or 24
            )
        final_qa_inputs = self.scaled_video_qa_inputs(goal, final_qa_inputs)
        final_qa_inputs.update(
            {"render_node": "native-h3-render", "video_node": video_node}
        )
        qa_tags = ["media-inspection", "manual-review", *tags]
        preview_tags = ["preview", *tags]
        package_dependencies = [
            "native-h3-render",
            video_node,
            "native-h3-qa",
            "native-h3-preview",
        ]
        if include_keyframe_source_check:
            package_dependencies.append(frame_source_check_node_id)
        nodes.extend(
            [
                ExecutionNode(
                    node_id="native-h3-qa",
                    skill_name="longvideo.qa_native_h3",
                    inputs={
                        "mode": "discord_review_evidence_only",
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

    @staticmethod
    def review_loop_enabled(goal: GoalRequest) -> bool:
        if bool(goal.constraints.get("enable_review_loop", False)):
            return True
        review_notes = str(goal.constraints.get("review_notes", "") or "").strip()
        return bool(review_notes)

    @staticmethod
    def review_selection_limit(goal: GoalRequest, default: int = 3) -> int:
        return int(
            goal.constraints.get("selection_limit")
            or goal.constraints.get("review_selection_limit")
            or default
        )

    @staticmethod
    def stage_review_enabled(goal: GoalRequest) -> bool:
        return bool(goal.constraints.get("enable_stage_review", False))

    @staticmethod
    def pre_video_review_enabled(goal: GoalRequest) -> bool:
        # Pure T2V routes intentionally keep their prompt-only conditioning.
        # A caller may still request an explicit stage review through
        # ``enable_stage_review``; the shared automatic image gate must not
        # silently change a T2V strategy into I2V.
        if goal.media_type in {"text2video", "native_h3_t2v_story"}:
            return False
        return bool(goal.constraints.get("pre_video_review_enabled", False))

    @staticmethod
    def pre_video_review_requires_human(goal: GoalRequest) -> bool:
        return bool(
            goal.constraints.get(
                "pre_video_review_require_human",
                PlanningContext.pre_video_review_enabled(goal),
            )
        )

    @staticmethod
    def pre_video_candidate_count(goal: GoalRequest, default: int = 6) -> int:
        value = goal.constraints.get("pre_video_candidate_count")
        return max(1, int(value or default))
